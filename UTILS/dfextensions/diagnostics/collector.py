#!/usr/bin/env python3
"""
collector.py - PHASE_13_74_ADF v8 D1.6: process / user / workload sampling.

Long-running sampler started by dfx_host_diagnostics.sh in sampling mode
(division per v8 line 669: the shell script remains the public collector
command; this module owns schema-aware process/accounting logic).

Writes into the bundle directory, long format, fixed headers:
  process_samples.csv   one row per SELECTED process per timestamp
  user_samples.csv      one row per visible user per timestamp (ALL processes)
  workload_rollup.csv   mutually exclusive partition per timestamp

Selection per sample = union of: top Nall by CPU, top Nall by RSS,
top Nuser by CPU (current user), top Nuser by RSS (current user),
plus EVERY target-job process regardless of rank, plus D-state processes
(bounded by --max-d-state-rows, target-job first).

Normative semantics implemented here:
  * rank fields are EMPTY when the process is not in that ranking -
    never 0 or a sentinel (architect anchor, v8 R8-1);
  * first scan is a silent warm-up baseline - no CPU ranks until a valid
    previous sample exists (v8 section 4.2);
  * identity = (boot_id, pid, starttime): PID-reuse safe (v8 section 0.4);
  * per-user totals aggregate ALL visible processes, not only top-N;
  * unreadable fields are counted per-sample coverage states, never
    silent zeros; per-user IO complete only if every process readable;
  * redaction (shareable default): other users/processes appear as stable
    bundle-local tokens (u_xxxx / [other]); --raw keeps real names locally;
  * CPU normalization: 100% = one logical CPU (v8 section 4.4).

Fixture injection: PROC_ROOT, CLK_TCK_OVERRIDE, DFX_UID_OVERRIDE,
DFX_BOOTID_OVERRIDE (same discipline as the bash collector, C-8).
"""
from __future__ import annotations
import argparse
import hashlib
import os
import pwd
import signal
import sys
import time
from pathlib import Path

PROC = Path(os.environ.get("PROC_ROOT", "/proc"))
CLK = float(os.environ.get("CLK_TCK_OVERRIDE") or os.sysconf("SC_CLK_TCK"))

PROC_HEADER = ("ts,run_id,tenant,proc,pid,starttime,is_target_job,state,"
               "cpu_pct,rss_kb,virt_kb,shr_kb,cpu_time_s,"
               "rank_cpu_all,rank_mem_all,rank_cpu_user,rank_mem_user,selected_reason")
USER_HEADER = ("ts,run_id,tenant,is_current_user,process_count,running_count,"
               "blocked_count,cpu_cores,cpu_time_s,rss_kb,virt_kb,"
               "io_read_bytes,io_write_bytes,io_coverage_state")
ROLL_HEADER = ("ts,run_id,scope,process_count,running_count,blocked_count,"
               "cpu_cores,rss_kb,io_read_bytes,io_write_bytes,"
               "visible_process_count,readable_process_count,permission_denied_count,"
               "vanished_during_scan_count,new_process_count,accounting_lower_bound")
SCOPES = ("target_job", "current_user_non_job", "other_visible_workloads",
          "kernel_or_system", "unknown_or_inaccessible")


def read_boot_id():
    ov = os.environ.get("DFX_BOOTID_OVERRIDE")
    if ov:
        return ov
    try:
        return (PROC / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown-boot"


def uid_name(uid, cache={}):
    if uid not in cache:
        try:
            cache[uid] = pwd.getpwuid(uid).pw_name
        except KeyError:
            cache[uid] = f"uid{uid}"
    return cache[uid]


class Redactor:
    """v8 938-940 (CRR-2): SHAREABLE mode emits bundle-local random tokens
    ONLY - no usernames (not even the caller's own), no process names, no
    PID/starttime. Analytical meaning survives via is_current_user /
    is_target_job flags and rank columns. RAW/local mode keeps real identity."""
    def __init__(self, own_user, raw=False, salt=None):
        self.own = own_user
        self.raw = raw
        self.salt = salt or os.urandom(8).hex()

    def user(self, name):
        if self.raw:
            return name
        h = hashlib.sha256((self.salt + name).encode()).hexdigest()[:8]
        return f"u_{h}"

    def proc(self, owner, comm, pid=0, starttime=0):
        if self.raw:
            return comm
        h = hashlib.sha256(f"{self.salt}p{pid}s{starttime}".encode()).hexdigest()[:8]
        return f"p_{h}"

    def pid_field(self, pid):
        return pid if self.raw else ""

    def start_field(self, st):
        return st if self.raw else ""


def scan_stage1(target_pids):
    """Inexpensive pass: /proc/<pid>/stat + statm + status-uid for EVERY visible
    process. Returns dict keyed by (pid, starttime). Two-stage rule v8 4.2/5."""
    procs = {}
    counters = {"visible": 0, "readable": 0, "permission_denied": 0, "vanished": 0}
    for p in PROC.iterdir():
        if not p.name.isdigit():
            continue
        counters["visible"] += 1
        pid = int(p.name)
        try:
            stat = (p / "stat").read_text()
            # comm may contain spaces/parens: split around last ')'
            lpar = stat.index("("); rpar = stat.rindex(")")
            comm = stat[lpar + 1:rpar]
            f = stat[rpar + 2:].split()
            state = f[0]
            utime, stime = int(f[11]), int(f[12])
            starttime = int(f[19])
            vsize_kb = int(f[20]) // 1024
            rss_kb = int(f[21]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
            shr_kb = ""
            try:
                sm = (p / "statm").read_text().split()
                shr_kb = int(sm[2]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
            except (OSError, IndexError, ValueError):
                pass
            owner_uid = None
            try:
                for line in (p / "status").read_text().splitlines():
                    if line.startswith("Uid:"):
                        owner_uid = int(line.split()[1]); break
            except OSError:
                pass
            if owner_uid is None:
                counters["permission_denied"] += 1
                continue
            counters["readable"] += 1
            procs[(pid, starttime)] = dict(
                pid=pid, starttime=starttime, comm=comm, state=state,
                cpu_ticks=utime + stime, rss_kb=rss_kb, virt_kb=vsize_kb,
                shr_kb=shr_kb, uid=owner_uid,
                is_target=(pid in target_pids))
        except (OSError, ValueError, IndexError):
            counters["vanished"] += 1
    return procs, counters


def scan_stage2_io(entry):
    """Expensive pass, selected + target rows only: /proc/<pid>/io (own procs)."""
    try:
        out = {}
        for line in (PROC / str(entry["pid"]) / "io").read_text().splitlines():
            k, v = line.split(":")
            out[k.strip()] = int(v)
        return out.get("read_bytes"), out.get("write_bytes")
    except OSError:
        return None, None


def target_pid_set(target_pid_file, job_cgroup):
    """Target-job discovery: cgroup procs where given, else PID tree from file."""
    pids = set()
    if job_cgroup:
        try:
            for tok in Path(job_cgroup, "cgroup.procs").read_text().split():
                pids.add(int(tok))
            return pids
        except OSError:
            pass  # fall through to tree - DEGRADED handled by caller manifest
    root = None
    if target_pid_file:
        try:
            root = int(Path(target_pid_file).read_text().strip())
        except (OSError, ValueError):
            return pids
    if root is None:
        return pids
    children = {}
    for p in PROC.iterdir():
        if p.name.isdigit():
            try:
                f = (p / "stat").read_text()
                ppid = int(f[f.rindex(")") + 2:].split()[1])
                children.setdefault(ppid, []).append(int(p.name))
            except (OSError, ValueError, IndexError):
                continue
    stack = [root]
    while stack:
        n = stack.pop()
        if n in pids:
            continue
        pids.add(n)
        stack.extend(children.get(n, []))
    return pids


def rank_map(entries, key, n):
    """(pid,starttime)->rank (1-based) for top-n by key; stable tie-break."""
    order = sorted(entries, key=lambda e: (-(e[key] or 0), e["pid"], e["starttime"]))
    return {(e["pid"], e["starttime"]): i + 1 for i, e in enumerate(order[:n])}


def sample_once(prev, dt, args, red, own_uid, run_id, ts, boot_id, out):
    procs, cov = scan_stage1(target_pid_set(args.target_pid_file, args.job_cgroup))
    new_count = sum(1 for k in procs if prev is not None and k not in prev)
    warm = prev is None
    # per-process CPU over the interval (v8 4.4: 100% = one logical CPU)
    for k, e in procs.items():
        if warm or k not in prev:
            e["cpu_pct"] = None
        else:
            e["cpu_pct"] = round(100.0 * (e["cpu_ticks"] - prev[k]["cpu_ticks"]) / CLK / dt, 2)
        e["owner"] = uid_name(e["uid"])
    entries = list(procs.values())

    # ---- selection: union of four rankings + target + D-state (v8 D1.6) ----
    sel = {}
    def add(k, e, reason, rankfield=None, rank=None):
        row = sel.setdefault(k, dict(e, reasons=set(),
                                     rank_cpu_all="", rank_mem_all="",
                                     rank_cpu_user="", rank_mem_user=""))
        row["reasons"].add(reason)
        if rankfield:
            row[rankfield] = rank
    # memory rankings are instantaneous - valid from the FIRST scan;
    # CPU rankings need a previous sample (v8 4.2 warm-up applies to rates only)
    for field, key, scope_own, n in (
            ("rank_cpu_all", "cpu_pct", False, args.nall),
            ("rank_mem_all", "rss_kb", False, args.nall),
            ("rank_cpu_user", "cpu_pct", True, args.nuser),
            ("rank_mem_user", "rss_kb", True, args.nuser)):
        if key == "cpu_pct" and warm:
            continue
        pool = [e for e in entries if e["uid"] == own_uid] if scope_own else entries
        if key == "cpu_pct":
            pool = [e for e in pool if e["cpu_pct"] is not None]
        for k2, r in rank_map(pool, key, n).items():
            add(k2, procs[k2], field, field, r)
    for k2, e in procs.items():
        if e["is_target"]:
            add(k2, e, "target_job")
    dstate = sorted((e for e in entries if e["state"] == "D"),
                    key=lambda e: (not e["is_target"], -(e["rss_kb"] or 0), e["pid"]))
    for e in dstate[:args.max_d_state_rows]:
        add((e["pid"], e["starttime"]), e, "d_state")

    # stage-2 expensive reads: selected + target only
    for k2, row in sel.items():
        rb, wb = scan_stage2_io(row)
        row["io_read"], row["io_write"] = rb, wb

    with open(out / "process_samples.csv", "a") as f:
        for row in sorted(sel.values(), key=lambda r: (-(r["cpu_pct"] or 0), r["pid"])):
            f.write(",".join(str(x) for x in (
                ts, run_id, red.user(row["owner"]),
                red.proc(row["owner"], row["comm"], row["pid"], row["starttime"]),
                red.pid_field(row["pid"]), red.start_field(row["starttime"]),
                int(row["is_target"]), row["state"],
                "" if row["cpu_pct"] is None else row["cpu_pct"],
                row["rss_kb"], row["virt_kb"], row["shr_kb"],
                round(row["cpu_ticks"] / CLK, 2),
                row["rank_cpu_all"], row["rank_mem_all"],
                row["rank_cpu_user"], row["rank_mem_user"],
                "|".join(sorted(row["reasons"])))) + "\n")

    # ---- per-user totals over ALL visible processes (v8 0.4) ----
    users = {}
    for e in entries:
        u = users.setdefault(e["owner"], dict(n=0, run=0, blk=0, ticks=0, cores=0.0,
                                              rss=0, virt=0, io_r=0, io_w=0,
                                              io_known=0))
        u["n"] += 1
        u["run"] += e["state"] == "R"
        u["blk"] += e["state"] == "D"
        u["ticks"] += e["cpu_ticks"]
        if e["cpu_pct"] is not None:
            u["cores"] += e["cpu_pct"] / 100.0
        u["rss"] += e["rss_kb"] or 0
        u["virt"] += e["virt_kb"] or 0
        k2 = (e["pid"], e["starttime"])
        if k2 in sel and sel[k2]["io_read"] is not None:
            u["io_r"] += sel[k2]["io_read"]; u["io_w"] += sel[k2]["io_write"]
            u["io_known"] += 1
    with open(out / "user_samples.csv", "a") as f:
        for name, u in sorted(users.items()):
            io_state = ("complete" if u["io_known"] == u["n"] and u["n"] > 0
                        else ("partial" if u["io_known"] else "unavailable"))
            f.write(",".join(str(x) for x in (
                ts, run_id, red.user(name), int(name == red.own), u["n"], u["run"],
                u["blk"], round(u["cores"], 3), round(u["ticks"] / CLK, 2), u["rss"],
                u["virt"],
                u["io_r"] if u["io_known"] else "", u["io_w"] if u["io_known"] else "",
                io_state)) + "\n")

    # ---- mutually exclusive workload partition (v8 4.1) ----
    part = {s: dict(n=0, run=0, blk=0, cores=0.0, rss=0, io_r=0, io_w=0) for s in SCOPES}
    for e in entries:
        if e["is_target"]:
            s = "target_job"
        elif e["uid"] == own_uid:
            s = "current_user_non_job"
        elif e["virt_kb"] == 0:
            s = "kernel_or_system"          # kthreads have no VM image (vsize==0), documented
        else:
            s = "other_visible_workloads"
        d = part[s]
        d["n"] += 1; d["run"] += e["state"] == "R"; d["blk"] += e["state"] == "D"
        if e["cpu_pct"] is not None:
            d["cores"] += e["cpu_pct"] / 100.0
        d["rss"] += e["rss_kb"] or 0
        k2 = (e["pid"], e["starttime"])
        if k2 in sel and sel[k2].get("io_read") is not None:
            d["io_r"] += sel[k2]["io_read"]; d["io_w"] += sel[k2]["io_write"]
    part["unknown_or_inaccessible"]["n"] = cov["permission_denied"] + cov["vanished"]
    with open(out / "workload_rollup.csv", "a") as f:
        for s in SCOPES:
            d = part[s]
            f.write(",".join(str(x) for x in (
                ts, run_id, s, d["n"], d["run"], d["blk"], round(d["cores"], 3),
                d["rss"], d["io_r"] or "", d["io_w"] or "",
                cov["visible"], cov["readable"], cov["permission_denied"],
                cov["vanished"], new_count, 1)) + "\n")
    return procs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-o", "--out", required=True, help="bundle directory")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--nsamples", type=int, default=0, help="0 = until SIGTERM")
    ap.add_argument("--nall", type=int, default=20)
    ap.add_argument("--nuser", type=int, default=20)
    ap.add_argument("--max-d-state-rows", type=int, default=20)
    ap.add_argument("--target-pid-file", default=None)
    ap.add_argument("--job-cgroup", default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--token-salt", default=None,
                    help="per-bundle token salt from the bash collector (P1-1: "
                         "one namespace so the same user has one token bundle-wide)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    own_uid = int(os.environ.get("DFX_UID_OVERRIDE", os.getuid()))
    red = Redactor(uid_name(own_uid), raw=args.raw, salt=args.token_salt)
    boot_id = read_boot_id()
    for name, hdr in (("process_samples.csv", PROC_HEADER),
                      ("user_samples.csv", USER_HEADER),
                      ("workload_rollup.csv", ROLL_HEADER)):
        p = out / name
        if not p.exists():
            p.write_text(hdr + "\n")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    prev, last_t = None, None
    i = 0
    # loop guarantees AT LEAST ONE scan even if SIGTERM arrived during startup
    # (slow venv import can exceed a short host run - 2026-07-17 alma2 race)
    while True:
        t0 = time.monotonic()
        ts = int(time.time())
        dt = (t0 - last_t) if last_t is not None else args.interval
        prev = sample_once(prev, dt, args, red, own_uid, args.run_id, ts, boot_id, out)
        last_t = t0
        i += 1
        if i == 1:
            (out / ".sampler_ready").touch()   # readiness marker: shell waits for
                                               # this before TERM (startup race fix)
        if stop["flag"] or (args.nsamples and i >= args.nsamples):
            break
        sleep_left = args.interval - (time.monotonic() - t0)
        if sleep_left > 0 and not stop["flag"]:
            time.sleep(sleep_left)
    return 0


if __name__ == "__main__":
    sys.exit(main())

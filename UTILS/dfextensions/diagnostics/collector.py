#!/usr/bin/env python3
"""
collector.py - PHASE_13_74_ADF v8 D1.6: process / user / workload sampling.

Long-running sampler started by dfx_host_diagnostics.sh in sampling mode
(division per v8 line 669: the shell script remains the public collector
command; this module owns schema-aware process/accounting logic).

Writes into the bundle directory, long format, fixed headers:
  process_samples.tsv   one row per SELECTED process per timestamp
  process_catalog.tsv   one row per observed selected/target process instance
  user_samples.tsv      one row per visible user per timestamp (ALL processes)
  workload_rollup.tsv   mutually exclusive partition per timestamp

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
  * PHASE_13_75 full-fidelity default: real users/processes/PID identity is
    preserved in canonical evidence; --raw remains a compatibility flag;
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
TABLE_TSV = os.environ.get("DFX_TABLE_FORMAT", "tsv").lower() == "tsv"
TABLE_SEP = "\t" if TABLE_TSV else ","
TABLE_EXT = "tsv" if TABLE_TSV else "csv"

PROC_HEADER = ("ts,run_id,tenant,proc,pid,starttime,is_target_job,state,"
               "cpu_pct,rss_kb,virt_kb,shr_kb,cpu_time_s,"
               "rank_cpu_all,rank_mem_all,rank_cpu_user,rank_mem_user,selected_reason")
CATALOG_HEADER = ("run_id,process_id,host_boot_id,user_name,uid,pid,starttime,ppid,"
                  "parent_process_id,command,exe,exe_state,cmdline,cmdline_state,"
                  "first_seen_ts,master_pid,master_process_id,is_master,is_target_job")
USER_HEADER = ("ts,run_id,tenant,is_current_user,process_count,running_count,"
               "blocked_count,cpu_cores,cpu_time_s,rss_kb,virt_kb,"
               "io_read_bytes,io_write_bytes,io_coverage_state")
ROLL_HEADER = ("ts,run_id,scope,process_count,running_count,blocked_count,"
               "cpu_cores,rss_kb,io_read_bytes,io_write_bytes,"
               "visible_process_count,readable_process_count,permission_denied_count,"
               "vanished_during_scan_count,new_process_count,accounting_lower_bound")
SCOPES = ("target_job", "current_user_non_job", "other_visible_workloads",
          "kernel_or_system", "unknown_or_inaccessible")


def process_id(boot_id, pid, starttime):
    """Stable process-instance key; PID reuse is separated by kernel starttime."""
    raw = f"{boot_id}\0{pid}\0{starttime}".encode("utf-8", "surrogateescape")
    return "pr_" + hashlib.sha256(raw).hexdigest()[:24]


def _tsv_field(value):
    r"""PHASE_13_75 TSV field: unquoted \N=NULL; quoted "\N" is text."""
    if value is None:
        return r"\N"
    text = str(value)
    must_quote = text == r"\N" or any(c in text for c in ("\t", "\n", "\r", '"'))
    if must_quote:
        return '"' + text.replace('"', '""') + '"'
    return text


def _write_tsv_row(path, values):
    """Append one row using the existing canonical PHASE_13_75 TSV field writer."""
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write("\t".join(_tsv_field(v) for v in values) + "\n")


def _write_table_row(path, values):
    """Append one canonical table row while preserving explicit CSV compatibility."""
    if TABLE_TSV:
        _write_tsv_row(path, values)
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(",".join(str(v) for v in values) + "\n")


def _rewrite_process_catalog(path, records):
    """Atomically rewrite the one-row-per-process-instance descriptive catalog."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(CATALOG_HEADER.replace(",", "\t") + "\n")
        for rec in sorted(records.values(), key=lambda r: (r["first_seen_ts"], r["pid"], r["starttime"])):
            f.write("\t".join(_tsv_field(v) for v in rec["values"]) + "\n")
    os.replace(tmp, path)


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

    def uid_token(self, uid):
        """ONE executable algorithm shared with the bash collector [GPT24
        blocker]: token(uid) = u_ + sha256(salt + "uid:" + uid)[:8].
        Canonical identity is the NUMERIC UID - ps truncates usernames over
        8 chars (averylongname -> averylo+), which made the same person hash to
        two tokens in one bundle; a uid cannot truncate."""
        if self.raw:
            return uid_name(uid)
        h = hashlib.sha256(f"{self.salt}uid:{uid}".encode()).hexdigest()[:8]
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
            ppid = int(f[1])
            procs[(pid, starttime)] = dict(
                pid=pid, starttime=starttime, ppid=ppid, comm=comm, state=state,
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


def read_master_pid(target_pid_file):
    """Return the explicitly supplied job-root PID, or None while not registered."""
    if not target_pid_file:
        return None
    try:
        return int(Path(target_pid_file).read_text().strip())
    except (OSError, ValueError):
        return None


def read_pid_starttime(pid):
    """Return Linux /proc starttime for *pid*, or None if the instance vanished."""
    if pid is None:
        return None
    try:
        stat = (PROC / str(pid) / "stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def resolve_master_pid(target_pid_file, master_state):
    """Pin the explicit master to one (pid,starttime) process instance.

    A pidfile contains only a numeric PID.  Once that PID has been observed,
    do not silently rebind the job root if Linux later reuses the number for a
    different process.  A deliberate change to a different pidfile PID starts
    a new pin.
    """
    requested = read_master_pid(target_pid_file)
    if requested is None:
        return None

    if master_state.get("pid") != requested:
        master_state.clear()
        master_state["pid"] = requested

    current_starttime = read_pid_starttime(requested)
    if current_starttime is None:
        return None

    pinned = master_state.get("starttime")
    if pinned is None:
        master_state["starttime"] = current_starttime
        return requested
    if current_starttime != pinned:
        return None
    return requested


def _read_process_text(pid):
    """Read descriptive /proc fields once for a catalog row, with evidence state."""
    p = PROC / str(pid)
    try:
        exe = os.readlink(p / "exe")
        exe_state = "available"
    except PermissionError:
        exe, exe_state = None, "permission_denied"
    except OSError:
        exe, exe_state = None, "unavailable"
    try:
        raw = (p / "cmdline").read_bytes()
        parts = [x for x in raw.split(b"\0") if x]
        cmdline = " ".join(x.decode("utf-8", "replace") for x in parts)
        cmdline_state = "available" if parts else "unavailable"
        if not parts:
            cmdline = None
    except PermissionError:
        cmdline, cmdline_state = None, "permission_denied"
    except OSError:
        cmdline, cmdline_state = None, "unavailable"
    return exe, exe_state, cmdline, cmdline_state


def target_pid_set(target_pid_file, job_cgroup, root=None):
    """Target-job discovery: cgroup procs where given, else PID tree from file."""
    pids = set()
    if job_cgroup:
        try:
            for tok in Path(job_cgroup, "cgroup.procs").read_text().split():
                pids.add(int(tok))
            return pids
        except OSError:
            pass  # fall through to tree - DEGRADED handled by caller manifest
    if root is None:
        root = read_master_pid(target_pid_file)
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


def sample_once(prev, dt, args, red, own_uid, run_id, ts, boot_id, out,
                catalog_seen=None, master_state=None):
    if catalog_seen is None:
        catalog_seen = set()  # compatibility for direct/unit callers; main() passes persistent set
    if master_state is None:
        master_state = {}
    master_pid = resolve_master_pid(args.target_pid_file, master_state)
    # The root has already been resolved to a pinned process instance.  Never
    # let target_pid_set reread a same-number reused PID from the pidfile.
    target_pids = target_pid_set(None, args.job_cgroup, root=master_pid)
    procs, cov = scan_stage1(target_pids)
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

    # Descriptive identity/hierarchy is one row per process instance.  In the
    # normal long-running sampler catalog_seen is a dict so an exec() within
    # the same (pid,starttime) refreshes COMMAND/EXE/CMDLINE without creating
    # a duplicate process_id.  Direct legacy/unit callers may still pass a set.
    pid_map = {e["pid"]: e for e in entries}
    master_entry = pid_map.get(master_pid) if master_pid is not None else None
    master_process_id = (process_id(boot_id, master_entry["pid"], master_entry["starttime"])
                         if master_entry else None)
    catalog_ready = (not args.target_pid_file) or (master_pid is not None)
    catalog_records = catalog_seen if isinstance(catalog_seen, dict) else None
    catalog_dirty = False
    for row in (sorted(sel.values(), key=lambda r: (r["pid"], r["starttime"]))
                if catalog_ready else []):
        proc_id = process_id(boot_id, row["pid"], row["starttime"])
        existing = catalog_records.get(proc_id) if catalog_records is not None else None
        if catalog_records is None and proc_id in catalog_seen:
            continue
        parent = pid_map.get(row["ppid"])
        parent_id = (process_id(boot_id, parent["pid"], parent["starttime"])
                     if parent else None)
        # Mutable descriptive metadata can change across exec() even when the
        # Linux comm string stays unchanged. Re-read it on every selected
        # observation; rewrite the one catalog row only when current metadata
        # changes. process_id and first_seen_ts remain immutable.
        exe, exe_state, cmdline, cmdline_state = _read_process_text(row["pid"])
        row_master_pid = master_pid if row["is_target"] else None
        row_master_process_id = master_process_id if row["is_target"] else None
        first_seen_ts = existing["first_seen_ts"] if existing is not None else ts
        values = (
            run_id, proc_id, boot_id, uid_name(row["uid"]), row["uid"],
            row["pid"], row["starttime"], row["ppid"], parent_id, row["comm"],
            exe, exe_state, cmdline, cmdline_state, first_seen_ts, row_master_pid,
            row_master_process_id, int(row["is_target"] and row["pid"] == master_pid),
            int(row["is_target"]))
        if catalog_records is not None:
            if existing is not None and existing["values"] == values:
                continue
            catalog_records[proc_id] = {
                "values": values,
                "first_seen_ts": first_seen_ts, "pid": row["pid"],
                "starttime": row["starttime"],
            }
            catalog_dirty = True
        else:
            _write_tsv_row(out / "process_catalog.tsv", values)
            catalog_seen.add(proc_id)
    if catalog_dirty:
        _rewrite_process_catalog(out / "process_catalog.tsv", catalog_records)

    for row in sorted(sel.values(), key=lambda r: (-(r["cpu_pct"] or 0), r["pid"])):
        _write_table_row(out / f"process_samples.{TABLE_EXT}", (
            ts, run_id, red.uid_token(row["uid"]),
            red.proc(row["owner"], row["comm"], row["pid"], row["starttime"]),
            red.pid_field(row["pid"]), red.start_field(row["starttime"]),
            int(row["is_target"]), row["state"],
            "" if row["cpu_pct"] is None else row["cpu_pct"],
            row["rss_kb"], row["virt_kb"], row["shr_kb"],
            round(row["cpu_ticks"] / CLK, 2),
            row["rank_cpu_all"], row["rank_mem_all"],
            row["rank_cpu_user"], row["rank_mem_user"],
            "|".join(sorted(row["reasons"]))))

    # ---- per-user totals over ALL visible processes (v8 0.4) ----
    users = {}
    for e in entries:
        u = users.setdefault(e["uid"], dict(n=0, run=0, blk=0, ticks=0, cores=0.0,
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
    for uid, u in sorted(users.items()):
        io_state = ("complete" if u["io_known"] == u["n"] and u["n"] > 0
                    else ("partial" if u["io_known"] else "unavailable"))
        _write_table_row(out / f"user_samples.{TABLE_EXT}", (
            ts, run_id, red.uid_token(uid), int(uid == own_uid), u["n"], u["run"],
            u["blk"], round(u["cores"], 3), round(u["ticks"] / CLK, 2), u["rss"],
            u["virt"],
            u["io_r"] if u["io_known"] else "", u["io_w"] if u["io_known"] else "",
            io_state))

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
    for scope_name in SCOPES:
        d = part[scope_name]
        _write_table_row(out / f"workload_rollup.{TABLE_EXT}", (
            ts, run_id, scope_name, d["n"], d["run"], d["blk"], round(d["cores"], 3),
            d["rss"], d["io_r"] or "", d["io_w"] or "",
            cov["visible"], cov["readable"], cov["permission_denied"],
            cov["vanished"], new_count, 1))
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
    ap.add_argument("--raw", action="store_true", default=True,
                    help="full-fidelity identities (PHASE_13_75 default; compatibility flag)")
    ap.add_argument("--token-salt", default=None,
                    help="per-bundle token salt from the bash collector (P1-1: "
                         "one namespace so the same user has one token bundle-wide)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    own_uid = int(os.environ.get("DFX_UID_OVERRIDE", os.getuid()))
    red = Redactor(uid_name(own_uid), raw=args.raw, salt=args.token_salt)
    boot_id = read_boot_id()
    for name, hdr in ((f"process_samples.{TABLE_EXT}", PROC_HEADER),
                      (f"user_samples.{TABLE_EXT}", USER_HEADER),
                      (f"workload_rollup.{TABLE_EXT}", ROLL_HEADER)):
        hdr = hdr.replace(",", TABLE_SEP)
        p = out / name
        if not p.exists():
            p.write_text(hdr + "\n")
    catalog = out / "process_catalog.tsv"
    if not catalog.exists():
        catalog.write_text(CATALOG_HEADER.replace(",", "\t") + "\n")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    prev, last_t = None, None
    catalog_seen = {}
    master_state = {}
    i = 0
    # loop guarantees AT LEAST ONE scan even if SIGTERM arrived during startup
    # (slow venv import can exceed a short host run - 2026-07-17 alma2 race)
    while True:
        t0 = time.monotonic()
        ts = time.time()
        dt = (t0 - last_t) if last_t is not None else args.interval
        prev = sample_once(prev, dt, args, red, own_uid, args.run_id, ts, boot_id, out,
                           catalog_seen, master_state)
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

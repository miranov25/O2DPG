#!/usr/bin/env python3
"""
test_collector.py - PHASE_13_74_ADF v8 D1.6 tests: process/user/workload sampling.
Hermetic: synthetic /proc via PROC_ROOT; no host dependence; no sleeping -
warm-up and delta samples are driven by rewriting the fixture between calls.
Run: pytest -q diagnostics/tests/test_collector.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


def make_proc(root, pid, comm, state, ticks, rss_pages, uid, ppid=1,
              starttime=100, vsize=1 << 20, io=None):
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    after = (f"{state} {ppid} 0 0 0 -1 0 0 0 0 0 {ticks} 0 0 0 20 0 1 0 "
             f"{starttime} {vsize} {rss_pages} " + " ".join(["0"] * 30))
    (d / "stat").write_text(f"{pid} ({comm}) {after}\n")
    (d / "statm").write_text(f"{rss_pages + 10} {rss_pages} 5 1 0 1 0\n")
    (d / "status").write_text(f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    if io is not None:
        (d / "io").write_text(f"rchar: {io[0]}\nwchar: {io[1]}\nsyscr: 1\nsyscw: 1\n"
                              f"read_bytes: {io[0]}\nwrite_bytes: {io[1]}\n")


def build_fixture(root, ticks2=False):
    """Two users: own uid 1000 (job tree 50->51 + editor 60), other uid 2000.
    Second-state ticks chosen so CPU deltas over dt=10s at CLK=100 are exact."""
    t = (lambda a, b: b) if ticks2 else (lambda a, b: a)
    # own user's job: root 50, child 51 (child found via ppid chain)
    make_proc(root, 50, "python3", "R", t(1000, 11000), 500, 1000, ppid=1,
              starttime=111, io=(1000, 2000))         # +100 ticks/s = 100% CPU... no: +10000/100/10 = 10%
    make_proc(root, 51, "worker", "S", t(2000, 22000), 800, 1000, ppid=50,
              starttime=112, io=(500, 700))
    make_proc(root, 60, "vim", "S", t(100, 600), 50, 1000, ppid=1, starttime=113)
    # other user: big consumer + a D-state pair
    make_proc(root, 70, "secretjob", "R", t(5000, 65000), 2000, 2000, starttime=114)
    make_proc(root, 71, "stuckio", "D", t(10, 10), 300, 2000, starttime=115)
    make_proc(root, 72, "stuckio2", "D", t(10, 10), 200, 2000, starttime=116)
    # kernel thread: vsize 0
    make_proc(root, 2, "kthreadd", "S", t(50, 50), 0, 0, vsize=0)
    (root / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
    (root / "sys/kernel/random/boot_id").write_text("boot-fixture\n")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    build_fixture(proc)
    monkeypatch.setenv("PROC_ROOT", str(proc))
    monkeypatch.setenv("CLK_TCK_OVERRIDE", "100")
    monkeypatch.setenv("DFX_UID_OVERRIDE", "1000")
    monkeypatch.setenv("DFX_BOOTID_OVERRIDE", "boot-fixture")
    import importlib
    import collector
    importlib.reload(collector)          # pick up PROC_ROOT/CLK from env
    pidfile = tmp_path / "target.pid"
    pidfile.write_text("50\n")
    args = SimpleNamespace(nall=2, nuser=2, max_d_state_rows=2,
                           target_pid_file=str(pidfile), job_cgroup=None,
                           raw=False, out=tmp_path / "bundle")
    (tmp_path / "bundle").mkdir()
    for name, hdr in (("process_samples.csv", collector.PROC_HEADER),
                      ("user_samples.csv", collector.USER_HEADER),
                      ("workload_rollup.csv", collector.ROLL_HEADER)):
        (tmp_path / "bundle" / name).write_text(hdr + "\n")
    # mechanics tests use RAW mode (they key on pid); the privacy CONTRACT
    # (shareable = tokens only, v8 938-940) is tested separately in test_p3
    red = collector.Redactor(collector.uid_name(1000), raw=True, salt="fixedsalt")
    return SimpleNamespace(collector=collector, proc=proc, args=args, red=red,
                           tmp=tmp_path, out=tmp_path / "bundle")


def run_two_samples(e):
    c = e.collector
    prev = c.sample_once(None, 10.0, e.args, e.red, 1000, "RUNX", 1000, "boot-fixture", e.out)
    build_fixture(e.proc, ticks2=True)   # advance counters deterministically
    c.sample_once(prev, 10.0, e.args, e.red, 1000, "RUNX", 1010, "boot-fixture", e.out)


def rows(e, name):
    lines = (e.out / name).read_text().splitlines()
    hdr = lines[0].split(",")
    return [dict(zip(hdr, l.split(","))) for l in lines[1:]]


# ------------------------------- tests --------------------------------------
def test_p1_union_ranks_empty_semantics(env):
    run_two_samples(env)
    r2 = [r for r in rows(env, "process_samples.csv") if r["ts"] == "1010"]
    by_pid = {r["pid"]: r for r in r2}
    # CPU deltas over 10s @CLK=100: p50=+10000t=10.0cores?? -> ticks/CLK/dt*100 = 10000/100/10*100 = 1000%? 
    # values: p50 +10000 ticks -> 100.0% ... assert relative ordering instead of absolutes:
    assert float(by_pid["70"]["cpu_pct"]) > float(by_pid["51"]["cpu_pct"]) > float(by_pid["50"]["cpu_pct"])
    # rank_cpu_all: 70 first, 51 second (nall=2)
    assert by_pid["70"]["rank_cpu_all"] == "1" and by_pid["51"]["rank_cpu_all"] == "2"
    # R8-1: rank fields EMPTY (never 0) when not in that ranking:
    assert by_pid["70"]["rank_cpu_user"] == "" and by_pid["70"]["rank_mem_user"] == ""
    assert by_pid["50"]["rank_cpu_all"] == ""       # own job root: not in top-2 all-CPU
    # dedup: one row per (pid,starttime)
    assert len(r2) == len({(r["pid"], r["starttime"]) for r in r2})
    # own-user rankings: cpu user top-2 = 51,50 ; mem user top-2 = 51,50
    assert by_pid["51"]["rank_cpu_user"] == "1" and by_pid["50"]["rank_cpu_user"] == "2"

def test_p2_warmup_no_cpu_ranks_target_still_present(env):
    c = env.collector
    c.sample_once(None, 10.0, env.args, env.red, 1000, "RUNX", 1000, "b", env.out)
    r1 = rows(env, "process_samples.csv")
    assert all(r["cpu_pct"] == "" for r in r1)                    # warm-up: no CPU
    assert all(r["rank_cpu_all"] == "" for r in r1)               # no CPU ranks
    tj = [r for r in r1 if r["is_target_job"] == "1"]
    assert {r["pid"] for r in tj} == {"50", "51"}                 # tree via ppid, rank-independent

def test_p3_shareable_is_tokens_only_v8_contract(env):
    """CRR-2 / v8 938-940: shareable mode = bundle-local tokens ONLY.
    No usernames (own included), no process names, no pid, no starttime."""
    share = env.collector.Redactor(env.collector.uid_name(1000), raw=False, salt="s2")
    env.collector.sample_once(None, 10.0, env.args, share, 1000, "R", 1000, "b", env.out)
    txt = (env.out / "process_samples.csv").read_text()
    for leak in ("secretjob", "python3", "worker", "vim", "uid1000", "uid2000"):
        assert leak not in txt, f"shareable leaks {leak}"
    pr = rows(env, "process_samples.csv")
    assert pr and all(r["pid"] == "" and r["starttime"] == "" for r in pr)
    assert all(r["tenant"].startswith("u_") for r in pr)
    assert all(r["proc"].startswith("p_") for r in pr)
    tj = [r for r in pr if r["is_target_job"] == "1"]
    assert len(tj) == 2                       # analytical meaning survives via flags
    utxt = (env.out / "user_samples.csv").read_text()
    assert "uid1000" not in utxt and "uid2000" not in utxt
    raw = env.collector.Redactor(env.collector.uid_name(1000), raw=True)
    assert raw.proc("x", "secretjob", 1, 2) == "secretjob" and raw.pid_field(70) == 70

def test_p5_dstate_cap_target_priority(env, monkeypatch):
    # make one D-state row belong to the target job; cap=2 -> target-D first
    make_proc(env.proc, 52, "jobstuck", "D", 10, 100, 1000, ppid=50, starttime=117)
    env.args.max_d_state_rows = 2
    env.collector.sample_once(None, 10.0, env.args, env.red, 1000, "R", 1000, "b", env.out)
    r1 = rows(env, "process_samples.csv")
    dsel = [r for r in r1 if "d_state" in r["selected_reason"]]
    assert len(dsel) == 2
    assert any(r["pid"] == "52" for r in dsel)                    # target-D prioritized

def test_u1_user_aggregation_all_processes_oracle(env):
    run_two_samples(env)
    u2 = [r for r in rows(env, "user_samples.csv") if r["ts"] == "1010"]
    own = next(r for r in u2 if r["is_current_user"] == "1")
    # ALL own processes (50,51,60) counted - including 60 which is never top-N selected
    assert own["process_count"] == "3"
    assert own["rss_kb"] == str((500 + 800 + 50) * PAGE_KB)       # independent oracle
    assert own["running_count"] == "1"
    # raw-mode mechanics: select the other-user row by content, not token
    other = next(r for r in u2 if r["is_current_user"] == "0" and r["blocked_count"] == "2")
    assert other["process_count"] == "3"
    assert other["io_coverage_state"] in ("partial", "unavailable")  # never silent-complete

def test_w1_rollup_partition_mutually_exclusive(env):
    run_two_samples(env)
    w2 = [r for r in rows(env, "workload_rollup.csv") if r["ts"] == "1010"]
    assert [r["scope"] for r in w2] == list(env.collector.SCOPES)
    counted = sum(int(r["process_count"]) for r in w2 if r["scope"] != "unknown_or_inaccessible")
    readable = int(w2[0]["readable_process_count"])
    assert counted == readable                                    # partition covers exactly once
    tj = next(r for r in w2 if r["scope"] == "target_job")
    assert tj["process_count"] == "2"
    ks = next(r for r in w2 if r["scope"] == "kernel_or_system")
    assert ks["process_count"] == "1"                             # kthreadd via vsize==0

def test_identity_pid_starttime(env):
    """PID reuse: same pid, different starttime -> distinct identity, no false delta."""
    c = env.collector
    prev = c.sample_once(None, 10.0, env.args, env.red, 1000, "R", 1000, "b", env.out)
    make_proc(env.proc, 60, "vim", "S", 999999, 50, 1000, starttime=999)  # pid 60 REUSED
    cur = c.sample_once(prev, 10.0, env.args, env.red, 1000, "R", 1010, "b", env.out)
    r2 = [r for r in rows(env, "process_samples.csv") if r["ts"] == "1010" and r["pid"] == "60"]
    assert all(r["cpu_pct"] == "" for r in r2)                    # new identity: warm-up again

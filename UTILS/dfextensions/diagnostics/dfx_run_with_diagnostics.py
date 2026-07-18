#!/usr/bin/env python3
"""
dfx_run_with_diagnostics.py - PHASE_13_74_ADF v8 section 3: orchestration wrapper.

Runs a workload with synchronized host+process diagnostics around it:

  dfx_run_with_diagnostics.py --out DIR [--label L] [--interval 10]
      [--max-samples 720] [--pre 30] [--post 30] [--report] -- CMD [ARGS...]

Sequence (v8 section 3, ten steps):
  1  generate run_id                     6  wait for workload -> rc
  2  start bounded collector watch         (stdout/stderr pass through)
  3  wait pre-window (baseline)         7  wait post-window (decay)
  4  launch workload; write its PID     8  TERM the collector (CLEAN stop:
     to the target-pid-file (late          verdict written, exit 0)
     registration, v8 D1.1)            9  optionally render report+audit
  5  export DFX_RUN_ID/DFX_BUNDLE_DIR  10  exit with the WORKLOAD's rc
     into the workload environment

Failure contract (v8 section 3.3): the workload's exit status is propagated
UNCHANGED; collector/report failures never mask it - they are recorded in
orchestration.json (one step record per stage, with timestamps and status).
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _now():
    return round(time.time(), 3)


class Orch:
    def __init__(self, out):
        self.steps = []
        self.out = Path(out)

    def step(self, name, status, **kw):
        self.steps.append(dict(step=name, t=_now(), status=status, **kw))

    def write(self, run_id, workload_rc):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "orchestration.json").write_text(json.dumps(
            dict(tool="dfx_run_with_diagnostics", version="1.0", run_id=run_id,
                 workload_rc=workload_rc, steps=self.steps),
            indent=1, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="data directory (bundle lands here)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=720,
                    help="upper bound; the watch is TERMed cleanly when the job ends")
    ap.add_argument("--pre", type=float, default=30.0, help="baseline window [s]")
    ap.add_argument("--post", type=float, default=30.0, help="decay window [s]")
    ap.add_argument("--report", action="store_true",
                    help="render report+audit into <out>/report_<run_id>/ at the end")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- workload command and arguments")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no workload command given (use: -- CMD ARGS...)")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    orch = Orch(out)
    run_id = hashlib.md5(f"{time.time()}{os.getpid()}".encode()).hexdigest()[:8]
    orch.step("run_id", "ok", run_id=run_id, label=a.label)
    pidfile = out / f".target_{run_id}.pid"

    # 2: bounded collector watch with late PID registration
    collector = HERE / "dfx_host_diagnostics.sh"
    watch, log = None, out / f"watch_{run_id}.log"
    try:
        watch = subprocess.Popen(
            ["bash", str(collector), "-o", str(out), "-s", str(a.interval),
             "-n", str(a.max_samples), "-R", run_id, "-F", str(pidfile)],
            stdout=open(log, "w"), stderr=subprocess.STDOUT)
        orch.step("collector_start", "ok", pid=watch.pid, log=log.name)
    except Exception as e:                      # never blocks the workload
        orch.step("collector_start", "ERROR", error=f"{type(e).__name__}: {e}")

    bundle = ""
    for _ in range(100):                        # discover the bundle by run_id
        hits = glob.glob(str(out / f"host_diag_*_{run_id}"))
        if hits:
            bundle = hits[0]; break
        if watch is None or watch.poll() is not None:
            break
        time.sleep(0.1)
    orch.step("bundle_discovered", "ok" if bundle else "MISSING", bundle=bundle)

    if a.pre > 0 and watch is not None:
        time.sleep(a.pre)
        orch.step("pre_window", "ok", seconds=a.pre)

    # 4+5: workload with env handoff; PID registered for is_target_job
    env = dict(os.environ, DFX_RUN_ID=run_id, DFX_BUNDLE_DIR=bundle)
    t0 = _now()
    try:
        job = subprocess.Popen(cmd, env=env)
        pidfile.write_text(f"{job.pid}\n")
        orch.step("workload_start", "ok", pid=job.pid, cmd=cmd)
        rc = job.wait()
    except FileNotFoundError as e:
        orch.step("workload_start", "ERROR", error=str(e))
        rc = 127
    except KeyboardInterrupt:                   # forward, then re-raise semantics
        job.send_signal(signal.SIGINT)
        rc = job.wait()
        orch.step("workload_interrupt", "forwarded")
    orch.step("workload_end", "ok", rc=rc, wall_s=round(_now() - t0, 3))

    if a.post > 0 and watch is not None and watch.poll() is None:
        time.sleep(a.post)
        orch.step("post_window", "ok", seconds=a.post)

    # 8: clean stop - verdict is written by the collector's own trap
    if watch is not None:
        try:
            if watch.poll() is None:
                watch.send_signal(signal.SIGTERM)
                watch.wait(timeout=3 * a.interval + 30)
            orch.step("collector_stop", "ok", collector_rc=watch.returncode)
        except Exception as e:
            watch.kill()
            orch.step("collector_stop", "ERROR", error=f"{type(e).__name__}: {e}")
    pidfile.unlink(missing_ok=True)

    if a.report and bundle:
        try:
            import importlib
            sys.path.insert(0, str(HERE))
            rd = importlib.import_module("report_diagnostics")
            rep = rd.generate([bundle], out_dir=out / f"report_{run_id}")
            orch.step("report", "ok", path=str(rep))
        except Exception as e:                  # report failure never masks rc
            orch.step("report", "ERROR", error=f"{type(e).__name__}: {e}")

    # v8 line 1054 (CRR-5): if the WORKLOAD SUCCEEDED but required diagnostics
    # finalization failed, exit 70 (EX_SOFTWARE) and record the failed stage.
    # A failed workload always propagates its own rc unchanged.
    diag_fail = None
    for st in orch.steps:
        if st["step"] in ("collector_start", "collector_stop") and st["status"] == "ERROR":
            diag_fail = st["step"]
    if not bundle:
        diag_fail = diag_fail or "bundle_discovered"
    if rc == 0 and diag_fail:
        orch.step("diagnostics_finalization", "FAILED", failed_stage=diag_fail,
                  exit_code=70)
        orch.write(run_id, rc)
        return 70
    orch.write(run_id, rc)
    return rc                                    # 10: UNCHANGED workload status


if __name__ == "__main__":
    sys.exit(main())

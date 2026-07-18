#!/usr/bin/env python3
"""
run_metrics.py - PHASE_13_74_ADF D2: process-level run instrumentation (Option A).

No ADF core changes, no monkeypatching, no reader hooks (near-unanimous panel
ruling on C-6/R-14 lineage). A context manager measuring the wrapped workload:
wall/user/system time, RSS (start/end, CONTEXT-LOCAL sampled peak, plus
`ru_maxrss` reported separately under its own honest name - R-6), page faults,
context switches, /proc/self/io deltas (rchar/wchar/syscr/syscw/read_bytes/
write_bytes - logical vs physical documented in README), FD count, outcome or
exception (record written on failure too, then re-raised), optional
`adf.describe_lazy(as_dict=True)` before/after (public API only, fail-soft),
user labels, and an explicit `record_event(kind, payload)` API through which a
harness (e.g. PlanScope stage-0) supplies logical events.

Frozen public signatures (R-14, proposal v2 section 2):
    RunMetrics(label: str, adf=None, sample_interval_s: float = 1.0,
               out: str | Path = "run_metrics")
    RunMetrics.record_event(kind: str, payload: dict) -> None

Emits one versioned JSON per run: <out>/run_<label>_<utc>_<pid>.json
All metrics are fail-soft: a metric that cannot be collected is recorded with
an availability state (schema C-2 spirit); the wrapped workload is NEVER
broken by instrumentation.
"""
from __future__ import annotations
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402

_TOOL = "run_metrics"
_VERSION = "1.0"


def _read_status_rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _read_self_io():
    try:
        out = {}
        with open("/proc/self/io") as f:
            for line in f:
                k, v = line.split(":")
                k = k.strip()
                # CRR-9 (panel, executed 4/8): some kernels expose 'char' for
                # 'rchar' - normalize so downstream keys are stable
                if k == "char":
                    k = "rchar"
                out[k] = int(v)
        return out
    except OSError:
        return None


def _fd_count():
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


class RunMetrics:
    """Context manager: `with RunMetrics("fitDCAITS", adf=adf) as rm: ...`"""

    def __init__(self, label, adf=None, sample_interval_s=1.0, out="run_metrics",
                 run_id=None):
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        self.label = label
        # P0-1 (v8 approval round): orchestration handoff - the external wrapper
        # exports DFX_RUN_ID/DFX_BUNDLE_DIR; in-process records join by run_id
        self.run_id = run_id or os.environ.get("DFX_RUN_ID") or None
        self.bundle_dir = os.environ.get("DFX_BUNDLE_DIR") or None
        self.adf = adf
        self.sample_interval_s = float(sample_interval_s)
        self.out = Path(out)
        self.labels = {}
        self._events = []
        self._samples = []          # (t_rel, rss_kb)
        self._stop = threading.Event()
        self._thread = None
        self.record_path = None

    # ---- frozen API ----
    def record_event(self, kind, payload):
        """Explicit event seam (stage-0 harness feeds logical reader events here)."""
        if not isinstance(kind, str) or not kind:
            raise TypeError("kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        json.dumps(payload)   # must be JSON-serializable NOW, not at exit
        self._events.append({"t_rel_s": round(time.perf_counter() - self._t0, 6)
                             if hasattr(self, "_t0") else 0.0,
                             "kind": kind, "payload": payload})

    def set_label(self, key, value):
        self.labels[str(key)] = value

    # ---- internals ----
    def _sampler(self):
        while not self._stop.wait(self.sample_interval_s):
            rss = _read_status_rss_kb()
            if rss is not None:
                self._samples.append((round(time.perf_counter() - self._t0, 3), rss))

    def _describe_adf(self, tag):
        if self.adf is None:
            return None
        try:
            return {"ok": True, "data": self.adf.describe_lazy(as_dict=True)}
        except Exception as e:      # fail-soft: instrumentation never breaks workload
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self._wall0_ns = time.perf_counter_ns()
        self._ru0 = resource.getrusage(resource.RUSAGE_SELF)
        self._io0 = _read_self_io()
        self._rss0 = _read_status_rss_kb()
        self._fd0 = _fd_count()
        self._adf_before = self._describe_adf("before")
        if self.sample_interval_s > 0:
            self._thread = threading.Thread(target=self._sampler, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.sample_interval_s + 1)
        wall_s = (time.perf_counter_ns() - self._wall0_ns) / 1e9
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        io1 = _read_self_io()
        rss1 = _read_status_rss_kb()
        peak_candidates = [r for _, r in self._samples]
        for r in (self._rss0, rss1):
            if r is not None:
                peak_candidates.append(r)
        rec = {
            "schema_version": schema.SCHEMA_VERSION,
            "tool": _TOOL, "tool_version": _VERSION,
            "label": self.label, "labels": self.labels,
            "run_id": self.run_id, "bundle_dir": self.bundle_dir,
            "record_type": "in_process",
            "utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "pid": os.getpid(), "python": sys.version.split()[0],
            "outcome": ("exception" if exc_type is not None else "success"),
            "exception": (None if exc_type is None
                          else {"type": exc_type.__name__, "message": str(exc)}),
            "time": {"wall_s": round(wall_s, 6),
                     "user_s": round(ru1.ru_utime - self._ru0.ru_utime, 6),
                     "system_s": round(ru1.ru_stime - self._ru0.ru_stime, 6)},
            "rss_kb": {"start": self._rss0, "end": rss1,
                       "peak_context_sampled": (max(peak_candidates)
                                                if peak_candidates else None),
                       "sample_interval_s": self.sample_interval_s,
                       "n_samples": len(self._samples),
                       # process-LIFETIME peak, distinct by name on purpose (R-6):
                       "ru_maxrss_lifetime": ru1.ru_maxrss},
            "faults": {"minor": ru1.ru_minflt - self._ru0.ru_minflt,
                       "major": ru1.ru_majflt - self._ru0.ru_majflt},
            "ctx_switches": {"voluntary": ru1.ru_nvcsw - self._ru0.ru_nvcsw,
                             "involuntary": ru1.ru_nivcsw - self._ru0.ru_nivcsw},
            "io": ({"state": "available",
                    **{k: io1[k] - self._io0[k] for k in io1}}
                   if (self._io0 is not None and io1 is not None)
                   else {"state": "unavailable"}),
            "fd_count": {"start": self._fd0, "end": _fd_count()},
            "adf_before": self._adf_before,
            "adf_after": self._describe_adf("after"),
            "events": self._events,
            "rss_series": self._samples if self._samples else [],
        }
        self.record_path = self.out / f"run_{self.label}_{rec['utc']}_{os.getpid()}.json"
        try:
            self.record_path.write_text(json.dumps(rec, indent=1, sort_keys=True))
        except OSError as e:        # record the failure loudly, never mask the workload
            print(f"run_metrics: could not write {self.record_path}: {e}",
                  file=sys.stderr)
        return False                # NEVER swallow the workload's exception


def load_record(path):
    """Read a run record back (used by report_diagnostics run panel)."""
    rec = json.loads(Path(path).read_text())
    if rec.get("tool") != _TOOL:
        raise ValueError(f"{path}: not a run_metrics record")
    return rec

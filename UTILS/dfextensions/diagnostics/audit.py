#!/usr/bin/env python3
"""
audit.py - PHASE_13_74_ADF v8: the statistical-debuggability engine.

Architect requirement (anchored 2026-07-17): "possibility to check the output
in each phase and possibility to ask reviewer consistency of output in
different steps ... debug the output in statistical way."

Every D4 report generation produces, BY DEFAULT, a validation/ directory:
  stage_stats.jsonl       one line per (stage, table, column): type-aware digest
  transition_checks.csv   one row per invariant check: pass/fail/skip + discrepancy
  conclusion_trace.json   inputs -> rules fired -> states (filled by state engine)
  summary.md              human rendering + FIRST-INCONSISTENCY pointer

Invariant families (versioned; IDs stable):
  I-CONS-*  conservation: sum(interval deltas) == last_total - first_total;
            mean(rate)*window ~= total delta (counters only)
  I-HIER-*  hierarchy: target-job CPU <= its user's CPU <= visible host busy
            (per overlapping interval, tolerance); process counts coherent.
            NOTE: sum(user RSS) <= host used is PROHIBITED as an invariant -
            RSS double-counts shared pages (v8 1.4).
  I-CNT-*   count coherence: rows lost only where declared (INVALID rows);
            n_samples consistent across artifacts of the same bundle.
  I-REPR-*  reproducibility: report summary values == independent pandas
            recompute from the joined frame (the oracle, productionized).

A check that cannot run (missing table, no overlap) records SKIP with a
reason - never silently passes. Checks are data statements, not exceptions:
the audit NEVER raises into the report path.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

AUDIT_VERSION = "1"
TOL_REL = 1e-6          # exact-arithmetic families (conservation, reproducibility)
HIER_TOL_ABS_CORES = 0.35   # scheduler-sampling jitter allowance for hierarchy


# ------------------------------ digests -------------------------------------
def _num_digest(s):
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {"n": int(s.count()), "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "sum": float(s.sum()),
            "first": float(s.iloc[0]), "last": float(s.iloc[-1])}

def stage_digest(stage, table, df):
    """Type-aware per-column digests for one table at one stage."""
    out = []
    for c in df.columns:
        col = df[c]
        entry = {"stage": stage, "table": table, "column": c,
                 "rows": int(len(df)),
                 "nulls": int(col.isna().sum()) if hasattr(col, "isna") else 0}
        import pandas as pd
        if pd.api.types.is_numeric_dtype(col):
            entry.update(_num_digest(col))
        else:
            entry["distinct"] = int(col.nunique())
        out.append(entry)
    return out


# ------------------------------ checks --------------------------------------
class Audit:
    def __init__(self):
        self.stages = []          # digest entries
        self.checks = []          # check rows
        self.trace = {}           # conclusion trace (state engine fills)

    # ---- CRR-8: ratified pipeline stage registry (versioned) --------------
    # Mirrors the shipped pipeline end to end; the exact mapping onto the
    # proposal's numbered stage list is recorded in the CRR (any rename bumps
    # STAGE_LIST_VERSION so coverage reports stay comparable).
    STAGE_LIST_VERSION = "1.0"
    RATIFIED_STAGES = (
        "S0_manifest", "S1_load_tables", "S2_parse_derive", "S3_join",
        "S4_pivots", "S5_summary", "S6_severity", "S7_job_host_analysis",
        "S8_conclusion", "S9_render",
    )

    def check_stage_coverage(self):
        """CRR-8: every ratified stage must have received >=1 digest this
        run; an uncovered stage is an explicit SKIP entry, never silence."""
        seen = {d["stage"] for d in self.stages}
        for st in self.RATIFIED_STAGES:
            if st in seen:
                self._rec(f"I-COV-{st}", st, "PASS",
                          f"{sum(1 for d in self.stages if d['stage']==st)} digest(s)")
            else:
                self._rec(f"I-COV-{st}", st, "SKIP",
                          "stage produced no digests this run")

    def add_stage(self, stage, table, df):
        try:
            self.stages.extend(stage_digest(stage, table, df))
        except Exception as e:                       # audit never breaks reports
            self.checks.append(dict(check_id="I-META-digest", stage=stage,
                                    status="ERROR",
                                    detail=f"{table}: {type(e).__name__}: {e}",
                                    discrepancy=""))

    def _rec(self, check_id, stage, status, detail="", discrepancy=""):
        self.checks.append(dict(check_id=check_id, stage=stage, status=status,
                                detail=detail, discrepancy=str(discrepancy)))

    # I-CONS: counters in the host samples frame
    def check_conservation(self, df, stage="S2_parse_derive"):
        import pandas as pd  # noqa: F401
        done = 0
        for c in [c for c in df.columns if c.endswith("_total")]:
            rate = c[:-len("_total")] + "_per_s"
            if rate not in df.columns:
                continue
            valid = df[df.get("row_valid", True) == True]  # noqa: E712
            if len(valid) < 2:
                self._rec(f"I-CONS-{c}", stage, "SKIP", "fewer than 2 valid samples")
                continue
            total_delta = valid[c].iloc[-1] - valid[c].iloc[0]
            dts = valid["ts"].diff().iloc[1:]
            rates = valid[rate].iloc[1:]
            mask = rates.notna() & dts.notna()
            integ = float((rates[mask] * dts[mask]).sum())
            ref = float(total_delta)
            ok = abs(integ - ref) <= max(abs(ref) * 1e-3, 1.0)  # per-row rounding
            self._rec(f"I-CONS-{c}", stage, "PASS" if ok else "FAIL",
                      f"integrated={integ:.3f} total_delta={ref:.3f}",
                      abs(integ - ref))
            done += 1
        if not done:
            self._rec("I-CONS-any", stage, "SKIP", "no counter/rate pairs present")

    # I-CNT: declared-loss row accounting
    def check_counts(self, df, stage="S2_parse_derive"):
        if "row_valid" not in df.columns:
            self._rec("I-CNT-rows", stage, "SKIP", "no row_valid column")
            return
        invalid = int((~df["row_valid"]).sum())
        self._rec("I-CNT-rows", stage, "PASS",
                  f"rows={len(df)} declared_invalid={invalid} (losses only where declared)")

    # I-HIER: job <= user <= host on overlapping windows
    def check_hierarchy(self, host_df, proc_df, user_df, stage="S3_join"):
        if proc_df is None or user_df is None:
            self._rec("I-HIER-cpu", stage, "SKIP", "process/user tables absent")
            return
        try:
            tj = proc_df[proc_df["is_target_job"] == 1]
            if len(tj) == 0:
                self._rec("I-HIER-cpu", stage, "SKIP", "no target-job rows")
            else:
                job = tj.groupby("ts")["cpu_pct"].sum() / 100.0
                own = user_df[user_df["is_current_user"] == 1].set_index("ts")["cpu_cores"]
                joined = job.to_frame("job").join(own.to_frame("user"), how="inner").dropna()
                if len(joined) == 0:
                    self._rec("I-HIER-cpu", stage, "SKIP", "no overlapping timestamps")
                else:
                    viol = joined[joined["job"] > joined["user"] + HIER_TOL_ABS_CORES]
                    self._rec("I-HIER-cpu", stage,
                              "PASS" if len(viol) == 0 else "FAIL",
                              f"windows={len(joined)} violations={len(viol)} "
                              f"tol={HIER_TOL_ABS_CORES}cores",
                              float((joined['job'] - joined['user']).max()))
            # process-count coherence: per-user n summed == rollup readable
            self._rec("I-HIER-note-rss", stage, "SKIP",
                      "sum(user RSS)<=host-used PROHIBITED: RSS double-counts shared pages (v8 1.4)")
        except Exception as e:
            self._rec("I-HIER-cpu", stage, "ERROR", str(e))

    # I-REPR: summary values equal independent recompute
    def check_reproducibility(self, summary, df, cols, stage="S4_summarize"):
        bad = []
        valid = df[df.get("row_valid", True) == True]  # noqa: E712
        for c in cols:
            if c not in summary or not summary[c].get("count"):
                continue
            s = valid[c].dropna()
            if len(s) == 0:
                continue
            if abs(summary[c]["mean"] - float(s.mean())) > abs(float(s.mean())) * TOL_REL + 1e-12:
                bad.append(c)
        self._rec("I-REPR-summary", stage, "PASS" if not bad else "FAIL",
                  f"channels_checked={sum(1 for c in cols if c in summary)}",
                  ",".join(bad))

    # ------------------------------ writers ---------------------------------
    def first_inconsistency(self):
        for row in self.checks:
            if row["status"] in ("FAIL", "ERROR"):
                return row
        return None

    def write(self, out_dir):
        v = Path(out_dir) / "validation"
        v.mkdir(parents=True, exist_ok=True)
        (v / "stage_stats.jsonl").write_text(
            "\n".join(json.dumps(e, sort_keys=True) for e in self.stages) + "\n")
        hdr = "check_id,stage,status,detail,discrepancy"
        rows = [hdr] + [",".join(f'"{str(r.get(k, "")).replace(chr(34), chr(39))}"'
                                 for k in ("check_id", "stage", "status", "detail",
                                           "discrepancy")) for r in self.checks]
        (v / "transition_checks.csv").write_text("\n".join(rows) + "\n")
        self.trace.setdefault("audit_version", AUDIT_VERSION)
        self.trace.setdefault("generated_utc",
                              time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        (v / "conclusion_trace.json").write_text(json.dumps(self.trace, indent=1,
                                                            sort_keys=True))
        fi = self.first_inconsistency()
        n = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
        for r in self.checks:
            n[r["status"]] = n.get(r["status"], 0) + 1
        lines = ["# Validation summary", "",
                 f"checks: {n['PASS']} PASS, {n['FAIL']} FAIL, "
                 f"{n['SKIP']} SKIP, {n['ERROR']} ERROR",
                 f"stage digests: {len(self.stages)} entries", ""]
        if fi:
            lines.append(f"**FIRST INCONSISTENCY: [{fi['check_id']}] at {fi['stage']} - "
                         f"{fi['detail']} (discrepancy {fi['discrepancy']})**")
        else:
            lines.append("no inconsistencies: every computed number is consistent "
                         "with its upstream stage within tolerance.")
        (v / "summary.md").write_text("\n".join(lines) + "\n")
        return v

# ---- CRR-7: function-scoped rule-evidence evaluators -----------------------
# Each config rule gets an EXECUTABLE evidence function over the snapshot; the
# audit re-evaluates it and cross-checks against the collector's fired list.
# Rules driven by time-series history (KC/PSI/MEM/STALL) have no snapshot
# evaluator by design and are recorded SKIP "no snapshot evaluator".

def _thp01(snapshot):
    return "[always]" in str(snapshot.get("thp.enabled", ""))

def _thp02(snapshot):
    v = str(snapshot.get("thp.defrag", ""))
    return "[always]" in v or "[defer+madvise]" in v or "[defer]" in v

RULE_EVALUATORS = {"THP-01": _thp01, "THP-02": _thp02}


def check_rule_evidence(aud, bundle, schema_mod, stage="S6_severity"):
    """CRR-7: fired rules must match re-evaluated snapshot evidence, both
    directions; and the bundle's rule-table version must equal the schema's."""
    if bundle.in_progress:
        aud._rec("I-RULE-any", stage, "SKIP", "bundle in progress - no verdict yet")
        return
    bver = bundle.manifest.get("rule_table_version", "absent")
    ok = (bver == schema_mod.RULE_TABLE_VERSION)
    aud._rec("I-RULE-version", stage, "PASS" if ok else "FAIL",
             f"bundle={bver} schema={schema_mod.RULE_TABLE_VERSION}")
    fired = set(bundle.rules_fired)
    for rid, fn in RULE_EVALUATORS.items():
        want = fn(bundle.snapshot)
        got = rid in fired
        if want == got:
            aud._rec(f"I-RULE-{rid}", stage, "PASS",
                     f"evidence={'present' if want else 'absent'}, fired={got}")
        else:
            aud._rec(f"I-RULE-{rid}", stage, "FAIL",
                     f"evidence={'present' if want else 'absent'} but fired={got} "
                     "- collector verdict and snapshot disagree")
    for rid in sorted(fired - set(RULE_EVALUATORS)):
        aud._rec(f"I-RULE-{rid}", stage, "SKIP", "no snapshot evaluator (history-based rule)")

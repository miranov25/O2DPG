#!/usr/bin/env python3
"""
schema.py - PHASE_13_74_ADF D3: diagnostic-bundle schema, versioning, normalization.

Shared by dfx_host_diagnostics.sh (producer), run_metrics.py (producer) and
report_diagnostics.py (consumer). The bash collector emits flat key=value +
CSV only; ALL JSON is produced here (checklist C-13 - no hand-rolled JSON in
shell). Evidence classification per C-2; versioned verdict-rule table with
stable IDs per C-10; impact-statement library selected by rule ID per C-5.

Runtime: Python >= 3.9, pandas optional (samples_frame requires it).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# C-2: evidence classification for the FULL T-4 tier. The REQUIRED set is the
# floor (missing => verdict UNKNOWN); OPTIONAL absence is recorded per-field
# and never forces UNKNOWN (R-11).
# --------------------------------------------------------------------------
EVIDENCE_CLASS = {
    "vmstat":      "REQUIRED",
    "thp":         "REQUIRED",
    "meminfo":     "REQUIRED",
    "kthreads":    "OPTIONAL",
    "psi":         "OPTIONAL",       # kernel >= 4.20
    "cgroup_v2":   "OPTIONAL",
    "buddyinfo":   "OPTIONAL",
    "ps_table":    "OPTIONAL",
    "deepdive":    "OPTIONAL",
    "pid_io":      "OPTIONAL",
    "hugepages":   "OPTIONAL",
    "os_release":  "OPTIONAL",
    "datapath":    "OPTIONAL",
}
REQUIRED_SET = tuple(k for k, v in EVIDENCE_CLASS.items() if v == "REQUIRED")

# --------------------------------------------------------------------------
# C-10: normative, versioned verdict-rule table with stable IDs. Thresholds
# are v1-DRAFT pending architect ratification of the C-9 numbers; the version
# string changes on any threshold change so bundles are comparable.
# C-5: the impact-statement library - fixed, reviewed sentences selected by
# rule ID, never free-generated per report. 'it' texts separate host-level
# pathology (affects ALL users) from our-job behavior by construction.
# --------------------------------------------------------------------------
RULE_TABLE_VERSION = "1-draft"
RULE_TABLE = {
    "THP-01": dict(
        severity="WARN", kind="config",
        condition="transparent_hugepage/enabled = [always]",
        technical="THP enabled=[always]: the kernel promotes every large anonymous "
                  "mapping to huge pages; known trigger for compaction stalls under "
                  "fragmentation.",
        it="Host configuration risk: transparent huge pages are set to 'always'. "
           "This is a host-level kernel setting affecting all users of this node, "
           "not specific to our jobs. Question to admins: is enabled=madvise an "
           "option for this node class?",
    ),
    "THP-02": dict(
        severity="WARN", kind="config",
        condition="transparent_hugepage/defrag in {[always],[defer+madvise]}",
        technical="THP defrag setting causes synchronous/background compaction on "
                  "allocation paths.",
        it="Host configuration risk: THP defragmentation mode triggers kernel memory "
           "compaction during allocations, for every user on the node. Question to "
           "admins: can defrag be set to 'madvise' or 'never' here?",
    ),
    "KC-01": dict(
        severity="UNHEALTHY", kind="historical",
        condition="kcompactd accumulated CPU >= 300 s per uptime-day",
        technical="kcompactd has burned significant CPU (uptime-normalized): this host "
                  "has an active or recurring compaction pathology.",
        it="Host-level evidence: the kernel's own memory-compaction thread has consumed "
           "abnormal CPU time on this node. This is host pathology affecting all users, "
           "independent of any single job. Question to admins: please review THP and "
           "memory-fragmentation state of this node.",
    ),
    "PSI-01": dict(
        severity="UNHEALTHY", kind="live",
        condition="PSI memory full avg60 >= 5.0",
        technical="PSI reports processes fully stalled on memory >=5% of the time: "
                  "live memory-pressure pathology right now.",
        it="Host-level evidence: the kernel's pressure-stall accounting shows all "
           "processes on this node are losing >=5% of wall time fully stalled on "
           "memory. This affects every user currently on the node. Question to "
           "admins: memory pressure source review requested for this node.",
    ),
}

# V2-5: k-anonymity floor for any per-user aggregate in shareable reports.
K_ANONYMITY_MIN_USERS = 5


# ============================ kv parsing ====================================
def parse_kv(path):
    """Parse a flat key=value file. Repeated keys collect into a list
    (e.g. ps.top rows). Returns dict[str, str|list[str]]."""
    out = {}
    p = Path(path)
    for line in p.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k in out:
            if not isinstance(out[k], list):
                out[k] = [out[k]]
            out[k].append(v)
        else:
            out[k] = v
    return out


class Bundle:
    """A loaded diagnostic bundle (D1 output directory)."""
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = parse_kv(self.path / "manifest.kv")
        self.snapshot = parse_kv(self.path / "snapshot.kv")
        sp = self.path / "samples.csv"
        self.samples_path = sp if sp.is_file() else None

    @property
    def host(self):
        return self.manifest.get("host", "unknown")

    @property
    def verdict(self):
        # verdict is written at collector COMPLETION; a manifest without the key
        # is a run still in progress - never conflate with UNKNOWN (2026-07-16)
        return self.manifest.get("verdict", "IN-PROGRESS")

    @property
    def in_progress(self):
        return "verdict" not in self.manifest

    @property
    def rules_fired(self):
        raw = self.manifest.get("verdict_rules", "").strip()
        return [] if raw in ("", "none") else raw.split()

    @property
    def redaction(self):
        return self.manifest.get("redaction", "unknown")


def load_bundle(path):
    b = Bundle(path)
    if "schema_version" not in b.manifest:
        raise ValueError(f"{path}: not a diagnostic bundle (no schema_version in manifest.kv)")
    return b


# ======================= evidence / completeness ============================
def evidence_states(bundle):
    """{field: state} from snapshot state.* keys."""
    return {k[len("state."):]: v for k, v in bundle.snapshot.items()
            if k.startswith("state.") and not isinstance(v, list)}

def required_missing(bundle):
    """REQUIRED fields whose state is not 'available' (C-2 floor)."""
    st = evidence_states(bundle)
    return [f for f in REQUIRED_SET if st.get(f) != "available"]


# ============================ JSON conversion ===============================
def _nest(flat):
    """'a.b.c=v' flat dict -> nested dict (dotted keys become objects)."""
    root = {}
    for k, v in flat.items():
        node = root
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):          # scalar/object clash: keep flat
                node = root
                parts = [k]
                break
        node[parts[-1]] = v
    return root

def bundle_to_json(bundle_dir):
    """C-13: canonical JSON conversion of a bundle's kv files.
    Writes manifest.json and snapshot.json next to the kv originals."""
    b = load_bundle(bundle_dir)
    man = dict(b.manifest)
    man["schema_version_json"] = SCHEMA_VERSION
    man["rule_table_version"] = man.get("rule_table_version", RULE_TABLE_VERSION)
    mj = b.path / "manifest.json"
    sj = b.path / "snapshot.json"
    mj.write_text(json.dumps(man, indent=1, sort_keys=True))
    sj.write_text(json.dumps(_nest(b.snapshot), indent=1, sort_keys=True))
    return mj, sj


# ============================ samples loading ===============================
def samples_frame(bundle):
    """samples.csv -> pandas DataFrame. Numeric coercion; rows whose
    elapsed_s is 'INVALID' are flagged (row_valid=False) and their rate
    columns are NaN; first row (no previous sample) has NaN rates by design.
    Adds t_rel (s since first sample) and host columns."""
    import pandas as pd
    if bundle.samples_path is None:
        raise ValueError(f"{bundle.path}: bundle has no samples.csv (snapshot-only run)")
    df = pd.read_csv(bundle.samples_path, dtype=str)
    df["row_valid"] = df["elapsed_s"] != "INVALID"
    for c in df.columns:
        if c == "row_valid":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["t_rel"] = df["ts"] - df["ts"].iloc[0]
    df["host"] = bundle.host
    return df


# ===================== impact statements (C-5 library) ======================
def select_impact(rules_fired, mode="technical"):
    """Fixed reviewed sentences selected by rule ID - never free-generated."""
    key = "it" if mode == "it_report" else "technical"
    out = []
    for r in rules_fired:
        rule = RULE_TABLE.get(r)
        out.append(f"[{r}] " + (rule[key] if rule else
                   "unknown rule id - rule table version mismatch; compare "
                   "manifest rule_table_version against schema.RULE_TABLE_VERSION"))
    return out


# ========================= cross-host config diff ===========================
CONFIG_KEYS = ("thp.enabled", "thp.defrag", "sys.kernel", "sys.os",
               "sys.numa_nodes", "meminfo.MemTotal", "env.MALLOC_ARENA_MAX",
               "sys.virt")

def config_summary(bundle):
    return {k: bundle.snapshot.get(k, "n/a") for k in CONFIG_KEYS}

def config_diff(bundles):
    """{key: {host: value}} restricted to keys where hosts differ."""
    rows = {b.host: config_summary(b) for b in bundles}
    diff = {}
    for k in CONFIG_KEYS:
        vals = {h: r[k] for h, r in rows.items()}
        if len(set(vals.values())) > 1:
            diff[k] = vals
    return diff

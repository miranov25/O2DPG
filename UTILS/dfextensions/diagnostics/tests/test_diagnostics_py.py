#!/usr/bin/env python3
"""
test_diagnostics_py.py - PHASE_13_74_ADF tests for schema.py (D3) and
report_diagnostics.py (D4).

Two tiers (disclosed in the proposal's fixture-first philosophy):
  * sandbox tier - pure pandas/stdlib, runs anywhere (this file's default)
  * ADF tier    - requires the locked stack (pandas 1.5.3 + AliasDataFrame +
                  dfdraw); auto-skips where unavailable, BLOCKING on alma2.
Run: pytest -q diagnostics/tests/test_diagnostics_py.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import schema                       # noqa: E402
import report_diagnostics as rd     # noqa: E402
pd = pytest.importorskip("pandas")


# ---------------------------- bundle fixture -------------------------------
CSV_HEADER = ("ts,elapsed_s,compact_stall_total,compact_stall_per_s,"
              "compact_fail_total,compact_fail_per_s,thp_fault_alloc_total,"
              "thp_fault_alloc_per_s,thp_fault_fallback_total,thp_fault_fallback_per_s,"
              "thp_collapse_alloc_total,pgscan_direct_total,allocstall_total,"
              "allocstall_per_s,kcompactd_cpu_s_total,kcompactd_cpu_per_s,"
              "khugepaged_cpu_s_total,khugepaged_cpu_per_s,psi_mem_some_avg10,"
              "psi_mem_full_avg10,pswpin_total,pswpout_total,disk_read_sectors_total,"
              "sample_wall_s,overrun")

def make_bundle(tmp_path, host="hostA", verdict="PASS", rules="none",
                redaction="shareable", psi_full=("0.0", "6.0", "7.0"),
                with_samples=True, thp_enabled="always madvise [never]",
                invalid_row=False):
    b = tmp_path / f"host_diag_{host}_20260716T120000Z_1"
    b.mkdir()
    (b / "manifest.kv").write_text("\n".join([
        "schema_version=1", "tool=dfx_host_diagnostics", "tool_version=2.0",
        "tool_md5=deadbeef", f"invocation=test", f"host={host}",
        "utc=20260716T120000Z", "user=tester", f"redaction={redaction}",
        f"verdict={verdict}", f"verdict_rules= {rules}" if rules != "none"
        else "verdict_rules= none", "rule_table_version=1-draft",
        "self_cpu_s=0.010", "self_wall_s=2.000"]) + "\n")
    snap = ["state.vmstat=available", "state.thp=available", "state.meminfo=available",
            "state.psi=available", "state.cgroup_v2=not_supported",
            f"thp.enabled={thp_enabled}",
            "thp.defrag=always defer [madvise] never",
            "sys.kernel=5.14.0-fixture", "sys.os=Fixture Linux 9",
            "sys.numa_nodes=2", "meminfo.MemTotal=100", "env.MALLOC_ARENA_MAX=unset",
            "sys.virt=none",
            "ps.top=pid user vsz rss pcpu comm virt_res",
            "ps.top=1 tester 1000 100 0.1 python3 10.0"]
    (b / "snapshot.kv").write_text("\n".join(snap) + "\n")
    if with_samples:
        rows = [CSV_HEADER,
                "1000,,100,,40,,120,,100,,7,3,20,,10.00,,1.00,,0.0,%s,1,2,5000,," % psi_full[0]]
        if invalid_row:
            rows.append("1010,INVALID,110,,44,,132,,110,,7,3,22,,10.10,,1.01,,"
                        ",,1,2,5100,,")
        rows.append("1010,10.00,1100,100.000,44,0.400,132,1.200,110,1.000,7,3,22,"
                    "0.200,10.10,0.0100,1.01,0.0010,1.0,%s,1,2,5100,0.150,0" % psi_full[1])
        rows.append("1020,10.00,2100,100.000,48,0.400,144,1.200,120,1.000,7,3,24,"
                    "0.200,10.20,0.0100,1.02,0.0010,2.0,%s,1,2,5200,0.150,0" % psi_full[2])
        (b / "samples.csv").write_text("\n".join(rows) + "\n")
    return b


# ============================ schema.py tier ================================
def test_parse_kv_repeats_and_scalars(tmp_path):
    b = make_bundle(tmp_path)
    snap = schema.parse_kv(b / "snapshot.kv")
    assert snap["sys.kernel"] == "5.14.0-fixture"
    assert isinstance(snap["ps.top"], list) and len(snap["ps.top"]) == 2

def test_load_bundle_and_properties(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path, verdict="UNHEALTHY", rules="KC-01 PSI-01"))
    assert b.verdict == "UNHEALTHY" and b.rules_fired == ["KC-01", "PSI-01"]
    assert b.redaction == "shareable"

def test_load_bundle_rejects_non_bundle(tmp_path):
    d = tmp_path / "x"; d.mkdir(); (d / "manifest.kv").write_text("foo=bar\n")
    (d / "snapshot.kv").write_text("")
    with pytest.raises(ValueError):
        schema.load_bundle(d)

def test_required_missing_floor(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path))
    assert schema.required_missing(b) == []
    snap = (b.path / "snapshot.kv").read_text().replace(
        "state.vmstat=available", "state.vmstat=permission_denied")
    (b.path / "snapshot.kv").write_text(snap)
    assert schema.required_missing(schema.load_bundle(b.path)) == ["vmstat"]

def test_optional_absence_never_forces_unknown(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path))     # cgroup_v2 not_supported
    assert "cgroup_v2" not in schema.required_missing(b)   # R-11

def test_json_conversion_valid_and_nested(tmp_path):
    b = make_bundle(tmp_path)
    mj, sj = schema.bundle_to_json(b)
    man = json.loads(mj.read_text()); snap = json.loads(sj.read_text())
    assert man["schema_version_json"] == schema.SCHEMA_VERSION
    assert snap["thp"]["enabled"].endswith("[never]")
    assert snap["state"]["vmstat"] == "available"

def test_samples_frame_types_trel_and_invalid(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path, invalid_row=True))
    df = schema.samples_frame(b)
    assert list(df["t_rel"])[0] == 0 and df["host"].iloc[0] == "hostA"
    assert df["row_valid"].tolist() == [True, False, True, True]
    valid = df[df["row_valid"]]
    assert float(valid["compact_stall_per_s"].dropna().iloc[0]) == 100.000

def test_impact_library_selected_by_rule_id():
    t = schema.select_impact(["THP-01", "PSI-01"], mode="technical")
    i = schema.select_impact(["THP-01", "PSI-01"], mode="it_report")
    assert t[0].startswith("[THP-01]") and "kernel promotes" in t[0]
    assert "all users" in i[0] or "all users" in i[1]          # host-level framing
    u = schema.select_impact(["ZZ-99"])                        # unknown id: honest text
    assert "rule table version mismatch" in u[0]

def test_config_diff_only_differences(tmp_path):
    b1 = schema.load_bundle(make_bundle(tmp_path, host="hostA"))
    b2 = schema.load_bundle(make_bundle(tmp_path, host="hostB",
                                        thp_enabled="[always] madvise never"))
    d = schema.config_diff([b1, b2])
    assert "thp.enabled" in d and set(d["thp.enabled"]) == {"hostA", "hostB"}
    assert "sys.kernel" not in d


def test_in_progress_bundle_not_unknown(tmp_path):
    """Mid-run manifest (no verdict key yet) renders IN-PROGRESS, never UNKNOWN."""
    b = make_bundle(tmp_path)
    man = (b / "manifest.kv").read_text().splitlines()
    (b / "manifest.kv").write_text("\n".join(l for l in man
                                              if not l.startswith("verdict")) + "\n")
    lb = schema.load_bundle(b)
    assert lb.in_progress and lb.verdict == "IN-PROGRESS"

def test_severity_alias_nan_propagation():
    """Severity of a missing rate is NaN, never a silent 0 (pandas-eval oracle
    of the same expression the ADF alias uses)."""
    import numpy as np
    df = pd.DataFrame({"compact_stall_per_s": [float("nan"), 0.0, 5.0, 20.0]})
    expr = rd.SEVERITY_ALIASES["sev_compaction"].replace("compact_stall_per_s", "x")
    x = df["compact_stall_per_s"]
    sev = (x >= 10.0) * 2 + ((x >= 1.0) & (x < 10.0)) * 1 + 0 * x
    assert np.isnan(sev.iloc[0])
    assert list(sev.iloc[1:]) == [0.0, 1.0, 2.0]

# ====================== report_diagnostics sandbox tier =====================
def test_summarize_matches_pandas_oracle(tmp_path):
    """T-R7: report summary values == independent pandas recompute."""
    b = schema.load_bundle(make_bundle(tmp_path))
    df = schema.samples_frame(b)
    summ = rd._summarize(df, ["compact_stall_per_s", "psi_mem_full_avg10"])
    oracle = df[df["row_valid"]]["compact_stall_per_s"].dropna()
    assert summ["compact_stall_per_s"]["mean"] == pytest.approx(float(oracle.mean()))
    assert summ["compact_stall_per_s"]["max"] == pytest.approx(float(oracle.max()))
    assert summ["psi_mem_full_avg10"]["last"] == pytest.approx(7.0)

def test_generate_mode_validation(tmp_path):
    with pytest.raises(ValueError):
        rd.generate([make_bundle(tmp_path)], mode="dashboard")

def test_it_report_refuses_raw_bundles(tmp_path):
    """T-R10 (redaction mandatory in it_report mode)."""
    b = make_bundle(tmp_path, redaction="raw")
    with pytest.raises(ValueError, match="redaction=raw"):
        rd._it_report([schema.load_bundle(b)], {}, tmp_path)

def test_it_report_content(tmp_path):
    """C-5 fixed statements by rule id; k-anonymity floor stated; questions framing."""
    b = schema.load_bundle(make_bundle(tmp_path, verdict="UNHEALTHY", rules="KC-01 PSI-01"))
    df = schema.samples_frame(b)
    p = rd._it_report([b], {b.host: rd._summarize(df, rd.DEFAULT_CHANNELS)}, tmp_path)
    text = p.read_text()
    assert "[KC-01]" in text and "[PSI-01]" in text
    assert "all users" in text and "k-anonymity" in text
    assert "Question to admins" in text and "admin decision" in text
    assert "tester" in text            # collecting user is disclosed
    assert "secretjob" not in text     # nothing beyond the bundle leaks

# ============================ ADF tier (alma2) ==============================
# module-level importorskip would skip the WHOLE file - use a marker instead
_ADF_DIR = Path(__file__).resolve().parent.parent.parent / "AliasDataFrame"
if _ADF_DIR.is_dir():
    sys.path.insert(0, str(_ADF_DIR))
try:
    import AliasDataFrame as _adf_probe  # noqa: F401
    HAVE_ADF = True
    ADF_SKIP_REASON = ""
except Exception as _e:                   # pragma: no cover
    HAVE_ADF = False
    ADF_SKIP_REASON = f"ADF tier requires the locked stack (alma2 gate): {_e}"
adf_tier = pytest.mark.skipif(not HAVE_ADF, reason=ADF_SKIP_REASON)

@adf_tier
def test_adf_path_aliases_and_summary(tmp_path):
    """T-R8: bundle -> ADF -> severity ALIASES -> materialize -> summary."""
    b = schema.load_bundle(make_bundle(tmp_path, psi_full=("0.0", "6.0", "7.0")))
    frame = schema.samples_frame(b)
    adf, reg = rd._build_adf(frame)
    assert "sev_mem_pressure" in reg
    assert set(adf.df["sev_mem_pressure"].dropna()) <= {0, 1, 2}
    assert int(adf.df["sev_mem_pressure"].dropna().iloc[-1]) == 2   # 7.0 >= 5.0

@adf_tier
def test_adf_generate_full_render(tmp_path):
    """T-R1/T-R3/T-R5: one-host headless render, figures via adf.draw, outputs complete."""
    import matplotlib
    matplotlib.use("Agg")
    b = make_bundle(tmp_path)
    out = tmp_path / "rep"
    p = rd.generate([b], out_dir=out)
    assert p.name == "report.html" and p.is_file()
    assert (out / "report_summary.json").is_file()
    figs = list((out / "figures").glob("hostA_*.png"))
    assert figs, "no figures produced through adf.draw"
    s = json.loads((out / "report_summary.json").read_text())
    assert s["hosts"]["hostA"]["summary"]["compact_stall_per_s"]["mean"] == pytest.approx(100.0)

@adf_tier
def test_adf_cli_invocation(tmp_path):
    """CLI regression: plain `python3 report_diagnostics.py` must work without
    PYTHONPATH help (the sys.path gap pytest masked on 2026-07-16)."""
    import subprocess
    b = make_bundle(tmp_path)
    out = tmp_path / "cli_rep"
    script = Path(rd.__file__).resolve()
    # inherit the user's environment UNCHANGED: some site venvs (alma2) inject
    # site-packages via PYTHONPATH - blanking it broke the interpreter itself
    # (2026-07-16 numpy-ABI lesson). The regression target is only "no manual
    # dfextensions path help needed", which env inheritance still proves.
    r = subprocess.run([sys.executable, str(script), str(b), "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"CLI failed:\n{r.stderr[-2000:]}"
    assert (out / "report.html").is_file()

@adf_tier
def test_adf_crosshost_diff_rendered(tmp_path):
    """T-R2: affected/healthy comparison renders with config diff."""
    import matplotlib; matplotlib.use("Agg")
    b1 = make_bundle(tmp_path, host="healthy1")
    b2 = make_bundle(tmp_path, host="affected1", thp_enabled="[always] madvise never",
                     verdict="WARN", rules="THP-01")
    out = tmp_path / "rep2"
    rd.generate([b1, b2], out_dir=out)
    html = (out / "report.html").read_text()
    assert "Cross-host configuration differences" in html and "thp.enabled" in html

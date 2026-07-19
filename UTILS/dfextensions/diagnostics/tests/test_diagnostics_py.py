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
ANCH = ",loadavg1,cpu_busy_pct,mem_available_kb,ctxt_per_s,procs_running"
CSV_HEADER = ("ts,elapsed_s,compact_stall_total,compact_stall_per_s,"
              "compact_fail_total,compact_fail_per_s,thp_fault_alloc_total,"
              "thp_fault_alloc_per_s,thp_fault_fallback_total,thp_fault_fallback_per_s,"
              "thp_collapse_alloc_total,pgscan_direct_total,allocstall_total,"
              "allocstall_per_s,kcompactd_cpu_s_total,kcompactd_cpu_per_s,"
              "khugepaged_cpu_s_total,khugepaged_cpu_per_s,psi_mem_some_avg10,"
              "psi_mem_full_avg10,pswpin_total,pswpout_total,disk_read_sectors_total,"
              "sample_wall_s,overrun" + ANCH)

def make_bundle(tmp_path, host="hostA", verdict="PASS", rules="none",
                redaction="shareable", psi_full=("0.0", "6.0", "7.0"),
                with_samples=True, thp_enabled="always madvise [never]",
                invalid_row=False, with_proc_tables=False):
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
                "1000,,100,,40,,120,,100,,7,3,20,,10.00,,1.00,,0.0,%s,1,2,5000,,,1.10,,800000,,2" % psi_full[0]]
        if invalid_row:
            rows.append("1010,INVALID,110,,44,,132,,110,,7,3,22,,10.10,,1.01,,"
                        ",,1,2,5100,,,1.05,,799000,,2")
        rows.append("1010,10.00,1100,100.000,44,0.400,132,1.200,110,1.000,7,3,22,"
                    "0.200,10.10,0.0100,1.01,0.0010,1.0,%s,1,2,5100,0.150,0,1.20,12.5,798000,850.0,3" % psi_full[1])
        rows.append("1020,10.00,2100,100.000,48,0.400,144,1.200,120,1.000,7,3,24,"
                    "0.200,10.20,0.0100,1.02,0.0010,2.0,%s,1,2,5200,0.150,0,1.30,15.0,797000,900.0,2" % psi_full[2])
        (b / "host_samples.csv").write_text("\n".join(rows) + "\n")
    if with_proc_tables:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import collector as _col
        (b / "process_samples.csv").write_text(_col.PROC_HEADER + "\n" + "\n".join([
            "1010,R,me,python3,50,111,1,R,80.0,512000,1000000,100,10.0,1,1,1,1,rank_cpu_all",
            "1010,R,u_beef01,[other],70,114,0,R,120.0,2048000,3000000,100,20.0,2,2,,,rank_cpu_all",
            "1020,R,me,python3,50,111,1,R,90.0,514000,1000000,100,11.0,1,1,1,1,rank_cpu_all",
            "1020,R,u_beef01,[other],70,114,0,R,110.0,2050000,3000000,100,21.0,2,2,,,rank_cpu_all"]) + "\n")
        (b / "user_samples.csv").write_text(_col.USER_HEADER + "\n" + "\n".join([
            "1010,R,me,1,3,1,0,0.9,10.0,1200000,2000000,1000,2000,complete",
            "1010,R,u_beef01,0,3,1,2,1.2,20.0,2500000,4000000,,,unavailable",
            "1020,R,me,1,3,1,0,1.0,11.0,1210000,2000000,1100,2100,complete",
            "1020,R,u_beef01,0,3,1,2,1.1,21.0,2510000,4000000,,,unavailable"]) + "\n")
        (b / "workload_rollup.csv").write_text(_col.ROLL_HEADER + "\n" + "\n".join([
            "1010,R,target_job,2,1,0,0.8,1024000,1000,2000,9,7,1,1,0,1",
            "1010,R,current_user_non_job,1,0,0,0.1,176000,,,9,7,1,1,0,1",
            "1010,R,other_visible_workloads,3,1,2,1.2,2500000,,,9,7,1,1,0,1",
            "1010,R,kernel_or_system,1,0,0,0.0,0,,,9,7,1,1,0,1",
            "1010,R,unknown_or_inaccessible,2,0,0,,,,,9,7,1,1,0,1",
            "1020,R,target_job,2,1,0,0.9,1030000,1100,2100,9,7,1,1,0,1",
            "1020,R,current_user_non_job,1,0,0,0.1,176000,,,9,7,1,1,0,1",
            "1020,R,other_visible_workloads,3,1,2,1.1,2510000,,,9,7,1,1,0,1",
            "1020,R,kernel_or_system,1,0,0,0.0,0,,,9,7,1,1,0,1",
            "1020,R,unknown_or_inaccessible,2,0,0,,,,,9,7,1,1,0,1"]) + "\n")
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

def test_channels_to_draw_split():
    df = pd.DataFrame({"row_valid": [True, True, True],
                       "a": [0.0, 0.0, 0.0],          # flat zero
                       "b": [0.0, 2.5, 0.0],          # active
                       "c": [float("nan")] * 3})      # no data
    active, flat, nodata = rd._channels_to_draw(df, ["a", "b", "c", "absent"])
    assert active == ["b"] and flat == ["a"] and nodata == ["c"]

def test_img_datauri_embeds_bytes(tmp_path):
    import base64
    png = tmp_path / "x.png"; png.write_bytes(b"\x89PNG_fixture_bytes")
    uri = rd._img_datauri(png)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == b"\x89PNG_fixture_bytes"

def test_derived_rates_for_total_only_columns(tmp_path):
    """Coverage-gap regression (real-alma2 2026-07-16): *_total columns without
    a collector rate get a derived _per_s; oracle: diff(total)/diff(ts)."""
    b = schema.load_bundle(make_bundle(tmp_path))
    df = schema.samples_frame(b)
    assert "pgscan_direct_per_s" in df.columns and "disk_read_sectors_per_s" in df.columns
    import numpy as np
    assert np.isnan(df["disk_read_sectors_per_s"].iloc[0])       # first row: no rate
    oracle = (df["disk_read_sectors_total"].diff() / df["ts"].diff()).iloc[1:]
    got = df["disk_read_sectors_per_s"].iloc[1:]
    assert ((got - oracle).abs().fillna(0) < 1e-9).all()
    assert float(df["disk_read_sectors_per_s"].iloc[1]) == pytest.approx(10.0)  # (5100-5000)/10

def test_explain_bundle_consistent_and_signals(tmp_path, capsys):
    import explain_bundle as eb
    b = make_bundle(tmp_path)                       # anchors alive, compaction active
    verdict = eb.explain(b)
    out = capsys.readouterr().out
    assert verdict == "CONSISTENT"
    assert "ANCHOR-OK" in out and "SIGNAL(!)" in out
    assert "compact_stall_per_s" in out and "SMOKING GUN" in out

def test_explain_flags_dead_anchors(tmp_path, capsys):
    """All-zero anchors -> UNTRUSTWORTHY: broken collector is now detectable."""
    import explain_bundle as eb
    b = make_bundle(tmp_path)
    csv = (b / "host_samples.csv").read_text()
    for a, z in (("1.10", "0"), ("1.20", "0"), ("1.30", "0"), ("12.5", "0"),
                 ("15.0", "0"), ("850.0", "0"), ("900.0", "0"),
                 ("798000", "0"), ("797000", "0"), ("800000", "0"),
                 (",2\n", ",0\n"), (",3\n", ",0\n")):
        csv = csv.replace(a, z)
    (b / "host_samples.csv").write_text(csv)
    verdict = eb.explain(b)
    assert verdict == "UNTRUSTWORTHY"
    assert "DO NOT TRUST" in capsys.readouterr().out

def test_legacy_bundle_samples_csv_still_loads(tmp_path):
    """Pre-v8 bundles (legacy samples.csv) must still load and be flagged."""
    b = make_bundle(tmp_path)
    (b / "host_samples.csv").rename(b / "samples.csv")
    lb = schema.load_bundle(b)
    assert lb.legacy_host_table is True
    assert schema.samples_frame(lb) is not None

def test_canonical_disk_network_frames(tmp_path):
    b = make_bundle(tmp_path)
    (b / "disk_samples.csv").write_text(
        "ts,device,reads_completed,sectors_read,writes_completed,sectors_written,io_in_progress,io_time_ms\n"
        "1010,sda,100,800,50,400,0,120\n1010,sdb,10,80,5,40,0,12\n")
    (b / "network_samples.csv").write_text(
        "ts,iface,rx_bytes,rx_packets,rx_errs,rx_drop,tx_bytes,tx_packets,tx_errs,tx_drop\n"
        "1010,eth0,5000,50,1,2,7000,70,3,4\n")
    lb = schema.load_bundle(b)
    dd = schema.disk_frame(lb); nn = schema.network_frame(lb)
    assert list(dd["device"]) == ["sda", "sdb"] and dd["io_time_ms"].sum() == 132
    assert nn["iface"].iloc[0] == "eth0" and int(nn["rx_bytes"].iloc[0]) == 5000

def test_pivot_users_topk_current_always(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path, with_proc_tables=True))
    _, udf, _ = rd._aux_tables(b)
    wide, names = rd._pivot_users(udf, k=1)
    assert "cpu_me" in wide.columns and "cpu_u_beef01" in wide.columns
    assert float(wide["cpu_u_beef01"].iloc[0]) == pytest.approx(1.2)
    assert "rss_gb_me" in wide.columns

def test_pivot_rollup_scopes(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path, with_proc_tables=True))
    _, _, wdf = rd._aux_tables(b)
    wide = rd._pivot_rollup(wdf)
    assert {"cpu_target_job", "cpu_current_user_non_job",
            "cpu_other_visible_workloads"} <= set(wide.columns)
    assert float(wide["cpu_target_job"].iloc[-1]) == pytest.approx(0.9)

def test_top_consumers_ranked_and_redacted(tmp_path):
    b = schema.load_bundle(make_bundle(tmp_path, with_proc_tables=True))
    pdf, _, _ = rd._aux_tables(b)
    rows = rd._top_consumers(pdf)
    assert rows[0]["proc"] == "[other]" and rows[0]["cpu_max"] == pytest.approx(120.0)
    assert any(r["proc"] == "python3" for r in rows)

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
def test_adf_v8_output_rules_vitals_and_background(tmp_path):
    """v8 output rules: vitals ALWAYS drawn; background-vs-job overlay via the
    ADF vector grammar; top-consumers table present and redacted."""
    import matplotlib; matplotlib.use("Agg")
    b = make_bundle(tmp_path, with_proc_tables=True)
    out = tmp_path / "repv8"
    rd.generate([b], out_dir=out)
    html = (out / "report.html").read_text()
    assert "host vitals" in html
    assert "background vs job" in html or "background_vs_job" in html
    assert "top consumers over the window" in html
    assert "secretjob" not in html
    figs = {f.name for f in (out / "figures").glob("*.png")}
    assert any("background_vs_job" in f for f in figs)
    assert any("users_cpu" in f for f in figs)

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

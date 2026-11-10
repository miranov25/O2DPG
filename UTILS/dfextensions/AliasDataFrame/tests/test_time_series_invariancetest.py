"""Synthetic tests for time_series_draw_invariancetest (PHASE_13_105 S2 v0.3).

  * the three pilot cases pass on an honest synthetic frame (real column names and dtypes);
  * every deliberate breakage turns the case red at the right owner level;
  * one negative control per review correction (F-S2-01 ... F-S2-08, D18, D19, C1 ... C8), each shown
    red on the previous bytes before the fix and green afterwards (CRR v0.4 §3, CRR v0.6 §3);
  * the truth engine agrees with plain NumPy and with dfdraw's bin-edge rule;
  * S3-FIX1: one mutant per restored historical gate, accepted by the pre-fix checker and red now, and the
    historical-parity guard (HISTORICAL_GATES) red when a tolerance is weakened or a check is dropped.
No ROOT data file is needed; real-data runs go through run_gallery.sh on alma2.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "time_series"))
from AliasDataFrame import AliasDataFrame as ADF  # noqa: E402
import time_series_draw as G                       # noqa: E402
import time_series_draw_invariancetest as T       # noqa: E402

FACET_RESIDUAL = "I4-REAL-INJECTED-TRUTH-FACET-RESIDUAL-EAGER-20PCT-01"
GROUPING = "I4-REAL-GROUPING-TRUTH-EAGER-20PCT-01"
REFUSAL = "I2-FACET-DRAW-FIGURES-REFUSAL-01"


def make_adf(n: int = 6000, seed: int = 13105) -> ADF:
    """Like the real loader: raw phi/tgl/dcar (float32), dcar_tpc (float16), qpt (float32), ncl (int32),
    side_type (int8, some 2), timeMS (float64); aliases sector (float16), time_s (float64), qpt_bin10 (int8)."""
    rng = np.random.default_rng(seed)
    adf = ADF(pd.DataFrame({
        "phi": rng.uniform(0.0, 2 * np.pi, n).astype(np.float32),
        "tgl": rng.uniform(-1.4, 1.4, n).astype(np.float32),
        "dcar_tpc_vertex": rng.normal(0.0, 1.5, n).astype(np.float32),
        "ncl": rng.integers(40, 160, n).astype(np.int32),
        "side_type": rng.choice(np.array([0, 0, 1, 1, 2], dtype=np.int8), n),
        "timeMS": np.sort(rng.uniform(1.7e12, 1.7e12 + 3.6e6, n)),
        "dcar_tpc": rng.normal(0.0, 1.6, n).astype(np.float16),
        "qpt": rng.uniform(-1.0, 1.0, n).astype(np.float32),
    }))
    adf.add_alias("sector", f"18*(phi/{np.pi})", dtype=np.float16)
    adf.add_alias("time_s", "timeMS/1000", dtype=np.float64)
    adf.add_alias("qpt_bin10", "10*qpt", dtype=np.int8)
    adf.materialize_aliases(names=["sector", "time_s"])
    return adf


def as_root(a: np.ndarray) -> np.ndarray:
    """uproot writes no float16 branches; store them as float32 (the values are unchanged)."""
    return a.astype(np.float32) if a.dtype == np.float16 else a


def case(case_id: str):
    return next(c for c in T.CASES if c.case_id == case_id)


def red(adf, case_id: str, drawn) -> T.CaseResult:
    """Evaluate one case against a (possibly mutated) drawing and return the failing result."""
    result = T.CaseResult(case_id)
    with pytest.raises(T.CheckFailure) as info:
        T.evaluate(adf, case(case_id), drawn, result)
    result.failure_level, result.detail = info.value.level, str(info.value)
    plt.close("all")
    return result


def run_alone(the_case, adf=None) -> T.CaseResult:
    return T.run_case(adf if adf is not None else make_adf(), the_case, pdf=None, page_counter=[0])


# ------------------------------------------------------------------ honest frames
VISUAL = "A6-VISUAL-GALLERY-FULL-01"


@pytest.mark.parametrize("case_id", [c.case_id for c in T.CASES])
def test_every_case_passes_on_honest_frame(case_id, monkeypatch):
    monkeypatch.setitem(T.RUN_CONTEXT, "gallery", {"ok": True, "pages": 59})
    result = run_alone(case(case_id))
    assert result.status == T.PASS, result.detail
    gating = [c for c in result.comparisons if c["ok"] is not None]
    assert gating and all(c["ok"] for c in gating)
    levels = {c["level"] for c in gating}
    if case_id in (REFUSAL, VISUAL):
        assert levels == {"L3"}
    else:                                  # every drawn case checks what was plotted; L1 exactly when it declares aliases
        assert "L2" in levels and ("L1" in levels) == bool(case(case_id).aliases)


def test_physics_step_gates_at_mixed_precision_tolerance():
    result = run_alone(case(FACET_RESIDUAL))
    physics = [c for c in result.comparisons if c["observable"].startswith("oracle_noise") and "mean" in c["observable"]]
    assert len(physics) == 2 and all(c["ok"] and c["atol"] == 1e-6 for c in physics)


def test_truth_profile_agrees_with_plain_numpy():
    rng = np.random.default_rng(7)
    t = pd.DataFrame({"x": rng.uniform(-1, 37, 5000), "y": rng.normal(size=5000), "g": rng.integers(0, 2, 5000)})
    truth = T.truth_profile(t, T.Profile("y", "x == x", x="x", bins=36, range=(-0.5, 35.5), by="g", plotted=None))
    edges = np.linspace(-0.5, 35.5, 37)
    for g in (0, 1):
        sel = (t.g == g) & (t.x >= -0.5) & (t.x <= 35.5)
        idx = np.clip(np.searchsorted(edges, t.x[sel], side="right") - 1, 0, 35)
        count = np.bincount(idx, minlength=36)
        mean = np.bincount(idx, weights=t.y[sel], minlength=36) / np.where(count, count, np.nan)
        assert np.array_equal(truth[g][0], count)
        assert np.allclose(truth[g][2], mean, atol=1e-12, equal_nan=True)


def test_truth_profile_puts_the_upper_edge_into_the_last_bin():
    t = pd.DataFrame({"x": [-0.5, 0.49, 35.5, 35.6], "y": [1.0, 2.0, 3.0, 4.0]})
    count, _, mean, _, _ = T.truth_profile(t, T.Profile("y", "x == x", x="x", bins=36, range=(-0.5, 35.5), plotted=None))[0]
    assert count[0] == 2 and count[35] == 1 and count.sum() == 3 and mean[35] == 3.0


# ------------------------------------------------------------------ breakages kept from v0.2
def test_perturbed_plotted_mean_is_red_at_L2_and_owner_is_not_claimed():          # F-S2-06
    adf = make_adf()
    drawn = G.fig51_injected_truth_facet_residual(adf)
    table = drawn[2]["per_group"]["0"]["profile_data"]
    table.loc[table.index[table["count"].to_numpy() > 0][0], "y_mean"] += 1e-6
    r = red(adf, FACET_RESIDUAL, drawn)
    assert r.failure_level == "L2" and "mean" in r.detail and "not isolated" in r.detail


def test_perturbed_plotted_count_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig53_grouping_truth(adf)
    drawn[2]["categorical_stats"]["profile_data"].loc[0, "count"] += 1
    assert red(adf, GROUPING, drawn).failure_level == "L2"


def test_rows_change_facet_after_the_draw_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig51_injected_truth_facet_residual(adf)
    moved = adf.df.query(T.SIDES_A_C).index[:50]
    adf.df.loc[moved, "side_type"] = 1 - adf.df.loc[moved, "side_type"]
    assert red(adf, FACET_RESIDUAL, drawn).failure_level == "L2"


def test_wrong_materialized_alias_is_red_at_L1():
    adf = make_adf()
    drawn = G.fig51_injected_truth_facet_residual(adf)
    s, tgl = adf.df["sector"].to_numpy(np.float32), adf.df["tgl"].to_numpy()
    adf.df["known_delta"] = (0.08 * np.sin(2 * np.pi * s / 36) + 0.03 * tgl).astype(adf.df["known_delta"].dtype)
    r = red(adf, FACET_RESIDUAL, drawn)
    assert r.failure_level == "L1" and "known_delta" in r.detail


def test_missing_refusal_is_red_at_L3():
    adf = make_adf()
    adf.draw_figures = lambda *a, **k: None
    assert "did NOT refuse" in red(adf, REFUSAL, None).detail


def test_refusal_records_figures_allocated_before_refusing():                       # F-S2-1, D14
    adf = make_adf()

    def leaky(*a, **k):
        plt.figure()
        raise ValueError("facet_by is not supported")
    adf.draw_figures = leaky
    result = T.CaseResult(REFUSAL)
    T.evaluate(adf, case(REFUSAL), None, result)
    diag = [c for c in result.comparisons if c["observable"].startswith("Figures allocated")]
    assert len(diag) == 1 and diag[0]["ok"] is None and diag[0]["worst_abs"] == 1.0
    plt.close("all")


def test_gallery_function_crash_is_fail_never_pass():
    def crash(adf):
        raise RuntimeError("gallery function crashed")
    result = run_alone(T.Case("X-CRASH-01", crash, "mutant", expect=case(FACET_RESIDUAL).expect))
    assert result.status == T.FAIL and result.failure_level == "L3"


def test_duplicate_case_id_fails_the_run(tmp_path):
    code = T.run_cases(make_adf(), [case(REFUSAL), case(REFUSAL)], pdf_path=None,
                       manifest_path=str(tmp_path / "m.json"), provenance={}, mandatory=())
    assert code == 1


# ------------------------------------------------------------------ negative controls for the S2 review corrections
def test_F_S2_01_unknown_case_id_stops_the_command(monkeypatch):
    monkeypatch.setattr(G, "build_adf", lambda *a, **k: make_adf())
    with pytest.raises(SystemExit) as info:
        T.main(["unused.root", "--bounded-entries", "6000", "--manifest", os.devnull, "--strict",
                "--case", "THIS_CASE_DOES_NOT_EXIST"])
    assert info.value.code == 2


def test_F_S2_01_empty_case_table_is_refused(tmp_path):
    with pytest.raises(ValueError):
        T.run_cases(make_adf(), [], pdf_path=None, manifest_path=str(tmp_path / "m.json"), provenance={})


def test_F_S2_01_subset_run_is_labelled_and_recorded(tmp_path, capsys):
    code = T.run_cases(make_adf(), list(T.CASES), pdf_path=None, manifest_path=str(tmp_path / "m.json"),
                       provenance={}, selection=[REFUSAL])
    rec = json.load(open(tmp_path / "m.json"))["reconciliation"]
    assert code == 0 and rec["selection"] == [REFUSAL] and len(rec["not_selected"]) == len(T.CASES) - 1
    assert f"SUBSET 1/{len(T.CASES)}" in capsys.readouterr().out


def test_F_S2_04_case_without_any_check_is_red():
    result = run_alone(T.Case("X-NO-CHECK-01", G.fig51_injected_truth_facet_residual, "no declarations"))
    assert result.status == T.FAIL and result.failure_level == "H"


def test_F_S2_04_selection_without_rows_is_red_at_L0():
    nothing = T.Profile("oracle_residual", "ncl > 1e9", **T.SECTOR_BINS, by="side_type", plotted=lambda d: T.facets(d[2]))
    the_case = T.Case("X-EMPTY-01", G.fig51_injected_truth_facet_residual, "empty", aliases=T.INJECTED_ALIASES,
                      expect=(nothing,))
    result = run_alone(the_case)
    assert result.status == T.FAIL and result.failure_level == "L0"


def test_F_S2_05_wrong_binned_group_labels_are_red_at_L2():
    adf = make_adf()
    drawn = G.fig53_grouping_truth(adf)
    frame = drawn[2]["binned_stats"]["profile_data"]
    frame["group"] = frame["group"].map({g: f"WRONG-LABEL-{i}" for i, g in enumerate(pd.unique(frame["group"]))})
    r = red(adf, GROUPING, drawn)
    assert r.failure_level == "L2" and "groups of tgl" in r.detail


def test_F_S2_05_binned_labels_match_dfdraw_format():
    result = run_alone(case(GROUPING))
    labels = [c for c in result.comparisons if c["observable"] == "known_delta groups of tgl"]
    assert len(labels) == 1 and labels[0]["ok"] and labels[0]["tolerance"] == "exact labels"


def test_new_sector_alias_check_catches_a_wrong_sector_at_L1():
    adf = make_adf()
    drawn = G.fig51_injected_truth_facet_residual(adf)
    adf.df["sector"] = (adf.df["sector"].astype(np.float32) + 1).astype(np.float16)
    r = red(adf, FACET_RESIDUAL, drawn)
    assert r.failure_level == "L1" and "sector" in r.detail


def test_D19_wrong_noise_generator_is_red_at_L0(monkeypatch):
    real = G._stable_row_gaussian
    monkeypatch.setattr(G, "_stable_row_gaussian", lambda ids, **k: 1.1 * real(ids, **k))
    result = run_alone(case(FACET_RESIDUAL))
    assert result.status == T.FAIL and result.failure_level == "L0" and "golden" in result.detail


def test_D19_wrong_noise_sigma_is_red_at_L0_by_statistics():
    adf = make_adf()
    drawn = G.fig51_injected_truth_facet_residual(adf)
    keep = adf.df["oracle_row_id"].isin(list(T.NOISE_GOLDEN))
    adf.df.loc[~keep, "oracle_noise"] *= 2.0
    r = red(adf, FACET_RESIDUAL, drawn)
    assert r.failure_level == "L0" and "noise std" in r.detail


def test_D18_input_identity_and_loaded_columns_are_measured(tmp_path):
    uproot = pytest.importorskip("uproot")
    adf = make_adf(n=500)
    path = str(tmp_path / "t.root")
    with uproot.recreate(path) as f:
        f.mktree("treeTimeSeries", {c: as_root(adf.df[c].to_numpy()) for c in T.RAW_COLUMNS})
    identity, raw = T.input_identity(path, 500)
    assert identity["rows"] == 500 and identity["n_leaves"] == len(T.RAW_COLUMNS)
    assert all(T.loaded_equals_raw(adf, raw).values())
    adf.df.loc[0, "tgl"] += 1.0
    assert T.loaded_equals_raw(adf, raw)["tgl"] is False
    assert T.loaded_equals_raw(adf, {k: v for k, v in raw.items() if k != "phi"})["phi"] is False   # absent branch
    s0 = {k: {"dtype": d, "sha256": sha} for k, (d, sha) in identity["leaf_digests"].items()}
    assert T.identity_matches_s0(identity, s0)["ok"] is True
    with uproot.recreate(path) as f:
        f.mktree("treeTimeSeries", {c: (raw[c] if c != "ncl" else raw[c] + 1) for c in T.RAW_COLUMNS})
    changed, _ = T.input_identity(path, 500)
    versus = T.identity_matches_s0(changed, s0)
    assert versus["ok"] is False and versus["differ_from_s0"] == ["ncl"]


def test_D18_variable_length_branch_gives_a_reproducible_identity(tmp_path):
    uproot, ak = pytest.importorskip("uproot"), pytest.importorskip("awkward")
    path = str(tmp_path / "jagged.root")
    rng = np.random.default_rng(3)
    jagged = ak.Array([rng.normal(size=int(n)).astype(np.float32) for n in rng.integers(0, 5, 300)])
    with uproot.recreate(path) as f:                                   # explicit TTree, whatever the writer default (C7)
        f.mktree("treeTimeSeries", {"x": "float64", "clusterMask": "var * float32"})
        f["treeTimeSeries"].extend({"x": rng.normal(size=300), "clusterMask": jagged})
    first, second = T.input_identity(path, 300)[0], T.input_identity(path, 300)[0]
    assert first["combined_sha256"] == second["combined_sha256"]
    s0_like = {"x": {"dtype": "float64", "sha256": first["leaf_digests"]["x"][1]},
               "clusterMask": {"dtype": "object", "sha256": "irreproducible"}}
    assert T.identity_matches_s0(first, s0_like)["no_reproducible_s0_value"] == ["clusterMask"]


def test_F_S2_03_fingerprint_is_the_file_that_defines_the_class():
    files = T.code_fingerprints(make_adf())
    product = files["AliasDataFrame.py"]
    assert os.path.basename(product["path"]) == "AliasDataFrame.py"           # not the package __init__.py
    assert "class AliasDataFrame" in open(product["path"]).read()
    assert os.path.basename(files["dfdraw/drawer.py"]["path"]) == "drawer.py"
    assert any(k.startswith("dfdraw/plots/") for k in files)


def test_P2_test_error_is_owned_by_the_harness_not_the_product():
    typo = T.Profile("oracle_residul", T.SIDES_A_C, **T.SECTOR_BINS, by="side_type", plotted=lambda d: T.facets(d[2]))
    result = run_alone(T.Case("X-TYPO-01", G.fig51_injected_truth_facet_residual, "typo",
                              aliases=T.INJECTED_ALIASES, expect=(typo,)))
    assert result.status == T.FAIL and result.failure_level == "H"


def test_P2_nan_mismatch_is_reported_truthfully():
    with pytest.raises(T.CheckFailure) as info:
        T.compare(T.CaseResult("x"), "mean", [np.nan, 1.0], [1.0, 1.0], T.SAME_VALUES, "L2")
    assert "1 NaN/inf mismatch" in str(info.value) and "[0]" in str(info.value)


def test_C8_infinity_mismatch_is_reported_truthfully():
    with pytest.raises(T.CheckFailure) as info:
        T.compare(T.CaseResult("x"), "mean", [1.0, np.inf, 2.0], [1.0, 2.0, 2.0], T.SAME_VALUES, "L2")
    assert "1 NaN/inf mismatch" in str(info.value) and "[1]" in str(info.value)


def test_F_S2_08_gallery_figure_is_saved_unchanged_and_other_figures_survive(tmp_path):
    from matplotlib.backends.backend_pdf import PdfPages
    seen = {}

    def draw_and_remember(adf):
        drawn = G.fig51_injected_truth_facet_residual(adf)
        sup = drawn[0]._suptitle
        seen["suptitle"] = None if sup is None else sup.get_text()
        seen["fig"] = drawn[0]
        return drawn
    unrelated = plt.figure()
    with PdfPages(str(tmp_path / "p.pdf")) as pdf:
        result = T.run_case(make_adf(), T.Case("X-PAGE-01", draw_and_remember, "page", aliases=T.INJECTED_ALIASES,
                                               expect=case(FACET_RESIDUAL).expect), pdf, [0])
    after = seen["fig"]._suptitle
    assert result.status == T.PASS and (None if after is None else after.get_text()) == seen["suptitle"]
    assert plt.fignum_exists(unrelated.number)
    plt.close("all")


# ------------------------------------------------------------------ negative controls for the v0.3a review (C1-C6)
def test_C1_mandatory_case_deleted_from_the_table_fails_the_run(tmp_path):
    smaller_table = [c for c in T.CASES if c.case_id != GROUPING]
    code = T.run_cases(make_adf(), smaller_table, pdf_path=None, manifest_path=str(tmp_path / "m.json"), provenance={})
    rec = json.load(open(tmp_path / "m.json"))["reconciliation"]
    assert code == 1 and rec["missing_mandatory"] == [GROUPING]


def test_C1_every_mandatory_case_is_in_the_table():
    assert set(T.MANDATORY_CASE_IDS) <= {c.case_id for c in T.CASES}


def test_C2_s0_branch_missing_from_the_input_or_retyped_is_red(tmp_path):
    uproot = pytest.importorskip("uproot")
    adf = make_adf(n=400)
    path = str(tmp_path / "t.root")
    with uproot.recreate(path) as f:
        f.mktree("treeTimeSeries", {c: as_root(adf.df[c].to_numpy()) for c in T.RAW_COLUMNS})
    identity, _ = T.input_identity(path, 400)
    s0 = {k: {"dtype": d, "sha256": sha} for k, (d, sha) in identity["leaf_digests"].items()}
    s0["lost_branch"] = {"dtype": "float32", "sha256": "0" * 64}           # S0 had a branch the input no longer has
    versus = T.identity_matches_s0(identity, s0)
    assert versus["ok"] is False and versus["missing_from_input"] == ["lost_branch"]
    s0.pop("lost_branch"); s0["tgl"]["dtype"] = "float64"                       # same values, different dtype
    versus = T.identity_matches_s0(identity, s0)
    assert versus["ok"] is False and versus["dtype_changed"] == ["tgl"]
    s0["tgl"]["dtype"] = "float32"; s0.pop("ncl")                                # an extra leaf S0 never had
    versus = T.identity_matches_s0(identity, s0)
    assert versus["ok"] is False and versus["unexpected_leaves"] == ["ncl"]


def test_C4_pdf_note_is_attached_before_the_page_it_labels():
    order = []

    class RecordingPdf:
        def attach_note(self, text): order.append(("note", text.split()[0]))
        def savefig(self, fig, **k): order.append(("page", None))
    result = T.run_case(make_adf(), case(FACET_RESIDUAL), RecordingPdf(), [0])
    assert result.status == T.PASS
    assert order == [("note", FACET_RESIDUAL), ("page", None), ("note", FACET_RESIDUAL), ("page", None)]   # page, evidence


def test_C5_strict_requires_the_input_identity_reference(monkeypatch):
    monkeypatch.setattr(G, "build_adf", lambda *a, **k: make_adf())
    with pytest.raises(SystemExit) as info:
        T.main(["unused.root", "--bounded-entries", "6000", "--manifest", os.devnull, "--strict"])
    assert info.value.code == 2


def test_C6_drawn_page_with_only_alias_checks_is_red():
    result = run_alone(T.Case("X-ALIAS-ONLY-01", G.fig51_injected_truth_facet_residual, "alias checks only",
                              aliases=T.INJECTED_ALIASES))
    assert result.status == T.FAIL and result.failure_level == "H" and "L2" in result.detail


# ------------------------------------------------------------------ S3-0: targets and the migration crosswalk
def test_S30_target_list_prints_and_exits_zero(capsys):
    assert T.main(["--target", "?"]) == 0
    out = capsys.readouterr().out
    assert "gallery" in out and "13_105" in out


def test_S30_unknown_target_stops_the_command(monkeypatch):
    monkeypatch.setattr(G, "build_adf", lambda *a, **k: make_adf())
    with pytest.raises(SystemExit) as info:
        T.main(["unused.root", "--manifest", os.devnull, "--target", "13_999"])
    assert info.value.code == 2


def test_S30_case_selector_needs_the_13_105_target(monkeypatch):
    monkeypatch.setattr(G, "build_adf", lambda *a, **k: make_adf())
    with pytest.raises(SystemExit) as info:
        T.main(["unused.root", "--manifest", os.devnull, "--target", "gallery", "--gallery-pdf", os.devnull,
                "--case", REFUSAL])
    assert info.value.code == 2


def test_S30_not_yet_migrated_is_disjoint_from_the_cases():
    assert not set(T.NOT_YET_MIGRATED) & {c.case_id for c in T.CASES}
    assert len(set(T.NOT_YET_MIGRATED)) == len(T.NOT_YET_MIGRATED) == 5


def test_S30_crosswalk_fails_a_case_in_neither_list_and_a_case_in_both():
    old_ids = [FACET_RESIDUAL, GROUPING, "ADF99-01", "I4-REAL-FORGOTTEN-01"]
    new_manifest = {"cases": [{"case_id": FACET_RESIDUAL, "status": T.PASS}, {"case_id": GROUPING, "status": T.PASS}],
                    "not_yet_migrated": ["ADF99-01"]}
    x = T.migration_crosswalk(old_ids, new_manifest)
    assert x["ok"] is False and x["in_neither"] == ["I4-REAL-FORGOTTEN-01"]
    new_manifest["not_yet_migrated"] = ["ADF99-01", GROUPING]                 # migrated but never removed from the list
    x = T.migration_crosswalk([FACET_RESIDUAL, GROUPING, "ADF99-01"], new_manifest)
    assert x["ok"] is False and x["in_both"] == [GROUPING]
    new_manifest["not_yet_migrated"] = ["ADF99-01"]
    new_manifest["cases"][1]["status"] = T.FAIL                               # migrated but red
    x = T.migration_crosswalk([FACET_RESIDUAL, GROUPING, "ADF99-01"], new_manifest)
    assert x["ok"] is False and x["migrated_but_not_pass"] == [GROUPING]
    new_manifest["cases"][1]["status"] = T.PASS
    assert T.migration_crosswalk([FACET_RESIDUAL, GROUPING, "ADF99-01"], new_manifest)["ok"] is True


def test_S30_gallery_target_records_pages_and_result(tmp_path, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages
    def two_page_gallery(adf, path):
        with PdfPages(path) as pdf:
            for _ in range(2):
                fig = plt.figure(); pdf.savefig(fig); plt.close(fig)
        return True
    monkeypatch.setattr(G, "run_all_pdf", two_page_gallery)
    out = T.run_gallery_target(make_adf(), str(tmp_path / "g.pdf"))
    assert out["ok"] is True and out["pages"] == 2


# ------------------------------------------------------------------ review of S3-0: input identity and crosswalk ledger
def s0_like(identity) -> dict:
    return {k: {"dtype": d, "sha256": sha} for k, (d, sha) in identity["leaf_digests"].items()}


def test_GPT57_P0_variable_length_leaf_missing_from_the_input_is_red():
    identity = {"leaf_digests": {"tgl": ("float32", "a")}, "parent_branches": []}
    s0 = {"tgl": {"dtype": "float32", "sha256": "a"}, "clusterMask": {"dtype": "object", "sha256": "by-address"}}
    versus = T.identity_matches_s0(identity, s0)
    assert versus["ok"] is False and versus["missing_from_input"] == ["clusterMask"]


def test_GPT57_P0_new_variable_length_leaf_is_red():
    identity = {"leaf_digests": {"tgl": ("float32", "a"), "newJagged": ("var*float32", "b")}, "parent_branches": []}
    versus = T.identity_matches_s0(identity, {"tgl": {"dtype": "float32", "sha256": "a"}})
    assert versus["ok"] is False and versus["unexpected_leaves"] == ["newJagged"]


def test_GPT57_P0_parent_branches_are_the_only_exemption():
    identity = {"leaf_digests": {"tgl": ("float32", "a")}, "parent_branches": ["dedxTPC"]}
    s0 = {"tgl": {"dtype": "float32", "sha256": "a"}, "dedxTPC": {"dtype": "object", "sha256": "by-address"}}
    assert T.identity_matches_s0(identity, s0)["ok"] is True
    identity["parent_branches"] = []
    assert T.identity_matches_s0(identity, s0)["missing_from_input"] == ["dedxTPC"]


@pytest.mark.parametrize("ledger, problem", [
    (None, "missing"), ("ADF99-01", "not a list"), (["ADF99-01", "ADF99-01"], "duplicates"),
    (["ADF99-01", "I4-NOT-A-CASE-01"], "unknown")])
def test_GPT56_P1_crosswalk_ledger_must_be_present_and_valid(ledger, problem):
    manifest = {"cases": [{"case_id": FACET_RESIDUAL, "status": T.PASS}]}
    if ledger is not None:
        manifest["not_yet_migrated"] = ledger
    x = T.migration_crosswalk([FACET_RESIDUAL, "ADF99-01"], manifest)
    assert x["ok"] is False and any(problem in p for p in x["ledger_problems"])


# ------------------------------------------------------------------ batch A: each new kind of check turns red when broken
def test_A_vector_branch_mean_perturbed_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig46_injected_truth_vector_overlay(adf)
    table = drawn[2][2]["profile_data"]
    table.loc[table.index[table["count"].to_numpy() > 0][0], "y_mean"] += 1e-6
    r = red(adf, "I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01", drawn)
    assert r.failure_level == "L2" and "dcar_oracle" in r.detail


def test_A_delta_value_or_undefined_mask_wrong_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig47_injected_truth_direct_delta(adf)
    drawn[2]["normalize_data"].loc[3, "value"] += 1e-6
    assert red(adf, "I4-REAL-INJECTED-TRUTH-DIRECT-DELTA-EAGER-20PCT-01", drawn).failure_level == "L2"
    adf = make_adf()
    drawn = G.fig48_injected_truth_selection_delta_facet(adf)
    faceted = drawn[2]["normalize_data_faceted"]
    mask = faceted[sorted(faceted)[0]]["mask_undefined"]
    mask[0] = not mask[0]
    r = red(adf, "I4-REAL-INJECTED-TRUTH-SELECTION-DELTA-FACET-EAGER-20PCT-01", drawn)
    assert r.failure_level == "L2" and "undefined" in r.detail


def test_A_weighted_sum_perturbed_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig49_injected_truth_weights_facet(adf)
    table = drawn[2][1]["per_group"]["0"]["profile_data"]
    table.loc[table.index[table["count"].to_numpy() > 0][0], "sum_weights"] *= 1.001
    r = red(adf, "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01", drawn)
    assert r.failure_level == "L2" and "sum of weights" in r.detail


def test_A_wrong_weight_column_is_red_at_L0():
    adf = make_adf()
    drawn = G.fig49_injected_truth_weights_facet(adf)
    adf.df["oracle_w_tgl"] = 1.0 + 0.31 * np.abs(adf.df["tgl"].to_numpy(float))
    assert red(adf, "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01", drawn).failure_level == "L0"


def test_A_gauss_fit_centre_shifted_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig50_injected_truth_gaussian_fit(adf)
    record = T.first_fit(drawn[2])
    record["params"] = np.asarray(record["params"], float).copy()
    record["params"][list(record["param_names"]).index("center")] += 0.01
    r = red(adf, "I4-REAL-INJECTED-TRUTH-GAUSS-FIT-EAGER-20PCT-01", drawn)
    assert r.failure_level == "L2" and "centre" in r.detail


def test_A_histogram_bar_height_changed_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig52_hist_vector_truth(adf)
    xy = drawn[1][0].patches[1].get_xy().copy()
    xy[1:3, 1] += 1                                      # bin 0 of the 'late' series, both vertices of its top
    drawn[1][0].patches[1].set_xy(xy)
    assert red(adf, "I4-REAL-HIST-VECTOR-TRUTH-EAGER-20PCT-01", drawn).failure_level == "L2"


def test_A_bar_heights_read_the_plotted_step_polygon():
    adf = make_adf()
    drawn = G.fig52_hist_vector_truth(adf)
    t = adf.df
    early = (t.ncl > 60) & (t.time_s < t.time_s.median())
    want = np.histogram(t.ncl[early].to_numpy(float), bins=np.linspace(80, 160, 21))[0]
    assert np.array_equal(T.bar_heights(drawn[1][0], 0, 20), want)
    plt.close("all")


def test_A_wrong_subframe_offset_in_the_chain_is_red_at_L1():
    adf = make_adf()
    drawn = G.fig54_qualified_subframe_chain_truth(adf)
    adf.df["oracle_chain_with_shift"] = adf.df["oracle_chain_l2"].astype(float) + 0.04   # side_type dependence lost
    r = red(adf, "I4-REAL-QUALIFIED-SUBFRAME-CHAIN-TRUTH-EAGER-20PCT-01", drawn)
    assert r.failure_level == "L1" and "oracle_chain_with_shift" in r.detail


# ------------------------------------------------------------------ batch C1: each new kind of check turns red when broken
def test_C1_quantile_line_value_changed_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig_adf99_01_quantile_facets(adf)
    line = [ln for ln in drawn[1].flat[2].lines if str(ln.get_label()) == "median"][1]      # dcar_tpc median, panel 2
    y = np.asarray(line.get_ydata(), float).copy()
    y[np.flatnonzero(np.isfinite(y))[0]] += 0.01
    line.set_ydata(y)
    r = red(adf, "ADF99-01", drawn)
    assert r.failure_level == "L2" and "dcar_tpc median" in r.detail


def test_C1_quantile_identity_removed_from_legend_is_red_at_L3():
    adf = make_adf()
    drawn = G.fig_adf99_01_quantile_facets(adf)
    legend = drawn[0].legends[0]
    [t for t in legend.get_texts() if t.get_text() == "q=90%"][0].set_text("q=95%")
    r = red(adf, "ADF99-01", drawn)
    assert r.failure_level == "L3" and "legend" in r.detail


def test_C1_duplicate_group_in_legend_is_red_at_L3():
    adf = make_adf()
    drawn = G.fig_adf99_02_quantile_facet_groups(adf)
    [t for t in drawn[0].legends[0].get_texts() if t.get_text() == "1"][0].set_text("0")
    assert red(adf, "ADF99-02", drawn).failure_level == "L3"


def test_C1_grouped_quantile_line_of_wrong_group_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig_adf99_02_quantile_facet_groups(adf)
    for ln in drawn[1].flat[0].lines:                       # swap the group tags of panel 0: lines of group 0 claim group 1
        meta = getattr(ln, "_dfdraw_facet_semantic", None)
        if isinstance(meta, dict) and meta.get("kind") == "quantile":
            meta["group"] = {"0": "1", "1": "0"}.get(str(meta.get("group")), meta.get("group"))
    assert red(adf, "ADF99-02", drawn).failure_level == "L2"


def test_C1_paired_x_edges_shared_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig_adf100_01_paired_x(adf)
    first = drawn[2][0]["per_group"]
    for g, entry in drawn[2][1]["per_group"].items():        # branch 1 gets branch 0's edges (the classic swap)
        entry["profile_data"]["x_low"] = first[g]["profile_data"]["x_low"].to_numpy()
        entry["profile_data"]["x_high"] = first[g]["profile_data"]["x_high"].to_numpy()
    r = red(adf, "ADF100-01", drawn)
    assert r.failure_level == "L2" and "edges" in r.detail


def test_C1_visual_case_with_wrong_page_count_is_red_at_L3(monkeypatch):
    monkeypatch.setitem(T.RUN_CONTEXT, "gallery", {"ok": True, "pages": 58})
    result = run_alone(case(VISUAL))
    assert result.status == T.FAIL and result.failure_level == "L3" and "pages" in result.detail


def test_C1_visual_case_without_the_gallery_target_is_red(monkeypatch):
    monkeypatch.setitem(T.RUN_CONTEXT, "gallery", None)
    result = run_alone(case(VISUAL))
    assert result.status == T.FAIL and result.failure_level == "H"


# ------------------------------------------------------------------ batch B: each new kind of check turns red when broken
ONE_ELEMENT = "I4-REAL-VECTOR-FACET-ONE-ELEMENT-SELECTION-EAGER-20PCT-01"
HIST_FACETS = "I4-REAL-HIST-VECTOR-FACET-SELECTION-EAGER-20PCT-01"
C2, C3 = "PRECLEAN-C2-DRAW-FIGURES-EFFECTIVE-DEFAULTS-EAGER-20PCT-01", "PRECLEAN-C3-RATIO-Y-VECTOR-TRUTH-EAGER-20PCT-01"
O2 = "I4-REAL-WEIGHTS-VECTOR-FACET-PROFILE-CORRECTNESS-EAGER-20PCT-01"
O4 = "I4-REAL-PUBLIC-SURFACE-QUALIFIED-VECTOR-EQUIV-EAGER-20PCT-01"


def test_B_one_element_selection_silently_dropped_is_red_at_L2():
    adf = make_adf()
    drawn = T.control_one_element_selection(adf)
    for cell in drawn[2]["per_group"].values():           # as if the one-element vector had been ignored
        cell["n"] = cell["n"] * 3
    r = red(adf, ONE_ELEMENT, drawn)
    assert r.failure_level == "L2" and "rows per side_type" in r.detail


def test_B_histogram_branch_in_one_facet_wrong_is_red_at_L2():
    adf = make_adf()
    drawn = T.control_hist_selection_facets(adf)
    patch = drawn[1].flat[1].patches[0]
    xy = patch.get_xy().copy()
    xy[1:3, 1] += 1
    patch.set_xy(xy)
    assert red(adf, HIST_FACETS, drawn).failure_level == "L2"


def test_B_ratio_value_or_count_wrong_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig57_preclean_ratio_y_vector_truth(adf)
    drawn[2]["normalize_data"].loc[5, "value"] = 0.5000001
    r = red(adf, C3, drawn)
    assert r.failure_level == "L2" and "ratio" in r.detail and "values" in r.detail
    adf = make_adf()
    drawn = G.fig57_preclean_ratio_y_vector_truth(adf)
    drawn[2]["normalize_data"].loc[5, "reference_count"] += 1
    assert "reference count" in red(adf, C3, drawn).detail


def test_B_weighted_std_wrong_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig44_weights_vector_facet_fit_oracle(adf)
    table = drawn[2][1]["per_group"]["1"]["profile_data"]
    table.loc[table.index[table["count"].to_numpy() > 1][0], "y_std"] *= 1.001
    r = red(adf, O2, drawn)
    assert r.failure_level == "L2" and "std" in r.detail


def test_B_surface_disagreement_is_red_at_L3():
    adf = make_adf()
    drawn = G.fig45_public_surface_equivalence_oracle(adf)
    table = drawn[2]["surface_stats"]["draw_batch"]["normalize_data"]
    table.loc[table.index[~table["mask_undefined"].to_numpy(bool)][0], "value"] += 1e-6
    r = red(adf, O4, drawn)
    assert r.failure_level == "L3" and "draw_batch value" in r.detail


def test_B_caller_spec_modified_is_red_at_L3():
    adf = make_adf()
    drawn = G.fig56_preclean_draw_figures_effective_defaults(adf)
    drawn[2]["caller_unchanged"] = False
    r = red(adf, C2, drawn)
    assert r.failure_level == "L3" and "unchanged" in r.detail


def test_B_evidence_page_follows_every_case_with_verdict_and_comparisons():
    from matplotlib.backends.backend_pdf import PdfPages
    texts = []

    class RecordingPdf:
        def attach_note(self, text): pass
        def savefig(self, fig, **k): texts.append(" ".join(t.get_text() for t in fig.texts))
    T.run_case(make_adf(), case(REFUSAL), RecordingPdf(), [0])               # machine-only: evidence page only
    assert len(texts) == 1 and REFUSAL in texts[0] and "verdict: PASS" in texts[0]
    texts.clear()
    T.run_case(make_adf(), case(C3), RecordingPdf(), [0])
    assert len(texts) == 2 and "verdict: PASS" in texts[1] and "ratio" in texts[1]


# ------------------------------------------------------------------ batch B v02: the three checks ported from the old harness
FACET_DELTA = "I4-REAL-SELECTION-VECTOR-FACET-DELTA-CORRECTNESS-EAGER-20PCT-01"


def test_B2_overlay_line_point_changed_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig22_delta_faceted(adf)
    line = T.profile_lines(T.facet_panels(drawn)[1])[1]           # reference curve of side_type=1
    y = np.asarray(line.get_ydata(), float).copy()
    y[3] += 1e-9
    line.set_ydata(y)
    r = red(adf, FACET_DELTA, drawn)
    assert r.failure_level == "L2" and "reference plotted line" in r.detail


def test_B2_overlay_point_off_the_bin_centres_is_red_at_L2():
    adf = make_adf()
    drawn = G.fig22_delta_faceted(adf)
    line = T.profile_lines(T.facet_panels(drawn)[0])[0]
    x = np.asarray(line.get_xdata(), float).copy()
    x[2] += 0.3
    line.set_xdata(x)
    assert "line x on bin centres" in red(adf, FACET_DELTA, drawn).detail


def test_B2_weights_branch_fit_differs_from_scalar_draw_is_red_at_L3():
    adf = make_adf()
    drawn = G.fig44_weights_vector_facet_fit_oracle(adf)
    cell = drawn[2][1]["per_group"]["0"]["fit"]
    while isinstance(cell[0], (list, tuple)):
        cell = cell[0]
    cell[0]["params"] = np.asarray(cell[0]["params"], float) + 1e-6
    r = red(adf, O2, drawn)
    assert r.failure_level == "L3" and "pol1 parameters" in r.detail and "branch 1" in r.detail


def test_B2_scalar_draw_runs_for_both_branches_and_both_facets():
    result = run_alone(case(O2))
    names = [c["observable"] for c in result.comparisons if c["level"] == "L3" and "pol1" in c["observable"]]
    assert result.status == T.PASS and len(names) == 4


def test_B2_branch_style_differs_between_facets_is_red_at_L3():
    adf = make_adf()
    drawn = G.fig44_weights_vector_facet_fit_oracle(adf)
    T.profile_lines(T.facet_panels(drawn)[1])[1].set_color("magenta")
    r = red(adf, O2, drawn)
    assert r.failure_level == "L3" and "branch styles" in r.detail


# ------------------------------------------------------------------ S3-FIX1: mutants the previous checker accepted
# Each test below uses only names that exist in the pre-fix module too, so against those bytes it is RED with
# "DID NOT RAISE": the old checker accepted the mutant. Against the fixed module it is GREEN: the case is red at the
# named comparison, at the plotted-result (L2) or public-call (L3) level.
C1_CASE = "PRECLEAN-C1-HIST-VECTOR-FACET-ACCEPTANCE-EAGER-20PCT-01"


def test_FIX1_mutant_C2_bin_centre_0_moved_by_5e13_is_red():           # historical 1e-14 absolute, was 1e-12
    adf = make_adf()
    drawn = G.fig56_preclean_draw_figures_effective_defaults(adf)
    table = drawn[2]["stats"]["normalize_data"]
    assert table.loc[0, "x_center"] == 0.0
    table.loc[0, "x_center"] = 5e-13
    assert "delta bin centres" in red(adf, C2, drawn).detail


def test_FIX1_mutant_C3_bin_centre_0_moved_by_5e13_is_red():
    adf = make_adf()
    drawn = G.fig57_preclean_ratio_y_vector_truth(adf)
    table = drawn[2]["normalize_data"]
    assert table.loc[0, "x_center"] == 0.0
    table.loc[0, "x_center"] = 5e-13
    assert "ratio bin centres" in red(adf, C3, drawn).detail


def test_FIX1_mutant_C3_signal_central_off_by_1p5e12_is_red():         # historical 1e-12 absolute, was 2e-12
    adf = make_adf()
    drawn = G.fig57_preclean_ratio_y_vector_truth(adf)
    table = drawn[2]["normalize_data"]
    i = table.index[table["signal_count"].to_numpy() > 0][0]
    table.loc[i, "signal_central"] += 1.5e-12 + 1e-12 * abs(table.loc[i, "signal_central"])
    r = red(adf, C3, drawn)
    assert r.failure_level == "L2" and "ratio signal central" in r.detail


def test_FIX1_mutant_O4_draw_figures_bin_centre_moved_by_5e13_is_red():  # historical 1e-14/1e-12, was 1e-12/1e-10
    adf = make_adf()
    drawn = G.fig45_public_surface_equivalence_oracle(adf)
    table = drawn[2]["surface_stats"]["draw_figures"]["normalize_data"]
    table.loc[14, "x_center"] += 5e-13                                   # the bin centred at -0.05
    r = red(adf, O4, drawn)
    assert r.failure_level == "L3" and "draw_figures x_center == draw" in r.detail


def test_FIX1_mutant_O4_draw_batch_count_not_equal_is_red():            # historical exact, was approximate
    adf = make_adf()
    drawn = G.fig45_public_surface_equivalence_oracle(adf)
    table = drawn[2]["surface_stats"]["draw_batch"]["normalize_data"]
    table["signal_count"] = table["signal_count"].astype(float)
    table.loc[table.index[table["signal_count"].to_numpy() > 0][0], "signal_count"] += 1e-9
    r = red(adf, O4, drawn)
    assert r.failure_level == "L3" and "draw_batch signal_count == draw" in r.detail


@pytest.mark.parametrize("case_id, facets", [(C1_CASE, lambda d: d[1]["facets"]), (HIST_FACETS, lambda d: d[1])])
def test_FIX1_mutant_facet_titles_swapped_is_red(case_id, facets):     # old: "never trust subplot order"
    adf = make_adf()
    drawn = case(case_id).draw(adf)
    first, second = np.asarray(facets(drawn), dtype=object).reshape(-1)[:2]
    first_title, second_title = first.get_title(), second.get_title()
    first.set_title(second_title)
    second.set_title(first_title)
    r = red(adf, case_id, drawn)
    assert r.failure_level == "L2" and "histogram ncl side_type=0" in r.detail


def test_FIX1_mutant_C1_facet_title_duplicated_is_red():
    adf = make_adf()
    drawn = G.fig55_preclean_hist_vector_facet_acceptance(adf)
    np.asarray(drawn[1]["facets"], dtype=object).reshape(-1)[2].set_title("side_type=1")
    r = red(adf, C1_CASE, drawn)
    assert r.failure_level == "L2" and "facet panels (by title)" in r.detail


@pytest.mark.parametrize("case_id", ["I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01",
                                     "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01", "ADF100-01", O2])
def test_FIX1_mutant_extra_vector_branch_is_red(case_id):              # old: "expected N branches"
    adf = make_adf()
    drawn = case(case_id).draw(adf)
    drawn[2].append(drawn[2][-1])
    r = red(adf, case_id, drawn)
    assert r.failure_level == "L2" and "number of vector branches" in r.detail


# ------------------------------------------------------------------ S3-FIX1: the historical-parity table and its guard
def test_FIX1_parity_table_has_the_66_declared_old_observables_of_the_21_migrated_cases():
    rows = [(c, row) for c, gates in T.HISTORICAL_GATES.items() for row in gates]
    assert sum(row[0] == "declared" for _, row in rows) == 66
    migrated = {c.case_id for c in T.CASES} - {REFUSAL}
    assert set(T.HISTORICAL_GATES) == migrated and len(migrated) == 21


def test_FIX1_parity_holds_for_every_case_on_the_honest_frame(tmp_path, monkeypatch):
    monkeypatch.setitem(T.RUN_CONTEXT, "gallery", {"ok": True, "pages": 59})
    code = T.run_cases(make_adf(), list(T.CASES), pdf_path=None, manifest_path=str(tmp_path / "m.json"),
                       provenance={"sample_mode": "FULL"})
    parity = json.load(open(tmp_path / "m.json"))["historical_parity"]
    assert code == 0 and parity["ok"] and parity["checked"] == len(parity["rows"]) == 84


def run_with(case_id: str, kind: type, edit=None) -> T.CaseResult:
    """Run one case with its declarations of type `kind` changed by `edit`, or dropped when edit is None."""
    import dataclasses
    the_case = case(case_id)
    expect = tuple(s if not isinstance(s, kind) else edit(s) for s in the_case.expect
                   if edit is not None or not isinstance(s, kind))
    return T.run_case(make_adf(), dataclasses.replace(the_case, expect=expect), pdf=None, page_counter=[0])


def test_FIX1_parity_guard_reports_a_weakened_tolerance():
    import dataclasses
    r = run_with(C3, T.Delta, lambda s: dataclasses.replace(s, signal_tol=T.C3_REFERENCE))   # the pre-fix C3 setting
    parity = T.historical_parity([r])
    assert r.status == T.PASS and not parity["ok"]
    assert parity["failed"] == [f"{C3} signal_central: WEAKER"]


def test_FIX1_parity_guard_reports_approximate_counts_across_surfaces():
    import dataclasses
    r = run_with(O4, T.SameAcrossSurfaces, lambda s: dataclasses.replace(s, columns={c: T.O4 for c in s.columns}))
    failed = T.historical_parity([r])["failed"]                                           # the pre-fix O4 setting
    assert r.status == T.PASS and sorted(failed) == sorted(f"{O4} {n}: WEAKER" for n in (
        "x_center", "signal_count", "reference_count", "mask_undefined"))


def test_FIX1_parity_guard_reports_a_dropped_check():
    r = run_with(C1_CASE, T.FacetPanels)
    assert r.status == T.PASS and T.historical_parity([r])["failed"] == [f"{C1_CASE} facet_identities: MISSING"]


def test_FIX1_parity_failure_fails_the_strict_gate(tmp_path, monkeypatch):
    import dataclasses
    weakened = dataclasses.replace(case(C3), expect=tuple(
        dataclasses.replace(s, signal_tol=T.C3_REFERENCE) if isinstance(s, T.Delta) else s for s in case(C3).expect))
    code = T.run_cases(make_adf(), [weakened], pdf_path=None, manifest_path=str(tmp_path / "m.json"),
                       provenance={}, mandatory=())
    rec = json.load(open(tmp_path / "m.json"))["reconciliation"]
    assert code == 1 and rec["gating"] == [] and rec["historical_parity_failed"] == [f"{C3} signal_central: WEAKER"]


# ------------------------------------------------------------------ manifest consumed by run_gallery.sh / parity
def test_manifest_has_the_fields_the_consumers_read(tmp_path, monkeypatch):
    monkeypatch.setitem(T.RUN_CONTEXT, "gallery", {"ok": True, "pages": 59})
    code = T.run_cases(make_adf(), list(T.CASES), pdf_path=str(tmp_path / "p.pdf"),
                       manifest_path=str(tmp_path / "m.json"), provenance={"sample_mode": "FULL"})
    manifest = json.load(open(tmp_path / "m.json"))
    assert code == 0 and set(manifest) >= {"cases", "provenance", "reconciliation", "schema_version", "historical_parity"}
    assert manifest["historical_parity"]["ok"] and manifest["reconciliation"]["historical_parity_failed"] == []
    rec = manifest["reconciliation"]
    assert rec["n_declared"] == rec["n_selected"] == rec["n_results"] == len(T.CASES) and rec["selection"] == "ALL"
    assert rec["missing_mandatory"] == []
    assert manifest["not_yet_migrated"] == list(T.NOT_YET_MIGRATED)
    for row in manifest["cases"]:
        assert set(row) >= {"case_id", "status", "failure_level", "comparisons", "detail", "page", "wall_time_s"}
    assert os.path.getsize(tmp_path / "p.pdf") > 1000

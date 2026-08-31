"""PHASE_13_77_ADF Stage A — fast self-checks for the invariance harness.

v1.2 §7.5: this suite is FAST and SYNTHETIC.  It must not require ROOT,
time_series_tracks_0.root, full gallery execution, or a multi-million-row
input.  The standalone harness owns the real-data ROOT execution.

What it proves:
    the surface adapter unwraps the three public envelopes correctly
    the adapter FAILS LOUDLY on a wrong path and never returns a silent default
    the shape-aware accessor raises on an unresolvable observable
    registry validation catches each violation class
    a corrupted result cannot report PASS   (self-falsification)
"""

import os
import sys
import json
from dataclasses import replace

import numpy as np
import pytest

# dfdraw and matplotlib are both required by the harness path under test.
# Name EVERY dependency: a guard that names one of two produces a red failure
# that looks like a contract defect on any environment missing the other.
pytest.importorskip("matplotlib")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("dfextensions.dfdraw")

import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(os.path.dirname(_HERE), "examples", "time_series")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

H = pytest.importorskip("time_series_draw_invariance")

from AliasDataFrame import AliasDataFrame as ADF  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

SPEC_HIST = {"expr": "x", "type": "hist", "bins": 12}
SPEC_FACET = {"expr": "y:x", "type": "profile", "bins": 10,
              "facet_by": "w", "facet_by_bins": 3}


@pytest.fixture
def frame():
    rng = np.random.default_rng(11)
    n = 600
    return pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n),
                         "w": rng.uniform(0.5, 1.5, n)})


@pytest.fixture
def mk(frame):
    return lambda: ADF(frame.copy())


def _full_figure_contract(case_id="X", **over):
    d = dict(expected_panels="one panel", panel_roles="main: the histogram",
             expected_traces="one bar series", expected_group_count="1",
             primary_comparison="hist statistics across public surfaces",
             residual_definition="per-observable difference vs the reference surface",
             accepted_envelope="counts exact; floats within declared rtol",
             case_ids=(case_id,), proof_kind="CONSISTENCY")
    d.update(over)
    return H.FigureContract(**d)


def _base_case(**over):
    """The MINIMAL case the registry accepts.

    A1 v03: this constructor is the thing that proved the omission — v02's
    version supplied fifteen fields, validation accepted it, and four ratified
    fields (applicability, setup_contract, preconditions, figure_contract) were
    therefore demonstrably not required.  It must now demonstrate the COMPLETE
    ratified minimum, or the same false-completeness returns.
    """
    d = dict(case_id="X", claim_id="C", title="t", claim="c",
             failure_means="f", expected_visual="v", owner_on_failure="ADF",
             purpose="INVARIANCE", gate="CORE_MANDATORY",
             oracle_kind="CONSISTENCY", loading_mode="EAGER",
             sample_mode="FULL", canonical_spec=SPEC_HIST,
             applicable=True, applicability_reason="",
             setup_contract="synthetic frame; no alias; EAGER; FULL",
             preconditions=("x is present and numeric",),
             figure_contract=_full_figure_contract(),
             negative_control="corrupt one surface's stats -> FAIL",
             surfaces_under_test=("draw", "draw_batch"),
             observables=(H.Observable("n", "STATS", "FLAT", "n"),))
    d.update(over)
    return H.CaseSpec(**d)


# ── 1. adapter: positive ────────────────────────────────────────────────────

@pytest.mark.parametrize("surface", H.SURFACES)
def test_a1_adapter_unwraps_each_public_surface(mk, surface):
    raw, kw = H._call(mk(), surface, SPEC_HIST)
    payload = H.unwrap(surface, raw, **kw)
    assert isinstance(payload.stats, dict), f"{surface}: stats is not a dict"
    assert "n" in payload.stats, f"{surface}: no 'n' in unwrapped stats"
    assert payload.path, f"{surface}: adapter recorded no payload path"


# ── 2. adapter: negative — must RAISE, never default ────────────────────────

def test_a1_adapter_raises_on_wrong_batch_case_key(mk):
    raw, _ = H._call(mk(), "draw_batch", SPEC_HIST)
    with pytest.raises(H.AdapterError):
        H.unwrap("draw_batch", raw, case_key="NO_SUCH_CASE")


def test_a1_adapter_raises_on_bad_plot_index(mk):
    raw, _ = H._call(mk(), "draw_figures", SPEC_HIST)
    with pytest.raises(H.AdapterError):
        H.unwrap("draw_figures", raw, plot_index=99)


def test_a1_adapter_raises_on_bad_figure_key(mk):
    raw, _ = H._call(mk(), "draw_figures", SPEC_HIST)
    with pytest.raises(H.AdapterError):
        H.unwrap("draw_figures", raw, figure_key="figure_9")


def test_a1_adapter_raises_on_wrong_return_shape():
    with pytest.raises(H.AdapterError):
        H.unwrap("draw", {"not": "a tuple"})


def test_a1_adapter_never_returns_a_silent_default(mk):
    """THE forbidden behaviour: wrong path -> empty dict -> PASS.

    Unwrapping a draw_figures envelope with the draw_batch rule must raise or
    return a non-empty payload; it must never yield {} that a comparator would
    then compare against {} and call equal.
    """
    raw, _ = H._call(mk(), "draw_figures", SPEC_HIST)
    try:
        payload = H.unwrap("draw_batch", raw, case_key="figure_0")
    except H.AdapterError:
        return  # raising is the correct outcome
    assert payload.stats, "adapter returned an empty default instead of raising"


# ── 3. shape-aware accessor ─────────────────────────────────────────────────

def test_a1_accessor_resolves_flat(mk):
    raw, kw = H._call(mk(), "draw", SPEC_HIST)
    stats = H.unwrap("draw", raw, **kw).stats
    assert H.resolve(stats, "n", "FLAT") > 0


def test_a1_accessor_raises_on_unresolvable_path(mk):
    raw, kw = H._call(mk(), "draw", SPEC_HIST)
    stats = H.unwrap("draw", raw, **kw).stats
    with pytest.raises(H.HarnessError):
        H.resolve(stats, "per_group.nope.n", "PER_GROUP")


def test_a1_accessor_rejects_array_declared_flat(mk):
    """autorange_used is a tuple; declaring it FLAT is a CaseSpec error."""
    raw, kw = H._call(mk(), "draw", SPEC_HIST)
    stats = H.unwrap("draw", raw, **kw).stats
    with pytest.raises(H.HarnessError):
        H.resolve(stats, "autorange_used", "FLAT")


def test_a1_faceted_stats_have_a_different_shape(mk):
    """Measured (survey a1_v01 §3): a faceted result carries n_groups/per_group
    and NONE of the flat statistical keys.  A flat accessor must not silently
    resolve nothing."""
    raw, kw = H._call(mk(), "draw", SPEC_FACET)
    stats = H.unwrap("draw", raw, **kw).stats
    assert H.resolve(stats, "n_groups", "FLAT") > 0
    with pytest.raises(H.HarnessError):
        H.resolve(stats, "n", "FLAT")


# ── 4. registry validation ──────────────────────────────────────────────────

def test_a1_registry_accepts_a_clean_case():
    assert H.validate_registry([_base_case()]) == []


@pytest.mark.parametrize("over,needle", [
    (dict(claim=""), "no claim"),
    (dict(owner_on_failure=""), "owner_on_failure"),
    (dict(loading_mode="LAZY", sample_mode="FRACTION"), "sampled-lazy"),
    (dict(known_bug_status="KNOWN_BUG"), "known_bug_id"),
    (dict(oracle_kind="CORRECTNESS"), "conflated"),
    (dict(observables=()), "declares no observable"),
])
def test_a1_registry_rejects_violation(over, needle):
    violations = H.validate_registry([_base_case(**over)])
    assert any(needle in v for v in violations), \
        f"expected a violation containing {needle!r}; got {violations}"


def test_a1_registry_rejects_floating_comparator_without_rationale():
    case = _base_case(observables=(
        H.Observable("mean", "STATS", "FLAT", "mean",
                     comparator="close", atol=1e-9),))
    assert any("rationale" in v for v in H.validate_registry([case]))


def test_a1_registry_rejects_duplicate_case_id():
    assert any("duplicate" in v
               for v in H.validate_registry([_base_case(), _base_case()]))


# ── 5. runner statuses ──────────────────────────────────────────────────────

def test_a1_consistency_case_passes_across_three_surfaces(mk):
    case = _base_case(case_id="INV-SURFACE-01",
                      surfaces_under_test=H.SURFACES,
                      observables=(H.Observable("n", "STATS", "FLAT", "n"),
                                   H.Observable("n_input", "STATS", "FLAT", "n_input")))
    res = H.run_consistency(case, mk)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)


def test_a1_undeclarable_observable_is_invalid_fixture_not_pass(mk):
    """§14: a case may not PASS when an observable it claims to compare was
    never extracted."""
    case = _base_case(observables=(
        H.Observable("ghost", "STATS", "FLAT", "no_such_field"),))
    res = H.run_consistency(case, mk)
    assert res.status == H.INVALID_FIXTURE, res.detail


def test_a1_correctness_case_agrees_with_numpy(mk):
    case = _base_case(case_id="COR-HIST-01", purpose="CORRECTNESS",
                      oracle_kind="CORRECTNESS", surfaces_under_test=("draw",),
                      observables=(
                          H.Observable("n", "INDEPENDENT", "FLAT", "n"),
                          H.Observable("mean", "INDEPENDENT", "FLAT", "mean",
                                       comparator="close", rtol=1e-10,
                                       rationale="independent reduction, different "
                                                 "summation order")))

    def anchor(df):
        v = np.asarray(df["x"], dtype=float)
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "mean": float(v.mean())}

    res = H.run_correctness(case, mk, anchor)
    assert res.status == H.PASS, res.detail


def test_a1_error_contract_draw_figures_refuses_facet_by(mk):
    """KNOWN_BUG: the ADF guard over BUG_dfdraw_20260611_facet_by_ax_ignored.

    If this ever FAILS the guard is gone and a faceted panel may render into
    the wrong axes.  When dfdraw fixes the layout defect this case is promoted,
    deliberately, to a three-surface invariance case.
    """
    case = _base_case(case_id="KB-FIGURES-FACET-01", purpose="ERROR_CONTRACT",
                      canonical_spec=SPEC_FACET,
                      known_bug_status="KNOWN_BUG",
                      known_bug_id="BUG_dfdraw_20260611_facet_by_ax_ignored")
    res = H.run_error_contract(case, mk, "draw_figures",
                               "facet_by is not supported")
    assert res.status == H.PASS, res.detail


# ── 6. self-falsification — a corrupted result must not green ───────────────

def test_a1_mutation_corrupted_surface_cannot_pass(mk, monkeypatch):
    """M1 (v1.2 §14): numeric corruption -> the gate FAILS.

    Without this the whole suite could be green because nothing is compared.
    """
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES)
    original = H.unwrap

    def poisoned(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_batch" and isinstance(payload.stats, dict):
            corrupted = dict(payload.stats)
            corrupted["n"] = corrupted["n"] + 1
            return H.Payload(surface, corrupted, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", poisoned)
    res = H.run_consistency(case, mk)
    assert res.status == H.FAIL, \
        "a corrupted surface reported PASS — the oracle compares nothing"
    assert "mismatch" in res.detail


def test_a1_strict_exit_code_is_non_zero_on_mandatory_failure():
    case = _base_case(case_id="M1")
    failing = H.CaseResult(case_id="M1", status=H.FAIL, detail="x")
    assert H.strict_exit_code([failing], [case]) == 1


def test_a1_strict_exit_code_ignores_a_failing_known_bug():
    case = _base_case(case_id="KB", known_bug_status="KNOWN_BUG",
                      known_bug_id="BUG_x")
    failing = H.CaseResult(case_id="KB", status=H.FAIL, detail="x")
    assert H.strict_exit_code([failing], [case]) == 0


def test_a1_strict_exit_code_always_gates_on_invalid_fixture():
    case = _base_case(case_id="IF", known_bug_status="KNOWN_BUG",
                      known_bug_id="BUG_x")
    bad = H.CaseResult(case_id="IF", status=H.INVALID_FIXTURE, detail="x")
    assert H.strict_exit_code([bad], [case]) == 1


# ── 7. v02 falsifiers — one per finding GPT32 reproduced against v01 ────────

def test_a1_p0_1_case_comparing_nothing_cannot_pass(mk):
    """A1-P0-1.  v01: every observable NOT_EXTRACTABLE -> registry [] -> PASS."""
    case = _base_case(observables=(
        H.Observable("ghost", "STATS", "FLAT", "nope", status="NOT_EXTRACTABLE"),))
    assert any("no EXECUTED observable" in v
               for v in H.validate_registry([case])), "registry accepted it"
    res = H.run_consistency(case, mk)
    assert res.status == H.INVALID_FIXTURE, res.detail
    assert res.executed_comparisons == 0


def test_a1_p0_2_anchor_is_immune_to_product_mutation():
    """A1-P0-2.  v01 computed anchor(adf.df) AFTER draw(); a product that
    mutated its own frame made the 'independent' oracle agree with the wrong
    answer.  Measured: pristine 15.0, public 0.0, independent 0.0, PASS."""
    class Poison(ADF):
        def draw(self, expr, **kw):
            self.df["x"] = 0.0
            return super().draw(expr, **kw)

    case = _base_case(case_id="COR-POISON", purpose="CORRECTNESS",
                      oracle_kind="CORRECTNESS", surfaces_under_test=("draw",),
                      canonical_spec={"expr": "x", "type": "hist", "bins": 2},
                      observables=(H.Observable(
                          "mean", "INDEPENDENT", "FLAT", "mean",
                          comparator="close", rtol=1e-9,
                          rationale="independent reduction"),))

    def anchor(df):
        return {"mean": float(np.asarray(df["x"], dtype=float).mean())}

    res = H.run_correctness(
        case, lambda: Poison(pd.DataFrame({"x": np.array([10.0, 20.0])})), anchor)
    assert res.status == H.FAIL, "the anchor read product-mutated state"
    assert res.observed["mean"]["independent"] == 15.0


def test_a1_p0_3_mandatory_skip_fails_closed():
    """A1-P0-3.  v01 returned 0, so a malformed mandatory case vanished."""
    case = _base_case(case_id="S1")
    skipped = H.CaseResult(case_id="S1", status=H.SKIP, detail="x")
    assert H.strict_exit_code([skipped], [case]) == 1


def test_a1_p0_3_environment_gated_skip_is_still_legal():
    """SUPERSEDED BY v06 — this test asserted the A1-v05-P0-1 DEFECT.

    v04 wrote it as `ENVIRONMENT_GATED + SKIP -> 0` with the default
    applicable=True, which is exactly the silent non-proof P0-1 reports.  The
    property v04 MEANT to protect is that an UNAVAILABLE environment may skip,
    and that is what it now asserts.  The applicable variant is covered by
    test_a1_v06_applicable_skip_always_gates.
    """
    case = _base_case(case_id="EG", gate="ENVIRONMENT_GATED", applicable=False,
                      applicability_reason="no ROOT")
    skipped = H.CaseResult(case_id="EG", status=H.SKIP, detail="no ROOT")
    assert H.strict_exit_code([skipped], [case]) == 0


def test_a1_p0_3_registry_rejects_a_mandatory_case_that_can_only_skip():
    case = _base_case(surfaces_under_test=("draw",))
    assert any("would never gate" in v for v in H.validate_registry([case]))


def test_a1_p1_1_unknown_comparator_is_refused(mk):
    """A1-P1-1.  v01 fell through to `close` for any unknown name."""
    case = _base_case(observables=(
        H.Observable("n", "STATS", "FLAT", "n", comparator="banana"),))
    assert any("unknown comparator" in v for v in H.validate_registry([case]))
    with pytest.raises(H.HarnessError):
        H.comparator_for(H.Observable("n", "STATS", "FLAT", "n",
                                      comparator="banana"))


def test_a1_p1_2_unimplemented_oracle_source_is_refused():
    """A1-P1-2.  ARTIST_FALLBACK was declarable while stats were actually read."""
    case = _base_case(observables=(
        H.Observable("n", "ARTIST_FALLBACK", "FLAT", "n", rationale="r"),))
    assert any("not executable by this runner" in v
               for v in H.validate_registry([case]))


def test_a1_p1_3_error_contract_failure_always_gates():
    """A1-P1-3.  known_bug_id is provenance for WHY a guard exists, not a
    licence to ignore the guard disappearing."""
    case = _base_case(case_id="EC", purpose="ERROR_CONTRACT",
                      known_bug_status="KNOWN_BUG", known_bug_id="BUG_x")
    failing = H.CaseResult(case_id="EC", status=H.FAIL, detail="guard gone")
    assert H.strict_exit_code([failing], [case]) == 1


def test_a1_p1_4_manifest_carries_the_comparison_contract(mk, tmp_path):
    """A1-P1-4.  v01 serialised observed values only, so the JSON could not
    say WHICH comparison ran, with what comparator or tolerance."""
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES,
                      observables=(H.Observable("n", "STATS", "FLAT", "n"),
                                   H.Observable("mean", "STATS", "FLAT", "mean",
                                                comparator="close", rtol=1e-12,
                                                rationale="float reduction")))
    res = H.run_consistency(case, mk)
    assert res.status == H.PASS, res.detail
    assert res.executed_comparisons > 0
    doc = H.write_manifest(str(tmp_path / "m.json"), [res], [case])
    rec = doc["cases"][0]
    assert rec["executed_comparisons"] > 0
    assert rec["declared_observables"], "manifest carries no observable contract"
    mean = [o for o in rec["declared_observables"] if o["name"] == "mean"][0]
    assert mean["comparator"] == "close" and mean["rtol"] == 1e-12
    assert mean["rationale"]


# ── 8. v03 — the ratified minimum contract and single-source footer ─────────

@pytest.mark.parametrize("over,needle", [
    (dict(setup_contract=""), "no setup_contract"),
    (dict(preconditions=()), "no preconditions"),
    (dict(figure_contract=None), "no figure_contract"),
    (dict(applicable=False, gate="ENVIRONMENT_GATED"), "applicability_reason"),
    (dict(applicable=False), "always applicable"),
])
def test_a1_v03_registry_requires_the_ratified_minimum(over, needle):
    """P0-COMPLETE-1.  v02 accepted a case missing all four ratified fields,
    and test_a1_registry_accepts_a_clean_case PROVED they were not required."""
    violations = H.validate_registry([_base_case(**over)])
    assert any(needle in v for v in violations), \
        f"expected a violation containing {needle!r}; got {violations}"


@pytest.mark.parametrize("field", H.FigureContract.REQUIRED)
def test_a1_v03_figure_contract_requires_each_of_its_nine_fields(field):
    """v1.2 §6 names nine required fields; a partial contract is not a contract."""
    fc = _full_figure_contract(**{field: ""})
    case = _base_case(figure_contract=fc)
    violations = H.validate_registry([case])
    assert any("figure_contract missing" in v and field in v
               for v in violations), violations


def test_a1_v03_footer_carries_the_three_required_blocks():
    case = _base_case()
    text = H.footer_text(case)
    for block in ("EXPECTED:", "MACHINE ORACLE:", "FAILURE MEANS:"):
        assert block in text, f"footer is missing {block}"
    assert case.case_id in text


def test_a1_v03_footer_and_manifest_share_one_source(mk, tmp_path):
    """P0-COMPLETE-2 — the REQUIREMENT is one structured source, not that a
    footer exists.  v1.2 §5: the PDF footer and the JSON explanation must be
    generated from the same structured source.

    This test fails if either is ever authored separately: change an
    observable's comparator and BOTH must move together.
    """
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES,
                      observables=(H.Observable("n", "STATS", "FLAT", "n"),
                                   H.Observable("mean", "STATS", "FLAT", "mean",
                                                comparator="close", rtol=1e-12,
                                                rationale="float reduction")))
    res = H.run_consistency(case, mk)
    assert res.status == H.PASS, res.detail
    doc = H.write_manifest(str(tmp_path / "m.json"), [res], [case])
    rec = doc["cases"][0]

    oracle = H.machine_oracle_text(case)
    assert rec["machine_oracle"] == oracle
    assert oracle in rec["footer"], "footer does not contain the manifest oracle"
    assert "rtol=1e-12" in oracle, "the generated text lost the declared tolerance"

    # the drift check: perturb the DECLARATION, both renderings must follow.
    moved = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES,
                       observables=(H.Observable("n", "STATS", "FLAT", "n"),
                                    H.Observable("mean", "STATS", "FLAT", "mean",
                                                 comparator="close", rtol=1e-3,
                                                 rationale="float reduction")))
    assert H.machine_oracle_text(moved) != oracle
    assert H.footer_text(moved) != H.footer_text(case)


def test_a1_v03_manifest_carries_the_new_contract_fields(mk, tmp_path):
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES)
    res = H.run_consistency(case, mk)
    doc = H.write_manifest(str(tmp_path / "m.json"), [res], [case])
    rec = doc["cases"][0]
    for key in ("applicable", "applicability_reason", "setup_contract",
                "preconditions", "figure_contract", "machine_oracle", "footer"):
        assert key in rec, f"manifest omits {key!r}"
    assert rec["figure_contract"]["primary_comparison"]


# ── 9. v04 — the three findings GPT31/GPT32 reproduced against v03 ──────────

def test_a1_v04_inapplicable_case_never_invokes_the_product():
    """A1-v03-P0-1.  v03 validated `applicable` and every runner ignored it:
    measured, the frame factory was called twice and the case returned PASS.

    The factory RAISES if called.  An unavailable environment is unavailable —
    touching it is the failure the SKIP exists to avoid.
    """
    def must_not_run():
        raise AssertionError("the product was invoked for an inapplicable case")

    case = _base_case(gate="ENVIRONMENT_GATED", applicable=False,
                      applicability_reason="no ROOT on this host")
    res = H.run_consistency(case, must_not_run)
    assert res.status == H.SKIP
    assert "no ROOT on this host" in res.detail


def test_a1_v04_inapplicable_short_circuits_every_runner():
    def must_not_run():
        raise AssertionError("the product was invoked for an inapplicable case")

    case = _base_case(gate="ENVIRONMENT_GATED", applicable=False,
                      applicability_reason="unavailable")
    assert H.run_consistency(case, must_not_run).status == H.SKIP
    assert H.run_correctness(case, must_not_run,
                             lambda df: {}).status == H.SKIP
    assert H.run_error_contract(case, must_not_run, "draw", "x").status == H.SKIP


@pytest.mark.parametrize("over,needle", [
    (dict(case_ids=()), "case_ids is empty"),
    (dict(case_ids=("SOMETHING_ELSE",)), "case_ids"),
    (dict(proof_kind="CORRECTNESS"), "proof_kind"),
])
def test_a1_v04_figure_contract_may_not_contradict_its_case(over, needle):
    """A1-v03-P0-2.  v03 required the nine fields to be NON-EMPTY and never
    asked whether their CONTENT agreed with the case.  A contract naming a
    different case is worse than none — a page describing something else,
    validating clean.

    IMPLEMENTATION RULING, not ratified text: v1.2 §6 requires the fields to
    exist and nowhere requires agreement.
    """
    case = _base_case(figure_contract=_full_figure_contract(**over))
    violations = H.validate_registry([case])
    assert any("contradicts the case" in v and needle in v
               for v in violations), violations


def test_a1_v04_aligned_figure_contract_is_accepted():
    """The positive control: a rule that refused everything would pass the
    three falsifiers above and be useless."""
    assert H.validate_registry([_base_case()]) == []


@pytest.mark.parametrize("nc,ok", [
    ("", False),
    ("corrupt one surface -> FAIL", True),
    ("GLOBAL_MUTATION:", False),
    ("GLOBAL_MUTATION:M1", True),
    ("FAMILY_MUTATION:hist-anchor", True),
])
def test_a1_v04_every_machine_gated_case_has_a_falsification_owner(nc, ok):
    """A1-v03-COMPLETE-3.  v1.2 §3.1 lists negative_control / required_mutation
    in the minimum schema.  The objective is not a mutation per case — it is
    that no machine-gated case silently has NO falsification owner.  A shared
    harness-level falsifier is a legitimate answer, but it must be NAMED.
    """
    violations = H.validate_registry([_base_case(negative_control=nc)])
    has = any("falsification owner" in v or "gives no id" in v
              for v in violations)
    assert has != ok, f"negative_control={nc!r}: {violations}"


# ── 10. v05 — the class, not only its instances ────────────────────────────

def test_a1_v05_applicable_failure_gates_whatever_the_gate_class():
    """A1-v04-P0-1.  ENVIRONMENT_GATED buys the right to SKIP when the
    environment is UNAVAILABLE, not the right to fail silently when the case
    actually ran.  v04 measured strict_exit_code 0 on a genuine FAIL."""
    case = _base_case(case_id="EG", gate="ENVIRONMENT_GATED")
    failing = H.CaseResult(case_id="EG", status=H.FAIL, detail="real failure")
    assert H.strict_exit_code([failing], [case]) == 1


def test_a1_v05_unavailable_environment_may_still_skip():
    """The positive control: the v04 fix must not be undone."""
    case = _base_case(case_id="EG", gate="ENVIRONMENT_GATED", applicable=False,
                      applicability_reason="no ROOT")
    skipped = H.CaseResult(case_id="EG", status=H.SKIP, detail="no ROOT")
    assert H.strict_exit_code([skipped], [case]) == 0


def test_a1_v05_declared_source_must_match_the_executed_path(mk):
    """A1-v04-P0-2.  Third instance of one class: a label that does not
    describe the executed path.  v04 validated that a source was
    *implementable* and never checked it against the runner."""
    case = _base_case(case_id="MIS", observables=(
        H.Observable("n", "INDEPENDENT", "FLAT", "n"),))
    res = H.run_consistency(case, mk)
    assert res.status == H.INVALID_FIXTURE, res.detail
    assert "executes STATS" in res.detail


def test_a1_v05_correctness_runner_rejects_a_stats_label(mk):
    case = _base_case(case_id="MIS2", purpose="CORRECTNESS",
                      oracle_kind="CORRECTNESS", surfaces_under_test=("draw",),
                      figure_contract=_full_figure_contract(
                          "MIS2", proof_kind="CORRECTNESS"),
                      observables=(H.Observable("n", "STATS", "FLAT", "n"),))
    res = H.run_correctness(case, mk, lambda df: {"n": len(df)})
    assert res.status == H.INVALID_FIXTURE, res.detail
    assert "executes INDEPENDENT" in res.detail


def test_a1_v05_matching_source_labels_still_execute(mk):
    """Positive control: the check must not refuse every case."""
    assert H.run_consistency(_base_case(), mk).status == H.PASS


def test_a1_v05_no_caseSpec_field_is_declared_but_unread():
    """A1-v04-P1-1, and the CLASS behind four increments of findings.

    v01 oracle_source, v03 applicable, v04 source, v04 slots_under_test /
    reference_policy — each fixed individually, the class never closed.  This
    test fails the day a field is added with no reader, instead of the defect
    surviving to the next review round.
    """
    orphans = H.audit_declared_state()
    assert orphans == [], (
        "CaseSpec fields declared but not read, validated or serialised:\n  "
        + "\n  ".join(orphans))


def test_a1_v05_audit_detects_a_planted_orphan():
    """SUPERSEDED BY v06 — the mechanism this tested no longer exists.

    v05's negative control removed a name from the hand-maintained
    CONSUMED_FIELDS set.  A1-v05-P1-1 found that allow-list was already
    false-certifying three fields, so v06 DERIVES consumption from the module
    AST instead and the set is gone.  The replacement plants a real field with
    no reader: test_a1_v06_audit_detects_a_field_with_no_reader.
    """
    assert not hasattr(H, "CONSUMED_FIELDS")


def test_a1_v05_future_staged_fields_are_recorded_with_their_stage(mk, tmp_path):
    """A4.1 ownership transfer: only genuinely future-owned fields remain here.

    Before A4 this test required slots_under_test to be FUTURE_STAGE:A4.  A4.1
    now gives that field (and anti_contamination_preconditions) executable
    readers, validation and manifest evidence, so keeping the old expectation
    would protect stale governance state rather than the A1 invariant.
    """
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES)
    res = H.run_consistency(case, mk)
    doc = H.write_manifest(str(tmp_path / "m.json"), [res], [case])
    fs = doc["cases"][0]["future_staged"]
    assert set(fs) == {"reference_policy"}
    assert fs["reference_policy"]["owning_stage"] == "A6"
    assert doc["cases"][0]["slots_under_test"] == []
    assert doc["cases"][0]["anti_contamination_preconditions"] == []


# ── 11. v06 — the derived audit and the enumerated gate matrix ─────────────

def test_a1_v06_applicable_skip_always_gates():
    """A1-v05-P0-1.  An applicable case that proved nothing is a silent
    non-proof.  v05 measured strict exit 0 for ENVIRONMENT_GATED + applicable
    + SKIP."""
    for gate in ("CORE_MANDATORY", "ENVIRONMENT_GATED"):
        case = _base_case(case_id="S", gate=gate)
        skipped = H.CaseResult(case_id="S", status=H.SKIP, detail="x")
        assert H.strict_exit_code([skipped], [case]) == 1, gate


def test_a1_v06_unavailable_environment_may_still_skip():
    case = _base_case(case_id="S", gate="ENVIRONMENT_GATED", applicable=False,
                      applicability_reason="no ROOT")
    skipped = H.CaseResult(case_id="S", status=H.SKIP, detail="no ROOT")
    assert H.strict_exit_code([skipped], [case]) == 0


def test_a1_v06_environment_blocked_may_not_be_applicable():
    """A1-v05-P1-2.  A blocked environment is not an applicable one."""
    case = _base_case(gate="ENVIRONMENT_GATED", known_bug_status="ENVIRONMENT_BLOCKED",
                      known_bug_id="ENV_x", applicable=True)
    assert any("contradictory" in v for v in H.validate_registry([case]))


@pytest.mark.parametrize("applicable,status", [
    (True, "PASS"), (False, "PASS"),
    (True, "FAIL"), (False, "FAIL"),
    (True, "SKIP"), (False, "SKIP"),
    (True, "INVALID_FIXTURE"), (False, "INVALID_FIXTURE"),
    (True, "DIAGNOSTIC"), (False, "DIAGNOSTIC"),
])
def test_a1_v06_every_gate_state_has_a_defined_verdict(applicable, status):
    """The class behind P0-1 and P1-2: I kept fixing ONE cell of
    gate x applicable x known_bug_status x status.  Every reachable
    combination must now yield a verdict AND a reason."""
    case = _base_case(gate="ENVIRONMENT_GATED", applicable=applicable,
                      applicability_reason="r" if not applicable else "")
    res = H.CaseResult(case_id=case.case_id, status=status)
    gates, why = H.gate_decision(case, res)
    assert isinstance(gates, bool)
    assert why, f"({applicable}, {status}) has no stated reason"


def test_a1_v06_unknown_status_fails_closed():
    case = _base_case()
    res = H.CaseResult(case_id=case.case_id, status="SOMETHING_NEW")
    gates, why = H.gate_decision(case, res)
    assert gates and "fail closed" in why


def test_a1_v06_audit_is_derived_not_declared():
    """A1-v05-P1-1.  v05's hand-maintained allow-list was not merely gameable
    — it was ALREADY WRONG: title, anti_contamination_preconditions and
    schema_version had no reader and the audit reported clean."""
    assert H.audit_declared_state() == [], H.audit_declared_state()
    assert not hasattr(H, "CONSUMED_FIELDS"), \
        "the hand-maintained allow-list is still present"
    read = H._fields_read_in_this_module()
    # v07: schema_version was REMOVED from CaseSpec (A1-v06-P1-2), so only
    # `title` remains from the pair v06 gave readers to.
    assert "title" in read, "title still has no reader"


def test_a1_v06_audit_detects_a_field_with_no_reader():
    """The audit's own negative control, against the DERIVED mechanism: a new
    field with no reader must be named."""
    import dataclasses
    orig = H.CaseSpec
    planted = dataclasses.make_dataclass(
        "CaseSpec", [("unread_new_field", str, dataclasses.field(default=""))],
        bases=(orig,))
    H.CaseSpec = planted
    try:
        orphans = H.audit_declared_state()
    finally:
        H.CaseSpec = orig
    assert any("unread_new_field" in o for o in orphans), orphans


def test_a1_v06_manifest_carries_title_and_case_schema_version(mk, tmp_path):
    """SUPERSEDED BY v07 — the per-case schema_version field is gone.

    A1-v06-P1-2: a field whose only legal value is a module constant carries no
    information and created a second schema authority.  provenance() records
    the module version once for the run.  `title` is still asserted.
    """
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES)
    res = H.run_consistency(case, mk)
    doc = H.write_manifest(str(tmp_path / "m.json"), [res], [case])
    assert doc["cases"][0]["title"] == case.title
    assert doc["provenance"]["schema_version"] == H.SCHEMA_VERSION


# ── 12. v07 — declared-set gating, receiver-narrowed audit, single schema ──

def test_a1_v07_declared_case_with_no_result_gates():
    """A1-v06-P0-1.  v06 iterated `results`, so a declared case that produced
    NO result was invisible: measured, 2 declared / 1 result -> exit 0."""
    cases = [_base_case(case_id="A"), _base_case(case_id="B_NEVER_RAN")]
    results = [H.CaseResult(case_id="A", status=H.PASS)]
    assert H.strict_exit_code(results, cases) == 1


def test_a1_v07_duplicate_result_gates():
    case = _base_case(case_id="A")
    dup = [H.CaseResult(case_id="A", status=H.PASS),
           H.CaseResult(case_id="A", status=H.PASS)]
    assert H.strict_exit_code(dup, [case]) == 1


def test_a1_v07_result_for_an_undeclared_case_gates():
    case = _base_case(case_id="A")
    stray = [H.CaseResult(case_id="A", status=H.PASS),
             H.CaseResult(case_id="GHOST", status=H.PASS)]
    assert H.strict_exit_code(stray, [case]) == 2


def test_a1_v07_complete_declared_set_still_passes():
    """Positive control: a rule that gated everything would pass the three
    falsifiers above and be useless."""
    cases = [_base_case(case_id="A"), _base_case(case_id="B")]
    results = [H.CaseResult(case_id="A", status=H.PASS, executed_comparisons=1),
               H.CaseResult(case_id="B", status=H.PASS, executed_comparisons=1)]
    assert H.strict_exit_code(results, cases) == 0


def test_a1_v07_coverage_gaps_names_the_dropped_case():
    cases = [_base_case(case_id="A"), _base_case(case_id="B_NEVER_RAN")]
    gaps = H.coverage_gaps([H.CaseResult(case_id="A", status=H.PASS)], cases)
    assert any("B_NEVER_RAN" in g and "no result" in g for g in gaps), gaps


def test_a1_v07_audit_is_receiver_narrowed():
    """A1-v06-P1-1.  v06 counted any same-named attribute anywhere, so
    `CaseResult.status` certified a hypothetical `CaseSpec.status`.  I named
    this hole in the v06 CRR as an attack point and shipped it anyway."""
    read = H._fields_read_in_this_module()
    assert "status" not in read, \
        "CaseResult.status is still leaking into the CaseSpec read-set"
    assert "detail" not in read
    assert H.audit_declared_state() == []


def test_a1_v07_audit_detects_a_name_colliding_field():
    """The negative control the collision demands: a NEW CaseSpec field whose
    name collides with a heavily-used CaseResult attribute."""
    import dataclasses
    orig = H.CaseSpec
    planted = dataclasses.make_dataclass(
        "CaseSpec", [("status", str, dataclasses.field(default=""))],
        bases=(orig,))
    H.CaseSpec = planted
    try:
        orphans = H.audit_declared_state()
    finally:
        H.CaseSpec = orig
    assert any("CaseSpec.status" in o for o in orphans), orphans


def test_a1_v07_iterating_a_field_does_not_promote_its_element():
    """`for o in case.observables` yields Observables, not CaseSpecs.  An
    earlier v07 draft walked through the Attribute and promoted `o`, putting
    Observable.status back into the read-set — the same receiver-blindness one
    level down."""
    read = H._fields_read_in_this_module()
    for observable_only in ("comparator", "atol", "rtol"):
        assert observable_only not in read, \
            f"{observable_only} is an Observable field and must not certify a CaseSpec field"


def test_a1_v07_no_per_case_schema_version():
    """A1-v06-P1-2.  Two schema authorities, one of them wrong."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(H.CaseSpec)}
    assert "schema_version" not in names
    with pytest.raises(TypeError):
        _base_case(schema_version="1.0.0-ANCIENT")


@pytest.mark.parametrize("applicable,status,expected", [
    (True,  "PASS",            False),
    (False, "PASS",            False),
    (True,  "FAIL",            True),
    (False, "FAIL",            False),
    (True,  "SKIP",            True),
    (False, "SKIP",            False),
    (True,  "INVALID_FIXTURE", True),
    (False, "INVALID_FIXTURE", True),
    (True,  "DIAGNOSTIC",      False),
    (False, "DIAGNOSTIC",      False),
    (True,  "SOMETHING_NEW",   True),
])
def test_a1_v07_gate_matrix_asserts_the_expected_verdict(applicable, status, expected):
    """A1-v06-P1-3.  v06 asserted only that a bool and a reason EXIST, so a
    wrong verdict passed.  This pins the policy, not its shape."""
    case = _base_case(gate="ENVIRONMENT_GATED", applicable=applicable,
                      applicability_reason="r" if not applicable else "")
    result = H.CaseResult(
        case_id=case.case_id,
        status=status,
        # A2 makes a PASS-with-zero-comparisons a false green.  This older
        # A1 gate-policy matrix is not testing that invariant; its PASS rows
        # therefore model a genuine successful numerical comparison.
        executed_comparisons=1 if status == "PASS" else 0,
    )
    gates, why = H.gate_decision(case, result)
    assert gates is expected, f"({applicable}, {status}): {why}"
    assert why


# ── 13. v08 — empty set, per-scope receivers, one reconciliation authority ──

def test_a1_v08_empty_declared_set_is_refused_at_both_doors():
    """A1-v07-P0-1.  v07 spent the whole increment making the gate iterate the
    DECLARED set, and never asked what happens when that set is EMPTY.
    Measured: validate_registry([]) == [] and strict_exit_code([], []) == 0 —
    a harness that ran nothing reporting that everything is fine."""
    assert any("EMPTY" in v for v in H.validate_registry([]))
    assert H.strict_exit_code([], []) == 1


def test_a1_v08_receiver_evidence_is_scope_local(tmp_path):
    """A1-v07-P1-1.  v07 built a MODULE-GLOBAL receiver-name set, so proving
    `case` is a CaseSpec in one function certified `case.<anything>` in every
    other function.  A variable name is not a type identity, and it is not one
    across lexical scopes either."""
    import importlib.util
    src = open(H.__file__).read() + (
        '\n\nclass Other:\n    status = "x"\n\n\n'
        'def unrelated(case):\n    return case.status\n')
    mod_path = tmp_path / "poisoned_harness.py"
    mod_path.write_text(src)
    spec = importlib.util.spec_from_file_location("poisoned_harness", mod_path)
    poisoned = importlib.util.module_from_spec(spec)
    sys.modules["poisoned_harness"] = poisoned
    spec.loader.exec_module(poisoned)
    try:
        assert "status" not in poisoned._fields_read_in_this_module(), \
            "an unrelated same-named receiver still certifies a CaseSpec field"
        import dataclasses
        orig = poisoned.CaseSpec
        planted = dataclasses.make_dataclass(
            "CaseSpec", [("status", str, dataclasses.field(default=""))],
            bases=(orig,))
        poisoned.CaseSpec = planted
        try:
            orphans = poisoned.audit_declared_state()
        finally:
            poisoned.CaseSpec = orig
        assert any("CaseSpec.status" in o for o in orphans), orphans
    finally:
        sys.modules.pop("poisoned_harness", None)


@pytest.mark.parametrize("ann,expected", [
    ("CaseSpec", True), ("Sequence[CaseSpec]", True), ("CaseSpec | None", True),
    ("CaseSpecView", False), ("NotACaseSpec", False), ("FakeCaseSpec", False),
    ("CaseResult", False),
])
def test_a1_v08_annotation_match_is_exact_not_substring(ann, expected):
    """A1-v07-P1-3.  v07 tested `"CaseSpec" in <string>`, admitting any name
    containing the token."""
    import ast
    node = ast.parse(repr(ann), mode="eval").body
    assert H._is_casespec_annotation(node) is expected


def test_a1_v08_missing_declared_case_appears_in_the_manifest(tmp_path):
    """A1-v07-P1-2.  v07's gate knew a case was dropped; the manifest did not.
    Measured: declared ['A','B_NEVER_RAN'], manifest listed ['A']."""
    cases = [_base_case(case_id="A"), _base_case(case_id="B_NEVER_RAN")]
    doc = H.write_manifest(str(tmp_path / "m.json"),
                           [H.CaseResult(case_id="A", status=H.PASS)], cases)
    ids = [c["case_id"] for c in doc["cases"]]
    assert "B_NEVER_RAN" in ids, ids
    dropped = [c for c in doc["cases"] if c["case_id"] == "B_NEVER_RAN"][0]
    assert dropped["status"] == "NO_RESULT"
    assert "B_NEVER_RAN" in doc["reconciliation"]["missing"]


def test_a1_v08_gate_and_coverage_derive_from_one_authority():
    """A1-v07-P1-2 / P2-1.  Two functions computing overlapping facts is how
    they drift — I named this in the v07 CRR §7 item 4 and shipped it."""
    cases = [_base_case(case_id="A"), _base_case(case_id="B_NEVER_RAN")]
    results = [H.CaseResult(case_id="A", status=H.PASS)]
    rec = H.reconcile(results, cases)
    assert H.strict_exit_code(results, cases) == rec["exit_code"]
    assert H.coverage_gaps(results, cases) == rec["gaps"]
    assert rec["missing"] == ["B_NEVER_RAN"]


def test_a1_v08_duplicate_declared_id_is_caught_by_reconcile():
    """A1-v07-P2-1, defence in depth: the gate no longer relies on the
    registry validator having run."""
    dup = [_base_case(case_id="A"), _base_case(case_id="A")]
    results = [H.CaseResult(case_id="A", status=H.PASS)]
    rec = H.reconcile(results, dup)
    assert any("duplicate declared case_id" in g for g in rec["gaps"])
    assert rec["exit_code"] == 1


def test_a1_v08_complete_run_still_passes():
    """Positive control: an authority that gated everything would satisfy every
    falsifier above and be useless."""
    cases = [_base_case(case_id="A"), _base_case(case_id="B")]
    results = [H.CaseResult(case_id="A", status=H.PASS, executed_comparisons=1),
               H.CaseResult(case_id="B", status=H.PASS, executed_comparisons=1)]
    rec = H.reconcile(results, cases)
    assert rec["exit_code"] == 0 and rec["gaps"] == [] and rec["gating"] == []


# ── 14. PHASE_13_77 A2 — numerical comparator/tolerance framework ──────────


def test_a2_01_scalar_exact_pass_and_fail():
    assert H.compare_scalar(7, 7, comparator="exact").ok
    bad = H.compare_scalar(7, 8, comparator="exact")
    assert not bad.ok and "exact mismatch" in bad.detail


def test_a2_02_scalar_close_atol_boundary():
    assert H.compare_scalar(1.0, 1.000001, comparator="close",
                            atol=2e-6, rtol=0.0).ok
    assert not H.compare_scalar(1.0, 1.000001, comparator="close",
                                atol=5e-7, rtol=0.0).ok


def test_a2_03_scalar_close_rtol_boundary():
    assert H.compare_scalar(100.0, 100.5, comparator="close",
                            atol=0.0, rtol=1e-2).ok
    assert not H.compare_scalar(100.0, 100.5, comparator="close",
                                atol=0.0, rtol=1e-3).ok


def test_a2_04_scalar_nan_and_infinity_rules():
    assert H.compare_scalar(np.nan, np.nan, comparator="exact").ok
    assert H.compare_scalar(np.nan, np.nan, comparator="close",
                            atol=1e-12, rtol=1e-12).ok
    assert H.compare_scalar(np.inf, np.inf, comparator="close",
                            atol=1e-12, rtol=1e-12).ok
    assert not H.compare_scalar(np.inf, -np.inf, comparator="close",
                                atol=1e-12, rtol=1e-12).ok


def test_a2_05_scalar_refuses_array_nonnumeric_and_unknown():
    with pytest.raises(H.HarnessError, match="scalar values only"):
        H.compare_scalar([1.0], [1.0], comparator="exact")
    with pytest.raises(H.HarnessError, match="real floating scalars"):
        H.compare_scalar("a", "a", comparator="close", atol=1e-6)
    with pytest.raises(H.HarnessError, match="unknown comparator"):
        H.compare_scalar(1, 1, comparator="banana")


def test_a2_06_array_shape_is_exact():
    bad = H.compare_array([1, 2], [[1, 2]], comparator="exact")
    assert not bad.ok and "shape mismatch" in bad.detail


def test_a2_07_array_exact_reports_mismatch_coordinates():
    bad = H.compare_array([[1, 2], [3, 4]], [[1, 9], [3, 8]],
                          comparator="exact")
    assert not bad.ok
    assert bad.mismatch_count == 2
    assert bad.mismatch_indices == ((0, 1), (1, 1))


def test_a2_08_array_close_elementwise_and_nan():
    a = np.array([1.0, np.nan, 100.0])
    b = np.array([1.0 + 5e-7, np.nan, 100.5])
    assert H.compare_array(a, b, comparator="close",
                           atol=1e-6, rtol=1e-2).ok
    bad = H.compare_array(a, b, comparator="close",
                          atol=1e-8, rtol=1e-3)
    assert not bad.ok and bad.mismatch_indices == ((2,),)


def test_a2_09_array_close_rejects_non_numeric():
    with pytest.raises(H.HarnessError, match="real floating arrays"):
        H.compare_array(["a"], ["a"], comparator="close", atol=1e-6)


def test_a2_10_a1_comparator_api_remains_compatible():
    assert H.cmp_exact([1, 2], [1, 2]) == (True, "")
    close = H.cmp_close(1e-6, 0.0)
    assert close([1.0, 2.0], [1.0, 2.0 + 5e-7]) == (True, "")


def test_a2_11_tolerance_exact_contract_is_valid():
    o = H.Observable("n", "STATS", "FLAT", "n")
    assert H.tolerance_for(o) == H.ToleranceSpec("exact", 0.0, 0.0, "")


def test_a2_12_tolerance_close_contract_is_valid():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=1e-9, rtol=1e-6, rationale="float reduction")
    assert H.tolerance_for(o) == H.ToleranceSpec(
        "close", 1e-9, 1e-6, "float reduction")


def test_a2_13_negative_atol_is_refused():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=-1.0, rtol=1e-6, rationale="x")
    assert any("non-negative" in x for x in H.tolerance_violations(o))


def test_a2_14_negative_rtol_is_refused():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=1e-6, rtol=-1.0, rationale="x")
    assert any("non-negative" in x for x in H.tolerance_violations(o))


def test_a2_15_nan_tolerance_is_refused():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=float("nan"), rtol=1e-6, rationale="x")
    assert any("finite" in x for x in H.tolerance_violations(o))


def test_a2_16_infinite_tolerance_is_refused():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=1e-6, rtol=float("inf"), rationale="x")
    assert any("finite" in x for x in H.tolerance_violations(o))


def test_a2_17_exact_cannot_carry_ignored_tolerance():
    o = H.Observable("n", "STATS", "FLAT", "n", comparator="exact", atol=1.0)
    assert any("exact comparator" in x for x in H.tolerance_violations(o))


def test_a2_18_close_zero_tolerance_is_refused():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     rationale="x")
    assert any("exact' in disguise" in x for x in H.tolerance_violations(o))


def test_a2_19_close_requires_rationale():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=1e-6)
    assert any("rationale" in x for x in H.tolerance_violations(o))


def test_a2_20_unknown_comparator_is_refused():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="banana")
    assert any("unknown comparator" in x for x in H.tolerance_violations(o))


def test_a2_21_compare_observable_uses_declared_scalar_contract():
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=1e-6, rtol=0.0, rationale="float reduction")
    assert H.compare_observable(o, 1.0, 1.0 + 5e-7).ok


def test_a2_22_compare_observable_uses_declared_array_contract():
    o = H.Observable("bins", "STATS", "ARRAY", "bins", comparator="close",
                     atol=1e-6, rtol=0.0, rationale="float bins")
    assert H.compare_observable(o, [1.0, 2.0], [1.0, 2.0 + 5e-7]).ok


def test_a2_23_comparison_evidence_is_json_ready():
    import json
    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=1e-6, rtol=1e-3, rationale="float reduction")
    result = H.compare_observable(o, 100.0, 100.5)
    rec = H.comparison_evidence(o, result, reference_label="draw",
                                candidate_label="draw_batch")
    assert rec["observable"] == "mean"
    assert rec["comparator"] == "close"
    assert rec["reference"] == "draw" and rec["candidate"] == "draw_batch"
    json.dumps(rec)


def test_a2_24_consistency_runner_records_structured_evidence(tmp_path):
    class SyntheticSurfaces:
        def draw(self, expr, **kw):
            return None, None, {"n": 3}
        def draw_batch(self, specs):
            return {"c": {"stats": {"n": 3}}}

    case = _base_case(case_id="A2-CONSISTENCY")
    res = H.run_consistency(case, SyntheticSurfaces)
    assert res.status == H.PASS, res.detail
    assert res.executed_comparisons == 1
    assert len(res.comparisons) == 1
    doc = H.write_manifest(str(tmp_path / "a2.json"), [res], [case])
    assert doc["cases"][0]["comparisons"][0]["ok"] is True


def test_a2_25_correctness_runner_records_structured_evidence():
    class SyntheticSurface:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        def draw(self, expr, **kw):
            return None, None, {"n": 3}

    case = _base_case(
        case_id="A2-CORRECTNESS", purpose="CORRECTNESS",
        oracle_kind="CORRECTNESS", surfaces_under_test=("draw",),
        figure_contract=_full_figure_contract(
            "A2-CORRECTNESS", proof_kind="CORRECTNESS"),
        observables=(H.Observable("n", "INDEPENDENT", "FLAT", "n"),))
    res = H.run_correctness(
        case, SyntheticSurface, lambda df: {"n": len(df)},
        raw_factory=lambda: pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    assert res.status == H.PASS, res.detail
    assert res.executed_comparisons == 1
    assert res.comparisons[0]["reference"] == "independent"


def test_a2_26_pass_without_comparison_fails_strict_gate_and_manifest(tmp_path):
    case = _base_case(case_id="A2-ZERO")
    result = H.CaseResult(case_id=case.case_id, status=H.PASS,
                          executed_comparisons=0)
    assert H.strict_exit_code([result], [case]) == 1
    doc = H.write_manifest(str(tmp_path / "zero.json"), [result], [case])
    assert doc["cases"][0]["gates"] is True
    assert "zero executed comparisons" in doc["cases"][0]["gate_reason"]


def test_a2_27_intentional_numeric_corruption_changes_gate_zero_to_one():
    case = _base_case(case_id="A2-MUTATION")
    o = case.observables[0]
    good_cmp = H.compare_observable(o, 10, 10)
    bad_cmp = H.compare_observable(o, 10, 11)
    good = H.CaseResult(case_id=case.case_id, status=H.PASS,
                        executed_comparisons=1,
                        comparisons=[H.comparison_evidence(
                            o, good_cmp, reference_label="reference",
                            candidate_label="candidate")])
    bad = H.CaseResult(case_id=case.case_id, status=H.FAIL,
                       detail=bad_cmp.detail, executed_comparisons=1,
                       comparisons=[H.comparison_evidence(
                           o, bad_cmp, reference_label="reference",
                           candidate_label="candidate")])
    assert H.strict_exit_code([good], [case]) == 0
    assert H.strict_exit_code([bad], [case]) == 1


def test_a2_28_direct_exact_comparison_rejects_nonzero_tolerance():
    with pytest.raises(H.HarnessError, match="exact comparator"):
        H.compare_scalar(1, 1, comparator="exact", atol=1.0)
    with pytest.raises(H.HarnessError, match="exact comparator"):
        H.compare_array([1], [1], comparator="exact", rtol=1.0)


def test_a2_29_invalid_comparator_is_rejected_before_shape_comparison():
    with pytest.raises(H.HarnessError, match="unknown comparator"):
        H.compare_array([1, 2], [[1, 2]], comparator="banana")


def _wider_than_float64_dtype():
    """Return a real floating dtype with more precision than float64, if any."""
    candidates = []
    for name in ("longdouble", "float128"):
        dtype = getattr(np, name, None)
        if dtype is None:
            continue
        try:
            finfo = np.finfo(dtype)
        except (TypeError, ValueError):
            continue
        if finfo.eps < np.finfo(np.float64).eps:
            candidates.append(dtype)
    return candidates[0] if candidates else None


def test_a2_30_extended_precision_scalar_mismatch_survives_close_and_gate():
    """A2-v02-P0-1: do not narrow wider real floating scalars to float64."""
    dtype = _wider_than_float64_dtype()
    if dtype is None:
        pytest.skip("platform has no real floating dtype wider than float64")

    reference = dtype(1)
    candidate = np.nextafter(reference, dtype(2), dtype=dtype)
    delta = candidate - reference
    atol = float(delta / dtype(2))
    assert delta > atol
    assert not bool(np.isclose(reference, candidate, atol=atol, rtol=0.0,
                               equal_nan=True))

    o = H.Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                     atol=atol, rtol=0.0, rationale="extended precision")
    cmp = H.compare_observable(o, reference, candidate)
    assert not cmp.ok, cmp

    case = _base_case(case_id="A2-EXTENDED-SCALAR", observables=(o,))
    result = H.CaseResult(
        case_id=case.case_id, status=H.FAIL, detail=cmp.detail,
        executed_comparisons=1,
        comparisons=[H.comparison_evidence(
            o, cmp, reference_label="reference", candidate_label="candidate")])
    assert H.strict_exit_code([result], [case]) == 1


def test_a2_31_extended_precision_array_mismatch_survives_close():
    """A2-v02-P0-1 array path: preserve the input floating dtype."""
    dtype = _wider_than_float64_dtype()
    if dtype is None:
        pytest.skip("platform has no real floating dtype wider than float64")

    reference = np.array([dtype(1)], dtype=dtype)
    candidate = np.nextafter(reference, np.array([dtype(2)], dtype=dtype))
    delta = candidate[0] - reference[0]
    atol = float(delta / dtype(2))
    assert not bool(np.isclose(reference, candidate, atol=atol, rtol=0.0,
                               equal_nan=True)[0])

    cmp = H.compare_array(reference, candidate, comparator="close",
                          atol=atol, rtol=0.0)
    assert not cmp.ok
    assert cmp.mismatch_count == 1
    assert cmp.mismatch_indices == ((0,),)


def test_a2_32_close_refuses_nonfloating_numeric_families_without_coercion():
    """Unsupported numeric families must refuse rather than narrow silently."""
    with pytest.raises(H.HarnessError, match="real floating scalars"):
        H.compare_scalar(1 + 2j, 1 + 3j, comparator="close", atol=1e-6)
    with pytest.raises(H.HarnessError, match="real floating arrays"):
        H.compare_array(np.array([1 + 2j]), np.array([1 + 3j]),
                        comparator="close", atol=1e-6)
    with pytest.raises(H.HarnessError, match="real floating scalars"):
        H.compare_scalar(np.int64(2**62), np.int64(2**62 + 1),
                         comparator="close", atol=0.5)
    with pytest.raises(H.HarnessError, match="real floating arrays"):
        H.compare_array(np.array([2**62], dtype=np.int64),
                        np.array([2**62 + 1], dtype=np.int64),
                        comparator="close", atol=0.5)


# ── A3.1/A3.2 — first canonical same-spec case ─────────────────────────────

def test_a3_01_hist_case_declares_one_spec_for_all_three_surfaces():
    case = H.a3_cases()[0]
    assert case.case_id == "I2-HIST-01"
    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert case.not_applicable == {}
    assert case.canonical_spec == {
        "expr": "ncl", "type": "hist", "bins": 50,
        "selection": "ncl>30", "auto_title": True,
    }
    assert H.validate_registry([case]) == []


def test_a3_02_hist_same_spec_passes_draw_draw_batch_draw_figures():
    # Synthetic execution of the REAL A3 CaseSpec.  Real-data execution belongs
    # to the standalone harness/A6 reference run; this test proves all three
    # public surfaces consume the exact same A3 canonical request.
    rng = np.random.default_rng(1377)
    frame = pd.DataFrame({"ncl": rng.integers(0, 160, size=800)})
    make_adf = lambda: ADF(frame.copy())
    case = H.a3_cases()[0]
    res = H.run_consistency(case, make_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)
    # draw is the reference; 2 candidate surfaces x 5 declared observables.
    assert res.executed_comparisons == 10
    assert len(res.comparisons) == 10
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons


def test_a3_03_draw_figures_is_an_executed_surface_not_a_documented_exception():
    case = H.a3_cases()[0]
    assert "draw_figures" in case.surfaces_under_test
    assert "draw_figures" not in case.not_applicable
    assert case.figure_contract is not None
    assert "draw_figures" in case.figure_contract.primary_comparison


def test_a3_04_profile_case_declares_one_spec_for_all_three_surfaces():
    case = next(c for c in H.a3_cases() if c.case_id == "I2-PROFILE-01")
    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert case.not_applicable == {}
    assert case.canonical_spec == {
        "expr": "y:x", "type": "profile", "bins": 25,
        "selection": "(x>-2.0)&(x<2.0)", "auto_title": True,
    }
    assert [o.name for o in case.observables] == [
        "n", "n_input", "n_filtered", "mean_x", "mean_y", "std_x", "std_y"
    ]
    assert H.validate_registry(H.a3_cases()) == []


def test_a3_05_profile_same_spec_passes_draw_draw_batch_draw_figures():
    # Synthetic dataframe, real public ADF surfaces.  No grouping/faceting yet:
    # A3.3 isolates the profile stats family from later A3/A4 dimensions.
    rng = np.random.default_rng(137703)
    x = rng.normal(0.0, 1.0, size=900)
    y = 1.5 + 0.7 * x + rng.normal(0.0, 0.2, size=900)
    frame = pd.DataFrame({"x": x, "y": y})
    make_adf = lambda: ADF(frame.copy())
    case = next(c for c in H.a3_cases() if c.case_id == "I2-PROFILE-01")
    res = H.run_consistency(case, make_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)
    # draw is reference; 2 candidate surfaces x 7 declared observables.
    assert res.executed_comparisons == 14
    assert len(res.comparisons) == 14
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons


def test_a3_06_surface_stats_corruption_changes_strict_gate_zero_to_one(monkeypatch):
    """A3.4 negative regression: one surface mismatch must gate the case.

    Use the canonical A3 histogram case itself.  The positive control executes
    all three declared surfaces with matching statistics and must keep the
    strict gate at zero.  Then corrupt exactly one numerical statistic on the
    draw_figures payload.  The A2 comparator must report the mismatch and the
    mandatory-case strict gate must become non-zero.
    """
    case = next(c for c in H.a3_cases() if c.case_id == "I2-HIST-01")

    rng = np.random.default_rng(137704)
    frame = pd.DataFrame({"ncl": rng.integers(0, 160, size=800)})
    make_adf = lambda: ADF(frame.copy())

    good = H.run_consistency(case, make_adf)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    original = H.unwrap

    def corrupted(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_figures" and isinstance(payload.stats, dict):
            stats = dict(payload.stats)
            stats["mean"] = float(stats["mean"]) + 1.0
            return H.Payload(surface, stats, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", corrupted)
    bad = H.run_consistency(case, make_adf)
    assert bad.status == H.FAIL, bad.detail
    assert "mean" in bad.detail
    assert "draw" in bad.detail and "draw_figures" in bad.detail
    assert H.strict_exit_code([bad], [case]) == 1



# ── A3.6 — canonical group_by invariance ────────────────────────────────────

def _a3_groupby_frame(seed=137706, n=1800):
    """Fast synthetic analogue of time_series_draw.py fig09_group_by_profile."""
    rng = np.random.default_rng(seed)
    side_type = rng.integers(0, 2, size=n)
    sector = rng.integers(0, 36, size=n).astype(float)
    dcar = (0.04 * (sector - 17.5)
            + np.where(side_type == 0, -0.35, 0.45)
            + rng.normal(0.0, 0.15, size=n))
    return pd.DataFrame({
        "sector": sector,
        "dcar_tpc_vertex": dcar,
        "side_type": side_type,
        "ncl": rng.integers(45, 160, size=n),
    })


def test_a3_07_groupby_case_declares_one_spec_and_group_resolved_observables():
    case = next(c for c in H.a3_cases() if c.case_id == "I2-GROUPBY-01")
    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert case.not_applicable == {}
    assert case.canonical_spec == {
        "expr": "dcar_tpc_vertex:sector",
        "type": "profile",
        "bins": 36,
        "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)&(side_type<2)",
        "group_by": "side_type",
        "return_data": True,
        "auto_title": True,
    }
    assert [(o.name, o.access, o.path) for o in case.observables] == [
        ("group", "ARRAY", "profile_data.group"),
        ("count", "ARRAY", "profile_data.count"),
        ("x_center", "ARRAY", "profile_data.x_center"),
        ("y_mean", "ARRAY", "profile_data.y_mean"),
    ]
    assert H.validate_registry(H.a3_cases()) == []


def test_a3_08_groupby_same_spec_passes_draw_draw_batch_draw_figures():
    frame = _a3_groupby_frame()
    make_adf = lambda: ADF(frame.copy())
    case = next(c for c in H.a3_cases() if c.case_id == "I2-GROUPBY-01")
    res = H.run_consistency(case, make_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)
    # draw is reference; 2 candidate surfaces x 4 declared array observables.
    assert res.executed_comparisons == 8
    assert len(res.comparisons) == 8
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons
    groups = np.asarray(res.observed["group"]["draw"])
    assert set(groups.tolist()) == {0, 1}


def test_a3_09_profile_data_missing_column_fails_loudly():
    profile_data = pd.DataFrame({"group": [0, 1], "y_mean": [1.0, 2.0]})
    with pytest.raises(H.HarnessError, match="no_such_column"):
        H.resolve({"profile_data": profile_data},
                  "profile_data.no_such_column", "ARRAY")


def test_a3_10_group_specific_mismatch_is_not_hidden_by_global_stats(monkeypatch):
    """Independent falsification test for A3.6.

    Construct a regression case that falsifies the implementation invariant:
    mutate one per-bin y_mean belonging to side_type==1 on draw_figures while
    leaving the global summary statistic untouched.  The existing A2 array
    comparator must identify that row and the strict numerical gate must fail.
    """
    frame = _a3_groupby_frame(seed=137710)
    make_adf = lambda: ADF(frame.copy())
    case = next(c for c in H.a3_cases() if c.case_id == "I2-GROUPBY-01")

    good = H.run_consistency(case, make_adf)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    # Independent public-surface baseline for the global statistic.  The
    # falsification below changes only profile_data.y_mean, not this summary.
    raw_global, kw_global = H._call(make_adf(), "draw_figures", case.canonical_spec)
    baseline_global_mean_y = H.unwrap(
        "draw_figures", raw_global, **kw_global).stats["mean_y"]
    H._close()

    original = H.unwrap
    mutation = {}

    def corrupted(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_figures" and isinstance(payload.stats, dict):
            stats = dict(payload.stats)
            table = stats["profile_data"].copy(deep=True)
            rows = np.flatnonzero(table["group"].to_numpy() == 1)
            assert len(rows) > 0
            row = int(rows[0])
            mutation["row"] = row
            mutation["group"] = int(table.iloc[row]["group"])
            mutation["global_mean_y"] = stats["mean_y"]
            table.iloc[row, table.columns.get_loc("y_mean")] += 0.25
            stats["profile_data"] = table
            return H.Payload(surface, stats, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", corrupted)
    bad = H.run_consistency(case, make_adf)
    assert bad.status == H.FAIL, bad.detail
    assert "y_mean" in bad.detail
    assert mutation["group"] == 1
    assert bad.observed["y_mean"]["draw_figures"][mutation["row"]] != (
        bad.observed["y_mean"]["draw"][mutation["row"]])
    # The deliberately untouched global statistic proves this mismatch cannot
    # be detected by a global-summary-only comparison.
    assert mutation["global_mean_y"] == baseline_global_mean_y
    failed = [rec for rec in bad.comparisons if not rec["ok"]]
    assert len(failed) == 1
    assert failed[0]["observable"] == "y_mean"
    assert failed[0]["mismatch_count"] == 1
    assert failed[0]["mismatch_indices"] == [[mutation["row"]]]
    assert H.strict_exit_code([bad], [case]) == 1


# ── A3.7 — canonical facet_by invariance on supported surfaces ─────────────

def test_a3_11_facet_case_declares_supported_surfaces_and_separate_refusal_contract():
    cases = H.a3_cases()
    case = next(c for c in cases if c.case_id == "I2-FACET-01")
    refusal = next(c for c in cases
                   if c.case_id == "I2-FACET-DRAW-FIGURES-REFUSAL-01")
    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert set(case.not_applicable) == {"draw_figures"}
    assert "facet_by" in case.not_applicable["draw_figures"]
    assert refusal.purpose == "ERROR_CONTRACT"
    assert tuple(refusal.surfaces_under_test) == ("draw_figures",)
    assert refusal.known_bug_id == "BUG_dfdraw_20260611_facet_by_ax_ignored"
    # The two proof obligations execute one identical request, not two similar
    # copies that can drift independently.
    assert case.canonical_spec is refusal.canonical_spec
    assert case.canonical_spec == {
        "expr": "dcar_tpc_vertex:sector",
        "type": "profile",
        "bins": 36,
        "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)&(side_type<2)",
        "facet_by": "side_type",
        "return_data": True,
        "auto_title": True,
    }
    assert [(o.name, o.access, o.path) for o in case.observables] == [
        ("facet_groups", "ARRAY", "groups"),
        ("facet0_count", "ARRAY", "per_group.0.profile_data.count"),
        ("facet0_x_center", "ARRAY", "per_group.0.profile_data.x_center"),
        ("facet0_y_mean", "ARRAY", "per_group.0.profile_data.y_mean"),
        ("facet1_count", "ARRAY", "per_group.1.profile_data.count"),
        ("facet1_x_center", "ARRAY", "per_group.1.profile_data.x_center"),
        ("facet1_y_mean", "ARRAY", "per_group.1.profile_data.y_mean"),
    ]
    assert H.validate_registry(cases) == []


def test_a3_12_facet_same_spec_passes_supported_surfaces_and_refuses_draw_figures():
    frame = _a3_groupby_frame(seed=137712)
    make_adf = lambda: ADF(frame.copy())
    cases = H.a3_cases()
    case = next(c for c in cases if c.case_id == "I2-FACET-01")
    refusal_case = next(c for c in cases
                        if c.case_id == "I2-FACET-DRAW-FIGURES-REFUSAL-01")

    res = H.run_consistency(case, make_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == {"draw", "draw_batch"}
    assert set(res.skipped_surfaces) == {"draw_figures"}
    # draw is reference; one supported candidate x seven facet-resolved observables.
    assert res.executed_comparisons == 7
    assert len(res.comparisons) == 7
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons
    assert np.array_equal(np.asarray(res.observed["facet_groups"]["draw"]),
                          np.asarray([0, 1]))

    refusal = H.run_error_contract(refusal_case, make_adf, "draw_figures",
                                   "facet_by is not supported")
    assert refusal.status == H.PASS, refusal.detail
    # The two obligations are separately declared and therefore reconcile as
    # two unique results rather than duplicating one case_id.
    assert H.strict_exit_code([res, refusal], [case, refusal_case]) == 0
    assert H.coverage_gaps([res, refusal], [case, refusal_case]) == []


def test_a3_13_facet_specific_mismatch_reaches_strict_gate(monkeypatch):
    """Independent falsification test for A3.7.

    Construct a regression case that falsifies the implementation invariant:
    mutate one profile_data.y_mean bin in facet ``side_type==1`` on draw_batch.
    Leave both the top-level population count and the facet's own summary mean
    untouched.  A summary-only adapter would remain green; the existing A2
    per-array comparator must report exactly one facet-specific mismatch.
    """
    frame = _a3_groupby_frame(seed=137713)
    make_adf = lambda: ADF(frame.copy())
    case = next(c for c in H.a3_cases() if c.case_id == "I2-FACET-01")

    good = H.run_consistency(case, make_adf)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    original = H.unwrap
    mutation = {}

    def corrupted(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_batch" and isinstance(payload.stats, dict):
            stats = dict(payload.stats)
            per_group = {k: dict(v) for k, v in stats["per_group"].items()}
            facet = per_group["1"]
            table = facet["profile_data"].copy(deep=True)
            rows = np.flatnonzero(table["count"].to_numpy() > 0)
            assert len(rows) > 0
            row = int(rows[0])
            mutation["row"] = row
            mutation["n_total"] = stats["n_total"]
            mutation["facet_mean_y"] = facet["mean_y"]
            table.iloc[row, table.columns.get_loc("y_mean")] += 0.25
            facet["profile_data"] = table
            per_group["1"] = facet
            stats["per_group"] = per_group
            return H.Payload(surface, stats, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", corrupted)
    bad = H.run_consistency(case, make_adf)
    assert bad.status == H.FAIL, bad.detail
    assert "facet1_y_mean" in bad.detail

    # These summaries were deliberately untouched by the falsification.
    raw_ref, kw_ref = H._call(make_adf(), "draw", case.canonical_spec)
    ref_stats = original("draw", raw_ref, **kw_ref).stats
    H._close()
    assert mutation["n_total"] == ref_stats["n_total"]
    assert mutation["facet_mean_y"] == ref_stats["per_group"]["1"]["mean_y"]

    failed = [rec for rec in bad.comparisons if not rec["ok"]]
    assert len(failed) == 1
    assert failed[0]["observable"] == "facet1_y_mean"
    assert failed[0]["mismatch_count"] == 1
    assert failed[0]["mismatch_indices"] == [[mutation["row"]]]
    assert H.strict_exit_code([bad], [case]) == 1


# ── A3.8 — canonical subframe-qualified expression invariance ──────────────

def _a3_subframe_adf(seed=137814):
    """Fast synthetic analogue of the real CalibVertex time-series workflow.

    The main frame has many rows per ``quantile_bin`` while the registered
    CalibVertex subframe has exactly one row per key.  The plotted y value is
    absent from the main frame, so success requires real subframe join/broadcast
    resolution rather than an ordinary same-frame column lookup.
    """
    rng = np.random.default_rng(seed)
    n_keys = 8
    rows_per_key = 30
    quantile_bin = np.repeat(np.arange(n_keys, dtype=np.int16), rows_per_key)
    n = len(quantile_bin)
    main = pd.DataFrame({
        "quantile_bin": quantile_bin,
        "time_s": np.linspace(0.0, 6.0 * 3600.0, n, dtype=np.float64),
        "noise": rng.normal(0.0, 1.0, n),
    })
    sub = pd.DataFrame({
        "quantile_bin": np.arange(n_keys, dtype=np.int16),
        "vertex_x_intercept": (
            0.15 + 0.07 * np.arange(n_keys, dtype=np.float64)
            + rng.normal(0.0, 0.002, n_keys)
        ),
    })

    # Fixture guards: this must genuinely be a keyed subframe lookup.
    assert "vertex_x_intercept" not in main.columns
    assert main["quantile_bin"].duplicated().any()
    assert sub["quantile_bin"].is_unique

    adf = ADF(main)
    adf.register_subframe("CalibVertex", ADF(sub), index_columns="quantile_bin")
    registered = adf.get_subframe("CalibVertex")
    assert "vertex_x_intercept" in registered.df.columns
    assert len(registered.df) == n_keys < len(adf.df)
    return adf


def test_a3_14_subframe_case_declares_one_spec_and_keyed_observables():
    case = next(c for c in H.a3_cases() if c.case_id == "I2-SUBFRAME-01")
    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert case.not_applicable == {}
    assert case.canonical_spec == {
        "expr": "CalibVertex.vertex_x_intercept:time_s",
        "type": "profile",
        "bins": 12,
        "return_data": True,
        "auto_title": True,
    }
    assert [(o.name, o.access, o.path) for o in case.observables] == [
        ("n", "FLAT", "n"),
        ("count", "ARRAY", "profile_data.count"),
        ("x_center", "ARRAY", "profile_data.x_center"),
        ("y_mean", "ARRAY", "profile_data.y_mean"),
    ]
    adf = _a3_subframe_adf()
    assert "vertex_x_intercept" not in adf.df.columns
    assert "vertex_x_intercept" in adf.get_subframe("CalibVertex").df.columns
    assert H.validate_registry(H.a3_cases()) == []


def test_a3_15_subframe_same_spec_passes_draw_draw_batch_draw_figures():
    case = next(c for c in H.a3_cases() if c.case_id == "I2-SUBFRAME-01")
    res = H.run_consistency(case, _a3_subframe_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)
    # draw is reference; 2 candidate surfaces x 4 declared observables.
    assert res.executed_comparisons == 8
    assert len(res.comparisons) == 8
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons
    counts = np.asarray(res.observed["count"]["draw"])
    assert counts.sum() == res.observed["n"]["draw"]


def test_a3_16_subframe_specific_mismatch_reaches_strict_gate(monkeypatch):
    """Independent falsification test for A3.8.

    Construct a regression case that falsifies the implementation invariant:
    change one per-bin ``y_mean`` derived from the CalibVertex subframe on
    ``draw_figures`` only.  Leave the global ``mean_y`` summary untouched.
    The existing A2 array comparator must identify exactly one bin and strict
    mode must fail, proving that subframe-derived numerical disagreement is not
    hidden by the adapter or by an unchanged global summary.
    """
    case = next(c for c in H.a3_cases() if c.case_id == "I2-SUBFRAME-01")

    good = H.run_consistency(case, _a3_subframe_adf)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    original = H.unwrap
    mutation = {}

    def corrupted(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_figures" and isinstance(payload.stats, dict):
            stats = dict(payload.stats)
            table = stats["profile_data"].copy(deep=True)
            rows = np.flatnonzero(table["count"].to_numpy() > 0)
            assert len(rows) > 0
            row = int(rows[len(rows) // 2])
            mutation["row"] = row
            mutation["global_mean_y"] = stats["mean_y"]
            table.iloc[row, table.columns.get_loc("y_mean")] += 0.125
            stats["profile_data"] = table
            return H.Payload(surface, stats, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", corrupted)
    bad = H.run_consistency(case, _a3_subframe_adf)
    assert bad.status == H.FAIL, bad.detail
    assert "y_mean" in bad.detail

    # The global summary is measured independently and deliberately untouched.
    raw_ref, kw_ref = H._call(_a3_subframe_adf(), "draw", case.canonical_spec)
    ref_mean_y = original("draw", raw_ref, **kw_ref).stats["mean_y"]
    H._close()
    assert mutation["global_mean_y"] == ref_mean_y

    failed = [rec for rec in bad.comparisons if not rec["ok"]]
    assert len(failed) == 1
    assert failed[0]["observable"] == "y_mean"
    assert failed[0]["mismatch_count"] == 1
    assert failed[0]["mismatch_indices"] == [[mutation["row"]]]
    assert H.strict_exit_code([bad], [case]) == 1

# ── A3.9 selection + selection_vector differential-profile checkpoint ──────

def _a3_selection_vector_frame(seed=137917):
    """Fast synthetic analogue of the real time-series selection-vector gallery.

    Exactly 84/120 rows survive the base selection.  Among those selected rows,
    sector alternates between 13 (signal) and 20 (reference), so the two
    selection-vector branches contain exactly 42 rows each and both populate
    every broad time bin.
    """
    rng = np.random.default_rng(seed)
    n = 120
    idx = np.arange(n)
    sector = np.where(idx % 2 == 0, 13.0, 20.0)
    selected = idx < 84
    time_s = np.linspace(0.0, 3600.0, n, dtype=np.float64)
    # Give the two vector branches distinct central values, with a gentle time
    # trend so wrong branch routing cannot be masked by a constant fixture.
    ncl_its = (5.0 + 0.0008 * time_s
               + np.where(sector == 13.0, 0.9, -0.6)
               + rng.normal(0.0, 0.08, n))
    frame = pd.DataFrame({
        "nClITS": ncl_its,
        "time_s": time_s,
        "ncl": np.where(selected, 80, 50),
        "dcar_tpc_vertex": rng.normal(0.0, 0.5, n),
        "hasITSTPC": np.ones(n, dtype=np.int8),
        "sector": sector,
    })

    base = ((frame["ncl"] > 60)
            & (frame["dcar_tpc_vertex"].abs() < 10)
            & frame["hasITSTPC"].astype(bool))
    signal = (frame["sector"] - 13).abs() < 2
    reference = ((frame["sector"] - 13).abs() >= 2) & (frame["sector"] < 36)
    assert int(base.sum()) == 84 < len(frame)
    assert int((base & signal).sum()) == 42
    assert int((base & reference).sum()) == 42
    assert not bool((base & signal & reference).any())
    assert bool((base == (base & (signal | reference))).all())
    return frame


def _a3_selection_vector_adf(seed=137917):
    return ADF(_a3_selection_vector_frame(seed=seed))


def test_a3_17_selection_vector_case_declares_branch_resolved_observables():
    case = next(c for c in H.a3_cases()
                if c.case_id == "I2-SELECTION-VECTOR-01")
    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert case.not_applicable == {}
    assert case.canonical_spec == {
        "expr": "nClITS:time_s",
        "type": "profile",
        "bins": 8,
        "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",
        "selection_vector": [
            "(abs(sector-13)<2)",
            "(abs(sector-13)>=2)&(sector<36)",
        ],
        "normalize": "delta",
        "return_data": True,
        "auto_title": True,
    }
    assert [(o.name, o.access, o.path) for o in case.observables] == [
        ("x_center", "ARRAY", "normalize_data.x_center"),
        ("signal_central", "ARRAY", "normalize_data.signal_central"),
        ("signal_count", "ARRAY", "normalize_data.signal_count"),
        ("reference_central", "ARRAY", "normalize_data.reference_central"),
        ("reference_count", "ARRAY", "normalize_data.reference_count"),
        ("value", "ARRAY", "normalize_data.value"),
    ]
    assert H.validate_registry(H.a3_cases()) == []


def test_a3_18_selection_vector_same_spec_passes_all_three_surfaces():
    case = next(c for c in H.a3_cases()
                if c.case_id == "I2-SELECTION-VECTOR-01")
    res = H.run_consistency(case, _a3_selection_vector_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)
    # draw is reference; 2 candidate surfaces x 6 declared observables.
    assert res.executed_comparisons == 12
    assert len(res.comparisons) == 12
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons
    assert int(np.asarray(res.observed["signal_count"]["draw"]).sum()) == 42
    assert int(np.asarray(res.observed["reference_count"]["draw"]).sum()) == 42
    assert H.strict_exit_code([res], [case]) == 0


def test_a3_19_selection_vector_branch_mismatch_is_not_hidden_by_derived_value(monkeypatch):
    """Independent falsification test for A3.9.

    Construct a regression case that falsifies the implementation invariant:
    change one signal-branch ``central`` bin on ``draw_figures`` only while
    deliberately leaving the already-derived normalized ``value`` unchanged.
    The existing A2 array comparator must still identify the branch-specific
    mismatch and strict mode must fail.  This proves that A3.9 does not reduce
    the vector contract to the final delta alone.
    """
    case = next(c for c in H.a3_cases()
                if c.case_id == "I2-SELECTION-VECTOR-01")

    good = H.run_consistency(case, _a3_selection_vector_adf)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    original = H.unwrap
    mutation = {}

    def corrupted(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_figures" and isinstance(payload.stats, dict):
            stats = dict(payload.stats)
            table = stats["normalize_data"].copy(deep=True)
            rows = np.flatnonzero(table["signal_count"].to_numpy() > 0)
            assert len(rows) > 0
            row = int(rows[len(rows) // 2])
            mutation["row"] = row
            mutation["derived_value_before"] = float(table.iloc[row]["value"])
            table.iloc[row, table.columns.get_loc("signal_central")] += 0.25
            # Intentionally DO NOT recompute ``value``.  A final-delta-only
            # oracle would therefore false-green this regression.
            mutation["derived_value_after"] = float(table.iloc[row]["value"])
            stats["normalize_data"] = table
            return H.Payload(surface, stats, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", corrupted)
    bad = H.run_consistency(case, _a3_selection_vector_adf)
    assert bad.status == H.FAIL, bad.detail
    assert "signal_central" in bad.detail

    failed = [rec for rec in bad.comparisons if not rec["ok"]]
    assert len(failed) == 1
    assert failed[0]["observable"] == "signal_central"
    assert failed[0]["mismatch_count"] == 1
    assert failed[0]["mismatch_indices"] == [[mutation["row"]]]

    # The derived delta is independently confirmed unchanged in the mutated
    # payload.  run_consistency intentionally returns at the first failed
    # observable, so later observables are not present in ``bad.observed``.
    assert mutation["derived_value_after"] == pytest.approx(
        mutation["derived_value_before"]
    )
    assert mutation["derived_value_before"] == pytest.approx(
        good.observed["value"]["draw_figures"][mutation["row"]]
    )
    assert H.strict_exit_code([bad], [case]) == 1

# ── A3.10 explicit profile-bin observables + histogram disposition ─────────

def _a3_profile_bins_frame():
    """Sparse five-bin fixture with only edge bins populated."""
    frame = pd.DataFrame({
        "x": np.asarray([0.2, 0.3, 0.4, 4.6, 4.7, 4.8], dtype=np.float64),
        "y": np.asarray([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], dtype=np.float64),
    })
    return frame


def _a3_profile_bins_adf():
    return ADF(_a3_profile_bins_frame().copy())


def test_a3_20_profile_bin_case_and_histogram_disposition_are_explicit(tmp_path):
    cases = H.a3_cases()
    hist = next(c for c in cases if c.case_id == "I2-HIST-01")
    case = next(c for c in cases if c.case_id == "I2-PROFILE-BINS-01")

    hist_disp = {o.name: o for o in hist.observables
                 if o.name in {"bin_edges", "bin_counts"}}
    assert set(hist_disp) == {"bin_edges", "bin_counts"}
    for observable in hist_disp.values():
        assert observable.source == "ARTIST_FALLBACK"
        assert observable.status == "NOT_EXTRACTABLE"
        assert "absent" in observable.rationale
        assert "ARTIST_FALLBACK" in observable.rationale

    # The disposition is not only source text: it must survive into the
    # machine-readable manifest so later closure cannot silently upgrade the
    # earlier histogram summary-statistics proof to a bin-level proof.
    rng = np.random.default_rng(137710)
    hist_frame = pd.DataFrame({"ncl": rng.integers(0, 160, size=300)})
    hist_res = H.run_consistency(hist, lambda: ADF(hist_frame.copy()))
    assert hist_res.status == H.PASS, hist_res.detail
    doc = H.write_manifest(str(tmp_path / "a3_10_hist_disposition.json"),
                           [hist_res], [hist])
    declared = {o["name"]: o for o in doc["cases"][0]["declared_observables"]}
    for name in ("bin_edges", "bin_counts"):
        assert declared[name]["source"] == "ARTIST_FALLBACK"
        assert declared[name]["status"] == "NOT_EXTRACTABLE"
        assert declared[name]["rationale"]

    assert tuple(case.surfaces_under_test) == H.SURFACES
    assert case.canonical_spec == {
        "expr": "y:x",
        "type": "profile",
        "bins": 5,
        "range": (0.0, 5.0),
        "return_data": True,
        "auto_title": True,
    }
    assert [(o.name, o.access, o.path) for o in case.observables] == [
        ("x_low", "ARRAY", "profile_data.x_low"),
        ("x_high", "ARRAY", "profile_data.x_high"),
        ("x_center", "ARRAY", "profile_data.x_center"),
        ("count", "ARRAY", "profile_data.count"),
        ("y_mean", "ARRAY", "profile_data.y_mean"),
        ("y_std", "ARRAY", "profile_data.y_std"),
        ("y_sem", "ARRAY", "profile_data.y_sem"),
        ("y_central", "ARRAY", "profile_data.y_central"),
    ]
    assert H.validate_registry(cases) == []


def test_a3_21_sparse_profile_bins_match_all_three_surfaces():
    case = next(c for c in H.a3_cases() if c.case_id == "I2-PROFILE-BINS-01")
    res = H.run_consistency(case, _a3_profile_bins_adf)
    assert res.status == H.PASS, res.detail
    assert set(res.payload_paths) == set(H.SURFACES)
    # draw is reference; 2 candidate surfaces x 8 bin-level observables.
    assert res.executed_comparisons == 16
    assert len(res.comparisons) == 16
    assert all(rec["ok"] for rec in res.comparisons), res.comparisons

    counts = np.asarray(res.observed["count"]["draw"])
    assert np.array_equal(counts, np.asarray([3, 0, 0, 0, 3]))
    empty = counts == 0
    populated = counts > 0
    assert int(empty.sum()) == 3
    for name in ("y_mean", "y_std", "y_sem", "y_central"):
        values = np.asarray(res.observed[name]["draw"])
        assert np.isnan(values[empty]).all(), (name, values)
        assert np.isfinite(values[populated]).all(), (name, values)

    assert np.allclose(res.observed["x_low"]["draw"], [0, 1, 2, 3, 4])
    assert np.allclose(res.observed["x_high"]["draw"], [1, 2, 3, 4, 5])
    assert np.allclose(res.observed["x_center"]["draw"], [.5, 1.5, 2.5, 3.5, 4.5])
    assert H.strict_exit_code([res], [case]) == 0


def test_a3_22_profile_bin_error_mismatch_reaches_strict_gate(monkeypatch):
    """Independent falsification test for A3.10.

    Construct a regression case that falsifies the implementation invariant:
    alter exactly one populated-bin ``y_sem`` value on ``draw_figures`` while
    leaving the global ``std_y`` summary and all per-bin central values
    untouched.  The existing A2 array comparator must report one mismatch and
    strict mode must fail.  This proves that error/effective-bin evidence is
    not reduced to global or central-value summaries.
    """
    case = next(c for c in H.a3_cases() if c.case_id == "I2-PROFILE-BINS-01")

    good = H.run_consistency(case, _a3_profile_bins_adf)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    original = H.unwrap
    mutation = {}

    def corrupted(surface, result, **kw):
        payload = original(surface, result, **kw)
        if surface == "draw_figures" and isinstance(payload.stats, dict):
            stats = dict(payload.stats)
            table = stats["profile_data"].copy(deep=True)
            rows = np.flatnonzero(table["count"].to_numpy() > 0)
            assert len(rows) >= 2
            row = int(rows[-1])
            mutation["row"] = row
            mutation["global_std_y"] = float(stats["std_y"])
            mutation["y_mean"] = float(table.iloc[row]["y_mean"])
            mutation["y_central"] = float(table.iloc[row]["y_central"])
            table.iloc[row, table.columns.get_loc("y_sem")] += 0.25
            stats["profile_data"] = table
            return H.Payload(surface, stats, payload.path)
        return payload

    monkeypatch.setattr(H, "unwrap", corrupted)
    bad = H.run_consistency(case, _a3_profile_bins_adf)
    assert bad.status == H.FAIL, bad.detail
    assert "y_sem" in bad.detail

    failed = [rec for rec in bad.comparisons if not rec["ok"]]
    assert len(failed) == 1
    assert failed[0]["observable"] == "y_sem"
    assert failed[0]["mismatch_count"] == 1
    assert failed[0]["mismatch_indices"] == [[mutation["row"]]]

    # Independently verify that global and central-value summaries were not
    # touched by the falsifier.  run_consistency returns at the first failure.
    raw_ref, kw_ref = H._call(_a3_profile_bins_adf(), "draw", case.canonical_spec)
    ref_stats = original("draw", raw_ref, **kw_ref).stats
    H._close()
    assert mutation["global_std_y"] == pytest.approx(ref_stats["std_y"])
    ref_table = ref_stats["profile_data"]
    assert mutation["y_mean"] == pytest.approx(ref_table.iloc[mutation["row"]]["y_mean"])
    assert mutation["y_central"] == pytest.approx(
        ref_table.iloc[mutation["row"]]["y_central"])
    assert H.strict_exit_code([bad], [case]) == 1


# ── A3.11 A3 reconciliation / closure-declaration checkpoint ────────────────

def test_a3_23_closure_reconciliation_covers_every_required_family_without_orphans():
    cases = H.a3_cases()
    rec = H.a3_closure_reconciliation(cases)

    assert rec["substage"] == "A3"
    assert rec["scope"] == "same-spec cross-surface consistency"
    assert rec["status"] == "READY_FOR_CLOSURE"
    assert rec["closure_ready"] is True
    assert rec["missing_case_ids"] == []
    assert rec["duplicate_case_ids"] == []
    assert rec["orphan_obligations"] == []
    assert rec["registry_errors"] == []

    families = {row["family"]: row for row in rec["required_families"]}
    assert set(families) == {
        "histogram", "profile", "group_by", "facet_by",
        "subframe_qualified", "selection_vector",
    }
    assert all(row["status"] == "PROVED" for row in families.values())
    assert families["facet_by"]["case_ids"] == [
        "I2-FACET-01", "I2-FACET-DRAW-FIGURES-REFUSAL-01"
    ]
    assert rec["closure_hardening_case_ids"] == ["I2-PROFILE-BINS-01"]

    hist = {row["observable"]: row for row in rec["histogram_bin_observables"]}
    assert set(hist) == {"bin_edges", "bin_counts"}
    for row in hist.values():
        assert row["source"] == "ARTIST_FALLBACK"
        assert row["status"] == "NOT_EXTRACTABLE"
        assert row["a3_disposition"] == "EXPLICIT_NON_CLAIM"
        assert row["rationale"]
        assert row["next_owner"] == "Stage A"
        assert "ARTIST_FALLBACK" in row["future_resolution"]

    # Closure is precise rather than expansive: named later work is carried as
    # a non-claim, not silently promoted into the A3 proof.
    joined = " ".join(rec["explicit_non_claims"])
    assert "raw-row mathematical correctness" in joined
    assert "weights_vector" in joined
    assert "lazy/eager" in joined


def test_a3_24_closure_reconciliation_fails_closed_on_missing_family_or_histogram_upgrade():
    """Independent falsification tests for the A3.11 closure record.

    Construct two regression cases that falsify the closure invariant:
    (1) remove a required A3 family; (2) silently upgrade histogram bin_edges
    from the explicit NOT_EXTRACTABLE non-claim to an EXECUTED claim without
    supplying an executable artist source.  Either mutation must block closure.
    """
    cases = H.a3_cases()

    missing_group = tuple(c for c in cases if c.case_id != "I2-GROUPBY-01")
    rec_missing = H.a3_closure_reconciliation(missing_group)
    assert rec_missing["closure_ready"] is False
    assert rec_missing["status"] == "BLOCKED"
    assert "I2-GROUPBY-01" in rec_missing["missing_case_ids"]
    assert next(row for row in rec_missing["required_families"]
                if row["family"] == "group_by")["status"] == "MISSING"

    mutated = []
    for c in cases:
        if c.case_id != "I2-HIST-01":
            mutated.append(c)
            continue
        changed_obs = tuple(
            replace(o, status="EXECUTED") if o.name == "bin_edges" else o
            for o in c.observables
        )
        mutated.append(replace(c, observables=changed_obs))

    rec_upgrade = H.a3_closure_reconciliation(tuple(mutated))
    assert rec_upgrade["closure_ready"] is False
    assert rec_upgrade["status"] == "BLOCKED"
    assert any("I2-HIST-01/bin_edges" in item
               for item in rec_upgrade["orphan_obligations"])
    # The ordinary registry also rejects the false executable artist claim.
    assert any("I2-HIST-01/bin_edges" in item
               for item in rec_upgrade["registry_errors"])


def test_a3_25_complete_a3_manifest_carries_closure_record(tmp_path):
    cases = H.a3_cases()
    results = []
    for case in cases:
        # This test exercises manifest/reconciliation serialization, not the
        # already-banked numerical cases.  Give each invariance result one
        # synthetic executed comparison so the generic strict gate cannot
        # false-green on PASS-with-zero-comparisons.
        ncmp = 1 if case.purpose in ("INVARIANCE", "CORRECTNESS") else 0
        results.append(H.CaseResult(case_id=case.case_id, status=H.PASS,
                                    executed_comparisons=ncmp))

    path = tmp_path / "a3_11_closure_manifest.json"
    doc = H.write_manifest(str(path), results, cases)
    assert doc["reconciliation"]["exit_code"] == 0
    assert doc["a3_closure"] == H.a3_closure_reconciliation(cases)
    assert doc["a3_closure"]["closure_ready"] is True

    loaded = json.loads(path.read_text())
    assert loaded["a3_closure"]["status"] == "READY_FOR_CLOSURE"
    assert loaded["a3_closure"]["orphan_obligations"] == []
    hist = {row["observable"]: row
            for row in loaded["a3_closure"]["histogram_bin_observables"]}
    assert hist["bin_edges"]["a3_disposition"] == "EXPLICIT_NON_CLAIM"
    assert hist["bin_counts"]["a3_disposition"] == "EXPLICIT_NON_CLAIM"

# ── A3.11 v02 closure-authority hardening ───────────────────────────────────

def test_a3_26_closure_contract_map_blocks_legal_required_case_drift():
    """Independent falsification tests for reviewed A3 proof-contract drift.

    Keep every required case ID present and make declarations that remain legal
    under the generic CaseSpec schema.  A3 closure must nevertheless block if
    the reviewed proof kind or public-surface/not-applicable scope changes.
    """
    cases = H.a3_cases()

    # Proof-kind drift: VISUAL_DIAGNOSTIC is a legal generic purpose, so this
    # mutation specifically tests the closure authority rather than registry
    # syntax validation.
    purpose_drift = tuple(
        replace(c, purpose="VISUAL_DIAGNOSTIC")
        if c.case_id == "I2-GROUPBY-01" else c
        for c in cases
    )
    assert H.validate_registry(purpose_drift) == []
    rec = H.a3_closure_reconciliation(purpose_drift)
    assert rec["status"] == "BLOCKED"
    assert rec["closure_ready"] is False
    assert {
        (row["case_id"], row["field"], row["expected"], row["observed"])
        for row in rec["contract_drift"]
    } >= {("I2-GROUPBY-01", "purpose", "INVARIANCE", "VISUAL_DIAGNOSTIC")}
    group = next(row for row in rec["required_families"]
                 if row["family"] == "group_by")
    assert group["status"] == "CONTRACT_DRIFT"
    assert group["contract_drift_case_ids"] == ["I2-GROUPBY-01"]

    # Surface-list drift can also remain generically valid because two
    # consistency surfaces still exist.  Closure must still reject the
    # narrower proof than the one that was banked.
    surface_drift = tuple(
        replace(c, surfaces_under_test=("draw", "draw_batch"))
        if c.case_id == "I2-PROFILE-01" else c
        for c in cases
    )
    assert H.validate_registry(surface_drift) == []
    rec_surface = H.a3_closure_reconciliation(surface_drift)
    assert rec_surface["status"] == "BLOCKED"
    assert any(row["case_id"] == "I2-PROFILE-01"
               and row["field"] == "surfaces_under_test"
               for row in rec_surface["contract_drift"])

    # not_applicable is independently part of the reviewed surface scope.
    na_drift = tuple(
        replace(c, not_applicable={"draw_figures": "synthetic narrowing"})
        if c.case_id == "I2-SUBFRAME-01" else c
        for c in cases
    )
    assert H.validate_registry(na_drift) == []
    rec_na = H.a3_closure_reconciliation(na_drift)
    assert rec_na["status"] == "BLOCKED"
    assert any(row["case_id"] == "I2-SUBFRAME-01"
               and row["field"] == "not_applicable_surfaces"
               for row in rec_na["contract_drift"])


def test_a3_27_missing_family_manifest_persists_blocked_closure(tmp_path):
    """A missing required A3 family must remain explicit in durable evidence."""
    cases = tuple(c for c in H.a3_cases() if c.case_id != "I2-GROUPBY-01")
    results = []
    for case in cases:
        ncmp = 1 if case.purpose in ("INVARIANCE", "CORRECTNESS") else 0
        results.append(H.CaseResult(case_id=case.case_id, status=H.PASS,
                                    executed_comparisons=ncmp))

    path = tmp_path / "a3_11_v02_blocked_manifest.json"
    doc = H.write_manifest(str(path), results, cases)
    assert "a3_closure" in doc
    assert doc["a3_closure"]["status"] == "BLOCKED"
    assert doc["a3_closure"]["closure_ready"] is False
    assert "I2-GROUPBY-01" in doc["a3_closure"]["missing_case_ids"]
    assert doc["a3_closure"]["execution_context"]["fresh_execution_verdict"] is False
    # The closure-record persistence contract survives later schema revisions;
    # this test must follow the module's current schema authority rather than
    # freeze the historical A3.11 marker after A4 adds manifest fields.
    assert doc["provenance"]["schema_version"] == H.SCHEMA_VERSION

    loaded = json.loads(path.read_text())
    assert loaded["a3_closure"]["status"] == "BLOCKED"
    assert "I2-GROUPBY-01" in loaded["a3_closure"]["missing_case_ids"]
    assert loaded["provenance"]["schema_version"] == H.SCHEMA_VERSION


# ── A4.1 — first expression-slot symmetry increment ─────────────────────────

class _A4TrackingLazyReader:
    """Small in-memory lazy-reader seam for the fast A4 self-check suite.

    It exercises ADF's real ensure_branches/materialize/draw path while keeping
    pytest independent of ROOT files, as required by proposal v1.2 §7.5.
    """

    def __init__(self, data):
        self.data = data.copy()
        self.available_branches = set(data.columns)
        self.loaded_branches = set()
        self.num_entries = len(data)
        self.adf_metadata = None

    def load_branches(self, names):
        names = set(names)
        missing = names - self.available_branches
        if missing:
            raise ValueError(f"missing tracking branches: {sorted(missing)}")
        to_load = names - self.loaded_branches
        self.loaded_branches.update(to_load)
        if not to_load:
            return pd.DataFrame(index=self.data.index)
        return self.data[sorted(to_load)].copy()


def _a4_selection_raw():
    n = 120
    x = np.linspace(0.05, 0.95, n)
    return pd.DataFrame({
        "x": x,
        "y": 1.0 + 0.4 * x,
        "dep_selection": np.tile(np.array([0.0, 1.0]), n // 2),
        "decoy": np.linspace(10.0, 20.0, n),
    })


def _a4_register_selection_alias(adf):
    adf.add_alias("slot_keep", "dep_selection > 0")
    # A4.4 closure falsifier: this unrelated alias shares the SAME physical
    # dependency.  LAZY branch-set evidence alone therefore cannot reveal a
    # buggy EAGER path that materializes every registered alias.
    adf.add_alias("slot_shadow", "dep_selection * 2")
    return adf


def _a4_make_eager(*, contaminate_alias=False):
    adf = _a4_register_selection_alias(ADF(_a4_selection_raw()))
    if contaminate_alias:
        adf.materialize_aliases(names=["slot_keep"])
    return adf


def _a4_make_lazy(*, preload=()):
    raw = _a4_selection_raw()
    adf = ADF(pd.DataFrame(index=range(len(raw))))
    reader = _A4TrackingLazyReader(raw)
    adf._lazy_reader = reader
    adf._chain = {
        "files": [], "entry_offsets": [0], "total_entries": len(raw),
        "validation_mode": None,
    }
    _a4_register_selection_alias(adf)
    if preload:
        adf.ensure_branches(list(preload))
    return adf


def test_a4_01_selection_slot_contract_is_executable_and_manifest_visible(tmp_path):
    case = H.a4_cases()[0]
    assert case.case_id == "I3-SELECTION-01"
    assert case.loading_mode == "BOTH"
    assert case.sample_mode == "FULL"
    assert tuple(case.slots_under_test) == ("selection",)
    assert case.anti_contamination_preconditions
    assert "slots_under_test" not in H.FUTURE_STAGE_FIELDS
    assert "anti_contamination_preconditions" not in H.FUTURE_STAGE_FIELDS
    assert H.validate_registry([case]) == []
    assert H.audit_declared_state() == []

    result = H.run_slot_symmetry(case, _a4_make_eager, _a4_make_lazy)
    assert result.status == H.PASS, result.detail

    doc = H.write_manifest(str(tmp_path / "a4.json"), [result], [case])
    rec = doc["cases"][0]
    assert rec["slots_under_test"] == ["selection"]
    assert rec["anti_contamination_preconditions"] == list(
        case.anti_contamination_preconditions)
    assert set(rec["future_staged"]) == {"reference_policy"}
    assert doc["provenance"]["schema_version"] == H.SCHEMA_VERSION


def test_a4_02_selection_slot_both_proves_materialization_and_exact_lazy_loads():
    case = H.a4_cases()[0]
    result = H.run_slot_symmetry(case, _a4_make_eager, _a4_make_lazy)
    assert result.status == H.PASS, result.detail
    assert result.executed_comparisons == 2
    assert len(result.comparisons) == 2
    assert all(c["ok"] for c in result.comparisons)

    evidence = result.observed["slot_evidence"]
    assert evidence["slot"] == "selection"
    assert evidence["alias"] == "slot_keep"
    assert evidence["eager_alias_materialized"] is True
    assert evidence["lazy_alias_materialized"] is True
    assert evidence["lazy_loaded_before"] == []
    assert set(evidence["lazy_loaded_after"]) == {"x", "y", "dep_selection"}
    assert evidence["required_physical_dependencies"] == ["dep_selection"]
    assert evidence["unrelated_physical_branches"] == ["decoy"]

    # The selection really fires: alternating dep_selection keeps exactly half.
    assert result.observed["n"]["EAGER"] == 60
    assert result.observed["n"]["LAZY"] == 60


def test_a4_03_m2_preload_contamination_is_invalid_fixture_not_pass():
    case = H.a4_cases()[0]

    # EAGER arm: the slot alias was materialized by setup, so the slot no longer
    # proves that selection= discovered it.
    bad_eager = H.run_slot_symmetry(
        case, lambda: _a4_make_eager(contaminate_alias=True), _a4_make_lazy)
    assert bad_eager.status == H.INVALID_FIXTURE, bad_eager.detail
    assert "pre-materialized" in bad_eager.detail

    # LAZY arm: the slot-only physical dependency was already loaded, so the
    # case cannot attribute that load to selection=.
    bad_lazy = H.run_slot_symmetry(
        case, _a4_make_eager,
        lambda: _a4_make_lazy(preload=("dep_selection",)))
    assert bad_lazy.status == H.INVALID_FIXTURE, bad_lazy.detail
    assert "preloaded physical" in bad_lazy.detail

    # M2 contamination can never be a strict green.
    assert H.strict_exit_code([bad_eager], [case]) == 1
    assert H.strict_exit_code([bad_lazy], [case]) == 1

# ── A4.2 — scalar expression-bearing slot expansion ─────────────────────────

def _a4_scalar_raw():
    n = 120
    x = np.linspace(0.05, 0.95, n)
    phase = np.arange(n)
    return pd.DataFrame({
        "x": x,
        "y": 1.0 + 0.4 * x,
        "dep_expr": 2.0 + 0.3 * x,
        "dep_weights": (phase % 4).astype(float) / 3.0,
        "dep_group": (phase % 2).astype(int),
        "dep_facet": ((phase // 2) % 2).astype(int),
        "dep_compound": 0.2 * np.sin(phase / 7.0),
        "decoy": np.linspace(10.0, 20.0, n),
    })


def _a4_register_scalar_aliases(adf):
    adf.add_alias("slot_expr", "dep_expr")
    adf.add_alias("slot_weight", "1.0 + dep_weights")
    adf.add_alias("slot_group", "dep_group")
    adf.add_alias("slot_facet", "dep_facet")
    adf.add_alias("slot_compound", "dep_compound")
    return adf


def _a4_make_scalar_eager():
    return _a4_register_scalar_aliases(ADF(_a4_scalar_raw()))


def _a4_make_scalar_lazy():
    raw = _a4_scalar_raw()
    adf = ADF(pd.DataFrame(index=range(len(raw))))
    reader = _A4TrackingLazyReader(raw)
    adf._lazy_reader = reader
    adf._chain = {
        "files": [], "entry_offsets": [0], "total_entries": len(raw),
        "validation_mode": None,
    }
    return _a4_register_scalar_aliases(adf)


def test_a4_04_scalar_catalogue_and_runner_binding_are_machine_authoritative():
    cases = {c.case_id: c for c in H.a4_cases()}
    expected = {
        "I3-SELECTION-01", "I3-EXPR-01", "I3-WEIGHTS-01",
        "I3-GROUP-BY-01", "I3-FACET-BY-01", "I3-COMPOUND-EXPR-01",
    }
    # A4.2 owns these six contracts; later A4 increments may append siblings.
    assert expected.issubset(set(cases))
    assert H.validate_registry(tuple(cases.values())) == []

    for cid in expected:
        contract = H.A4_SLOT_CONTRACTS[cid]
        assert contract["runner"] == "run_slot_symmetry"
        assert tuple(cases[cid].slots_under_test) == tuple(contract["slots_under_test"])

    # Review hardening: the one-surface CORE_MANDATORY exemption is justified
    # only by an execution contract actually bound to run_slot_symmetry.
    saved = H.A4_SLOT_CONTRACTS["I3-EXPR-01"]
    H.A4_SLOT_CONTRACTS["I3-EXPR-01"] = dict(saved, runner="run_consistency")
    try:
        bad = H.validate_registry([cases["I3-EXPR-01"]])
        assert any("not bound to run_slot_symmetry" in e for e in bad)
        assert any("1 applicable surface" in e for e in bad)
    finally:
        H.A4_SLOT_CONTRACTS["I3-EXPR-01"] = saved


@pytest.mark.parametrize(
    "case_id,expected_loaded",
    [
        ("I3-EXPR-01", {"x", "dep_expr"}),
        ("I3-WEIGHTS-01", {"x", "y", "dep_weights"}),
        ("I3-GROUP-BY-01", {"x", "y", "dep_group"}),
        ("I3-FACET-BY-01", {"x", "y", "dep_facet"}),
        ("I3-COMPOUND-EXPR-01", {"x", "y", "dep_compound"}),
    ],
)
def test_a4_05_to_09_scalar_slots_prove_eager_materialization_and_exact_lazy_loads(
        case_id, expected_loaded):
    case = {c.case_id: c for c in H.a4_cases()}[case_id]
    result = H.run_slot_symmetry(case, _a4_make_scalar_eager, _a4_make_scalar_lazy)
    assert result.status == H.PASS, result.detail
    assert result.executed_comparisons == len(case.observables)
    assert len(result.comparisons) == len(case.observables)
    assert all(c["ok"] for c in result.comparisons)

    evidence = result.observed["slot_evidence"]
    contract = H.A4_SLOT_CONTRACTS[case_id]
    assert evidence["slot"] == contract["slots_under_test"][0]
    assert evidence["alias"] == contract["slot_alias"]
    assert evidence["eager_alias_materialized"] is True
    assert evidence["lazy_alias_materialized"] is True
    assert evidence["lazy_loaded_before"] == []
    assert set(evidence["lazy_loaded_after"]) == expected_loaded
    assert set(evidence["required_physical_dependencies"]) <= expected_loaded
    assert set(evidence["unrelated_physical_branches"]).isdisjoint(expected_loaded)


def test_a4_10_source_contract_is_checked_before_observable_path_resolution():
    case = {c.case_id: c for c in H.a4_cases()}["I3-EXPR-01"]
    bad_observable = replace(
        case.observables[0], source="INDEPENDENT", path="definitely.missing.path")
    bad_case = replace(case, observables=(bad_observable,))

    result = H.run_slot_symmetry(
        bad_case, _a4_make_scalar_eager, _a4_make_scalar_lazy)
    assert result.status == H.INVALID_FIXTURE
    assert "declares source 'INDEPENDENT'" in result.detail
    assert "does not resolve" not in result.detail

# ── A4.3 — vector slots + explicit subframe-vector capability boundary ────────

def _a4_vector_raw():
    n = 120
    x = np.linspace(0.05, 0.95, n)
    phase = np.arange(n)
    return pd.DataFrame({
        "x": x,
        "y": 1.0 + 0.4 * x + 0.03 * np.sin(phase / 9.0),
        "dep_selection_vector": (phase % 2).astype(float),
        "dep_weights_vector": (phase % 4).astype(float) / 3.0,
        "decoy": np.linspace(10.0, 20.0, n),
    })


def _a4_register_vector_aliases(adf):
    adf.add_alias("slot_selection_vector", "dep_selection_vector")
    adf.add_alias("slot_weights_vector", "1.0 + dep_weights_vector")
    return adf


def _a4_make_vector_eager():
    return _a4_register_vector_aliases(ADF(_a4_vector_raw()))


def _a4_make_vector_lazy():
    raw = _a4_vector_raw()
    adf = ADF(pd.DataFrame(index=range(len(raw))))
    reader = _A4TrackingLazyReader(raw)
    adf._lazy_reader = reader
    adf._chain = {
        "files": [], "entry_offsets": [0], "total_entries": len(raw),
        "validation_mode": None,
    }
    return _a4_register_vector_aliases(adf)


def _a4_make_subframe_vector_eager():
    n = 120
    phase = np.arange(n)
    base = ADF(pd.DataFrame({
        "x": np.linspace(0.05, 0.95, n),
        "y": 1.0 + 0.4 * np.linspace(0.05, 0.95, n),
        "kbin": (phase % 4).astype(int),
    }))
    sub = ADF(pd.DataFrame({
        "kbin": np.arange(4, dtype=int),
        "count": np.array([1.0, 2.0, 3.0, 4.0]),
    }))
    base.register_subframe("S", sub, index_columns="kbin")
    return base


def _a4_make_subframe_vector_lazy():
    n = 120
    phase = np.arange(n)
    raw = pd.DataFrame({
        "x": np.linspace(0.05, 0.95, n),
        "y": 1.0 + 0.4 * np.linspace(0.05, 0.95, n),
        "kbin": (phase % 4).astype(int),
    })
    base = ADF(pd.DataFrame(index=range(n)))
    reader = _A4TrackingLazyReader(raw)
    base._lazy_reader = reader
    base._chain = {
        "files": [], "entry_offsets": [0], "total_entries": len(raw),
        "validation_mode": None,
    }
    sub = ADF(pd.DataFrame({
        "kbin": np.arange(4, dtype=int),
        "count": np.array([1.0, 2.0, 3.0, 4.0]),
    }))
    base.register_subframe("S", sub, index_columns="kbin")
    return base


def test_a4_11_vector_catalogue_and_refusal_contracts_are_machine_visible(tmp_path):
    cases = {c.case_id: c for c in H.a4_cases()}
    expected = {
        "I3-SELECTION-VECTOR-01",
        "I3-WEIGHTS-VECTOR-01",
        "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
        "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01",
    }
    assert expected.issubset(set(cases))
    assert H.validate_registry(tuple(cases.values())) == []

    for cid in ("I3-SELECTION-VECTOR-01", "I3-WEIGHTS-VECTOR-01"):
        c = cases[cid]
        assert c.purpose == "INVARIANCE"
        assert c.loading_mode == "BOTH"
        assert H.A4_SLOT_CONTRACTS[cid]["runner"] == "run_slot_symmetry"

    for cid in ("I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
                "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01"):
        c = cases[cid]
        assert c.purpose == "ERROR_CONTRACT"
        assert c.known_bug_status == "KNOWN_BUG"
        assert c.known_bug_id == "BUG_20260701_ADF_subframe_ref_slot_symmetry"
        assert H.A4_SLOT_CONTRACTS[cid]["runner"] == "run_error_contract"

    # The current capability boundary must survive the same manifest path used
    # by normal Stage-A evidence, not only exist as a source-code comment.
    result = H.run_error_contract(
        cases["I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01"],
        _a4_make_subframe_vector_eager, "draw",
        "BUG_20260701_ADF_subframe_ref_slot_symmetry")
    assert result.status == H.PASS, result.detail
    doc = H.write_manifest(str(tmp_path / "a4_3.json"), [result],
                           [cases["I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01"]])
    rec = doc["cases"][0]
    assert rec["known_bug_id"] == "BUG_20260701_ADF_subframe_ref_slot_symmetry"
    assert rec["slots_under_test"] == ["selection_vector"]


@pytest.mark.parametrize(
    "case_id,expected_loaded",
    [
        ("I3-SELECTION-VECTOR-01", {"x", "y", "dep_selection_vector"}),
        ("I3-WEIGHTS-VECTOR-01", {"x", "y", "dep_weights_vector"}),
    ],
)
def test_a4_12_13_vector_slots_prove_eager_materialization_and_exact_lazy_loads(
        case_id, expected_loaded):
    case = {c.case_id: c for c in H.a4_cases()}[case_id]
    result = H.run_slot_symmetry(case, _a4_make_vector_eager, _a4_make_vector_lazy)
    assert result.status == H.PASS, result.detail
    assert result.executed_comparisons == len(case.observables)
    assert len(result.comparisons) == len(case.observables)
    assert all(c["ok"] for c in result.comparisons)

    evidence = result.observed["slot_evidence"]
    contract = H.A4_SLOT_CONTRACTS[case_id]
    assert evidence["slot"] == contract["slots_under_test"][0]
    assert evidence["alias"] == contract["slot_alias"]
    assert evidence["eager_alias_materialized"] is True
    assert evidence["lazy_alias_materialized"] is True
    assert evidence["lazy_loaded_before"] == []
    assert set(evidence["lazy_loaded_after"]) == expected_loaded
    assert set(evidence["unrelated_physical_branches"]).isdisjoint(expected_loaded)

    # Both vector arms must carry data; a degenerate empty branch would make
    # the slot-causality proof weaker even if EAGER and LAZY agreed.
    assert np.nansum(result.observed["signal_count"]["EAGER"]) > 0
    assert np.nansum(result.observed["reference_count"]["EAGER"]) > 0


def test_a4_14_15_subframe_vector_refusals_hold_in_eager_and_lazy_modes():
    cases = {c.case_id: c for c in H.a4_cases()}
    for cid in ("I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
                "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01"):
        case = cases[cid]
        for maker in (_a4_make_subframe_vector_eager, _a4_make_subframe_vector_lazy):
            result = H.run_error_contract(
                case, maker, "draw", "BUG_20260701_ADF_subframe_ref_slot_symmetry")
            # PASS here already means run_error_contract found the exact bug ID
            # in the full exception text; result.detail is intentionally truncated.
            assert result.status == H.PASS, (cid, result.detail)
            assert case.known_bug_id == "BUG_20260701_ADF_subframe_ref_slot_symmetry"
            assert H.strict_exit_code([result], [case]) == 0

# ── A4.4 — cumulative slot-causality closure hardening ──────────────────────

def _a4_make_subframe_expr_eager():
    n = 120
    phase = np.arange(n)
    base = ADF(pd.DataFrame({
        "x": np.linspace(0.05, 0.95, n),
        "kbin": (phase % 4).astype(int),
        "decoy": np.linspace(10.0, 20.0, n),
    }))
    sub = ADF(pd.DataFrame({
        "kbin": np.arange(4, dtype=int),
        "count": np.array([1.0, 2.0, 3.0, 4.0]),
        "count2": np.array([11.0, 12.0, 13.0, 14.0]),
    }))
    base.register_subframe("S", sub, index_columns="kbin")
    return base


def _a4_make_subframe_expr_lazy():
    n = 120
    phase = np.arange(n)
    raw = pd.DataFrame({
        "x": np.linspace(0.05, 0.95, n),
        "kbin": (phase % 4).astype(int),
        "decoy": np.linspace(10.0, 20.0, n),
    })
    base = ADF(pd.DataFrame(index=range(n)))
    reader = _A4TrackingLazyReader(raw)
    base._lazy_reader = reader
    base._chain = {
        "files": [], "entry_offsets": [0], "total_entries": len(raw),
        "validation_mode": None,
    }
    # Established mixed-lazy subframe contract: the structural join key is a
    # setup baseline.  A4.4 does not falsely claim that the expr slot discovers
    # this key; it proves the remaining qualified-expression causality.
    base.ensure_branches(["kbin"])
    sub = ADF(pd.DataFrame({
        "kbin": np.arange(4, dtype=int),
        "count": np.array([1.0, 2.0, 3.0, 4.0]),
        "count2": np.array([11.0, 12.0, 13.0, 14.0]),
    }))
    base.register_subframe("S", sub, index_columns="kbin")
    return base


def test_a4_16_complete_catalogue_and_contract_reconciliation_is_bidirectional(tmp_path):
    cases = H.a4_cases()
    closure = H.a4_closure_reconciliation(cases)
    assert closure["status"] == "READY_FOR_CLOSURE", closure
    assert closure["closure_ready"] is True
    assert set(closure["positive_case_ids"]) == set(H.A4_POSITIVE_CASE_IDS)
    assert set(closure["error_case_ids"]) == set(H.A4_ERROR_CASE_IDS)
    assert set(closure["actual_a4_case_ids"]) == (
        set(H.A4_POSITIVE_CASE_IDS) | set(H.A4_ERROR_CASE_IDS))
    assert set(closure["contract_ids"]) == set(closure["actual_a4_case_ids"])
    assert closure["missing_case_ids"] == []
    assert closure["unexpected_case_ids"] == []
    assert closure["stale_contract_ids"] == []
    assert closure["uncontracted_case_ids"] == []
    assert closure["contract_drift"] == []

    # Durable healthy closure record: this is declaration reconciliation, not
    # a fresh execution verdict.
    doc = H.write_manifest(str(tmp_path / "a4_4_closure.json"), [], cases)
    assert doc["a4_closure"]["status"] == "READY_FOR_CLOSURE"
    assert doc["a4_closure"]["execution_context"]["fresh_execution_verdict"] is False

    # Reverse-direction falsifier: one stale contract must block closure.
    H.A4_SLOT_CONTRACTS["I3-STALE-A4-CONTRACT"] = {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("expr",),
        "slot_alias": "stale_alias",
        "expected_eager_new_aliases": ("stale_alias",),
        "required_physical_dependencies": ("stale_dep",),
        "expected_lazy_loaded_after": ("stale_dep", "x"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": ("synthetic stale contract",),
    }
    try:
        blocked = H.a4_closure_reconciliation(cases)
        assert blocked["status"] == "BLOCKED"
        assert "I3-STALE-A4-CONTRACT" in blocked["stale_contract_ids"]
    finally:
        H.A4_SLOT_CONTRACTS.pop("I3-STALE-A4-CONTRACT", None)


def test_a4_17_slot_alias_exclusivity_is_machine_locked_and_second_slot_fails():
    case = {c.case_id: c for c in H.a4_cases()}["I3-EXPR-01"]
    mutated_spec = dict(case.canonical_spec)
    mutated_spec["selection"] = "slot_expr > 0"
    bad_case = replace(case, canonical_spec=mutated_spec)

    errors = H.validate_registry((bad_case,))
    assert any("slot-exclusivity drift" in e for e in errors)

    result = H.run_slot_symmetry(
        bad_case, _a4_make_scalar_eager, _a4_make_scalar_lazy)
    assert result.status == H.INVALID_FIXTURE
    assert "slot-exclusivity drift" in result.detail
    assert H.strict_exit_code([result], [bad_case]) == 1


def test_a4_18_eager_target_only_materialization_is_proven_with_shared_dependency_alias():
    case = {c.case_id: c for c in H.a4_cases()}["I3-SELECTION-01"]
    result = H.run_slot_symmetry(case, _a4_make_eager, _a4_make_lazy)
    assert result.status == H.PASS, result.detail
    evidence = result.observed["slot_evidence"]
    assert set(evidence["eager_registered_aliases"]) >= {"slot_keep", "slot_shadow"}
    assert evidence["eager_newly_materialized_aliases"] == ["slot_keep"]
    # The non-target alias shares dep_selection, so this is genuinely stronger
    # than the exact physical-load-set oracle.
    assert "slot_shadow" not in _a4_make_eager().df.columns
    assert result.observed["n"]["EAGER"] == 60
    assert result.observed["n"]["LAZY"] == 60


def test_a4_19_shared_physical_dependency_all_alias_materialization_false_green_is_caught(monkeypatch):
    case = {c.case_id: c for c in H.a4_cases()}["I3-SELECTION-01"]
    original = ADF.materialize_aliases

    def materialize_all_registered(self, *args, **kwargs):
        names = kwargs.get("names")
        if names is not None and "slot_keep" in set(names):
            kwargs = dict(kwargs)
            kwargs["names"] = list(self.aliases)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ADF, "materialize_aliases", materialize_all_registered)
    result = H.run_slot_symmetry(case, _a4_make_eager, _a4_make_lazy)
    assert result.status == H.FAIL
    assert "EAGER alias materialization set mismatch" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a4_20_subframe_scalar_causality_and_vector_refusal_ownership_are_complete():
    cases = {c.case_id: c for c in H.a4_cases()}
    positive = cases["I3-SUBFRAME-EXPR-01"]
    result = H.run_subframe_slot_symmetry(
        positive, _a4_make_subframe_expr_eager, _a4_make_subframe_expr_lazy)
    assert result.status == H.PASS, result.detail
    assert result.executed_comparisons == len(positive.observables) == 2
    evidence = result.observed["slot_evidence"]
    assert evidence["qualified_reference"] == "S.count"
    assert evidence["lazy_loaded_before"] == ["kbin"]
    assert set(evidence["lazy_loaded_after"]) == {"kbin", "x"}
    assert "decoy" not in evidence["lazy_loaded_after"]
    assert evidence["subframe_reference_resolved"] is True

    for cid in ("I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
                "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01"):
        c = cases[cid]
        contract = H.A4_SLOT_CONTRACTS[cid]
        assert c.purpose == "ERROR_CONTRACT"
        assert c.known_bug_id == "BUG_20260701_ADF_subframe_ref_slot_symmetry"
        assert contract["runner"] == "run_error_contract"
        assert contract["bug_id_text_is_deliberate_contract"] is True

    # Removing the supported scalar ownership must block cumulative A4 closure.
    without_scalar = tuple(c for c in cases.values() if c.case_id != "I3-SUBFRAME-EXPR-01")
    blocked = H.a4_closure_reconciliation(without_scalar)
    assert blocked["status"] == "BLOCKED"
    assert "I3-SUBFRAME-EXPR-01" in blocked["missing_case_ids"]
    assert "I3-SUBFRAME-EXPR-01" in blocked["subframe_ownership_missing"]


def test_a4_21_historical_closure_ledger_has_no_implicit_obligation_and_blocks_on_loss():
    cases = H.a4_cases()
    closure = H.a4_closure_reconciliation(cases)
    assert closure["ledger_missing"] == []
    assert closure["ledger_unexpected"] == []
    assert closure["ledger_open_blockers"] == []
    ledger = {row["id"]: row for row in closure["historical_ledger"]}
    assert set(ledger) == set(H.A4_REQUIRED_LEDGER_IDS)
    assert ledger["facet_overlay_real_user_bug"]["owner"] == "BUG_dfdraw_20260822_facet_by_overlay_unsupported"
    assert ledger["interval_label_nan_real_user_bug"]["owner"] == "BUG_dfdraw_20260822_format_interval_label_nan_crash"
    assert ledger["subframe_vector_boundary"]["status"] == "ACCEPTED_ERROR_CONTRACT"

    saved = H.A4_CLOSURE_LEDGER
    H.A4_CLOSURE_LEDGER = tuple(
        row for row in saved if row["id"] != "eager_non_target_selectivity")
    try:
        blocked = H.a4_closure_reconciliation(cases)
        assert blocked["status"] == "BLOCKED"
        assert "eager_non_target_selectivity" in blocked["ledger_missing"]
    finally:
        H.A4_CLOSURE_LEDGER = saved

# ── A4.4 v02 — closure-authority semantic/identity hardening ────────────────

def test_a4_22_closure_ledger_status_and_owner_semantics_are_locked(tmp_path):
    cases = H.a4_cases()
    healthy = H.a4_closure_reconciliation(cases)
    assert healthy["status"] == "READY_FOR_CLOSURE"
    assert healthy["ledger_drift"] == []

    mutations = (
        ("eager_non_target_selectivity", "status", "LATER_NOT_A4"),
        ("subframe_vector_boundary", "status", "LATER_NOT_A4"),
        ("subframe_vector_boundary", "owner", "SOME_OTHER_BUG"),
        ("facet_overlay_real_user_bug", "owner", "SOME_OTHER_OWNER"),
        ("interval_label_nan_real_user_bug", "owner", None),
    )
    saved = H.A4_CLOSURE_LEDGER
    try:
        for ledger_id, field_name, wrong_value in mutations:
            mutated = []
            for row in saved:
                row = dict(row)
                if row["id"] == ledger_id:
                    if wrong_value is None:
                        row.pop(field_name, None)
                    else:
                        row[field_name] = wrong_value
                mutated.append(row)
            H.A4_CLOSURE_LEDGER = tuple(mutated)
            blocked = H.a4_closure_reconciliation(cases)
            assert blocked["status"] == "BLOCKED", (ledger_id, field_name, blocked)
            assert any(
                d["id"] == ledger_id and d["field"] == field_name
                for d in blocked["ledger_drift"]
            )

        # Durable negative-state check through the real manifest path.
        mutated = [dict(row) for row in saved]
        for row in mutated:
            if row["id"] == "eager_non_target_selectivity":
                row["status"] = "LATER_NOT_A4"
        H.A4_CLOSURE_LEDGER = tuple(mutated)
        path = tmp_path / "a4_4_v02_ledger_blocked.json"
        doc = H.write_manifest(str(path), [], cases)
        reloaded = json.loads(path.read_text())
        for payload in (doc, reloaded):
            assert payload["a4_closure"]["status"] == "BLOCKED"
            assert any(
                d["id"] == "eager_non_target_selectivity" and d["field"] == "status"
                for d in payload["a4_closure"]["ledger_drift"]
            )
    finally:
        H.A4_CLOSURE_LEDGER = saved


def test_a4_23_refusal_contract_locks_exact_bug_identity_and_proof_scope():
    cases = list(H.a4_cases())
    target_id = "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01"
    index = next(i for i, c in enumerate(cases) if c.case_id == target_id)
    original = cases[index]
    contract = H.A4_SLOT_CONTRACTS[target_id]
    assert contract["expected_known_bug_id"] == "BUG_20260701_ADF_subframe_ref_slot_symmetry"
    assert contract["expected_surfaces_under_test"] == ("draw",)
    assert contract["expected_loading_mode"] == "BOTH"
    assert contract["expected_sample_mode"] == "FULL"

    bad = replace(original, known_bug_id="SOME_OTHER_NONEMPTY_BUG_ID")
    assert bad.known_bug_id
    errors = H.validate_registry((bad,))
    assert any("refusal known_bug_id drift" in e for e in errors)

    cases[index] = bad
    closure = H.a4_closure_reconciliation(tuple(cases))
    assert closure["status"] == "BLOCKED"
    assert any(
        row["case_id"] == target_id and "refusal known_bug_id drift" in row["detail"]
        for row in closure["contract_drift"]
    )

    # The same authority also locks the reviewed draw/BOTH/FULL proof scope.
    for field_name, wrong_value, expected_detail in (
        ("surfaces_under_test", ("draw_batch",), "refusal surfaces_under_test drift"),
        ("loading_mode", "EAGER", "refusal loading_mode drift"),
        ("sample_mode", "FRACTION", "refusal sample_mode drift"),
    ):
        scoped_bad = replace(original, **{field_name: wrong_value})
        scoped_errors = H.validate_registry((scoped_bad,))
        assert any(expected_detail in e for e in scoped_errors), (field_name, scoped_errors)


def test_a4_24_qualified_reference_matching_is_token_exact_and_near_name_replacement_blocks():
    assert H._a4_text_contains_target("S.count", "S.count")
    assert H._a4_text_contains_target("S.count+1", "S.count")
    assert H._a4_text_contains_target("S.count:x", "S.count")
    assert not H._a4_text_contains_target("S.count2", "S.count")
    assert not H._a4_text_contains_target("XS.count", "S.count")
    assert not H._a4_text_contains_target("S.count_more", "S.count")

    # Prove the near-name replacement is a legal supported subframe reference,
    # so the closure failure below is identity-sensitive rather than a syntax error.
    probe = _a4_make_subframe_expr_eager()
    _, _, stats = probe.draw(
        "S.count2:x", type="profile", bins=4, return_data=True, auto_title=True)
    assert stats["n"] == 120

    cases = list(H.a4_cases())
    target_id = "I3-SUBFRAME-EXPR-01"
    index = next(i for i, c in enumerate(cases) if c.case_id == target_id)
    original = cases[index]
    mutated_spec = dict(original.canonical_spec)
    mutated_spec["expr"] = "S.count2:x"
    bad = replace(original, canonical_spec=mutated_spec)

    assert H.a4_actual_target_slots(bad, H.A4_SLOT_CONTRACTS[target_id]) == ()
    errors = H.validate_registry((bad,))
    assert any("slot-exclusivity drift" in e for e in errors)

    cases[index] = bad
    closure = H.a4_closure_reconciliation(tuple(cases))
    assert closure["status"] == "BLOCKED"
    assert any(
        row["case_id"] == target_id and "slot-exclusivity drift" in row["detail"]
        for row in closure["contract_drift"]
    )


# ── A4.4 v03 — duplicate historical-ledger identity hardening ──────────────

def test_a4_25_duplicate_historical_ledger_ids_block_and_persist(tmp_path):
    cases = H.a4_cases()
    healthy = H.a4_closure_reconciliation(cases)
    assert healthy["status"] == "READY_FOR_CLOSURE"
    assert healthy["closure_ready"] is True
    assert healthy["ledger_duplicate_ids"] == []

    saved = H.A4_CLOSURE_LEDGER
    target_id = "eager_non_target_selectivity"
    expected = next(dict(row) for row in saved if row["id"] == target_id)

    try:
        # Strong falsifier: prepend a contradictory duplicate while leaving the
        # later authoritative-looking row untouched.  A last-write-wins map
        # would silently erase the contradiction and false-green closure.
        conflicting = dict(expected)
        conflicting["status"] = "LATER_NOT_A4"
        H.A4_CLOSURE_LEDGER = (conflicting,) + tuple(saved)
        blocked = H.a4_closure_reconciliation(cases)
        assert blocked["status"] == "BLOCKED"
        assert blocked["closure_ready"] is False
        assert blocked["ledger_duplicate_ids"] == [target_id]
        assert sum(row["id"] == target_id for row in blocked["historical_ledger"]) == 2
        assert any("duplicate closure-ledger item" in item for item in blocked["blockers"])

        # Durable negative-state check through the real manifest / JSON path.
        path = tmp_path / "a4_4_v03_duplicate_ledger_blocked.json"
        doc = H.write_manifest(str(path), [], cases)
        reloaded = json.loads(path.read_text())
        for payload in (doc, reloaded):
            closure = payload["a4_closure"]
            assert closure["status"] == "BLOCKED"
            assert closure["closure_ready"] is False
            assert closure["ledger_duplicate_ids"] == [target_id]
            assert sum(row["id"] == target_id for row in closure["historical_ledger"]) == 2

        # Even a byte-for-semantics identical duplicate is ambiguous history:
        # one closure finding must have exactly one authoritative ledger row.
        H.A4_CLOSURE_LEDGER = (dict(expected),) + tuple(saved)
        identical = H.a4_closure_reconciliation(cases)
        assert identical["status"] == "BLOCKED"
        assert identical["closure_ready"] is False
        assert identical["ledger_duplicate_ids"] == [target_id]
    finally:
        H.A4_CLOSURE_LEDGER = saved

# ── A5.1 — first lazy/eager + keyed-subframe + group_by full stack ──────────

def _a5_full_stack_raw():
    """Deterministic parent fixture with two groups and four populated x bins."""
    rows = []
    values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    for group in (0, 1):
        for xbin in range(4):
            kbin = (xbin + group) % 4
            # Stay well inside each explicit bin while giving x nonzero spread.
            for rep in range(6):
                x = (xbin + 0.5) / 4.0 + (rep - 2.5) * 0.002
                rows.append((x, kbin, group, 1000.0 + len(rows)))
    frame = pd.DataFrame(rows, columns=["x", "kbin", "group", "decoy"])
    assert len(frame) == 48
    assert set(frame["group"]) == {0, 1}
    assert set(frame["kbin"]) == {0, 1, 2, 3}
    # Fixture truth: S.count will be this lookup, never a parent column.
    assert "count" not in frame.columns
    assert np.array_equal(values, np.asarray([1.0, 2.0, 4.0, 8.0]))
    return frame


def _a5_full_stack_subframe():
    return pd.DataFrame({
        "kbin": np.arange(4, dtype=int),
        "count": np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64),
        "decoy_s": np.asarray([101.0, 102.0, 104.0, 108.0], dtype=np.float64),
    })


def _a5_make_full_stack_eager():
    base = ADF(_a5_full_stack_raw().copy())
    base.register_subframe(
        "S", ADF(_a5_full_stack_subframe().copy()), index_columns="kbin")
    return base


def _a5_make_full_stack_lazy(*, preload=()):
    raw = _a5_full_stack_raw()
    base = ADF(pd.DataFrame(index=range(len(raw))))
    reader = _A4TrackingLazyReader(raw)
    base._lazy_reader = reader
    base._chain = {
        "files": [], "entry_offsets": [0], "total_entries": len(raw),
        "validation_mode": None,
    }
    # Keyed-subframe joins require the structural key as an explicit setup
    # baseline; A5.1 proves the remaining composed dependency discovery.
    base.ensure_branches(["kbin"])
    if preload:
        base.ensure_branches(list(preload))
    base.register_subframe(
        "S", ADF(_a5_full_stack_subframe().copy()), index_columns="kbin")
    return base


def _a5_full_stack_anchor():
    """Independent NumPy/pandas oracle; never constructs or calls ADF/dfdraw."""
    parent = _a5_full_stack_raw().copy(deep=True)
    sub = _a5_full_stack_subframe().copy(deep=True)
    lookup = dict(zip(sub["kbin"].tolist(), sub["count"].tolist()))
    y = parent["kbin"].map(lookup).to_numpy(dtype=np.float64)
    x = parent["x"].to_numpy(dtype=np.float64)
    group = parent["group"].to_numpy(dtype=int)

    edges = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    out_group = []
    out_count = []
    out_x_center = []
    out_y_mean = []
    for g in sorted(np.unique(group).tolist()):
        gm = group == g
        for ibin in range(4):
            # No fixture value is exactly 1.0; standard half-open bins suffice.
            bm = gm & (x >= edges[ibin]) & (x < edges[ibin + 1])
            assert int(bm.sum()) > 0
            out_group.append(int(g))
            out_count.append(int(bm.sum()))
            out_x_center.append(float(0.5 * (edges[ibin] + edges[ibin + 1])))
            out_y_mean.append(float(np.mean(y[bm])))
    return {
        "group": np.asarray(out_group, dtype=int),
        "count": np.asarray(out_count, dtype=int),
        "x_center": np.asarray(out_x_center, dtype=np.float64),
        "y_mean": np.asarray(out_y_mean, dtype=np.float64),
    }


def test_a5_01_full_stack_case_is_bounded_and_registry_valid():
    case = H.a5_cases()[0]
    assert case.case_id == "I4-SUBFRAME-GROUPBY-01"
    assert case.purpose == "CORRECTNESS"
    assert case.oracle_kind == "CORRECTNESS"
    assert case.loading_mode == "BOTH"
    assert case.sample_mode == "FULL"
    assert tuple(case.surfaces_under_test) == ("draw",)
    assert case.canonical_spec == {
        "expr": "S.count:x",
        "type": "profile",
        "bins": 4,
        "range": (0.0, 1.0),
        "group_by": "group",
        "return_data": True,
        "auto_title": True,
    }
    assert [(o.name, o.source, o.access, o.path) for o in case.observables] == [
        ("group", "INDEPENDENT", "ARRAY", "profile_data.group"),
        ("count", "INDEPENDENT", "ARRAY", "profile_data.count"),
        ("x_center", "INDEPENDENT", "ARRAY", "profile_data.x_center"),
        ("y_mean", "INDEPENDENT", "ARRAY", "profile_data.y_mean"),
    ]
    assert H.validate_registry(H.a5_cases()) == []
    assert H.audit_declared_state() == []


def test_a5_02_full_stack_matches_independent_oracle_in_eager_and_lazy_modes():
    case = H.a5_cases()[0]
    result = H.run_a5_full_stack(
        case, _a5_make_full_stack_eager, _a5_make_full_stack_lazy,
        _a5_full_stack_anchor)
    assert result.status == H.PASS, result.detail

    # Four observables x three comparisons:
    # independent->EAGER, independent->LAZY, EAGER->LAZY.
    assert result.executed_comparisons == 12
    assert len(result.comparisons) == 12
    assert all(row["ok"] for row in result.comparisons), result.comparisons

    evidence = result.observed["full_stack_evidence"]
    assert evidence["qualified_reference"] == "S.count"
    assert evidence["group_by"] == "group"
    assert evidence["lazy_loaded_before"] == ["kbin"]
    assert set(evidence["lazy_loaded_after"]) == {"kbin", "x", "group"}
    assert "decoy" not in evidence["lazy_loaded_after"]
    assert evidence["independent_anchor_computed_before_product"] is True

    expected = _a5_full_stack_anchor()
    assert np.array_equal(result.observed["group"]["independent"], expected["group"])
    assert np.array_equal(result.observed["count"]["independent"], expected["count"])
    assert np.array_equal(expected["count"], np.full(8, 6, dtype=int))
    assert np.allclose(expected["x_center"],
                       np.asarray([.125, .375, .625, .875] * 2))
    assert np.allclose(
        expected["y_mean"],
        np.asarray([1.0, 2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 1.0]))
    assert H.strict_exit_code([result], [case]) == 0


def test_a5_03_independent_oracle_catches_one_lazy_group_bin_corruption(monkeypatch):
    """A5.1 family falsifier: one grouped-bin mutation must gate the case."""
    case = H.a5_cases()[0]

    good = H.run_a5_full_stack(
        case, _a5_make_full_stack_eager, _a5_make_full_stack_lazy,
        _a5_full_stack_anchor)
    assert good.status == H.PASS, good.detail
    assert H.strict_exit_code([good], [case]) == 0

    original_draw = ADF.draw
    mutation = {}

    def corrupted_draw(self, *args, **kwargs):
        raw = original_draw(self, *args, **kwargs)
        if getattr(self, "_lazy_reader", None) is None:
            return raw
        fig, ax, stats = raw
        stats = dict(stats)
        table = stats["profile_data"].copy(deep=True)
        rows = np.flatnonzero(
            (table["group"].to_numpy() == 1)
            & (table["count"].to_numpy() > 0))
        assert len(rows) > 0
        row = int(rows[0])
        mutation["row"] = row
        table.iloc[row, table.columns.get_loc("y_mean")] += 0.5
        stats["profile_data"] = table
        return fig, ax, stats

    monkeypatch.setattr(ADF, "draw", corrupted_draw)
    bad = H.run_a5_full_stack(
        case, _a5_make_full_stack_eager, _a5_make_full_stack_lazy,
        _a5_full_stack_anchor)
    assert bad.status == H.FAIL, bad.detail
    assert "y_mean" in bad.detail
    assert "independent" in bad.detail and "LAZY" in bad.detail
    failed = [row for row in bad.comparisons if not row["ok"]]
    assert len(failed) == 1
    assert failed[0]["observable"] == "y_mean"
    assert failed[0]["candidate"] == "LAZY"
    assert failed[0]["mismatch_count"] == 1
    assert failed[0]["mismatch_indices"] == [[mutation["row"]]]
    assert H.strict_exit_code([bad], [case]) == 1


def test_a5_04_preloaded_unrelated_branch_is_invalid_fixture_not_pass():
    case = H.a5_cases()[0]
    bad = H.run_a5_full_stack(
        case, _a5_make_full_stack_eager,
        lambda: _a5_make_full_stack_lazy(preload=("decoy",)),
        _a5_full_stack_anchor)
    assert bad.status == H.INVALID_FIXTURE, bad.detail
    assert "structural baseline mismatch" in bad.detail
    assert H.strict_exit_code([bad], [case]) == 1

# ── A5.2 — environment-gated real-data G7.32 acceptance plumbing ───────────

class _A52FakeSubframe:
    def __init__(self):
        self.df = pd.DataFrame({
            "quantile_bin": np.arange(4, dtype=np.int16),
            "vertex_x_intercept": np.asarray([0.1, 0.2, 0.3, 0.4]),
        })


class _A52FakeADF:
    def __init__(self, df):
        self.df = df
        self._lazy_reader = None
        self._subframes = {}

    def get_subframe(self, name):
        return self._subframes.get(name)


class _A52FakeGallery:
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False):
        assert lazy is False
        frame = pd.DataFrame({
            "time_s": np.linspace(0.0, 100.0, 20),
            "quantile_bin": np.arange(20, dtype=np.int16) % 4,
            "noise": np.linspace(-1.0, 1.0, 20),
        })
        adf = _A52FakeADF(frame)
        if sample is not None:
            adf.df = adf.df.sample(frac=sample, random_state=42).reset_index(drop=True)
        return adf

    @staticmethod
    def fig32_subframe_vertex(adf):
        adf._subframes["CalibVertex"] = _A52FakeSubframe()
        n = int(len(adf.df))
        stats = {
            "n": n,
            "mean_y": 0.25,
            "profile_data": pd.DataFrame({
                "count": np.asarray([n], dtype=np.int64),
                "y_mean": np.asarray([0.25], dtype=np.float64),
            }),
        }
        return None, None, stats


class _A52NoneGallery(_A52FakeGallery):
    @staticmethod
    def fig32_subframe_vertex(adf):
        return None


class _A52RaiseGallery(_A52FakeGallery):
    @staticmethod
    def fig32_subframe_vertex(adf):
        raise RuntimeError("deliberate G7 failure")


class _A52MissingG7Gallery:
    build_adf = staticmethod(_A52FakeGallery.build_adf)


class _A52NonFiniteProfileGallery(_A52FakeGallery):
    @staticmethod
    def fig32_subframe_vertex(adf):
        adf._subframes["CalibVertex"] = _A52FakeSubframe()
        n = int(len(adf.df))
        stats = {
            "n": n,
            "mean_y": np.nan,
            "profile_data": pd.DataFrame({
                "count": np.asarray([n], dtype=np.int64),
                "y_mean": np.asarray([np.nan], dtype=np.float64),
            }),
        }
        return None, None, stats


class _A52BuildRaiseGallery(_A52FakeGallery):
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
        frame = pd.DataFrame({
            "time_s": np.linspace(0.0, 100.0, 20),
            "quantile_bin": np.arange(20, dtype=np.int16) % 4,
        })
        if sample is not None:
            frame.sample(frac=sample, random_state=42)
        raise RuntimeError("deliberate build failure after sample")


def test_a5_05_unavailable_realdata_environment_skips_without_running_product(tmp_path):
    missing = tmp_path / "missing.root"
    case = H.a5_2_realdata_case(str(missing), gallery_module=_A52RaiseGallery)
    assert case.case_id == H.A5_2_CASE_ID
    assert case.gate == "ENVIRONMENT_GATED"
    assert case.applicable is False
    assert "unavailable" in case.applicability_reason

    result = H.run_a5_2_realdata(
        case, str(missing), gallery_module=_A52RaiseGallery)
    assert result.status == H.SKIP, result.detail
    assert H.strict_exit_code([result], [case]) == 0


def test_a5_06_realdata_case_pins_eager_fraction_20pct_seed42(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 fake ROOT identity")
    case = H.a5_2_realdata_case(str(root_path), gallery_module=_A52FakeGallery)

    assert case.applicable is True
    assert case.purpose == "COVERAGE"
    assert case.gate == "ENVIRONMENT_GATED"
    assert case.loading_mode == "EAGER"
    assert case.sample_mode == "FRACTION"
    assert case.canonical_spec["gallery_function"] == "fig32_subframe_vertex"
    assert case.canonical_spec["sample_fraction"] == 0.20
    assert case.canonical_spec["sample_seed"] == 42
    assert H.validate_registry([case]) == []
    assert H.audit_declared_state() == []


def test_a5_07_realdata_runner_records_actual_sample_identity_and_g7_evidence(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 deterministic input")
    case = H.a5_2_realdata_case(str(root_path), gallery_module=_A52FakeGallery)

    first = H.run_a5_2_realdata(case, str(root_path), gallery_module=_A52FakeGallery)
    second = H.run_a5_2_realdata(case, str(root_path), gallery_module=_A52FakeGallery)
    assert first.status == H.PASS, first.detail
    assert second.status == H.PASS, second.detail
    assert H.strict_exit_code([first], [case]) == 0

    prov = first.observed["realdata_provenance"]
    assert prov["source_rows"] == 20
    assert prov["selected_rows"] == 4
    assert prov["sample_fraction"] == 0.20
    assert prov["sample_seed"] == 42
    assert len(prov["index_digest_sha256"]) == 64
    assert second.observed["realdata_provenance"]["index_digest_sha256"] == (
        prov["index_digest_sha256"])

    g7 = first.observed["g7_32_evidence"]
    assert g7["gallery_function"] == "fig32_subframe_vertex"
    assert g7["calibvertex_subframe_registered"] is True
    assert g7["parent_subframe_column_isolated"] is True
    assert g7["public_n"] == 4
    assert g7["numeric_summary"]["numeric_values"] > 0
    assert g7["numeric_summary"]["finite_values"] > 0
    assert g7["profile_numeric_evidence"]["source"] == "profile_data.y_mean"
    assert g7["profile_numeric_evidence"]["populated_bins"] == 1
    assert g7["profile_numeric_evidence"]["finite_profile_y_values"] == 1


def test_a5_08_realdata_gate_persists_sample_provenance_in_manifest(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 manifest input")
    manifest = tmp_path / "a5_2_manifest.json"

    result, doc, code = H.run_a5_2_realdata_gate(
        str(root_path), manifest_path=str(manifest), gallery_module=_A52FakeGallery)
    assert result.status == H.PASS, result.detail
    assert code == 0
    assert manifest.exists()
    assert doc["provenance"]["sample_fraction"] == 0.20
    assert doc["provenance"]["sample_seed"] == 42
    assert len(doc["provenance"]["index_digest_sha256"]) == 64
    assert doc["provenance"]["selected_rows"] == 4
    assert doc["cases"][0]["case_id"] == H.A5_2_CASE_ID
    assert doc["cases"][0]["status"] == H.PASS


def test_a5_09_applicable_optional_gallery_none_is_fail_not_skip(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 optional-none input")
    case = H.a5_2_realdata_case(str(root_path), gallery_module=_A52NoneGallery)

    result = H.run_a5_2_realdata(
        case, str(root_path), gallery_module=_A52NoneGallery)
    assert result.status == H.FAIL, result.detail
    assert "optional gallery skip is not an acceptance PASS" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_10_applicable_g7_exception_fails_closed_and_wrong_seed_is_invalid(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 failure input")

    case = H.a5_2_realdata_case(str(root_path), gallery_module=_A52RaiseGallery)
    failed = H.run_a5_2_realdata(
        case, str(root_path), gallery_module=_A52RaiseGallery)
    assert failed.status == H.FAIL, failed.detail
    assert "deliberate G7 failure" in failed.detail
    assert H.strict_exit_code([failed], [case]) == 1

    good_case = H.a5_2_realdata_case(str(root_path), gallery_module=_A52FakeGallery)
    invalid = H.run_a5_2_realdata(
        good_case, str(root_path), gallery_module=_A52FakeGallery, seed=7)
    assert invalid.status == H.INVALID_FIXTURE, invalid.detail
    assert "fraction=0.20, seed=42" in invalid.detail
    assert H.strict_exit_code([invalid], [good_case]) == 1

def test_a5_11_missing_required_gallery_callable_is_not_a_skip(tmp_path):
    """A5.2-P1-ENVIRONMENT: trusted-gallery API drift must fail closed."""
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 missing-callable input")

    case = H.a5_2_realdata_case(
        str(root_path), gallery_module=_A52MissingG7Gallery)
    assert case.applicable is True
    assert case.applicability_reason == ""

    result = H.run_a5_2_realdata(
        case, str(root_path), gallery_module=_A52MissingG7Gallery)
    assert result.status == H.INVALID_FIXTURE, result.detail
    assert "missing required callable" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_12_unexpected_gallery_import_exception_is_not_environment_skip(
        tmp_path, monkeypatch):
    """A5.2-P1-ENVIRONMENT: unexpected gallery code/import drift must gate."""
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 import-contract input")

    def broken_import():
        raise RuntimeError("deliberate gallery import contract failure")

    monkeypatch.setattr(H, "_a5_2_import_gallery", broken_import)
    case = H.a5_2_realdata_case(str(root_path))
    assert case.applicable is True

    result = H.run_a5_2_realdata(case, str(root_path))
    assert result.status == H.FAIL, result.detail
    assert "deliberate gallery import contract failure" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_13_finite_bookkeeping_cannot_hide_all_nan_profile_y(tmp_path):
    """A5.2-P1-NUMERIC: n/count finite is not plotted-y evidence."""
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 nonfinite-profile input")
    case = H.a5_2_realdata_case(
        str(root_path), gallery_module=_A52NonFiniteProfileGallery)

    result = H.run_a5_2_realdata(
        case, str(root_path), gallery_module=_A52NonFiniteProfileGallery)
    assert result.status == H.FAIL, result.detail
    assert "no finite populated plotted/profile y evidence" in result.detail
    assert "finite_values" in result.detail  # bookkeeping is visibly finite
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_14_sample_monkeypatch_restored_when_build_fails(tmp_path):
    """Recommended hardening: DataFrame.sample restoration is exception-safe."""
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.2 restoration input")
    case = H.a5_2_realdata_case(
        str(root_path), gallery_module=_A52BuildRaiseGallery)

    original = pd.DataFrame.sample
    result = H.run_a5_2_realdata(
        case, str(root_path), gallery_module=_A52BuildRaiseGallery)
    assert result.status == H.FAIL, result.detail
    assert "deliberate build failure after sample" in result.detail
    assert pd.DataFrame.sample is original
    assert H.strict_exit_code([result], [case]) == 1

# ── A5.3 — real-data G7.32 LAZY/FULL acceptance ──────────────────────────────

class _A53FakeLazyReader:
    def __init__(self, loaded=()):
        self.loaded_branches = set(loaded)


class _A53FakeGallery:
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
        assert sample is None
        assert lazy is True
        assert tree_name == H.A5_3_TREE_NAME
        frame = pd.DataFrame({
            "time_s": np.linspace(0.0, 100.0, 20),
            "quantile_bin": np.arange(20, dtype=np.int16) % 4,
        })
        adf = _A52FakeADF(frame)
        adf._lazy_reader = _A53FakeLazyReader({"timeMS", "sector"})
        return adf

    @staticmethod
    def fig32_subframe_vertex(adf):
        adf._lazy_reader.loaded_branches.update({"vertex_x", "vertex_z"})
        adf._subframes["CalibVertex"] = _A52FakeSubframe()
        n = int(len(adf.df))
        return None, None, {
            "n": n,
            "mean_y": 0.25,
            "profile_data": pd.DataFrame({
                "count": np.asarray([n], dtype=np.int64),
                "y_mean": np.asarray([0.25], dtype=np.float64),
            }),
        }


class _A53EagerDisguiseGallery(_A53FakeGallery):
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
        assert tree_name == H.A5_3_TREE_NAME
        frame = pd.DataFrame({
            "time_s": np.linspace(0.0, 100.0, 20),
            "quantile_bin": np.arange(20, dtype=np.int16) % 4,
        })
        return _A52FakeADF(frame)


class _A53NoExpansionGallery(_A53FakeGallery):
    @staticmethod
    def fig32_subframe_vertex(adf):
        adf._subframes["CalibVertex"] = _A52FakeSubframe()
        n = int(len(adf.df))
        return None, None, {"n": n, "mean_y": 0.25}


class _A53SamplingGallery(_A53FakeGallery):
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
        assert tree_name == H.A5_3_TREE_NAME
        frame = pd.DataFrame({
            "time_s": np.linspace(0.0, 100.0, 20),
            "quantile_bin": np.arange(20, dtype=np.int16) % 4,
        })
        # Deliberate contract violation: FULL lazy setup must never sample.
        frame = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        adf = _A52FakeADF(frame)
        adf._lazy_reader = _A53FakeLazyReader({"timeMS", "sector"})
        return adf


def test_a5_15_lazy_full_case_is_bounded_and_registry_valid(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 lazy/full input")
    case = H.a5_3_realdata_case(str(root_path), gallery_module=_A53FakeGallery)

    assert case.case_id == H.A5_3_CASE_ID
    assert case.purpose == "COVERAGE"
    assert case.gate == "ENVIRONMENT_GATED"
    assert case.loading_mode == "LAZY"
    assert case.sample_mode == "FULL"
    assert case.canonical_spec["sample"] is None
    assert case.canonical_spec["lazy"] is True
    assert case.canonical_spec["tree_name"] == "treeTimeSeries"
    assert H.validate_registry([case]) == []
    assert H.audit_declared_state() == []


def test_a5_16_lazy_full_g7_records_real_lazy_branch_expansion(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 lazy/full execution")
    case = H.a5_3_realdata_case(str(root_path), gallery_module=_A53FakeGallery)

    result = H.run_a5_3_realdata(
        case, str(root_path), gallery_module=_A53FakeGallery)
    assert result.status == H.PASS, result.detail
    assert H.strict_exit_code([result], [case]) == 0

    prov = result.observed["realdata_provenance"]
    assert prov["loading_mode"] == "LAZY"
    assert prov["sample_mode"] == "FULL"
    assert prov["sample_fraction"] is None
    assert prov["sample_seed"] is None
    assert prov["tree_name"] == "treeTimeSeries"
    assert set(prov["lazy_loaded_before"]) == {"sector", "timeMS"}
    assert set(prov["lazy_newly_loaded"]) == {"vertex_x", "vertex_z"}
    assert set(prov["lazy_loaded_after"]) == {
        "sector", "timeMS", "vertex_x", "vertex_z"}

    g7 = result.observed["g7_32_evidence"]
    assert g7["calibvertex_subframe_registered"] is True
    assert g7["parent_subframe_column_isolated"] is True
    assert g7["profile_numeric_evidence"]["finite_profile_y_values"] == 1


def test_a5_17_lazy_full_eager_in_disguise_is_invalid_fixture(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 eager-disguise input")
    case = H.a5_3_realdata_case(
        str(root_path), gallery_module=_A53EagerDisguiseGallery)

    result = H.run_a5_3_realdata(
        case, str(root_path), gallery_module=_A53EagerDisguiseGallery)
    assert result.status == H.INVALID_FIXTURE, result.detail
    assert "_lazy_reader.loaded_branches" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_18_lazy_full_requires_on_demand_branch_expansion(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 no-expansion input")
    case = H.a5_3_realdata_case(
        str(root_path), gallery_module=_A53NoExpansionGallery)

    result = H.run_a5_3_realdata(
        case, str(root_path), gallery_module=_A53NoExpansionGallery)
    assert result.status == H.FAIL, result.detail
    assert "no on-demand physical branch expansion" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_19_lazy_full_forbids_sampling_and_restores_sample(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 sampling-violation input")
    case = H.a5_3_realdata_case(
        str(root_path), gallery_module=_A53SamplingGallery)

    original = pd.DataFrame.sample
    result = H.run_a5_3_realdata(
        case, str(root_path), gallery_module=_A53SamplingGallery)
    assert result.status == H.INVALID_FIXTURE, result.detail
    assert "unexpectedly called pandas.DataFrame.sample" in result.detail
    assert pd.DataFrame.sample is original
    assert H.strict_exit_code([result], [case]) == 1

class _A53TimeMSBlockerGallery:
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
        assert sample is None
        assert lazy is True
        assert tree_name == H.A5_3_TREE_NAME
        raise KeyError("timeMS")

    fig32_subframe_vertex = staticmethod(_A53FakeGallery.fig32_subframe_vertex)


class _A53WrongKeyBlockerGallery(_A53TimeMSBlockerGallery):
    @staticmethod
    def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
        raise KeyError("notTimeMS")


def test_a5_20_realdata_lazy_setup_exact_timems_blocker_is_error_contract_pass(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 exact timeMS blocker")
    case = H.a5_3_lazy_setup_error_case(
        str(root_path), gallery_module=_A53TimeMSBlockerGallery)

    assert case.purpose == "ERROR_CONTRACT"
    assert case.known_bug_status == "KNOWN_BUG"
    assert case.known_bug_id == H.A5_3_BLOCKER_BUG_ID
    result = H.run_a5_3_lazy_setup_error_contract(
        case, str(root_path), gallery_module=_A53TimeMSBlockerGallery)
    assert result.status == H.PASS, result.detail
    assert result.observed["known_bug_evidence"]["exception_key"] == "timeMS"
    assert H.strict_exit_code([result], [case]) == 0


def test_a5_21_realdata_lazy_setup_wrong_key_does_not_false_green(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 wrong blocker")
    case = H.a5_3_lazy_setup_error_case(
        str(root_path), gallery_module=_A53WrongKeyBlockerGallery)

    result = H.run_a5_3_lazy_setup_error_contract(
        case, str(root_path), gallery_module=_A53WrongKeyBlockerGallery)
    assert result.status == H.FAIL, result.detail
    assert "not the owned timeMS key" in result.detail
    assert H.strict_exit_code([result], [case]) == 1


def test_a5_22_realdata_lazy_setup_success_forces_contract_review(tmp_path):
    root_path = tmp_path / "fake.root"
    root_path.write_bytes(b"A5.3 blocker unexpectedly gone")
    case = H.a5_3_lazy_setup_error_case(
        str(root_path), gallery_module=_A53FakeGallery)

    result = H.run_a5_3_lazy_setup_error_contract(
        case, str(root_path), gallery_module=_A53FakeGallery)
    assert result.status == H.FAIL, result.detail
    assert "unexpectedly disappeared" in result.detail
    assert H.strict_exit_code([result], [case]) == 1

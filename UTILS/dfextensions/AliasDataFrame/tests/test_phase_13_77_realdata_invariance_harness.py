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
    case = _base_case(case_id="INV-SURFACE-01", surfaces_under_test=H.SURFACES)
    res = H.run_consistency(case, mk)
    doc = H.write_manifest(str(tmp_path / "m.json"), [res], [case])
    fs = doc["cases"][0]["future_staged"]
    assert fs["slots_under_test"]["owning_stage"] == "A4"
    assert fs["reference_policy"]["owning_stage"] == "A6"


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

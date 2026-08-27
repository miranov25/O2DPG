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
    results = [H.CaseResult(case_id="A", status=H.PASS),
               H.CaseResult(case_id="B", status=H.PASS)]
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
    gates, why = H.gate_decision(case, H.CaseResult(case_id=case.case_id,
                                                    status=status))
    assert gates is expected, f"({applicable}, {status}): {why}"
    assert why

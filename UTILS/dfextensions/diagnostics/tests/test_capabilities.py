"""PHASE_13_74_ADF - capability registry acceptance tests (T-P9 + count lock).

The registry lives in capabilities.py; the human-readable matrix is committed at
docs/CAPABILITY_MATRIX.md.  These tests enforce, in order of importance:

  T-P9            the committed matrix is reproducible from the registry
                  (regenerate and compare byte-for-byte; fails on any drift).
  COUNT-LOCK      exactly COUNT_LOCK capabilities exist (v8 §13.2: 3).
  MAP-INTEGRITY   every proving-test path the registry references actually
                  exists - the registry cannot claim coverage by a test that is
                  not there.  This is the honest core: it is what stops the
                  matrix from asserting a mechanism nobody wrote.

FM#12: each fails against a tree without this increment - T-P9 and COUNT-LOCK
because capabilities.py does not exist; MAP-INTEGRITY because it guards a file
introduced here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

DIAG = Path(__file__).resolve().parent.parent


def _load_capabilities():
    """Load capabilities.py directly by path, so the test does not depend on the
    package being installed on sys.path in the runner."""
    import sys
    spec = importlib.util.spec_from_file_location(
        "diag_capabilities", DIAG / "capabilities.py")
    mod = importlib.util.module_from_spec(spec)
    # register BEFORE exec: @dataclass resolves cls.__module__ via sys.modules,
    # which fails if the module is not registered under its own name first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_TP9_committed_matrix_reproducible_from_registry():
    cap = _load_capabilities()
    committed = (DIAG / "docs" / "CAPABILITY_MATRIX.md")
    assert committed.is_file(), (
        "committed matrix missing; run `python capabilities.py` to generate it")
    regenerated = cap.generate_matrix()
    assert regenerated == committed.read_text(), (
        "committed CAPABILITY_MATRIX.md has drifted from capabilities.py.\n"
        "Regenerate with `python capabilities.py` and commit the result.")


def test_capability_count_lock():
    cap = _load_capabilities()
    assert len(cap.CAPABILITIES) == cap.COUNT_LOCK == 3, (
        f"count lock: {len(cap.CAPABILITIES)} capabilities vs lock "
        f"{cap.COUNT_LOCK}; update COUNT_LOCK deliberately when the set changes")


def test_capability_ids_are_the_expected_three():
    cap = _load_capabilities()
    assert {c.id for c in cap.CAPABILITIES} == {
        "DIAGNOSTICS.host_health",
        "DIAGNOSTICS.run_metrics",
        "DIAGNOSTICS.analytics_report",
    }


def test_mapping_integrity_every_proving_test_exists():
    """Every proving-test path the registry cites must exist on disk.  Prevents
    the registry from claiming a capability is covered by a phantom test."""
    cap = _load_capabilities()
    missing = [p for p in cap.referenced_test_paths()
               if not (DIAG / p).is_file()]
    assert not missing, f"registry references non-existent proving tests: {missing}"


def test_planned_oracles_match_v8_section_13_2_set_exactly():
    """P1-1: the union of every capability's planned_oracles must equal the ten
    named oracles in v8 section 13.2 - no silent drop (4 were missing), no
    invention outside the ratified list."""
    cap = _load_capabilities()
    declared = set()
    for c in cap.CAPABILITIES:
        declared |= set(c.planned_oracles)
    assert declared == set(cap.V8_SECTION_13_2_ORACLES), (
        f"planned oracles != v8 section 13.2 set.\n"
        f"missing: {set(cap.V8_SECTION_13_2_ORACLES) - declared}\n"
        f"unexpected: {declared - set(cap.V8_SECTION_13_2_ORACLES)}")


def test_planned_oracles_are_declared_not_claimed_present():
    """The v8 §13.2 named oracles are recorded as planned.  If a name later
    becomes a real test, that is fine; this test only guarantees the registry
    never lists a planned oracle in the proving-tests set, which would overclaim
    coverage."""
    cap = _load_capabilities()
    for c in cap.CAPABILITIES:
        overlap = set(c.planned_oracles) & set(c.proving_tests)
        assert not overlap, (
            f"{c.id}: planned oracle also listed as a proving test: {overlap}")

"""PHASE_13_101: bounded unsampled Stage-A contract tests.

The ROOT-backed science gate still runs on alma2 via ./run_gallery.sh.
These tests guard the 23-case registry, single-build provenance, and fail-closed
bounded semantics without inventing numerical answers.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNNER = Path(__file__).resolve().parents[1] / "examples/time_series/time_series_draw_invariance.py"


@pytest.fixture
def h():
    spec = importlib.util.spec_from_file_location("adf101_bounded_harness", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Gallery:
    def __init__(self, n=6, ids=None):
        self.n = n
        self.ids = np.arange(n) if ids is None else np.asarray(ids)
        self.calls = []

    def build_adf(self, root_path, **kwargs):
        self.calls.append(kwargs)
        return type("FakeADF", (), {"df": pd.DataFrame({"oracle_row_id": self.ids})})()

    def __getattr__(self, name):
        if name.startswith("fig") or name.startswith("calib"):
            return lambda *a, **kw: None
        raise AttributeError(name)


def test_cli_exposes_bounded_not_sample(h):
    parser = h.build_stage_a_cli_parser()
    ns = parser.parse_args(["fixture.root", "--bounded-entries", "1000000",
                            "--manifest", "x", "--pdf", "y", "--strict"])
    assert ns.bounded_entries == 1_000_000
    assert ns.sample is None
    with pytest.raises(SystemExit):
        parser.parse_args(["fixture.root", "--bounded-entries", "20",
                           "--sample", ".2"])


def test_existing_scientific_registry_has_all_22_cases_and_22_runners(h, tmp_path):
    path = tmp_path / "fixture.root"
    path.write_bytes(b"root")
    gallery = Gallery()
    cases = h._a6_3_fraction_cases(str(path), gallery_module=gallery)
    assert len(cases) == 22
    assert len(h._a6_3_fraction_runners()) == 22
    assert len({c.case_id for c in cases}) == 22
    assert all(c.sample_mode == "FRACTION" for c in cases)


def test_bounded_provenance_only_accepts_explicit_unsampled_contract(h):
    good = {"sample_mode": "FULL", "bounded_fixture_kind": "FIRST_N_NO_FRACTION",
            "sample_fraction": None, "sample_seed": None,
            "entry_start": 0, "entry_stop": 6}
    assert h._a6_4_is_bounded_fixture(good)
    assert not h._a6_4_is_bounded_fixture({**good, "sample_fraction": .2})
    assert not h._a6_4_is_bounded_fixture({**good, "entry_stop": 0})
    assert not h._a6_4_is_bounded_fixture({**good, "entry_start": 1})
    assert not h._a6_4_is_bounded_fixture({**good, "bounded_fixture_kind": "FRACTION"})


def test_bounded_provenance_rejects_case_input_mismatch(h):
    bounded = {"sample_mode": "FULL", "bounded_fixture_kind": "FIRST_N_NO_FRACTION",
               "sample_fraction": None, "sample_seed": None,
               "entry_start": 0, "entry_stop": 6}
    dummy = type("Case", (), {"sample_mode": "FRACTION"})()
    assert not h._a6_4_case_matches_input(dummy, bounded)
    dummy.sample_mode = "FULL"
    assert h._a6_4_case_matches_input(dummy, bounded)
    assert not h._a6_4_case_matches_input(dummy, {"sample_mode": "FRACTION"})


def test_bounded_stage_a_executes_22_plus_one_without_sampling(h, tmp_path, monkeypatch):
    root = tmp_path / "fixture.root"
    root.write_bytes(b"fixture")
    gallery = Gallery()
    seen = []

    def fake_run(case, path, *, gallery_module, prepared_adf, prepared_provenance):
        assert prepared_adf.df["oracle_row_id"].tolist() == list(range(6))
        assert case.sample_mode == "FULL"
        assert prepared_provenance["entry_stop"] == 6
        assert prepared_provenance["sample_fraction"] is None
        assert "entry_stop" not in case.canonical_spec
        assert "entry_start" not in case.canonical_spec
        assert "sample_fraction" not in case.canonical_spec
        seen.append(case.case_id)
        return h.CaseResult(
            case_id=case.case_id, status=h.PASS,
            executed_comparisons=max(1, len(case.observables)),
            observed={"realdata_provenance": dict(prepared_provenance)})

    def fake_visual(case, path, *, pdf_path, gallery_module, sample_fraction, prepared_adf):
        assert case.sample_mode == "FULL"
        assert sample_fraction is None
        Path(pdf_path).write_bytes(b"%PDF-1.4\n%%EOF")
        return h.CaseResult(case_id=case.case_id, status=h.PASS)

    monkeypatch.setattr(h, "_a6_3_fraction_runners", lambda: (fake_run,) * 22)
    monkeypatch.setattr(h, "_run_a6_3_visual_case", fake_visual)
    results, manifest, status = h.run_stage_a_bounded_gate(
        str(root), entry_stop=6,
        manifest_path=str(tmp_path / "manifest.json"),
        pdf_path=str(tmp_path / "gallery.pdf"), gallery_module=gallery)
    assert gallery.calls == [{"sample": None, "lazy": False, "entry_stop": 6}]
    assert len(seen) == 22 and len(set(seen)) == 22
    assert len(results) == 23
    assert len(manifest["cases"]) == 23
    assert manifest["provenance"]["sample_mode"] == "FULL"
    assert manifest["provenance"]["entry_start"] == 0
    assert manifest["provenance"]["entry_stop"] == 6
    assert manifest["provenance"]["sample_fraction"] is None
    assert all(row["sample_mode"] == "FULL" for row in manifest["cases"])
    assert Path(tmp_path / "manifest.json").is_file()
    assert status == 0


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_bound_fails_before_io(h, tmp_path, count):
    root = tmp_path / "fixture.root"
    root.write_bytes(b"fixture")
    gallery = Gallery()
    with pytest.raises(h.HarnessError, match="positive integer"):
        h.run_stage_a_bounded_gate(str(root), entry_stop=count,
            manifest_path=str(tmp_path / "m.json"), pdf_path=str(tmp_path / "f.pdf"),
            gallery_module=gallery)
    assert not gallery.calls


def test_short_read_fails_closed(h, tmp_path):
    root = tmp_path / "fixture.root"
    root.write_bytes(b"fixture")
    gallery = Gallery(n=5)
    with pytest.raises(h.HarnessError, match="expected 6"):
        h.run_stage_a_bounded_gate(str(root), entry_stop=6,
            manifest_path=str(tmp_path / "m.json"), pdf_path=str(tmp_path / "f.pdf"),
            gallery_module=gallery)
    assert gallery.calls == [{"sample": None, "lazy": False, "entry_stop": 6}]


def test_noncontiguous_source_indices_fail_closed(h, tmp_path):
    root = tmp_path / "fixture.root"
    root.write_bytes(b"fixture")
    gallery = Gallery(n=6, ids=[0, 1, 2, 8, 4, 5])
    with pytest.raises(h.HarnessError, match="original entry IDs"):
        h.run_stage_a_bounded_gate(str(root), entry_stop=6,
            manifest_path=str(tmp_path / "m.json"), pdf_path=str(tmp_path / "f.pdf"),
            gallery_module=gallery)


def test_run_gallery_default_includes_all_bounded_scientific_cases():
    script = (RUNNER.parents[2] / "run_gallery.sh").read_text()
    historical = script.index('run_stage historical-23-bounded ')
    new_cases = script.index('run_stage adf101-three-oracles ')
    assert historical < new_cases
    assert 'run_stage legacy-23-full-source' not in script
    assert '"$root_path" --bounded-entries 1000000' in script
    assert '"$root_path" --adf101-fast' in script
    assert 'old_13_101_scientific_cases=26' in script   # PHASE_13_105 S3-0: the old harness's 26 cases are one target of several
    assert 'historical_23_case_stageA=NOT_RUN' not in script


def test_run_gallery_fast_does_not_invoke_sample_20pct():
    script = (RUNNER.parents[2] / "run_gallery.sh").read_text()
    run_block = script[script.index('run_stage focused-pytest'):]
    assert '--sample 0.20' not in run_block
    assert '&&' not in script  # user shell working style
    assert 'exit ' not in script

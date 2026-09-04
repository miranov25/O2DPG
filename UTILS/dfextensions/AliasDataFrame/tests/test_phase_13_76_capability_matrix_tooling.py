"""
PHASE_13_76_ADF — Capability Matrix / runner tooling meta-tests.

Tests-first contract for:
  MATRIX-XFAIL-1
  MATRIX-PARITY-1
  MATRIX-MATCHED-1
  MATRIX-PHASE-1
  RUNNER-FOCUS-1
  reviewer-packet custody ordering

This file is tooling-only.  It does not import AliasDataFrame.py.

The first run against the pre-fix tooling is intentionally RED.  The tests
define the bounded API/behavior that the tooling implementation must satisfy.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "generate_capability_matrix.py"
RUNNER_PATH = PROJECT_ROOT / "run_tests.sh"
HTML_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "generate_matrix_html.py"


def _load_generator():
    """Load the real project generator from its canonical path."""
    spec = importlib.util.spec_from_file_location(
        "_adf_capability_matrix_generator_under_test",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_html_generator():
    """Load the established rich HTML renderer from its canonical path."""
    spec = importlib.util.spec_from_file_location(
        "_adf_capability_matrix_html_generator_under_test",
        HTML_GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature(*, invariance=False):
    """Minimal synthetic taxonomy feature understood by the real matcher."""
    return {
        "id": "META.synthetic",
        "name": "Synthetic meta-test capability",
        "category": "META",
        "test_patterns": ["synthetic.py::TestSynthetic"],
        "_invariance": invariance,
    }


def _markers(*node_ids, invariance=False):
    marker_set = {"invariance"} if invariance else set()
    return {node_id: set(marker_set) for node_id in node_ids}


def _feature_result(generator, feature, results, markers):
    """
    v02 tooling contract: normalized per-feature result.

    The implementation must expose build_feature_result() so all renderers
    consume one status/count object rather than independently recomputing it.
    """
    assert hasattr(generator, "build_feature_result"), (
        "MATRIX-PARITY-1: generator must expose build_feature_result() "
        "as the shared normalized feature-result primitive"
    )
    return generator.build_feature_result(feature, results, markers)


def _matrix_model(generator, results, markers):
    """
    v02 tooling contract: one semantic model feeds Markdown and HTML.
    """
    assert hasattr(generator, "build_matrix_model"), (
        "MATRIX-PARITY-1: generator must expose build_matrix_model()"
    )
    return generator.build_matrix_model(
        results,
        markers,
        phase="PHASE_13_76_ADF_TOOLING_META",
        features=[
            {
                "id": "META.synthetic",
                "name": "Synthetic meta-test capability",
                "category": "META",
                "test_patterns": ["synthetic.py::TestSynthetic"],
            }
        ],
    )


def _status_from_markdown(markdown: str) -> str:
    for label in ("Broken", "Verified", "Smoke-only", "Planned"):
        if re.search(rf"\|\s*(?:🧨|✅|☑️|📋)\s+{re.escape(label)}\s*\|", markdown):
            # Summary can contain all labels; use feature row below instead.
            continue
    m = re.search(
        r"\|\s*(🧨|✅|☑️|📋)\s*\|\s*\*\*META\.synthetic\*\*",
        markdown,
    )
    assert m, markdown
    return {"🧨": "Broken", "✅": "Verified", "☑️": "Smoke-only", "📋": "Planned"}[m.group(1)]


def _status_from_html(html: str) -> str:
    # The shared renderer contract requires feature id + semantic status to be
    # present in the HTML.  Keep parser deliberately format-light.
    assert "META.synthetic" in html
    for status in ("Broken", "Verified", "Smoke-only", "Planned"):
        if re.search(
            rf"META\.synthetic(?:(?!META\.synthetic).){{0,500}}{status}",
            html,
            flags=re.S,
        ) or re.search(
            rf"{status}(?:(?!META\.synthetic).){{0,500}}META\.synthetic",
            html,
            flags=re.S,
        ):
            return status
    pytest.fail("Could not locate semantic feature status in HTML")


class TestMatrixOutcomeSemantics:
    def test_T1_mapped_xfail_cannot_be_verified(self):
        gen = _load_generator()
        node_pass = "synthetic.py::TestSynthetic::test_pass"
        node_xfail = "synthetic.py::TestSynthetic::test_known_defect"
        result = _feature_result(
            gen,
            _feature(),
            {node_pass: "passed", node_xfail: "xfailed"},
            {
                node_pass: {"invariance"},
                node_xfail: {"invariance"},
            },
        )

        assert result["status"] == "Broken"
        assert result["n_xfail"] == 1
        assert result["n_fail"] == 0

    def test_T2_clean_invariance_feature_remains_verified(self):
        gen = _load_generator()
        node = "synthetic.py::TestSynthetic::test_clean_invariance"
        result = _feature_result(
            gen,
            _feature(),
            {node: "passed"},
            {node: {"invariance"}},
        )

        assert result["status"] == "Verified"
        assert result["n_pass"] == 1
        assert result["n_xfail"] == 0

    def test_T3_smoke_plus_mapped_xfail_is_broken(self):
        gen = _load_generator()
        node_pass = "synthetic.py::TestSynthetic::test_smoke"
        node_xfail = "synthetic.py::TestSynthetic::test_known_defect"
        result = _feature_result(
            gen,
            _feature(),
            {node_pass: "passed", node_xfail: "xfailed"},
            {
                node_pass: {"smoke"},
                node_xfail: {"smoke"},
            },
        )

        assert result["status"] == "Broken"
        assert result["n_xfail"] == 1

    def test_T4_skip_is_neutral_and_does_not_promote(self):
        gen = _load_generator()
        skipped = "synthetic.py::TestSynthetic::test_skipped"
        result = _feature_result(
            gen,
            _feature(),
            {skipped: "skipped"},
            {skipped: {"invariance"}},
        )

        assert result["status"] == "Planned"
        assert result["n_skip"] == 1
        assert result["n_pass"] == 0

    def test_T5_xpass_is_visible_but_does_not_promote(self):
        gen = _load_generator()
        xpass = "synthetic.py::TestSynthetic::test_stale_xfail"
        result = _feature_result(
            gen,
            _feature(),
            {xpass: "xpassed"},
            {xpass: {"invariance"}},
        )

        assert result["n_xpass"] == 1
        assert result["status"] != "Verified"


class TestMatrixRendererParity:
    def test_T6_markdown_html_summary_parity(self):
        gen = _load_generator()
        html_gen = _load_html_generator()
        clean = "synthetic.py::TestSynthetic::test_clean"
        broken = "synthetic.py::TestSynthetic::test_known_defect"
        model = _matrix_model(
            gen,
            {clean: "passed", broken: "xfailed"},
            {clean: {"invariance"}, broken: {"invariance"}},
        )

        assert hasattr(gen, "render_markdown")
        assert hasattr(html_gen, "generate_html_from_model")
        md = gen.render_markdown(model)
        html = html_gen.generate_html_from_model(model)

        for status in ("Verified", "Smoke-only", "Broken", "Planned"):
            assert f"{status}" in md
            assert f"{status}" in html

        summary = model["summary"]["status_counts"]
        for status, count in summary.items():
            assert isinstance(count, int)

        assert _status_from_markdown(md) == _status_from_html(html)

    def test_T7_per_feature_markdown_html_status_parity(self):
        gen = _load_generator()
        html_gen = _load_html_generator()
        clean = "synthetic.py::TestSynthetic::test_clean"
        broken = "synthetic.py::TestSynthetic::test_known_defect"
        model = _matrix_model(
            gen,
            {clean: "passed", broken: "xfailed"},
            {clean: {"invariance"}, broken: {"invariance"}},
        )

        md_status = _status_from_markdown(gen.render_markdown(model))
        html_status = _status_from_html(html_gen.generate_html_from_model(model))

        assert md_status == "Broken"
        assert html_status == md_status

    def test_T11_historical_interactive_html_surface_is_preserved(self):
        gen = _load_generator()
        html_gen = _load_html_generator()
        node = "synthetic.py::TestSynthetic::test_clean"
        model = _matrix_model(
            gen,
            {node: "passed"},
            {node: {"invariance"}},
        )
        rendered = html_gen.generate_html_from_model(model)

        # HTML-PRESENTATION-1: parity work must not replace the established
        # human navigation surface with a primitive static table.
        for token in (
            'class="filters"',
            'class="feature-row"',
            'class="tests-row"',
            'class="expand-toggle"',
            'id="cat-filter"',
            "applyFilters()",
            "JavaScript is disabled",
            "permalink",
        ):
            assert token in rendered, token

    def test_T12_feature_permalink_expands_supporting_tests(self):
        gen = _load_generator()
        html_gen = _load_html_generator()
        node = "synthetic.py::TestSynthetic::test_clean"
        model = _matrix_model(
            gen,
            {node: "passed"},
            {node: {"invariance"}},
        )
        rendered = html_gen.generate_html_from_model(model)

        # HTML-NAV-1: a feature permalink is not merely a scroll target.
        # Clicking it or opening #feature-... directly must expand the
        # corresponding tests-row so the supporting proof is visible.
        for token in (
            "function setFeatureExpanded",
            "function expandFeatureFromHash",
            "a[href^=\"#feature-\"]",
            "window.addEventListener('hashchange', expandFeatureFromHash)",
            "expandFeatureFromHash();",
        ):
            assert token in rendered, token

        # Inline stopPropagation was the regression: it made the permalink
        # bypass row expansion without providing an alternate expansion path.
        assert 'title="permalink" onclick="event.stopPropagation()"' not in rendered




class TestMatrixAccountingAndPhase:
    def test_T13_unique_matched_nodes_are_distinct_from_feature_associations(self):
        gen = _load_generator()
        shared = "synthetic.py::TestSynthetic::test_shared"
        only_a = "synthetic.py::TestSynthetic::test_only_a"
        only_b = "synthetic.py::TestSynthetic::test_only_b"
        features = [
            {
                "id": "META.a",
                "name": "A",
                "category": "META",
                "test_patterns": [
                    "synthetic.py::TestSynthetic::test_shared",
                    "synthetic.py::TestSynthetic::test_only_a",
                ],
            },
            {
                "id": "META.b",
                "name": "B",
                "category": "META",
                "test_patterns": [
                    "synthetic.py::TestSynthetic::test_shared",
                    "synthetic.py::TestSynthetic::test_only_b",
                ],
            },
        ]
        results = {shared: "passed", only_a: "passed", only_b: "passed"}
        markers = {node: {"smoke"} for node in results}
        model = gen.build_matrix_model(results, markers, phase="PHASE_13_76_ADF", features=features)
        summary = model["summary"]
        assert summary["matched_tests"] == 3
        assert summary["feature_test_associations"] == 4
        rendered = gen.render_markdown(model)
        assert "Unique matched tests" in rendered
        assert "Feature-test associations" in rendered

    def test_T14_runner_phase_provenance_is_adf_specific(self):
        text = RUNNER_PATH.read_text(encoding="utf-8")
        assert "PHASE_[0-9]*_DF*_END" not in text
        assert 'ADF_MATRIX_PHASE:-PHASE_13_79_ADF' in text
        assert 'PHASE_FOR_MATRIX="${ADF_MATRIX_PHASE:-PHASE_13_79_ADF}"' in text


class TestFocusedRunnerContract:
    def _runner_text(self):
        return RUNNER_PATH.read_text(encoding="utf-8")

    def test_T8_focused_normal_and_raw_use_same_selection(self):
        text = self._runner_text()

        assert 'focused_nodes_' in text
        assert 'mapfile -t FOCUSED_NODES < "$FOCUSED_NODE_LOG"' in text

        normal = re.search(
            r'python3\s+-m\s+pytest\s+"\$\{FOCUSED_NODES\[@\]\}"(?P<args>[^\n]*)',
            text,
        )
        raw = re.search(
            r'python3\s+-m\s+pytest\s+"\$\{FOCUSED_NODES\[@\]\}"(?P<args>[^\n]*--runxfail[^\n]*)',
            text,
        )
        assert normal, "focused normal exact-node pytest command not found"
        assert raw, "focused --runxfail exact-node pytest command not found"

    def test_T9_both_focused_lanes_use_configured_workers(self):
        text = self._runner_text()
        commands = re.findall(
            r'python3\s+-m\s+pytest\s+"\$\{FOCUSED_NODES\[@\]\}"[^\n]*(?:\\\n[^\n]*)*',
            text,
        )
        assert len(commands) >= 2, commands
        normal = next(c for c in commands if "--runxfail" not in c)
        raw = next(c for c in commands if "--runxfail" in c)
        for label, command in (("normal", normal), ("runxfail", raw)):
            assert '-n "$PYTEST_WORKERS"' in command, (
                f"RUNNER-FOCUS-1: {label} focused lane must use "
                f'the configured -n "$PYTEST_WORKERS" xdist setting'
            )

    def test_T15_manifest_is_execution_authority_for_both_focused_lanes(self):
        text = self._runner_text()
        collect_pos = text.find('--collect-only')
        mapfile_pos = text.find('mapfile -t FOCUSED_NODES < "$FOCUSED_NODE_LOG"')
        normal_pos = text.find('python3 -m pytest "${FOCUSED_NODES[@]}" -n "$PYTEST_WORKERS"')
        raw_pos = text.find('python3 -m pytest "${FOCUSED_NODES[@]}" --runxfail -n "$PYTEST_WORKERS"')
        assert -1 not in (collect_pos, mapfile_pos, normal_pos, raw_pos)
        assert collect_pos < mapfile_pos < normal_pos < raw_pos
        assert 'python3 -m pytest $FOCUSED_TESTS -n "$PYTEST_WORKERS"' not in text
        assert 'python3 -m pytest $FOCUSED_TESTS --runxfail' not in text


class TestReviewerPacketCustody:
    def test_T10_manifest_and_zip_are_finalized_after_matrix_rendering(self):
        text = RUNNER_PATH.read_text(encoding="utf-8")

        matrix_pos = text.find('echo "--- Generating capability matrix ---"')
        md5_write_pos = text.find('> "$MD5_MANIFEST"')
        zip_pos = text.find('echo "--- Packaging reviewer.zip ---"')

        assert matrix_pos >= 0
        assert md5_write_pos >= 0
        assert zip_pos >= 0

        assert matrix_pos < md5_write_pos < zip_pos, (
            "Packet custody contract: generate/render matrices first, then "
            "compute the MD5 manifest, then package reviewer.zip"
        )

        # The historical rich HTML renderer must remain active, but its
        # canonical runner path must consume pytest JSON/shared semantics, not
        # independently recompute status from the terminal log.
        assert "generate_matrix_html.py" in text
        assert '--test-results "$MATRIX_JSON"' in text
        assert '--log "$LOG_FILE"' not in text


class TestPhase1379DiagnosticIndex:
    def test_T16_json_projection_is_ai_readable_and_has_source_locators(self):
        gen = _load_generator()
        node = "synthetic.py::TestSynthetic::test_clean_invariance[param]"
        model = gen.build_matrix_model(
            {node: "passed"},
            {node: set()},
            phase="PHASE_13_79_ADF",
            features=[{
                "id": "META.synthetic",
                "name": "Synthetic",
                "description": "Human-readable diagnostic description.",
                "category": "META",
                "test_patterns": ["synthetic.py::TestSynthetic"],
            }],
        )
        payload = gen.export_json_model(model)
        assert payload["schema"] == "AliasDataFrame.CapabilityMatrix"
        assert payload["schema_version"] == 1
        assert payload["features"][0]["description"].startswith("Human-readable")
        rec = payload["features"][0]["tests"][0]
        assert rec["node_id"] == node
        assert rec["evidence_layer"] == "invariance"
        assert rec["file"] == "tests/synthetic.py"
        assert "line" in rec
        assert "provenance" in payload

    def test_T17_markdown_contains_description_and_expandable_owned_nodes(self):
        gen = _load_generator()
        node = "synthetic.py::TestSynthetic::test_clean"
        model = gen.build_matrix_model(
            {node: "passed"}, {node: {"smoke"}}, phase="PHASE_13_79_ADF",
            features=[{
                "id": "META.synthetic", "name": "Synthetic",
                "description": "Findable capability description.",
                "category": "META", "test_patterns": ["synthetic.py::TestSynthetic"],
            }],
        )
        md = gen.render_markdown(model)
        assert "Findable capability description." in md
        assert "<details>" in md
        assert node in md
        assert "tests/synthetic.py" in md

    def test_T18_html_contains_description_and_source_locator_surface(self):
        gen = _load_generator()
        html_gen = _load_html_generator()
        node = "synthetic.py::TestSynthetic::test_clean"
        model = gen.build_matrix_model(
            {node: "passed"}, {node: {"smoke"}}, phase="PHASE_13_79_ADF",
            features=[{
                "id": "META.synthetic", "name": "Synthetic",
                "description": "HTML diagnostic description.",
                "category": "META", "test_patterns": ["synthetic.py::TestSynthetic"],
            }],
        )
        html = html_gen.generate_html_from_model(model)
        assert "HTML diagnostic description." in html
        assert "feature-description" in html
        assert "source-locator" in html
        assert "tests/synthetic.py" in html

    def test_T19_runner_defaults_focus_to_phase_13_79_and_packages_candidates(self):
        text = RUNNER_PATH.read_text(encoding="utf-8")
        assert 'FOCUSED_TESTS="${FOCUSED_TESTS:-tests/test_phase_13_79_slot_grid*.py}"' in text
        assert 'docs/CAPABILITY_MATRIX.json' in text
        assert 'for f in $CANDIDATE_FILES' in text
        assert 'ZIP_FILES="$ZIP_FILES $f"' in text


class TestPhase1379SlotGridRegistration:
    def test_T20_slot_grid_contract_summary_is_machine_derived(self):
        gen = _load_generator()
        feature = next(f for f in gen.FEATURES if f["id"] == "DRAW.slot_grid")
        summary = gen.feature_contract_summary(feature)
        assert summary["declared_cells"] == 88
        assert summary["cell_state_counts"] == {"KNOWN_GAP": 18, "PASSING": 70}
        assert summary["interface_contract_counts"] == {"SUPPORTED": 88}
        assert summary["seams"] == 4
        assert len(summary["sha256"]) == 64
        assert feature["test_patterns"] == [
            "test_phase_13_79_slot_grid.py",
            "test_phase_13_79_slot_grid_invariance.py",
        ]


class TestPhase1379MachineSurfaceExport:
    def test_T21_json_carries_full_slot_grid_cells_and_seams(self):
        gen = _load_generator()
        feature = next(f for f in gen.FEATURES if f["id"] == "DRAW.slot_grid")
        surfaces = gen.feature_contract_surfaces(feature)

        cells = surfaces["DRAW.slot_grid"]
        seams = surfaces["DRAW.slot_grid.seams"]
        assert cells["evidence_class"] == "cell"
        assert seams["evidence_class"] == "seam"
        assert len(cells["rows"]) == 88
        assert len(seams["rows"]) == 4
        assert {row["cell_id"] for row in cells["rows"]} == {
            row["cell_id"] for row in gen._load_feature_contract(feature)[0]["cells"]
        }
        assert {row["seam_id"] for row in seams["rows"]} == {
            row["seam_id"] for row in gen._load_feature_contract(feature)[0]["seams"]
        }

        model = gen.build_matrix_model({}, {}, phase="PHASE_13_79_ADF", features=[feature])
        payload = gen.export_json_model(model)
        assert payload["surfaces"] == surfaces
        assert payload["semantic_payload_digest"] == gen._semantic_payload_digest(surfaces)

    def test_T22_semantic_payload_digest_is_deterministic_and_surface_sensitive(self):
        gen = _load_generator()
        feature = next(f for f in gen.FEATURES if f["id"] == "DRAW.slot_grid")
        surfaces = gen.feature_contract_surfaces(feature)
        digest = gen._semantic_payload_digest(surfaces)
        assert len(digest) == 64
        assert digest == gen._semantic_payload_digest(surfaces)

        changed = {key: dict(value) for key, value in surfaces.items()}
        changed["DRAW.slot_grid"] = dict(changed["DRAW.slot_grid"])
        changed["DRAW.slot_grid"]["rows"] = [
            dict(row) for row in changed["DRAW.slot_grid"]["rows"]
        ]
        changed["DRAW.slot_grid"]["rows"][0]["current_state"] = "__MUTATED__"
        assert gen._semantic_payload_digest(changed) != digest

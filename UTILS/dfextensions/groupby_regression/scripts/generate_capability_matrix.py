#!/usr/bin/env python3
"""
Capability Matrix Generator — Phase 13.7.GB

Reads feature taxonomy + test layer classification to produce a two-tier
capability matrix showing which features are truly verified vs smoke-only.

Implements:
  - Two-tier status (§3.1): Verified, Smoke-only, Broken, Partial, Planned
  - Fail-closed rule (§3.5): unclassified → smoke, warns UNCLASSIFIED
  - Verbose deduplication (§3.6): verbose SW tests counted once
  - Benchmark proof column (§4.3): informational bench references
  - Classification decision tree results (§3.3)

Usage:
  python scripts/generate_capability_matrix.py [--json] [--output PATH]

Reads:
  tests/feature_taxonomy.py
  tests/test_layer_classification.py
  (Optional) pytest JSON report for pass/fail status

Outputs:
  docs/CAPABILITY_MATRIX.md (default)
"""

import sys
import os
import json
import argparse
from datetime import datetime
from collections import Counter

# Add tests/ to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TESTS_DIR = os.path.join(PROJECT_DIR, "tests")
sys.path.insert(0, TESTS_DIR)

from feature_taxonomy import FEATURE_TAXONOMY, VERBOSE_DUPLICATES, IMPL_TAGS
from test_layer_classification import TEST_LAYERS


# ==================================================================
# Layer lookup with fail-closed default
# ==================================================================

def get_layer(test_id):
    """Get layer for a test ID. Fail-closed: unclassified → smoke."""
    layer = TEST_LAYERS.get(test_id)
    if layer is None:
        return "smoke", True  # (layer, is_unclassified)
    return layer, False


def is_verbose_duplicate(test_id):
    """Check if test_id is from a verbose duplicate file."""
    for verbose_file in VERBOSE_DUPLICATES:
        if test_id.startswith(verbose_file + "::"):
            return True
    return False


# ==================================================================
# Status computation
# ==================================================================

VERIFIED_LAYERS = {"invariance", "integration"}

def compute_feature_status(feature_id, feature_def, test_results=None):
    """
    Compute two-tier status for a feature.

    Returns dict with:
      status: str — one of ✅ ☑️ 🧨 ⚠️ 📋
      status_label: str — Verified, Smoke-only, Broken, Partial, Planned
      tests: list of {test_id, layer, is_unclassified, is_verbose, passed}
      has_invariance: bool
      has_bench_proof: bool
      unclassified_count: int
    """
    proof_tests = feature_def["proof"]
    bench_proof = feature_def.get("bench_proof", [])

    if not proof_tests:
        return {
            "status": "📋",
            "status_label": "Planned",
            "tests": [],
            "has_invariance": False,
            "has_bench_proof": bool(bench_proof),
            "unclassified_count": 0,
        }

    tests = []
    has_invariance = False
    unclassified_count = 0

    for tid in proof_tests:
        is_verbose = is_verbose_duplicate(tid)
        layer, is_unclassified = get_layer(tid)
        if is_unclassified:
            unclassified_count += 1

        # Determine pass/fail from test results (if available)
        passed = None
        if test_results is not None:
            passed = test_results.get(tid)

        # Check for verified layers (skip verbose duplicates for promotion)
        if not is_verbose and layer in VERIFIED_LAYERS:
            has_invariance = True

        tests.append({
            "test_id": tid,
            "layer": layer,
            "is_unclassified": is_unclassified,
            "is_verbose": is_verbose,
            "passed": passed,
        })

    # Determine status
    if test_results is not None:
        all_passed = all(t["passed"] for t in tests if t["passed"] is not None)
        any_failed = any(t["passed"] is False for t in tests)
        any_skipped = any(t["passed"] is None for t in tests)

        if any_failed:
            status = "🧨"
            label = "Broken"
        elif not all_passed and any_skipped:
            status = "⚠️"
            label = "Partial"
        elif has_invariance:
            status = "✅"
            label = "Verified"
        else:
            status = "☑️"
            label = "Smoke-only"
    else:
        # No test results — assume all pass, report layer-based status
        if has_invariance:
            status = "✅"
            label = "Verified"
        else:
            status = "☑️"
            label = "Smoke-only"

    return {
        "status": status,
        "status_label": label,
        "tests": tests,
        "has_invariance": has_invariance,
        "has_bench_proof": bool(bench_proof),
        "unclassified_count": unclassified_count,
    }


# ==================================================================
# Matrix generation
# ==================================================================

def generate_matrix(test_results=None):
    """Generate the complete capability matrix."""
    matrix = {}
    for fid, fdef in FEATURE_TAXONOMY.items():
        matrix[fid] = compute_feature_status(fid, fdef, test_results)
        matrix[fid]["feature"] = fdef
    return matrix


def format_markdown(matrix):
    """Format matrix as markdown document."""
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Capability Matrix — groupby_regression")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Phase:** 13.7.GB — Test Quality Classification")
    lines.append(f"**Generator:** `scripts/generate_capability_matrix.py`")
    lines.append("")

    # Summary
    statuses = Counter(m["status_label"] for m in matrix.values())
    total = len(matrix)
    verified = statuses.get("Verified", 0)
    smoke = statuses.get("Smoke-only", 0)
    broken = statuses.get("Broken", 0)
    partial = statuses.get("Partial", 0)
    planned = statuses.get("Planned", 0)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count | % |")
    lines.append(f"|--------|------:|--:|")
    lines.append(f"| ✅ Verified | {verified} | {100*verified/total:.1f}% |")
    lines.append(f"| ☑️ Smoke-only | {smoke} | {100*smoke/total:.1f}% |")
    lines.append(f"| 🧨 Broken | {broken} | {100*broken/total:.1f}% |")
    lines.append(f"| ⚠️ Partial | {partial} | {100*partial/total:.1f}% |")
    lines.append(f"| 📋 Planned | {planned} | {100*planned/total:.1f}% |")
    lines.append(f"| **Total** | **{total}** | |")
    lines.append("")

    # Layer summary
    all_layers = Counter()
    unclassified_total = 0
    for m in matrix.values():
        for t in m["tests"]:
            if not t["is_verbose"]:
                all_layers[t["layer"]] += 1
                if t["is_unclassified"]:
                    unclassified_total += 1

    lines.append("## Test Layer Distribution (excluding verbose duplicates)")
    lines.append("")
    lines.append("| Layer | Count |")
    lines.append("|-------|------:|")
    for layer in ["invariance", "integration", "performance", "smoke", "validation"]:
        lines.append(f"| {layer} | {all_layers.get(layer, 0)} |")
    lines.append(f"| **Total unique** | **{sum(all_layers.values())}** |")
    lines.append("")

    if unclassified_total > 0:
        lines.append(f"⚠️ **UNCLASSIFIED tests: {unclassified_total}** — "
                     f"defaulted to smoke (fail-closed rule §3.5)")
        lines.append("")

    # Feature tables by module
    modules = {}
    for fid, m in matrix.items():
        mod = m["feature"]["module"]
        if mod not in modules:
            modules[mod] = []
        modules[mod].append((fid, m))

    for mod in sorted(modules.keys()):
        features = modules[mod]
        lines.append(f"## {mod}")
        lines.append("")
        lines.append("| Status | Feature | Tests | Inv/Int | Bench | Tag |")
        lines.append("|--------|---------|------:|--------:|-------|-----|")

        for fid, m in sorted(features, key=lambda x: x[0]):
            feat = m["feature"]
            n_tests = len([t for t in m["tests"] if not t["is_verbose"]])
            n_inv = len([t for t in m["tests"]
                        if not t["is_verbose"] and t["layer"] in VERIFIED_LAYERS])
            bench = "✓" if m["has_bench_proof"] else ""
            tag = feat.get("impl_tag") or ""
            name = feat["name"]
            lines.append(f"| {m['status']} | **{fid}** — {name} | {n_tests} | {n_inv} | {bench} | {tag} |")

        lines.append("")

    # Benchmark proof catalog
    lines.append("## Benchmark Proof Catalog")
    lines.append("")
    lines.append("| Feature | Benchmark Check | Gate? |")
    lines.append("|---------|-----------------|-------|")
    for fid, fdef in sorted(FEATURE_TAXONOMY.items()):
        for bp in fdef.get("bench_proof", []):
            gated = "✅ GATED" if "[GATED]" in bp else "📊 MONITOR"
            lines.append(f"| {fid} | {bp} | {gated} |")
    lines.append("")

    # Unclassified warnings
    unclassified_tests = []
    for m in matrix.values():
        for t in m["tests"]:
            if t["is_unclassified"]:
                unclassified_tests.append(t["test_id"])

    if unclassified_tests:
        lines.append("## ⚠️ UNCLASSIFIED Tests (fail-closed → smoke)")
        lines.append("")
        for tid in sorted(unclassified_tests):
            lines.append(f"- `{tid}`")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Two-tier verification per Phase 13.7.GB v02 proposal.*")
    lines.append(f"*✅ = invariance/integration test exists. "
                 f"☑️ = smoke tests only — does not catch numerical regressions.*")
    lines.append(f"*Verbose SW duplicates ({len(VERBOSE_DUPLICATES)} files) "
                 f"deduplicated per §3.6.*")

    return "\n".join(lines)


def format_json(matrix):
    """Format matrix as JSON for programmatic consumption."""
    output = {
        "generated": datetime.utcnow().isoformat(),
        "phase": "13.7.GB",
        "features": {},
        "summary": {},
    }

    statuses = Counter()
    for fid, m in matrix.items():
        statuses[m["status_label"]] += 1
        output["features"][fid] = {
            "name": m["feature"]["name"],
            "module": m["feature"]["module"],
            "status": m["status"],
            "status_label": m["status_label"],
            "impl_tag": m["feature"].get("impl_tag"),
            "test_count": len([t for t in m["tests"] if not t["is_verbose"]]),
            "invariance_count": len([t for t in m["tests"]
                                    if not t["is_verbose"]
                                    and t["layer"] in VERIFIED_LAYERS]),
            "has_bench_proof": m["has_bench_proof"],
            "bench_proof": m["feature"].get("bench_proof", []),
            "unclassified_count": m["unclassified_count"],
        }

    output["summary"] = dict(statuses)
    output["summary"]["total"] = len(matrix)

    return json.dumps(output, indent=2)


# ==================================================================
# Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate capability matrix for groupby_regression"
    )
    parser.add_argument("--json", action="store_true",
                       help="Output JSON instead of Markdown")
    parser.add_argument("--output", "-o", default=None,
                       help="Output file path (default: docs/CAPABILITY_MATRIX.md)")
    parser.add_argument("--test-results", default=None,
                       help="Path to pytest JSON report for pass/fail status")
    args = parser.parse_args()

    # Load test results if provided
    test_results = None
    if args.test_results and os.path.exists(args.test_results):
        with open(args.test_results) as f:
            report = json.load(f)
        test_results = {}
        for test in report.get("tests", []):
            node_id = test.get("nodeid", "")
            outcome = test.get("outcome", "")
            test_results[node_id] = (outcome == "passed")

    # Generate matrix
    matrix = generate_matrix(test_results)

    # Format output
    if args.json:
        content = format_json(matrix)
        default_path = os.path.join(PROJECT_DIR, "docs", "capability_matrix.json")
    else:
        content = format_markdown(matrix)
        default_path = os.path.join(PROJECT_DIR, "docs", "CAPABILITY_MATRIX.md")

    output_path = args.output or default_path

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Capability matrix written to: {output_path}")

    # Print summary to stdout
    statuses = Counter(m["status_label"] for m in matrix.values())
    total = len(matrix)
    print(f"\n  Features: {total}")
    for label in ["Verified", "Smoke-only", "Broken", "Partial", "Planned"]:
        count = statuses.get(label, 0)
        print(f"  {label}: {count} ({100*count/total:.1f}%)")

    # Unclassified warning
    unclassified = sum(m["unclassified_count"] for m in matrix.values())
    if unclassified:
        print(f"\n  ⚠️  UNCLASSIFIED tests: {unclassified} (defaulted to smoke)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Capability Matrix Generator — dfdraw

Auto-generates CAPABILITY_MATRIX.md from:
  1. pytest JSON report (.pytest_report.json)
  2. tests/feature_taxonomy.py (feature → test mapping)
  3. tests/test_layer_classification.py (test → invariance/smoke)

Usage:
  python scripts/generate_capability_matrix.py --test-results .pytest_report.json
  python scripts/generate_capability_matrix.py --test-results path.json --output docs/CAPABILITY_MATRIX.md

Follows GroupByRegression reference implementation.
"""

import json
import sys
import os
import argparse
from datetime import datetime
from collections import Counter
from pathlib import Path

# Add project root to path so we can import from tests/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from tests.feature_taxonomy import FEATURES
from tests.test_layer_classification import TEST_LAYERS


def load_test_results(json_path):
    """Load pytest JSON report → {nodeid: outcome}."""
    with open(json_path) as f:
        report = json.load(f)
    results = {}
    for test in report.get("tests", []):
        nodeid = test["nodeid"]
        # Normalize: "tests/test_foo.py::TestBar::test_baz" → "test_foo.py::TestBar::test_baz"
        if "/" in nodeid:
            nodeid = nodeid.split("/")[-1]
        results[nodeid] = test["outcome"]  # 'passed', 'failed', 'skipped'
    return results


def compute_feature_status(feature, test_results):
    """Determine feature status from its tests + layer classification."""
    tests = feature.get("tests", [])
    if not tests:
        return "📋", "Planned", 0, 0

    n_pass = 0
    n_fail = 0
    has_invariance = False

    for test_id in tests:
        outcome = test_results.get(test_id, "missing")
        if outcome == "passed":
            n_pass += 1
            layer = TEST_LAYERS.get(test_id, "smoke")
            if layer in ("invariance", "integration"):
                has_invariance = True
        elif outcome == "failed":
            n_fail += 1

    if n_fail > 0:
        return "🧨", "Broken", n_pass, n_fail
    if n_pass == 0:
        return "📋", "Planned", 0, 0
    if has_invariance:
        return "✅", "Verified", n_pass, n_fail
    return "☑️", "Smoke-only", n_pass, n_fail


def generate_matrix(test_results, phase="13.15.DF", output_path=None):
    """Generate capability matrix markdown."""
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Capability Matrix — dfdraw")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Phase:** {phase}")
    lines.append(f"**Generator:** `scripts/generate_capability_matrix.py`")
    lines.append(f"**Sources:** `tests/feature_taxonomy.py` + `tests/test_layer_classification.py`")
    lines.append("")

    # Compute statuses
    counts = {"Verified": 0, "Smoke-only": 0, "Broken": 0, "Planned": 0}
    feature_statuses = []
    for feature in FEATURES:
        icon, status, n_pass, n_fail = compute_feature_status(feature, test_results)
        counts[status] += 1
        feature_statuses.append((feature, icon, status, n_pass, n_fail))

    # Summary
    total = sum(counts.values())
    total_tests = sum(len(f["tests"]) for f in FEATURES)
    inv_tests = sum(1 for t in TEST_LAYERS.values() if t == "invariance")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | % |")
    lines.append("|--------|------:|--:|")
    for label, emoji in [("Verified", "✅"), ("Smoke-only", "☑️"), ("Broken", "🧨"), ("Planned", "📋")]:
        count = counts.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"| {emoji} {label} | {count} | {pct:.0f}% |")
    lines.append(f"| **Total features** | **{total}** | |")
    lines.append(f"| **Total proof tests** | **{total_tests}** | |")
    lines.append(f"| **Invariance tests** | **{inv_tests}** | |")
    lines.append("")
    lines.append("**Status key:**")
    lines.append("- ✅ Verified — has at least one invariance test (A ≡ B check)")
    lines.append("- ☑️ Smoke-only — tests pass but only check 'no crash'")
    lines.append("- 🧨 Broken — at least one test failing")
    lines.append("- 📋 Planned — no tests mapped yet")
    lines.append("")

    # Feature table grouped by category
    lines.append("## Features")
    lines.append("")
    lines.append("| Status | Feature | Pass | Fail |")
    lines.append("|--------|---------|-----:|-----:|")

    current_category = None
    for feature, icon, status, n_pass, n_fail in feature_statuses:
        cat = feature.get("category", "")
        if cat != current_category:
            lines.append(f"| | **{cat}** | | |")
            current_category = cat
        lines.append(f"| {icon} | **{feature['id']}** — {feature['name']} | {n_pass} | {n_fail} |")

    lines.append("")

    # Broken details
    broken = [(f, s) for f, i, s, p, n in feature_statuses if s == "Broken"]
    if broken:
        lines.append("## 🧨 Broken Features — Details")
        lines.append("")
        for feature, _ in broken:
            lines.append(f"### {feature['id']} — {feature['name']}")
            lines.append("")
            for test_id in feature["tests"]:
                outcome = test_results.get(test_id, "missing")
                if outcome == "failed":
                    lines.append(f"- ❌ `{test_id}`")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Auto-generated. ✅ = invariance test (A ≡ B). ☑️ = smoke only.*")

    text = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"Capability matrix written to: {output_path}")

    return text


def main():
    parser = argparse.ArgumentParser(description="Generate capability matrix for dfdraw")
    parser.add_argument("--test-results", required=True, help="Path to .pytest_report.json")
    parser.add_argument("--output", default=os.path.join(PROJECT_DIR, "docs", "CAPABILITY_MATRIX.md"))
    parser.add_argument("--phase", default="13.15.DF")
    args = parser.parse_args()

    if not os.path.exists(args.test_results):
        print(f"ERROR: {args.test_results} not found. Run tests first.")
        sys.exit(1)

    test_results = load_test_results(args.test_results)
    generate_matrix(test_results, phase=args.phase, output_path=args.output)

    # Print summary
    counts = Counter()
    for feature in FEATURES:
        _, status, _, _ = compute_feature_status(feature, test_results)
        counts[status] += 1
    print(f"\n  Features: {len(FEATURES)}  |  Invariance tests: {sum(1 for v in TEST_LAYERS.values() if v == 'invariance')}")
    for label in ["Verified", "Smoke-only", "Broken", "Planned"]:
        print(f"  {label}: {counts.get(label, 0)}")


if __name__ == "__main__":
    main()

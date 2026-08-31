#!/usr/bin/env python3
"""
Capability Matrix Generator v4 — AliasDataFrame (taxonomy-based)

Single normalized semantic model for Capability Matrix renderers.

Inputs:
  1. pytest JSON report (.pytest_report.json)
  2. tests/feature_taxonomy.py (feature -> test pattern mapping)
  3. pytest keywords/markers

PHASE_13_76 tooling correction:
  - mapped xfailed/error/failed outcomes make a feature Broken;
  - only PASSED invariance tests count as invariance proof;
  - skipped/xpassed nodes are visible but do not promote a feature;
  - Markdown and the historical HTML renderer consume the same normalized matrix model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "tests"))

try:
    from feature_taxonomy import FEATURES
except ImportError:
    print("ERROR: Cannot import feature_taxonomy. Place feature_taxonomy.py in tests/")
    sys.exit(1)


STATUS_ICON = {
    "Verified": "✅",
    "Smoke-only": "☑️",
    "Broken": "🧨",
    "Planned": "📋",
}

BROKEN_OUTCOMES = {"failed", "error", "xfailed"}


def load_pytest_report(path):
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    results, markers = {}, {}
    for test in report.get("tests", []):
        node_id = test.get("nodeid", "")
        if "/" in node_id:
            if "::" in node_id:
                file_part, rest = node_id.split("::", 1)
                node_id = f"{os.path.basename(file_part)}::{rest}"
            else:
                node_id = os.path.basename(node_id)

        results[node_id] = test.get("outcome", "unknown")
        markers[node_id] = {
            kw
            for kw in test.get("keywords", [])
            if kw in ("invariance", "smoke", "slow", "integration")
        }

    return results, markers


def match_tests(patterns, all_ids):
    matched = set()
    for pat in patterns:
        for nid in all_ids:
            if pat.endswith(".py"):
                if nid.startswith(pat + "::") or nid == pat:
                    matched.add(nid)
            elif "::" in pat:
                if nid.startswith(pat + "::") or nid == pat:
                    matched.add(nid)
    return matched


def build_feature_result(feature, results, markers):
    """
    Build the single normalized semantic result for one capability feature.

    Status precedence:
      Broken     any mapped failed/error/xfailed node
      Verified   at least one passed node and one PASSED invariance proof
      Smoke-only at least one passed node, no broken evidence, no inv proof
      Planned    no positive passing proof

    skipped/xpassed/missing/unknown remain visible counts but do not promote.
    """
    matched = match_tests(feature.get("test_patterns", []), results.keys())

    counts = Counter()
    for node_id in matched:
        counts[results.get(node_id, "missing")] += 1

    n_pass = counts.get("passed", 0)
    n_fail = counts.get("failed", 0)
    n_error = counts.get("error", 0)
    n_xfail = counts.get("xfailed", 0)
    n_xpass = counts.get("xpassed", 0)
    n_skip = counts.get("skipped", 0)

    # Invariance is proof only when the mapped invariance test actually PASSED.
    n_inv = sum(
        1
        for node_id in matched
        if results.get(node_id) == "passed"
        and "invariance" in markers.get(node_id, set())
    )

    if any((n_fail, n_error, n_xfail)):
        status = "Broken"
    elif n_pass > 0 and n_inv > 0:
        status = "Verified"
    elif n_pass > 0:
        status = "Smoke-only"
    else:
        status = "Planned"

    return {
        "feature": feature,
        "icon": STATUS_ICON[status],
        "status": status,
        "n_tests": len(matched),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_error,
        "n_xfail": n_xfail,
        "n_xpass": n_xpass,
        "n_skip": n_skip,
        "n_inv": n_inv,
        "matched": tuple(sorted(matched)),
        "outcomes": {node_id: results.get(node_id, "missing")
                     for node_id in sorted(matched)},
    }


def feature_status(feature, results, markers):
    """
    Backward-compatible compact status API used by older callers.

    The legacy 'fail' slot now represents all load-bearing broken evidence:
    failed + error + xfailed.
    """
    d = build_feature_result(feature, results, markers)
    n_broken = d["n_fail"] + d["n_error"] + d["n_xfail"]
    return (
        d["icon"],
        d["status"],
        d["n_pass"],
        n_broken,
        d["n_inv"],
        set(d["matched"]),
    )


def build_matrix_model(results, markers, phase="13.11.B", features=None):
    features = FEATURES if features is None else features
    data = [
        build_feature_result(feature, results, markers)
        for feature in features
    ]

    counts = Counter(d["status"] for d in data)
    matched_union = set().union(*(set(d["matched"]) for d in data)) if data else set()
    unmatched = sorted(set(results.keys()) - matched_union)

    return {
        "generated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "phase": phase,
        "taxonomy_size": len(features),
        "features": data,
        "summary": {
            "status_counts": {
                label: counts.get(label, 0)
                for label in ("Verified", "Smoke-only", "Broken", "Planned")
            },
            "total_features": len(data),
            # Unique pytest node identities claimed by at least one feature.
            "matched_tests": len(matched_union),
            # Feature-test ownership associations; one pytest node may own
            # multiple semantic features, so this is intentionally >= the
            # unique matched-node count.
            "feature_test_associations": sum(d["n_tests"] for d in data),
            # Now explicitly passing invariance proof count.
            "invariance_tests": sum(d["n_inv"] for d in data),
            "xfail_tests": sum(d["n_xfail"] for d in data),
            "xpass_tests": sum(d["n_xpass"] for d in data),
            "skipped_tests": sum(d["n_skip"] for d in data),
            "unmatched_tests": len(unmatched),
        },
        "unmatched": unmatched,
        # Renderer-neutral evidence.  The historical HTML renderer consumes
        # these exact normalized outcomes/markers instead of re-parsing the
        # terminal pytest log and recomputing feature status independently.
        "results": dict(results),
        "markers": {node_id: set(markers.get(node_id, set()))
                    for node_id in results},
    }


def render_markdown(model):
    lines = [
        "# Capability Matrix — AliasDataFrame",
        "",
        f"**Generated:** {model['generated']}",
        f"**Phase:** {model['phase']}",
        f"**Taxonomy:** {model['taxonomy_size']} features",
        "**Generator:** `scripts/generate_capability_matrix.py` v4 "
        "(shared semantic model)",
        "",
    ]

    summary = model["summary"]
    total = summary["total_features"]
    lines += [
        "## Summary",
        "",
        "| Status | Count | % |",
        "|--------|------:|--:|",
    ]
    for label in ("Verified", "Smoke-only", "Broken", "Planned"):
        count = summary["status_counts"][label]
        pct = 100 * count // total if total else 0
        lines.append(f"| {STATUS_ICON[label]} {label} | {count} | {pct}% |")

    lines += [
        f"| **Total features** | **{total}** | |",
        f"| **Unique matched tests** | **{summary['matched_tests']}** | |",
        f"| **Feature-test associations** | **{summary['feature_test_associations']}** | |",
        f"| **Invariance tests** | **{summary['invariance_tests']}** | |",
        f"| **Mapped XFAIL evidence** | **{summary['xfail_tests']}** | |",
        f"| **Mapped XPASS evidence** | **{summary['xpass_tests']}** | |",
        f"| **Mapped skipped tests** | **{summary['skipped_tests']}** | |",
        "",
    ]

    if model["unmatched"]:
        lines += [
            f"**Unmatched tests:** {summary['unmatched_tests']} "
            "(not mapped to any feature)",
            "",
        ]

    cats = defaultdict(list)
    category_order = []
    for d in model["features"]:
        category = d["feature"]["category"]
        if category not in cats:
            category_order.append(category)
        cats[category].append(d)

    for category in category_order:
        lines += [
            f"## {category}",
            "",
            "| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |",
            "|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|",
        ]
        for d in cats[category]:
            lines.append(
                f"| {d['icon']} | **{d['feature']['id']}** — {d['feature']['name']} "
                f"| {d['n_tests']} | {d['n_pass']} | {d['n_fail']} "
                f"| {d['n_error']} | {d['n_xfail']} | {d['n_xpass']} "
                f"| {d['n_skip']} | {d['n_inv'] or ''} |"
            )
        lines.append("")

    broken = [d for d in model["features"] if d["status"] == "Broken"]
    if broken:
        lines += ["## 🧨 Broken Features — Details", ""]
        outcome_icon = {"failed": "❌", "error": "💥", "xfailed": "🧨"}
        for d in broken:
            lines.append(f"### {d['feature']['id']}")
            for node_id, outcome in d["outcomes"].items():
                if outcome in BROKEN_OUTCOMES:
                    lines.append(
                        f"- {outcome_icon[outcome]} `{node_id}` — `{outcome}`"
                    )
            lines.append("")

    if model["unmatched"]:
        lines += [
            "## Unmatched Tests",
            "",
            f"{len(model['unmatched'])} tests not mapped to any feature.",
            "",
        ]
        for node_id in model["unmatched"][:30]:
            lines.append(f"- `{node_id}`")
        if len(model["unmatched"]) > 30:
            lines.append(f"- ... +{len(model['unmatched']) - 30} more")
        lines.append("")

    lines += [
        "---",
        "*Generated from pytest JSON + feature_taxonomy.py using the shared "
        "Capability Matrix semantic model.*",
    ]
    return "\n".join(lines)



def generate_matrix(results, markers, phase="13.11.B"):
    """Backward-compatible Markdown API."""
    return render_markdown(build_matrix_model(results, markers, phase=phase))


def _write_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-results", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--phase", default="13.11.B")
    args = parser.parse_args()

    report_path = args.test_results
    if not report_path or not os.path.exists(report_path):
        default = os.path.join(PROJECT_DIR, ".pytest_report.json")
        if os.path.exists(default):
            report_path = default
        else:
            print(
                "No test results. Run: pytest --json-report "
                "--json-report-file=.pytest_report.json"
            )
            sys.exit(1)

    results, markers = load_pytest_report(report_path)
    print(f"Loaded {len(results)} tests from {report_path}")

    model = build_matrix_model(results, markers, phase=args.phase)

    output = args.output or os.path.join(
        PROJECT_DIR, "docs", "CAPABILITY_MATRIX.md"
    )
    _write_text(output, render_markdown(model))
    print(f"Matrix Markdown: {output}")

    counts = model["summary"]["status_counts"]
    print(
        f"\n  Features: {model['summary']['total_features']} "
        f"| Unique matched: {model['summary']['matched_tests']} "
        f"| Associations: {model['summary']['feature_test_associations']} "
        f"| Unmatched: {model['summary']['unmatched_tests']}"
    )
    for label in ("Verified", "Smoke-only", "Broken", "Planned"):
        print(f"  {label}: {counts.get(label, 0)}")
    print(
        f"  Passing invariance proofs: {model['summary']['invariance_tests']} "
        f"| Mapped XFAIL: {model['summary']['xfail_tests']}"
    )


if __name__ == "__main__":
    main()

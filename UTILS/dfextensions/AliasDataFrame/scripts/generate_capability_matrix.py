#!/usr/bin/env python3
"""
Capability Matrix Generator v2 — AliasDataFrame (taxonomy-based)

Generates CAPABILITY_MATRIX.md from:
  1. pytest JSON report (.pytest_report.json)
  2. tests/feature_taxonomy.py (feature → test pattern mapping)
  3. @pytest.mark.invariance markers (from pytest keywords)

Phase 13.11.B — 41 approved features.
"""

import sys, os, json, argparse
from datetime import datetime
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'tests'))

try:
    from feature_taxonomy import FEATURES
except ImportError:
    print("ERROR: Cannot import feature_taxonomy. Place feature_taxonomy.py in tests/")
    sys.exit(1)


def load_pytest_report(path):
    with open(path) as f:
        report = json.load(f)
    results, markers = {}, {}
    for test in report.get("tests", []):
        node_id = test.get("nodeid", "")
        if '/' in node_id:
            if '::' in node_id:
                file_part, rest = node_id.split('::', 1)
                node_id = f"{os.path.basename(file_part)}::{rest}"
            else:
                node_id = os.path.basename(node_id)
        results[node_id] = test.get("outcome", "unknown")
        markers[node_id] = {kw for kw in test.get("keywords", [])
                            if kw in ("invariance", "smoke", "slow", "integration")}
    return results, markers


def match_tests(patterns, all_ids):
    matched = set()
    for pat in patterns:
        for nid in all_ids:
            if pat.endswith('.py'):
                if nid.startswith(pat + '::') or nid == pat:
                    matched.add(nid)
            elif '::' in pat:
                if nid.startswith(pat + '::') or nid == pat:
                    matched.add(nid)
    return matched


def feature_status(feature, results, markers):
    matched = match_tests(feature.get("test_patterns", []), results.keys())
    if not matched and not feature.get("test_patterns"):
        return '📋', 'Planned', 0, 0, 0, matched
    n_pass = n_fail = n_inv = 0
    for t in matched:
        out = results.get(t, 'missing')
        if out == 'passed':
            n_pass += 1
        elif out == 'failed':
            n_fail += 1
        if 'invariance' in markers.get(t, set()):
            n_inv += 1
    if not matched:
        return '📋', 'Planned', 0, 0, 0, matched
    if n_fail > 0:
        return '🧨', 'Broken', n_pass, n_fail, n_inv, matched
    if n_pass == 0:
        return '📋', 'Planned', 0, 0, 0, matched
    if n_inv > 0:
        return '✅', 'Verified', n_pass, n_fail, n_inv, matched
    return '☑️', 'Smoke-only', n_pass, n_fail, 0, matched


def generate_matrix(results, markers, phase="13.11.B"):
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        "# Capability Matrix — AliasDataFrame", "",
        f"**Generated:** {now}", f"**Phase:** {phase}",
        f"**Taxonomy:** {len(FEATURES)} features (PHASE_13_11_B approved)",
        f"**Generator:** `scripts/generate_capability_matrix.py` v2 (taxonomy-based)", "",
    ]
    data = []
    for f in FEATURES:
        icon, status, np_, nf, ni, matched = feature_status(f, results, markers)
        data.append(dict(feature=f, icon=icon, status=status,
                         n_pass=np_, n_fail=nf, n_inv=ni, n_tests=len(matched)))
    counts = Counter(d['status'] for d in data)
    total = len(data)
    lines += ["## Summary", "", "| Status | Count | % |", "|--------|------:|--:|"]
    for label, emoji in [("Verified","✅"),("Smoke-only","☑️"),("Broken","🧨"),("Planned","📋")]:
        c = counts.get(label, 0)
        lines.append(f"| {emoji} {label} | {c} | {100*c//total if total else 0}% |")
    lines += [
        f"| **Total features** | **{total}** | |",
        f"| **Matched tests** | **{sum(d['n_tests'] for d in data)}** | |",
        f"| **Invariance tests** | **{sum(d['n_inv'] for d in data)}** | |", "",
    ]
    unmatched = sorted(set(results.keys()) - set().union(
        *(match_tests(f.get("test_patterns", []), results.keys()) for f in FEATURES)))
    if unmatched:
        lines += [f"**Unmatched tests:** {len(unmatched)} (not mapped to any feature)", ""]

    cats = defaultdict(list)
    for d in data:
        cats[d['feature']['category']].append(d)
    for cat in dict.fromkeys(f['category'] for f in FEATURES):
        if cat not in cats:
            continue
        lines += [f"## {cat}", "", "| Status | Feature | Tests | Pass | Fail | Inv |",
                  "|--------|---------|------:|-----:|-----:|:---:|"]
        for d in cats[cat]:
            iv = str(d['n_inv']) if d['n_inv'] > 0 else ""
            lines.append(f"| {d['icon']} | **{d['feature']['id']}** — {d['feature']['name']} "
                         f"| {d['n_tests']} | {d['n_pass']} | {d['n_fail']} | {iv} |")
        lines.append("")

    broken = [d for d in data if d['status'] == 'Broken']
    if broken:
        lines += ["## 🧨 Broken Features — Details", ""]
        for d in broken:
            lines.append(f"### {d['feature']['id']}")
            for t in match_tests(d['feature'].get("test_patterns", []), results.keys()):
                if results.get(t) == 'failed':
                    lines.append(f"- ❌ `{t}`")
            lines.append("")

    if unmatched:
        lines += ["## Unmatched Tests", "",
                   f"{len(unmatched)} tests not mapped to any feature.", ""]
        for t in unmatched[:30]:
            lines.append(f"- `{t}`")
        if len(unmatched) > 30:
            lines.append(f"- ... +{len(unmatched)-30} more")
        lines.append("")

    lines += ["---", "*Generated from pytest JSON + feature_taxonomy.py (v2 taxonomy-based).*"]
    return "\n".join(lines)


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
            print("No test results. Run: pytest --json-report --json-report-file=.pytest_report.json")
            sys.exit(1)

    results, markers = load_pytest_report(report_path)
    print(f"Loaded {len(results)} tests from {report_path}")
    matrix = generate_matrix(results, markers, phase=args.phase)

    output = args.output or os.path.join(PROJECT_DIR, "docs", "CAPABILITY_MATRIX.md")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(matrix)
    print(f"Matrix: {output}")

    counts = Counter()
    for feat in FEATURES:
        _, s, *_ = feature_status(feat, results, markers)
        counts[s] += 1
    unmatched = sorted(set(results.keys()) - set().union(
        *(match_tests(f.get("test_patterns", []), results.keys()) for f in FEATURES)))
    print(f"\n  Features: {len(FEATURES)} | Matched: {len(results)-len(unmatched)} | Unmatched: {len(unmatched)}")
    for l in ["Verified", "Smoke-only", "Broken", "Planned"]:
        print(f"  {l}: {counts.get(l, 0)}")


if __name__ == "__main__":
    main()

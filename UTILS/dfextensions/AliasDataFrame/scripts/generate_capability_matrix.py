#!/usr/bin/env python3
"""
Capability Matrix Generator — AliasDataFrame

Auto-generates CAPABILITY_MATRIX.md from pytest JSON reports.
Uses @pytest.mark.invariance markers (set on test classes) to determine
two-tier verification status: Verified vs Smoke-only.

Usage:
  python scripts/generate_capability_matrix.py [--test-results path.json]
  python scripts/generate_capability_matrix.py --json
"""

import sys
import os
import json
import re
import argparse
from datetime import datetime
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

FILE_MODULE_MAP = {
    'test_alias_dataframe': 'Core',
    'test_constructor': 'Core',
    'test_batch_materialization': 'Core',
    'test_proxy_pattern': 'Core',
    'test_cycle_detection': 'Core',
    'test_self_referential': 'Core',
    'test_branch_detection': 'Core',
    'test_dependency_tree': 'Core',
    'test_fill_handling': 'Core',
    'test_clean_temporary': 'Core',
    'test_lazy': 'Lazy Loading',
    'test_chain': 'Lazy Loading',
    'test_invariance_load': 'Lazy Loading',
    'test_invariance_smoke': 'Invariance',
    'test_invariance_subframe': 'Subframes',
    'test_invariance_backend': 'Backend',
    'test_invariance_compression': 'Compression',
    'test_draw': 'Drawing',
    'test_register_evaluator': 'Registered Functions',
    'test_polynomial': 'Registered Functions',
    'test_register_fit': 'Fit Registration',
    'test_schema': 'Schema',
    'test_data_schema': 'Schema',
    'test_validation': 'Schema',
    'test_alias_data_frame_schema': 'Schema',
    'test_alias_subframe': 'Subframes',
    'test_subframe_alias': 'Subframes',
    'test_join': 'Subframes',
    'test_materialize_subframe': 'Subframes',
    'test_composite_keys': 'Subframes',
    'test_rdf': 'RDataFrame',
    'test_AliasDataFrameRDF': 'RDataFrame',
    'test_ttree': 'RDataFrame',
    'test_compression': 'Compression',
    'test_numba': 'Backend',
    'test_arrow': 'Backend',
    'test_profiling': 'Core',
}

CLASS_FEATURE_MAP = {
    'TestRegisterEvaluatorBasic': 'register_evaluator — basic',
    'TestRegisterEvaluatorCollision': 'register_evaluator — collision/overwrite',
    'TestRegisterEvaluatorValidation': 'register_evaluator — validation',
    'TestRegisterEvaluatorMultiPredictor': 'register_evaluator — multi-predictor',
    'TestRegisterEvaluatorComposition': 'register_evaluator — composition',
    'TestRegisterEvaluatorInvariance': 'register_evaluator — invariance',
    'TestRegisterEvaluatorEdgeCases': 'register_evaluator — edge cases',
    'TestBasisExpressions': 'PolynomialSpec — basis expressions',
    'TestSchemaRoundtrip': 'PolynomialSpec — schema roundtrip',
    'TestRootExpression': 'PolynomialSpec — ROOT expression',
    'TestRegisterFunction': 'register_function',
    'TestRegisterPolynomial': 'register_polynomial_from_subframe',
    'TestInvariancePolynomial': 'PolynomialSpec — invariance',
    'TestDrawSubframeResolution': 'draw() subframe resolution',
    'TestDrawBatchSubframeResolution': 'draw_batch() subframe resolution',
    'TestDrawFiguresSubframeResolution': 'draw_figures() subframe resolution',
    'TestDrawSubframeEdgeCases': 'draw() subframe edge cases',
    'TestCoreInvariants': 'Core data invariants',
    'TestSubframeJoinCorrectness': 'Subframe join correctness',
    'TestChainSubframeIntegration': 'Chain + subframe integration',
    'TestDtypePreservation': 'Dtype preservation',
    'TestErrorScenarios': 'Error scenarios',
    'TestDrawInvariance': 'Draw vs materialize invariance',
    'TestInvarianceSmoke': 'Invariance smoke (I0)',
    'TestInvarianceLoadMode': 'Load mode invariance (I1)',
    'TestInvarianceBackend': 'Backend invariance (I2)',
    'TestInvarianceSubframe': 'Subframe join invariance (I3)',
    'TestInvarianceCompression': 'Compression invariance (I4)',
}


def classify_test(node_id, markers=None):
    parts = node_id.split('::')
    # Extract just the filename — handle full paths like
    # dfextensions/AliasDataFrame/tests/test_foo.py
    file_path = parts[0]
    file_basename = os.path.basename(file_path).replace('.py', '')
    file_part = file_basename
    class_name = parts[1] if len(parts) > 1 else None
    method_name = parts[-1] if len(parts) > 1 else parts[0]

    module = 'Other'
    for prefix, mod in FILE_MODULE_MAP.items():
        if file_part.startswith(prefix):
            module = mod
            break

    feature = None
    if class_name and class_name in CLASS_FEATURE_MAP:
        feature = CLASS_FEATURE_MAP[class_name]
    elif class_name:
        feature = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', class_name.replace('Test', '')).strip()
    else:
        feature = file_part.replace('test_', '').replace('_', ' ').title()

    markers = markers or set()
    if 'invariance' in markers:
        layer = 'invariance'
    elif 'integration' in markers:
        layer = 'integration'
    else:
        layer = 'smoke'

    return {
        'module': module, 'feature': feature, 'layer': layer,
        'file': file_part, 'class': class_name, 'method': method_name,
    }


def generate_matrix(test_results, test_markers=None):
    test_markers = test_markers or {}
    features = defaultdict(lambda: {
        'module': 'Other', 'name': '', 'tests': [],
        'passed': 0, 'failed': 0, 'skipped': 0, 'has_invariance': False,
    })

    for node_id, passed in test_results.items():
        markers = test_markers.get(node_id, set())
        info = classify_test(node_id, markers)
        feature_key = f"{info['module']}::{info['feature']}"
        feat = features[feature_key]
        feat['module'] = info['module']
        feat['name'] = info['feature']
        feat['tests'].append({'node_id': node_id, 'passed': passed, 'layer': info['layer']})

        if passed is True: feat['passed'] += 1
        elif passed is False: feat['failed'] += 1
        else: feat['skipped'] += 1
        if info['layer'] in ('invariance', 'integration'):
            feat['has_invariance'] = True

    for key, feat in features.items():
        if feat['failed'] > 0:
            feat['status'], feat['status_label'] = '🧨', 'Broken'
        elif feat['passed'] == 0:
            feat['status'], feat['status_label'] = '📋', 'Planned'
        elif feat['has_invariance']:
            feat['status'], feat['status_label'] = '✅', 'Verified'
        else:
            feat['status'], feat['status_label'] = '☑️', 'Smoke-only'

    return dict(features)


def format_markdown(matrix, phase="13.11.ADF"):
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# Capability Matrix — AliasDataFrame")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Phase:** {phase}")
    lines.append(f"**Generator:** `scripts/generate_capability_matrix.py`")
    lines.append(f"**Verification:** `@pytest.mark.invariance` markers on test classes")
    lines.append("")

    statuses = Counter(f['status_label'] for f in matrix.values())
    total = len(matrix)
    total_tests = sum(f['passed'] + f['failed'] + f['skipped'] for f in matrix.values())
    inv_tests = sum(sum(1 for t in f['tests'] if t['layer'] == 'invariance') for f in matrix.values())

    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | % |")
    lines.append("|--------|------:|--:|")
    for label, emoji in [("Verified", "✅"), ("Smoke-only", "☑️"), ("Broken", "🧨"), ("Planned", "📋")]:
        count = statuses.get(label, 0)
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"| {emoji} {label} | {count} | {pct:.0f}% |")
    lines.append(f"| **Total features** | **{total}** | |")
    lines.append(f"| **Total tests** | **{total_tests}** | |")
    lines.append(f"| **Invariance tests** | **{inv_tests}** | |")
    lines.append("")
    lines.append("**Status key:**")
    lines.append("- ✅ Verified — has `@pytest.mark.invariance` tests")
    lines.append("- ☑️ Smoke-only — functional tests pass, no invariance verification")
    lines.append("- 🧨 Broken — at least one test failing")
    lines.append("- 📋 Planned — no tests yet")
    lines.append("")

    modules = defaultdict(list)
    for key, feat in sorted(matrix.items()):
        modules[feat['module']].append((key, feat))

    for mod in sorted(modules.keys()):
        features = modules[mod]
        lines.append(f"## {mod}")
        lines.append("")
        lines.append("| Status | Feature | Passed | Failed | Invariance |")
        lines.append("|--------|---------|-------:|-------:|:----------:|")
        for key, feat in sorted(features, key=lambda x: x[1]['name']):
            n_inv = sum(1 for t in feat['tests'] if t['layer'] == 'invariance')
            inv = f"{n_inv}" if n_inv > 0 else ""
            lines.append(f"| {feat['status']} | {feat['name']} | {feat['passed']} | {feat['failed']} | {inv} |")
        lines.append("")

    broken = [(k, f) for k, f in matrix.items() if f['status_label'] == 'Broken']
    if broken:
        lines.append("## 🧨 Broken Features — Details")
        lines.append("")
        for key, feat in broken:
            lines.append(f"### {feat['name']}")
            lines.append("")
            for t in feat['tests']:
                if t['passed'] is False:
                    lines.append(f"- ❌ `{t['node_id']}`")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Auto-generated from pytest results. ✅ = @pytest.mark.invariance test exists. ☑️ = smoke tests only.*")
    return "\n".join(lines)


def format_json(matrix, phase="13.11.ADF"):
    output = {"generated": datetime.utcnow().isoformat(), "phase": phase, "features": {}, "summary": {}}
    statuses = Counter()
    for key, feat in matrix.items():
        statuses[feat['status_label']] += 1
        n_inv = sum(1 for t in feat['tests'] if t['layer'] == 'invariance')
        output["features"][key] = {
            "name": feat['name'], "module": feat['module'], "status": feat['status'],
            "status_label": feat['status_label'], "passed": feat['passed'],
            "failed": feat['failed'], "has_invariance": feat['has_invariance'], "invariance_count": n_inv,
        }
    output["summary"] = dict(statuses)
    output["summary"]["total"] = len(matrix)
    return json.dumps(output, indent=2)


def load_pytest_report(path):
    with open(path) as f:
        report = json.load(f)
    results = {}
    markers = {}
    for test in report.get("tests", []):
        node_id = test.get("nodeid", "")
        outcome = test.get("outcome", "")
        results[node_id] = True if outcome == "passed" else (False if outcome == "failed" else None)
        test_markers = set()
        for kw in test.get("keywords", []):
            if kw in ("invariance", "smoke", "slow", "integration"):
                test_markers.add(kw)
        markers[node_id] = test_markers
    return results, markers


def load_from_multiple_reports(paths):
    merged_results, merged_markers = {}, {}
    for path in paths:
        if os.path.exists(path):
            r, m = load_pytest_report(path)
            merged_results.update(r)
            merged_markers.update(m)
    return merged_results, merged_markers


def main():
    parser = argparse.ArgumentParser(description="Generate capability matrix for AliasDataFrame")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--test-results", default=None)
    parser.add_argument("--from-reports", nargs="*", default=None)
    parser.add_argument("--phase", default="13.11.ADF")
    args = parser.parse_args()

    test_results, test_markers = {}, {}
    if args.from_reports:
        test_results, test_markers = load_from_multiple_reports(args.from_reports)
    elif args.test_results and os.path.exists(args.test_results):
        test_results, test_markers = load_pytest_report(args.test_results)
    else:
        default_report = os.path.join(PROJECT_DIR, ".pytest_report.json")
        if os.path.exists(default_report):
            test_results, test_markers = load_pytest_report(default_report)
        else:
            print("No test results found. Run tests first.")
            sys.exit(1)

    matrix = generate_matrix(test_results, test_markers)

    if args.json:
        content = format_json(matrix, phase=args.phase)
        default_path = os.path.join(PROJECT_DIR, "docs", "capability_matrix.json")
    else:
        content = format_markdown(matrix, phase=args.phase)
        default_path = os.path.join(PROJECT_DIR, "docs", "CAPABILITY_MATRIX.md")

    output_path = args.output or default_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Capability matrix written to: {output_path}")
    statuses = Counter(f['status_label'] for f in matrix.values())
    total = len(matrix)
    inv = sum(sum(1 for t in f['tests'] if t['layer'] == 'invariance') for f in matrix.values())
    print(f"\n  Features: {total}  |  Tests: {sum(f['passed']+f['failed']+f['skipped'] for f in matrix.values())}  |  Invariance: {inv}")
    for label in ["Verified", "Smoke-only", "Broken", "Planned"]:
        print(f"  {label}: {statuses.get(label, 0)}")


if __name__ == "__main__":
    main()

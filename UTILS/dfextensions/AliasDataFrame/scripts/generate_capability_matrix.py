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
import ast
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "tests"))

try:
    from feature_taxonomy import FEATURES, CATEGORY_DESCRIPTIONS
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


def infer_evidence_layer(node_id, marker_set):
    """Renderer-neutral evidence layer, with markers authoritative when present."""
    marker_set = set(marker_set or ())
    for layer in ("invariance", "integration", "smoke"):
        if layer in marker_set:
            return layer
    low = node_id.lower()
    if "invariance" in low:
        return "invariance"
    if "smoke" in low:
        return "smoke"
    return "smoke"


@lru_cache(maxsize=None)
def _source_ast_body(rel_path):
    """Parse one repository-relative test source once per generator process.

    Capability Matrix generation resolves thousands of pytest node IDs back to
    source locations, but those nodes belong to a much smaller set of source
    files.  Keep the established AST-based locator semantics while avoiding a
    full read+parse of the same test module for every node.
    """
    path = Path(PROJECT_DIR) / rel_path
    if not path.is_file():
        return ()
    try:
        return tuple(ast.parse(path.read_text(encoding="utf-8")).body)
    except (OSError, UnicodeError, SyntaxError):
        return ()


def _node_source_location(node_id):
    """Best-effort repository-relative file/line locator for a pytest node."""
    parts = node_id.split("::")
    file_name = parts[0]
    rel = Path("tests") / file_name
    line = None
    if len(parts) >= 2:
        target = [part.split("[", 1)[0] for part in parts[1:]]
        body = _source_ast_body(rel.as_posix())
        for name in target:
            found = None
            for node in body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    found = node
                    break
            if found is None:
                break
            line = found.lineno
            body = getattr(found, "body", ())
    return {"file": rel.as_posix(), "line": line}


def _sha256_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git_provenance():
    def run(*args):
        try:
            return subprocess.run(
                list(args), cwd=PROJECT_DIR, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""
    commit = run("git", "rev-parse", "HEAD") or None
    status = run("git", "status", "--porcelain", "--", ".")
    return {
        "commit": commit,
        "tree_state": "DIRTY" if status else "CLEAN",
        "generator_sha256": _sha256_file(__file__),
        "taxonomy_sha256": _sha256_file(Path(PROJECT_DIR) / "tests" / "feature_taxonomy.py"),
        "source_bundle_manifest_sha256": os.environ.get("ADF_SOURCE_BUNDLE_MANIFEST_SHA256"),
    }


def _load_feature_contract(feature):
    """Load a declared machine contract without re-interpreting its semantics."""
    rel = feature.get("contract_file")
    if not rel:
        return None, None
    path = Path(PROJECT_DIR) / rel
    if not path.is_file():
        return {"file": rel, "status": "MISSING"}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"file": rel, "status": "INVALID", "error": str(exc)}, None
    return payload, _sha256_file(path)


def feature_contract_summary(feature):
    """Load optional machine-contract summary declared by a taxonomy feature."""
    payload, digest = _load_feature_contract(feature)
    rel = feature.get("contract_file")
    if payload is None:
        return None
    if payload.get("status") in {"MISSING", "INVALID"} and "schema" not in payload:
        return payload
    states = Counter(row.get("current_state", "UNKNOWN") for row in payload.get("cells", []))
    contracts = Counter(row.get("interface_contract", "UNKNOWN") for row in payload.get("cells", []))
    return {
        "file": rel,
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "axes": payload.get("axes"),
        "declared_cells": len(payload.get("cells", [])),
        "cell_state_counts": dict(sorted(states.items())),
        "interface_contract_counts": dict(sorted(contracts.items())),
        "seams": len(payload.get("seams", [])),
        "measurement_summary": payload.get("measurement_summary"),
        "b2_invariance": payload.get("b2_invariance"),
        "sha256": digest,
    }


def feature_contract_surfaces(feature):
    """Project the full ratified contract into AI-facing cell/seam surfaces.

    The source contract keeps its own internal evidence_class values (for
    example core vs bounded-mode rows).  The matrix adds a wrapper-level
    evidence_class so an AI consumer cannot blur Cartesian cells and
    interaction seams.  No cell semantics are recomputed here.
    """
    payload, digest = _load_feature_contract(feature)
    if not payload or (payload.get("status") in {"MISSING", "INVALID"} and "schema" not in payload):
        return {}
    fid = feature["id"]
    out = {}
    cells = payload.get("cells", [])
    if cells:
        out[fid] = {
            "evidence_class": "cell",
            "source_contract": feature.get("contract_file"),
            "source_contract_sha256": digest,
            "rows": cells,
        }
    seams = payload.get("seams", [])
    if seams:
        out[f"{fid}.seams"] = {
            "evidence_class": "seam",
            "source_contract": feature.get("contract_file"),
            "source_contract_sha256": digest,
            "rows": seams,
        }
    return out


def _semantic_payload_digest(surfaces):
    """Formatting-independent SHA256 over normalized machine surfaces."""
    canonical = json.dumps(
        surfaces, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def export_json_model(model):
    """Stable JSON-safe projection for AI/reviewer navigation."""
    node_records = model.get("node_records", {})
    features = []
    surfaces = {}
    for d in model["features"]:
        f = d["feature"]
        surfaces.update(feature_contract_surfaces(f))
        features.append({
            "id": f["id"],
            "name": f["name"],
            "description": f.get("description", ""),
            "category": f["category"],
            "surface": f.get("surface"),
            "contract": feature_contract_summary(f),
            "status": d["status"],
            "counts": {
                "tests": d["n_tests"], "passed": d["n_pass"], "failed": d["n_fail"],
                "errors": d["n_error"], "xfailed": d["n_xfail"], "xpassed": d["n_xpass"],
                "skipped": d["n_skip"], "invariance": d["n_inv"],
            },
            "tests": [node_records[n] for n in d["matched"]],
        })
    return {
        "schema": "AliasDataFrame.CapabilityMatrix",
        "schema_version": 1,
        "generated": model["generated"],
        "phase": model["phase"],
        "taxonomy_size": model["taxonomy_size"],
        "provenance": model.get("provenance", {}),
        "semantic_payload_digest": _semantic_payload_digest(surfaces),
        "summary": model["summary"],
        "categories": [
            {"id": category, "description": model.get("category_descriptions", {}).get(category, "")}
            for category in model.get("category_order", [])
        ],
        "features": features,
        "surfaces": surfaces,
        "unmatched": [node_records[n] for n in model["unmatched"]],
    }


def render_json(model):
    return json.dumps(export_json_model(model), indent=2, sort_keys=False) + "\n"


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
        and infer_evidence_layer(node_id, markers.get(node_id, set())) == "invariance"
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
    node_records = {}
    for node_id in sorted(results):
        loc = _node_source_location(node_id)
        node_records[node_id] = {
            "node_id": node_id,
            "outcome": results.get(node_id, "missing"),
            "markers": sorted(markers.get(node_id, set())),
            "evidence_layer": infer_evidence_layer(node_id, markers.get(node_id, set())),
            "file": loc["file"],
            "line": loc["line"],
        }

    category_order = []
    for feature in features:
        category = feature.get("category", "")
        if category and category not in category_order:
            category_order.append(category)

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "phase": phase,
        "taxonomy_size": len(features),
        "category_order": category_order,
        "category_descriptions": {
            category: CATEGORY_DESCRIPTIONS.get(category, "")
            for category in category_order
        },
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
        "node_records": node_records,
        "provenance": _git_provenance(),
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
        ]
        category_description = model.get("category_descriptions", {}).get(category, "")
        if category_description:
            lines += [category_description, ""]
        lines += [
            "| Status | Feature | Tests | Pass | Fail | Err | XFail | XPass | Skip | Inv |",
            "|--------|---------|------:|-----:|-----:|----:|------:|------:|-----:|----:|",
        ]
        for d in cats[category]:
            desc = d["feature"].get("description", "")
            label = f"**{d['feature']['id']}** — {d['feature']['name']}"
            if desc:
                label += f"<br><sub>{desc}</sub>"
            contract = feature_contract_summary(d["feature"])
            if contract and contract.get("declared_cells") is not None:
                states = contract.get("cell_state_counts", {})
                state_text = ", ".join(f"{k}={v}" for k, v in states.items())
                label += (
                    f"<br><sub>Surface: {contract['declared_cells']} cells; "
                    f"{state_text}; seams={contract.get('seams', 0)}</sub>"
                )
            lines.append(
                f"| {d['icon']} | {label} "
                f"| {d['n_tests']} | {d['n_pass']} | {d['n_fail']} "
                f"| {d['n_error']} | {d['n_xfail']} | {d['n_xpass']} "
                f"| {d['n_skip']} | {d['n_inv'] or ''} |"
            )
        lines.append("")
        lines.append("### Supporting tests")
        lines.append("")
        for d in cats[category]:
            lines.append("<details>")
            lines.append(
                f"<summary><code>{d['feature']['id']}</code> — {d['n_tests']} owned pytest nodes</summary>"
            )
            lines.append("")
            if not d["matched"]:
                lines.append("_(no tests claimed yet)_")
            else:
                for node_id in d["matched"]:
                    record = model["node_records"][node_id]
                    locator = record["file"]
                    if record["line"] is not None:
                        locator += f":L{record['line']}"
                    lines.append(
                        f"- `{node_id}` — `{record['outcome']}` — "
                        f"`{record['evidence_layer']}` — `{locator}`"
                    )
            lines.append("")
            lines.append("</details>")
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
    parser.add_argument("--json-output", default=None)
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

    json_output = args.json_output
    if json_output:
        _write_text(json_output, render_json(model))
        print(f"Matrix JSON: {json_output}")

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

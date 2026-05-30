#!/usr/bin/env python3
"""
Capability Matrix Generator — dfdraw

Auto-generates CAPABILITY_MATRIX.md AND CAPABILITY_MATRIX.html from:
  1. pytest JSON report (.pytest_report.json)
  2. tests/feature_taxonomy.py (feature → test mapping; explicit node-IDs)
  3. tests/test_layer_classification.py (test → invariance/smoke/visual_primitive)

The .md is diff-friendly (counts only) for code review; the .html is the
navigable view where each feature row expands to its full test list — the
capability→test traceability link that v1.0 of this work was missing.

Visual coverage (`visual_primitive` layer) is tracked as an ORTHOGONAL
column + 👁 badge — distinct evidence from invariance, never folded into
Verified.  Per Phase 13.49.DF §3.5.

Usage:
  python scripts/generate_capability_matrix.py --test-results .pytest_report.json
  python scripts/generate_capability_matrix.py --test-results path.json \\
         --output docs/CAPABILITY_MATRIX.md --phase PHASE_13_49_DF_END

Follows GroupByRegression reference implementation; extends with link view.
"""

import argparse
import html
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from tests.feature_taxonomy import FEATURES
from tests.test_layer_classification import TEST_LAYERS


# ---------------------------------------------------------------------------
# Test results loading
# ---------------------------------------------------------------------------

def load_test_results(json_path):
    """Load pytest JSON report → {nodeid: outcome}.

    Pytest may report node IDs with varying prefixes depending on where the
    suite is invoked from (e.g. `tests/test_foo.py::...` when run from the
    dfdraw root, `UTILS/dfextensions/dfdraw/tests/test_foo.py::...` when run
    from the O2DPG root). `feature_taxonomy.py` stores tests as basenames
    (`test_foo.py::Class::test`), so the normalization must always reduce to
    the basename regardless of prefix depth. Earlier versions of this code
    special-cased only the `tests/` prefix, which silently failed for any
    deeper invocation path and left every feature unmatched — caught in the
    Phase 13.49 panel review by Opus2 + Sonnet53_R2.
    """
    with open(json_path) as f:
        report = json.load(f)
    results = {}
    for test in report.get("tests", []):
        nodeid = test["nodeid"]
        if "/" in nodeid:
            nodeid = nodeid.split("/")[-1]
        results[nodeid] = test["outcome"]
    return results


def load_collected_tests(json_path):
    """Set of every node-ID pytest collected, normalized to taxonomy format."""
    with open(json_path) as f:
        report = json.load(f)
    ids = set()
    for test in report.get("tests", []):
        nodeid = test["nodeid"]
        if "/" in nodeid:
            nodeid = nodeid.split("/")[-1]
        ids.add(nodeid)
    return ids


# ---------------------------------------------------------------------------
# Per-feature status + counts (Tests / Pass / Fail / Inv / Visual)
# ---------------------------------------------------------------------------

def compute_feature_stats(feature, test_results):
    """Return dict with status, icon, n_tests, n_pass, n_fail, n_inv, n_visual."""
    tests = feature.get("tests", [])
    n_tests = len(tests)
    n_pass = n_fail = n_inv = n_visual = 0
    has_invariance = False

    for test_id in tests:
        outcome = test_results.get(test_id, "missing")
        layer = TEST_LAYERS.get(test_id, "smoke")
        if outcome == "passed":
            n_pass += 1
            if layer in ("invariance", "integration"):
                has_invariance = True
                n_inv += 1
            elif layer == "visual_primitive":
                n_visual += 1
        elif outcome == "failed":
            n_fail += 1
        # 'missing'/'skipped' do not count toward pass/fail but layer counts unaffected

    # Status (visual_primitive deliberately does NOT influence Verified — §3.5)
    if n_fail > 0:
        icon, status = "🧨", "Broken"
    elif n_pass == 0:
        icon, status = "📋", "Planned"
    elif has_invariance:
        icon, status = "✅", "Verified"
    else:
        icon, status = "☑️", "Smoke-only"

    return {
        "icon": icon, "status": status,
        "n_tests": n_tests, "n_pass": n_pass, "n_fail": n_fail,
        "n_inv": n_inv, "n_visual": n_visual,
        "has_visual": n_visual > 0,
    }


# back-compat: keep the old function signature/API
def compute_feature_status(feature, test_results):
    s = compute_feature_stats(feature, test_results)
    return s["icon"], s["status"], s["n_pass"], s["n_fail"]


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------

def generate_matrix(test_results, phase="unknown", output_path=None,
                    collected_tests=None, known_unclaimed=None):
    """Generate capability matrix markdown.

    `collected_tests` (optional): set of every pytest-collected node-ID, used
    to compute the Unmatched-Tests section. When None, the section is omitted.
    `known_unclaimed` (optional): list of KNOWN_UNCLAIMED dicts; surfaced as
    an "Allow-listed (unclaimed)" sub-section under Unmatched.
    """
    lines = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Capability Matrix — dfdraw")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Phase:** {phase}")
    lines.append("**Generator:** `scripts/generate_capability_matrix.py`")
    lines.append("**Sources:** `tests/feature_taxonomy.py` + `tests/test_layer_classification.py`")
    lines.append("")

    counts = {"Verified": 0, "Smoke-only": 0, "Broken": 0, "Planned": 0}
    feature_stats = []
    for feature in FEATURES:
        s = compute_feature_stats(feature, test_results)
        counts[s["status"]] += 1
        feature_stats.append((feature, s))

    total = sum(counts.values())
    total_tests = sum(s["n_tests"] for _, s in feature_stats)
    inv_tests = sum(1 for t in TEST_LAYERS.values() if t == "invariance")
    visual_tests = sum(1 for t in TEST_LAYERS.values() if t == "visual_primitive")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | % |")
    lines.append("|--------|------:|--:|")
    for label, emoji in [("Verified", "✅"), ("Smoke-only", "☑️"),
                          ("Broken", "🧨"), ("Planned", "📋")]:
        c = counts.get(label, 0)
        pct = 100 * c / total if total > 0 else 0
        lines.append(f"| {emoji} {label} | {c} | {pct:.0f}% |")
    lines.append(f"| **Total features** | **{total}** | |")
    lines.append(f"| **Total proof tests** | **{total_tests}** | |")
    lines.append(f"| **Invariance tests** | **{inv_tests}** | |")
    if visual_tests > 0:
        lines.append(f"| **Visual tests** | **{visual_tests}** | |")
    lines.append("")
    lines.append("**Status key:**")
    lines.append("- ✅ Verified — has at least one passing invariance test (A ≡ B check)")
    lines.append("- ☑️ Smoke-only — tests pass but only check 'no crash'")
    lines.append("- 🧨 Broken — at least one test failing")
    lines.append("- 📋 Planned — no tests mapped yet")
    lines.append("- 👁 Visual — orthogonal: feature has ≥1 visual_primitive test (see HTML view)")
    lines.append("")

    # Feature table (ADF parity + Visual column)
    lines.append("## Features")
    lines.append("")
    lines.append("| Status | Feature | Tests | Pass | Fail | Inv | Visual |")
    lines.append("|--------|---------|------:|-----:|-----:|----:|-------:|")
    current_category = None
    for feature, s in feature_stats:
        cat = feature.get("category", "")
        if cat != current_category:
            lines.append(f"| | **{cat}** | | | | | |")
            current_category = cat
        visual_cell = str(s["n_visual"]) if s["n_visual"] > 0 else ""
        eye = " 👁" if s["has_visual"] else ""
        lines.append(
            f"| {s['icon']} | **{feature['id']}** — {feature['name']}{eye} "
            f"| {s['n_tests']} | {s['n_pass']} | {s['n_fail']} | {s['n_inv']} | {visual_cell} |"
        )
    lines.append("")

    # Broken details (unchanged)
    broken = [(f, s) for f, s in feature_stats if s["status"] == "Broken"]
    if broken:
        lines.append("## 🧨 Broken Features — Details")
        lines.append("")
        for feature, _ in broken:
            lines.append(f"### {feature['id']} — {feature['name']}")
            lines.append("")
            for test_id in feature["tests"]:
                if test_results.get(test_id, "missing") == "failed":
                    lines.append(f"- ❌ `{test_id}`")
            lines.append("")

    # Unmatched-Tests section — surfaces the coverage gap (the 451-class)
    if collected_tests is not None:
        claimed = {t for f in FEATURES for t in f["tests"]}
        unmatched = sorted(collected_tests - claimed)
        # Group by test-file prefix so triage is scannable (per Claude48 P3)
        if not unmatched:
            lines.append("## Unmatched Tests")
            lines.append("")
            lines.append("✓ 0 unmatched tests.")
            lines.append("")
        else:
            lines.append(f"## Unmatched Tests ({len(unmatched)})")
            lines.append("")
            lines.append(f"{len(unmatched)} tests pytest collected that no feature claims.")
            lines.append("Grouped by test-file prefix.")
            lines.append("")
            by_file = defaultdict(list)
            for t in unmatched:
                by_file[t.split("::", 1)[0]].append(t)
            for fname in sorted(by_file):
                lines.append(f"<details><summary><code>{fname}</code> ({len(by_file[fname])})</summary>")
                lines.append("")
                for t in by_file[fname]:
                    lines.append(f"- `{t}`")
                lines.append("")
                lines.append("</details>")
                lines.append("")

        # Allow-listed sub-section (the bounded debt)
        if known_unclaimed:
            lines.append(f"### Allow-listed (unclaimed) — {len(known_unclaimed)}")
            lines.append("")
            lines.append("Tests classified in `TEST_LAYERS` but not yet feature-claimed.")
            lines.append("Each carries a `target_phase` for claim-back tracking.")
            lines.append("")
            by_phase = defaultdict(list)
            for entry in known_unclaimed:
                by_phase[entry.get("target_phase", "OPEN")].append(entry)
            for tp in sorted(by_phase):
                lines.append(f"<details><summary>target_phase: <code>{tp}</code> ({len(by_phase[tp])})</summary>")
                lines.append("")
                for e in by_phase[tp]:
                    lines.append(f"- `{e['test_id']}` — {e.get('reason','(no reason)')}")
                lines.append("")
                lines.append("</details>")
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*For per-feature test lists, see `CAPABILITY_MATRIX.html`.*")
    lines.append("")
    lines.append("*Auto-generated. ✅ = invariance (A ≡ B). ☑️ = smoke. 👁 = visual_primitive (orthogonal).*")

    text = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"Capability matrix (.md) written to: {output_path}")

    return text


# ---------------------------------------------------------------------------
# HTML emitter (the navigable link view)
# ---------------------------------------------------------------------------

_LAYER_MARKER = {
    "invariance": "✓",
    "integration": "✓",
    "smoke": "☑",
    "visual_primitive": "👁",
}


def generate_html_matrix(test_results, features=None, test_layers=None,
                          known_unclaimed=None, collected_tests=None,
                          phase="unknown"):
    """Return a single-file HTML matrix as a string.

    Each feature row is anchored at id="feature-{feature_id}" and expands to
    a div.tests-panel listing every claimed test_id with layer marker and
    outcome. Filters (status × visual × category) AND-combine; expansion
    state is preserved across filter changes (§9 D-D / D-E).
    """
    features = features if features is not None else FEATURES
    test_layers = test_layers if test_layers is not None else TEST_LAYERS
    known_unclaimed = known_unclaimed or []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Compute stats
    counts = {"Verified": 0, "Smoke-only": 0, "Broken": 0, "Planned": 0}
    rows = []
    categories = []
    seen_cat = set()
    for f in features:
        s = compute_feature_stats(f, test_results)
        counts[s["status"]] += 1
        rows.append((f, s))
        cat = f.get("category", "")
        if cat and cat not in seen_cat:
            seen_cat.add(cat)
            categories.append(cat)

    total = sum(counts.values())

    # Unmatched (only if we have collected_tests)
    if collected_tests is not None:
        claimed = {t for f in features for t in f["tests"]}
        unmatched = sorted(collected_tests - claimed)
    else:
        unmatched = None

    def esc(s):
        return html.escape(str(s), quote=True)

    # ---- HTML ----
    out = []
    out.append("<!DOCTYPE html>")
    out.append('<html lang="en">')
    out.append("<head>")
    out.append('<meta charset="utf-8">')
    out.append("<title>Capability Matrix — dfdraw</title>")
    out.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
    out.append('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    out.append('<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Newsreader:wght@400;500;700&display=swap" rel="stylesheet">')
    out.append("<style>")
    out.append("""
:root {
  --fg: #111; --bg: #fff; --muted: #666; --rule: #ddd; --accent: #1a4a8a;
  --hover: #f4f4f4; --pass: #2a7a2a; --fail: #b03030; --miss: #999;
}
* { box-sizing: border-box; }
body {
  font-family: 'Newsreader', Georgia, serif; color: var(--fg); background: var(--bg);
  max-width: 1180px; margin: 32px auto; padding: 0 24px; line-height: 1.5;
}
h1, h2, h3 { font-weight: 500; margin-top: 1.8em; }
h1 { font-size: 2em; margin-top: 0; }
code, .mono { font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.92em; }
.meta { color: var(--muted); font-size: 0.95em; margin-bottom: 1.2em; }
.summary table { border-collapse: collapse; margin: 8px 0; }
.summary th, .summary td { text-align: left; padding: 4px 16px 4px 0; border-bottom: 1px solid var(--rule); }
.summary td.n { text-align: right; font-variant-numeric: tabular-nums; }
.filters { display: flex; flex-wrap: wrap; gap: 20px; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--rule); margin: 16px 0; }
.filter-group { display: flex; gap: 4px; }
.filter-group .label { color: var(--muted); margin-right: 6px; align-self: center; }
button, select {
  font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.85em;
  padding: 4px 10px; border: 1px solid var(--rule); background: var(--bg); color: var(--fg); cursor: pointer;
}
button.active { background: var(--fg); color: var(--bg); }
button:hover:not(.active) { background: var(--hover); }
table.features { border-collapse: collapse; width: 100%; margin: 1em 0; font-variant-numeric: tabular-nums; }
table.features th, table.features td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
table.features th { font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.85em; color: var(--muted); font-weight: 600; }
table.features td.n { text-align: right; }
tr.feature-row { cursor: pointer; }
tr.feature-row:hover { background: var(--hover); }
tr.category-row td { font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.9em; color: var(--accent); font-weight: 600; padding-top: 14px; border-bottom: 1px solid var(--fg); }
.expand-toggle { display: inline-block; width: 14px; color: var(--muted); }
.eye-badge { color: var(--accent); margin-left: 6px; }
tr.tests-row td { padding: 0; }
tr.tests-row[hidden] { display: none; }
.tests-panel { background: var(--hover); padding: 10px 32px; border-left: 3px solid var(--accent); margin: 4px 0 12px 0; }
.tests-panel ul { list-style: none; padding: 0; margin: 0; }
.tests-panel li { padding: 3px 0; font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.85em; }
.layer-marker { display: inline-block; width: 18px; text-align: center; }
.outcome { margin-left: 8px; font-size: 0.85em; }
.outcome.passed { color: var(--pass); }
.outcome.failed { color: var(--fail); font-weight: 600; }
.outcome.missing { color: var(--miss); }
details { margin: 8px 0; }
summary { cursor: pointer; font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.9em; color: var(--accent); }
.empty { color: var(--muted); font-style: italic; padding: 8px 0; }
.no-results { padding: 16px 0; color: var(--muted); font-style: italic; }
.no-results[hidden] { display: none; }
footer { color: var(--muted); font-size: 0.85em; margin-top: 3em; border-top: 1px solid var(--rule); padding-top: 1em; }
""")
    out.append("</style>")
    out.append("</head>")
    out.append("<body>")
    out.append("<h1>Capability Matrix — dfdraw</h1>")
    out.append(f'<p class="meta">Generated: <span class="mono">{esc(now)}</span> · Phase: <span class="mono">{esc(phase)}</span></p>')

    # Summary
    out.append('<section class="summary">')
    out.append("<h2>Summary</h2>")
    out.append("<table>")
    for label, emoji in [("Verified", "✅"), ("Smoke-only", "☑️"),
                          ("Broken", "🧨"), ("Planned", "📋")]:
        c = counts.get(label, 0)
        pct = 100 * c / total if total > 0 else 0
        out.append(f'<tr><th>{emoji} {esc(label)}</th><td class="n">{c}</td><td class="n">{pct:.0f}%</td></tr>')
    total_tests = sum(s["n_tests"] for _, s in rows)
    inv_tests = sum(1 for t in test_layers.values() if t == "invariance")
    vis_tests = sum(1 for t in test_layers.values() if t == "visual_primitive")
    out.append(f'<tr><th>Total features</th><td class="n"><strong>{total}</strong></td><td></td></tr>')
    out.append(f'<tr><th>Total proof tests</th><td class="n"><strong>{total_tests}</strong></td><td></td></tr>')
    out.append(f'<tr><th>Invariance tests</th><td class="n"><strong>{inv_tests}</strong></td><td></td></tr>')
    if vis_tests:
        out.append(f'<tr><th>Visual tests</th><td class="n"><strong>{vis_tests}</strong></td><td></td></tr>')
    out.append("</table>")
    out.append("</section>")

    # Filters
    out.append('<section class="filters">')
    out.append('<div class="filter-group"><span class="label">Status:</span>'
               '<button class="active" data-status="all">All</button>'
               '<button data-status="Verified">✅ Verified</button>'
               '<button data-status="Smoke-only">☑️ Smoke</button>'
               '<button data-status="Broken">🧨 Broken</button>'
               '<button data-status="Planned">📋 Planned</button></div>')
    out.append('<div class="filter-group"><span class="label">Visual:</span>'
               '<button class="active" data-visual="all">All</button>'
               '<button data-visual="visual-only">👁 With visual</button>'
               '<button data-visual="no-visual">No visual</button></div>')
    out.append('<div class="filter-group"><span class="label">Category:</span>')
    out.append('<select id="cat-filter"><option value="all">All categories</option>')
    for c in sorted(categories):
        out.append(f'<option value="{esc(c)}">{esc(c)}</option>')
    out.append('</select></div>')
    out.append("</section>")

    # Features table
    out.append('<section><h2>Features</h2>')
    out.append('<table class="features"><thead>')
    out.append("<tr><th></th><th>Status</th><th>Feature</th>"
               "<th class='n'>Tests</th><th class='n'>Pass</th><th class='n'>Fail</th>"
               "<th class='n'>Inv</th><th class='n'>Visual</th></tr>")
    out.append("</thead><tbody>")

    current_cat = None
    for f, s in rows:
        cat = f.get("category", "")
        if cat != current_cat:
            out.append(f'<tr class="category-row"><td colspan="8">{esc(cat)}</td></tr>')
            current_cat = cat
        fid = f["id"]
        eye = ' <span class="eye-badge">👁</span>' if s["has_visual"] else ""
        visual_cell = s["n_visual"] if s["n_visual"] > 0 else ""
        out.append(
            f'<tr id="feature-{esc(fid)}" class="feature-row" '
            f'data-status="{esc(s["status"])}" '
            f'data-visual="{"yes" if s["has_visual"] else "no"}" '
            f'data-category="{esc(cat)}">'
            f'<td><span class="expand-toggle">▶</span></td>'
            f'<td>{s["icon"]}</td>'
            f'<td><strong class="mono">{esc(fid)}</strong> — {esc(f["name"])}{eye}</td>'
            f'<td class="n">{s["n_tests"]}</td>'
            f'<td class="n">{s["n_pass"]}</td>'
            f'<td class="n">{s["n_fail"]}</td>'
            f'<td class="n">{s["n_inv"]}</td>'
            f'<td class="n">{visual_cell}</td>'
            f'</tr>'
        )
        # Tests-panel row (hidden by default; toggled by JS)
        panel = ['<tr class="tests-row" hidden><td colspan="8">',
                 '<div class="tests-panel">']
        if not f["tests"]:
            panel.append('<div class="empty">(no tests claimed yet)</div>')
        else:
            panel.append("<ul>")
            for test_id in f["tests"]:
                layer = test_layers.get(test_id, "smoke")
                marker = _LAYER_MARKER.get(layer, "·")
                outcome = test_results.get(test_id, "missing")
                outcome_label = {
                    "passed": "passed", "failed": "failed", "missing": "missing",
                    "skipped": "skipped", "xfailed": "xfailed",
                }.get(outcome, outcome)
                panel.append(
                    f'<li><span class="layer-marker">{marker}</span>'
                    f'<code>{esc(test_id)}</code>'
                    f'<span class="outcome {esc(outcome)}">{esc(outcome_label)}</span></li>'
                )
            panel.append("</ul>")
        panel.append("</div></td></tr>")
        out.extend(panel)

    out.append("</tbody></table>")
    out.append('<div id="no-results" class="no-results" hidden>No features match the current filters.</div>')
    out.append("</section>")

    # Unmatched + Allow-listed
    if unmatched is not None:
        if unmatched:
            out.append(f'<section><h2>Unmatched Tests ({len(unmatched)})</h2>')
            out.append(f'<p>{len(unmatched)} tests pytest collected that no feature claims. Grouped by test-file prefix.</p>')
            by_file = defaultdict(list)
            for t in unmatched:
                by_file[t.split("::", 1)[0]].append(t)
            for fname in sorted(by_file):
                out.append(f'<details><summary><code>{esc(fname)}</code> ({len(by_file[fname])})</summary><ul>')
                for t in by_file[fname]:
                    out.append(f'<li><code>{esc(t)}</code></li>')
                out.append("</ul></details>")
            out.append("</section>")
        else:
            out.append('<section><h2>Unmatched Tests</h2><p>✓ 0 unmatched tests.</p></section>')

        if known_unclaimed:
            by_phase = defaultdict(list)
            for e in known_unclaimed:
                by_phase[e.get("target_phase", "OPEN")].append(e)
            out.append(f'<section><h3>Allow-listed (unclaimed) — {len(known_unclaimed)}</h3>')
            out.append('<p>Tests classified in <code>TEST_LAYERS</code> but not yet feature-claimed. '
                       'Each carries a <code>target_phase</code> for claim-back tracking.</p>')
            for tp in sorted(by_phase):
                out.append(f'<details><summary>target_phase: <code>{esc(tp)}</code> ({len(by_phase[tp])})</summary><ul>')
                for e in by_phase[tp]:
                    out.append(f'<li><code>{esc(e["test_id"])}</code><br>'
                               f'<span class="meta">{esc(e.get("reason","(no reason)"))}</span></li>')
                out.append("</ul></details>")
            out.append("</section>")

    out.append("<footer>Auto-generated by <code>scripts/generate_capability_matrix.py</code>. "
               "✓ = invariance (A ≡ B). ☑ = smoke. 👁 = visual_primitive (orthogonal). "
               "Visual coverage is independent of Verified/Smoke status — see Phase 13.49.DF §3.5.</footer>")

    # ---- JS: filter logic + expansion (D-D AND, D-E preserve expansion) ----
    out.append("<script>")
    out.append(r"""
(function() {
  // Expansion: clicking a feature-row toggles the following tests-row only.
  // Filter changes never affect tests-row expansion state (preserved per D-E).
  document.querySelectorAll('tr.feature-row').forEach(function(row) {
    row.addEventListener('click', function(e) {
      if (e.target.tagName === 'A') return;
      var next = row.nextElementSibling;
      var toggle = row.querySelector('.expand-toggle');
      if (next && next.classList.contains('tests-row')) {
        if (next.hasAttribute('hidden')) {
          next.removeAttribute('hidden');
          if (toggle) toggle.textContent = '▼';
        } else {
          next.setAttribute('hidden', '');
          if (toggle) toggle.textContent = '▶';
        }
      }
    });
  });

  // Filter state
  var state = { status: 'all', visual: 'all', category: 'all' };

  function applyFilters() {
    var anyVisible = false;
    document.querySelectorAll('tr.feature-row').forEach(function(row) {
      var s = row.dataset.status, v = row.dataset.visual, c = row.dataset.category;
      var match =
        (state.status === 'all' || s === state.status) &&
        (state.visual === 'all' ||
          (state.visual === 'visual-only' && v === 'yes') ||
          (state.visual === 'no-visual' && v === 'no')) &&
        (state.category === 'all' || c === state.category);
      // Hide both the feature row and its expansion row. Hidden ≠ collapsed:
      // when the filter is removed the row reappears in whatever expansion
      // state it was in (D-E).
      if (match) {
        row.style.display = '';
        var next = row.nextElementSibling;
        if (next && next.classList.contains('tests-row') && !next.hasAttribute('hidden')) {
          next.style.display = '';
        }
        anyVisible = true;
      } else {
        row.style.display = 'none';
        var next2 = row.nextElementSibling;
        if (next2 && next2.classList.contains('tests-row')) {
          next2.style.display = 'none';
        }
      }
    });
    document.getElementById('no-results')
      .toggleAttribute('hidden', anyVisible);
    // Category rows: hide if no feature-row in that category is visible
    var lastCatRow = null;
    var anyInCat = false;
    document.querySelectorAll('tbody tr').forEach(function(tr) {
      if (tr.classList.contains('category-row')) {
        if (lastCatRow) lastCatRow.style.display = anyInCat ? '' : 'none';
        lastCatRow = tr;
        anyInCat = false;
      } else if (tr.classList.contains('feature-row') && tr.style.display !== 'none') {
        anyInCat = true;
      }
    });
    if (lastCatRow) lastCatRow.style.display = anyInCat ? '' : 'none';
  }

  document.querySelectorAll('[data-status]').forEach(function(b) {
    b.addEventListener('click', function() {
      document.querySelectorAll('[data-status]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.status = b.dataset.status;
      applyFilters();
    });
  });
  document.querySelectorAll('[data-visual]').forEach(function(b) {
    b.addEventListener('click', function() {
      document.querySelectorAll('[data-visual]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.visual = b.dataset.visual;
      applyFilters();
    });
  });
  document.getElementById('cat-filter').addEventListener('change', function(e) {
    state.category = e.target.value;
    applyFilters();
  });
})();
""")
    out.append("</script>")
    out.append("</body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate capability matrix for dfdraw")
    parser.add_argument("--test-results", required=True,
                        help="Path to .pytest_report.json")
    parser.add_argument("--output",
                        default=os.path.join(PROJECT_DIR, "docs", "CAPABILITY_MATRIX.md"))
    parser.add_argument("--html-output",
                        default=os.path.join(PROJECT_DIR, "docs", "CAPABILITY_MATRIX.html"))
    parser.add_argument("--phase", default="unknown",
                        help="Phase tag (derived by run_tests.sh from latest PHASE_*_END)")
    args = parser.parse_args()

    if not os.path.exists(args.test_results):
        print(f"ERROR: {args.test_results} not found. Run tests first.")
        sys.exit(1)

    test_results = load_test_results(args.test_results)
    collected_tests = load_collected_tests(args.test_results)

    # Pull KNOWN_UNCLAIMED so the Unmatched section can surface allow-listed entries
    try:
        from tests.test_meta_capability_matrix import KNOWN_UNCLAIMED
    except Exception:
        KNOWN_UNCLAIMED = []

    generate_matrix(test_results, phase=args.phase, output_path=args.output,
                    collected_tests=collected_tests, known_unclaimed=KNOWN_UNCLAIMED)

    html_text = generate_html_matrix(
        test_results=test_results, features=FEATURES, test_layers=TEST_LAYERS,
        known_unclaimed=KNOWN_UNCLAIMED, collected_tests=collected_tests, phase=args.phase,
    )
    os.makedirs(os.path.dirname(args.html_output), exist_ok=True)
    with open(args.html_output, "w") as fp:
        fp.write(html_text)
    print(f"Capability matrix (.html) written to: {args.html_output}")

    # Summary print
    counts = Counter()
    for f in FEATURES:
        s = compute_feature_stats(f, test_results)
        counts[s["status"]] += 1
    inv = sum(1 for v in TEST_LAYERS.values() if v == "invariance")
    vis = sum(1 for v in TEST_LAYERS.values() if v == "visual_primitive")
    print(f"\n  Features: {len(FEATURES)}  |  Invariance tests: {inv}  |  Visual tests: {vis}")
    for label in ["Verified", "Smoke-only", "Broken", "Planned"]:
        print(f"  {label}: {counts.get(label, 0)}")
    claimed = {t for f in FEATURES for t in f["tests"]}
    unmatched = collected_tests - claimed
    print(f"  Unmatched (pytest collected, no feature claim): {len(unmatched)}")


if __name__ == "__main__":
    main()

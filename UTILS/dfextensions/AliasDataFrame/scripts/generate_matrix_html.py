#!/usr/bin/env python3
"""generate_matrix_html.py — CAPABILITY_MATRIX.html for AliasDataFrame.
BUG_ADF_20260705_matrix_html. Standalone: parses the pytest full log, matches
tests to features via feature_taxonomy.FEATURES test_patterns (ADF prefix
model), infers layers (invariance by name), renders the dfdraw-ancestry
single-file HTML with environment banner (A3) and git-derived phase (A2 —
no hardwired PHASE; latest ADF BEGIN tag per architect 2026-07-05).
Usage: generate_matrix_html.py --log test_logs/test_full_X.log \
         --output docs/CAPABILITY_MATRIX.html [--snapshot path] [--phase X]
"""
import argparse, html, os, platform, re, socket, subprocess, sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "tests"))
try:
    from feature_taxonomy import FEATURES
except ImportError as e:
    sys.exit(f"generate_matrix_html: cannot import feature_taxonomy: {e}")


def detect_phase(default="untagged"):
    """Latest ADF BEGIN tag merged to HEAD (architect decision 2026-07-05);
    falls back through the historical constant BEGIN tags; never hardwired."""
    for pattern in ("PHASE_*_ADF_BEGIN", "PHASE_BEGIN_AliasDataFrame", "PHASE_BEGIN_ADF"):
        try:
            r = subprocess.run(["git", "tag", "--merged", "HEAD", "--list", pattern],
                               capture_output=True, text=True, timeout=5, cwd=PROJECT)
            tags = sorted([t for t in r.stdout.split() if t])
            if tags:
                tag = tags[-1]
                sha = subprocess.run(["git", "rev-parse", "--short", tag],
                                     capture_output=True, text=True, timeout=5,
                                     cwd=PROJECT).stdout.strip()
                return f"{tag} @ {sha}" if sha else tag
        except Exception:
            pass
    return default


OUTCOME_RE = re.compile(r"\b(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(tests/\S+)")
_OUTCOME_MAP = {"PASSED": "passed", "FAILED": "failed", "ERROR": "failed",
                "SKIPPED": "skipped", "XFAIL": "xfailed", "XPASS": "passed"}


def parse_log(path):
    results = {}
    with open(path, errors="replace") as fp:
        for line in fp:
            m = OUTCOME_RE.search(line)
            if m:
                results[m.group(2).strip()] = _OUTCOME_MAP[m.group(1)]
    return results


_INV_RE = re.compile(r"test_I\d+_|invariance", re.I)


def infer_layer(test_id):
    return "invariance" if _INV_RE.search(test_id) else "smoke"


def expand_features(features, results):
    """ADF taxonomy: test_patterns are prefixes (file or file::Class).
    Returns features with exact matched test ids + the layer map."""
    ids = sorted(results)
    expanded, layers = [], {}
    for f in features:
        pats = f.get("test_patterns", f.get("tests", []))
        matched = sorted({t for t in ids for p in pats if t.startswith("tests/" + p) or t.startswith(p)})
        for t in matched:
            layers[t] = infer_layer(t)
        expanded.append({**f, "tests": matched})
    return expanded, layers


def environment_info():
    return {"hostname": socket.gethostname(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version()}


_LAYER_MARKER = {
    "invariance": "✓",
    "integration": "✓",
    "smoke": "☑",
    "visual_primitive": "👁",
}




_LAYER_MARKER = {
    "invariance": "✓",
    "integration": "✓",
    "smoke": "☑",
    "visual_primitive": "👁",
}


def compute_feature_stats(f, test_results):
    """ADF-side stats (dfdraw ancestry, layer via infer_layer)."""
    n_tests = len(f["tests"])
    n_pass = n_fail = n_inv = n_visual = 0
    for t in f["tests"]:
        oc = test_results.get(t, "missing")
        if oc == "passed":
            n_pass += 1
        elif oc == "failed":
            n_fail += 1
        if infer_layer(t) == "invariance":
            n_inv += 1
    if n_tests == 0:
        status, icon = "Planned", "\U0001F4CB"
    elif n_fail > 0:
        status, icon = "Broken", "\U0001F9E8"
    elif n_inv > 0 and n_pass > 0:
        status, icon = "Verified", "\u2705"
    elif n_pass > 0:
        status, icon = "Smoke-only", "\u2611\uFE0F"
    else:
        status, icon = "Planned", "\U0001F4CB"
    return {"status": status, "icon": icon, "n_tests": n_tests, "n_pass": n_pass,
            "n_fail": n_fail, "n_inv": n_inv, "n_visual": n_visual, "has_visual": False}


def generate_html_matrix(test_results, features=None, test_layers=None,
                          known_unclaimed=None, collected_tests=None,
                          phase=None, project="AliasDataFrame",
                          expected_env=("Linux", "aarch64")):
    """Return a single-file HTML matrix as a string.

    Each feature row is anchored at id="feature-{feature_id}" and expands to
    a div.tests-panel listing every claimed test_id with layer marker and
    outcome. Filters (status × visual × category) AND-combine; expansion
    state is preserved across filter changes (§9 D-D / D-E).
    """
    if phase is None:
        phase = detect_phase()
    features = features if features is not None else []
    test_layers = test_layers or {}
    known_unclaimed = known_unclaimed or []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Compute stats
    counts = {"Verified": 0, "Smoke-only": 0, "Broken": 0, "Planned": 0}
    rows = []
    categories = []
    seen_cat = set()
    # H-3 fix: stable-sort by category (same fix as the MD path) so each
    # category header is emitted exactly once.
    sorted_features = sorted(features, key=lambda f: f.get("category", ""))
    for f in sorted_features:
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
    out.append(f"<title>Capability Matrix — {esc(project)}</title>")
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
.envbanner { font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.85em;
  padding: 8px 12px; margin: 10px 0 18px 0; border: 1px solid var(--rule); }
.envbanner.offgate { background: #fdecec; border-color: var(--fail); color: var(--fail); font-weight: 600; }
footer { color: var(--muted); font-size: 0.85em; margin-top: 3em; border-top: 1px solid var(--rule); padding-top: 1em; }
""")
    out.append("</style>")
    # JS-off fallback (review viewers often disable scripts): show ALL test
    # panels expanded and say so — reviewers always see the supporting tests.
    out.append("<noscript><style>tr.tests-row[hidden]{display:table-row !important}"
               ".expand-toggle{visibility:hidden}.filters{display:none}</style>"
               "<div class='envbanner'>JavaScript is disabled in this viewer — "
               "all test panels are shown expanded; filters are hidden. "
               "For interactive filtering, open the file in a normal browser."
               "</div></noscript>")
    out.append("</head>")
    out.append("<body>")
    out.append(f"<h1>Capability Matrix — {esc(project)}</h1>")
    out.append(f'<p class="meta">Generated: <span class="mono">{esc(now)}</span> · Phase: <span class="mono">{esc(phase)}</span></p>')
    env = environment_info()
    ongate = (env["system"], env["machine"]) == tuple(expected_env)
    cls = "envbanner" if ongate else "envbanner offgate"
    tag = "" if ongate else " — NOT the blocking-gate environment; matrix states here are NOT release evidence (BUG_20260610 class)"
    out.append(f'<div class="{cls}">Environment: {esc(env["hostname"])} · {esc(env["system"])}-{esc(env["machine"])} · Python {esc(env["python"])}{esc(tag)}</div>')

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
            f'<td><strong class="mono">{esc(fid)}</strong> — {esc(f["name"])}{eye} <a class="mono" href="#feature-{esc(fid)}" title="permalink" onclick="event.stopPropagation()">&para;</a></td>'
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

  document.querySelectorAll('.filter-group [data-status]').forEach(function(b) {
    b.addEventListener('click', function() {
      document.querySelectorAll('.filter-group [data-status]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      state.status = b.dataset.status;
      applyFilters();
    });
  });
  document.querySelectorAll('.filter-group [data-visual]').forEach(function(b) {
    b.addEventListener('click', function() {
      document.querySelectorAll('.filter-group [data-visual]').forEach(x => x.classList.remove('active'));
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--output", default=os.path.join(PROJECT, "docs", "CAPABILITY_MATRIX.html"))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--phase", default=None, help="override; default = latest ADF BEGIN tag")
    a = ap.parse_args()
    results = parse_log(a.log)
    feats, layers = expand_features(FEATURES, results)
    text = generate_html_matrix(results, features=feats, test_layers=layers,
                                collected_tests=set(results), phase=a.phase)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    open(a.output, "w").write(text)
    print(f"Capability matrix (.html) written to: {a.output}  [{len(results)} test results]")
    if a.snapshot:
        open(a.snapshot, "w").write(text)


if __name__ == "__main__":
    main()

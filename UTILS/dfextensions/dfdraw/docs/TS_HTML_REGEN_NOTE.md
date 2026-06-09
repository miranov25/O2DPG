# TS HTML Regeneration — Working Command (Phase 13.51)

**File:** `TS_HTML_REGEN_NOTE.md`
**Date:** 2026-06-09

## The single working command

```bash
cd ~/alicesw/O2DPG/UTILS/dfextensions/dfdraw/docs

pandoc dfdraw_Technical_Summary.md \
  -f markdown-yaml_metadata_block -t html5 \
  --standalone \
  --toc --toc-depth=3 --section-divs \
  --highlight-style=tango \
  --metadata title="dfdraw Technical Summary" \
  -V maxwidth=72em \
  -H custom_css.html \
  -o dfdraw_Technical_Summary.html
```

Save `custom_css.html` (delivered alongside) somewhere stable — e.g. `docs/custom_css.html` — and reference it with the `-H` flag.

## What each flag does and why prior runs needed iteration

| Flag | Without it | Why needed |
|---|---|---|
| `-f markdown-yaml_metadata_block` | `Error parsing YAML metadata: while scanning an alias` | The TS starts with `**Library version:**` — `**` at line start trips pandoc's YAML parser, which sees `*` as alias syntax. Disables the YAML-block extension; standard markdown emphasis still works. |
| `--standalone` | Output is HTML fragment, no `<head>`/`<body>` wrapper | Need full HTML page with embedded CSS. |
| `--toc --toc-depth=3` | No `<nav id="TOC">` element | Custom CSS targets `nav#TOC` — without `--toc` it's dead CSS and the page has no table of contents. |
| `--section-divs` | All content as flat `<h2>` + paragraphs | Original wraps sections in `<section>` elements; custom CSS references them. |
| `--highlight-style=tango` | Code blocks use pygments default (greens, reds) | Original uses tango palette (`#204a87` blue, `#0000cf` numeric, `#8f5902` brown) — match by name. `kate` is *similar* but uses `#a40000`/`#902000` reds; visibly different. |
| `-V maxwidth=72em` | `max-width: 36em` (pandoc 3.x default) | Original is wider; reading flow needs ≥72em for the table-heavy content. |
| `-H custom_css.html` | Default pandoc styling only — no TOC box, no two-column nav, no reset | Phase 13.50 (or earlier) coder injected a 139-line custom stylesheet for TOC styling, font reset, and section dividers. This is the bulk of the "looks polished" effect. |
| `--metadata title=...` | Empty `<title>` | Browser tab title. |

## Verification

Diff vs prior HTML (Phase 13.50.DF FIX2): all 368 lines of `<head>` are byte-identical (just whitespace re-indentation in the custom-CSS injection). `<section>` count is 167 vs 181 — 14 fewer because Phase 13.51 added blockquote callouts (not new H2 sections). Tango palette colors match exactly.

## Where the custom CSS came from

I extracted lines 229-367 of the prior `dfdraw_Technical_Summary.html` (the pre-Phase-13.51 file in the repo) into `custom_css.html`. It contains:
- CSS reset (`* { box-sizing: border-box }`, font-family stack)
- Hidden `header#title-block-header`
- `body { max-width: 1100px }` (overrides the pandoc-template variable for finer control)
- `nav#TOC` box styling (background, border-radius, padding)
- Section dividers
- Print/media queries

If you ever lose `custom_css.html`, re-extract via:
```bash
sed -n '229,367p' <prior_TS_html_from_git_history> > custom_css.html
```

## Apply

```bash
mv ~/Downloads/dfdraw_Technical_Summary.html \
   ~/alicesw/O2DPG/UTILS/dfextensions/dfdraw/docs/dfdraw_Technical_Summary.html

mv ~/Downloads/custom_css.html \
   ~/alicesw/O2DPG/UTILS/dfextensions/dfdraw/docs/custom_css.html

mv ~/Downloads/TS_HTML_REGEN_NOTE.md \
   ~/alicesw/O2DPG/UTILS/dfextensions/dfdraw/docs/

# Stage everything Phase 13.51 needs:
cd ~/alicesw/O2DPG/UTILS/dfextensions/dfdraw
git add docs/dfdraw_Technical_Summary.html \
        docs/dfdraw_Technical_Summary.md \
        docs/custom_css.html \
        docs/TS_HTML_REGEN_NOTE.md

git status .   # confirm clean staging
```

## Recommendation: bake into the test runner

Consider adding the pandoc command to `run_tests.sh` or a separate `scripts/regen_TS_html.sh` so the next coder doesn't need 6 iterations to rediscover the flag combination. Suggested:

```bash
# scripts/regen_TS_html.sh
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
pandoc docs/dfdraw_Technical_Summary.md \
  -f markdown-yaml_metadata_block -t html5 \
  --standalone --toc --toc-depth=3 --section-divs \
  --highlight-style=tango \
  --metadata title="dfdraw Technical Summary" \
  -V maxwidth=72em -H docs/custom_css.html \
  -o docs/dfdraw_Technical_Summary.html
echo "✅ Regenerated docs/dfdraw_Technical_Summary.html"
```

---

*Opus1 — TS HTML regeneration note — 2026-06-09*

*(No internal quota issues observed.)*

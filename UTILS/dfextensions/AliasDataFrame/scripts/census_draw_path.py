#!/usr/bin/env python3
"""
census_draw_path.py — PHASE_13_76_ADF Stage A deliverable A3.

Reproducible, deterministic AST census of the draw-path corpus, per Proposal
Rev 2 §4.2. Reports SEPARATE categories (never one ambiguous total):

  draw_call            direct  .draw(...) calls          (executable)
  draw_batch_call      .draw_batch(...) calls            (executable)
  draw_figures_call    .draw_figures(...) calls          (executable)
  typed_call           typed convenience calls           (executable; e.g. .drawProfile)
  wrapper_def          function defs whose body contains draw-family calls
                       (definitions are NOT runtime call sites; listed, not
                       added to the executable totals)
  dynamic_call         getattr()/alias-resolved draw-family calls that static
                       AST cannot bind (listed separately, never totalled)
  string_mention       '.draw'-family text inside string literals/docstrings
                       (excluded from executable counts by construction —
                       AST never parses strings as calls; counted for audit)
  materialize_call     manual materialize/ensure calls
  release_call         release/dematerialize calls
  ax_usage             draw-family call carrying ax= (kwarg or in a dict literal)
  modifier_usage       draw-family call carrying any §8.7 modifier kwarg

Rules honored (§4.2): AST for Python; comments never parse (AST-immune);
docstring/string examples excluded from executable totals; wrapper definitions
not counted as call sites; filename+line retained per row; deterministic CSV +
JSON + summary; exit code 0 on success so Gate A can fingerprint outputs.

Usage:
  python3 census_draw_path.py <file-or-dir> [more...] --out-prefix census_out
Outputs:
  <prefix>.csv   one row per finding (sorted: path, line, category, symbol)
  <prefix>.json  category totals + per-file totals + methodology block
  stdout         summary table
"""

import argparse
import ast
import csv
import json
import os
import sys

DRAW_SURFACES = ("draw", "draw_batch", "draw_figures")
# typed convenience prefixes observed in the corpus/API (extend via --typed)
TYPED_PREFIXES = ("drawProfile", "drawHist", "drawScatter")
MATERIALIZE_NAMES = (
    "materialize_aliases", "materialize", "ensure_columns", "ensure_branches",
    "materialize_subframes",
)
RELEASE_NAMES = ("dematerialize", "release_struct", "release", "clear_materialized")
MODIFIER_KWARGS = (
    "group_by_bins", "facet_by_bins", "facet_by_quantiles", "bins", "range",
    "ncols", "sharex", "sharey", "figsize", "ax",
)


def _call_attr_name(node):
    """Return attribute name for obj.name(...) calls, else None."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _receiver_repr(node):
    """Best-effort static receiver ('adf', 'self', ...); '<dynamic>' if unbound."""
    f = node.func
    if isinstance(f, ast.Attribute):
        v = f.value
        if isinstance(v, ast.Name):
            return v.id
        if isinstance(v, ast.Attribute):
            parts = []
            while isinstance(v, ast.Attribute):
                parts.append(v.attr)
                v = v.value
            if isinstance(v, ast.Name):
                parts.append(v.id)
                return ".".join(reversed(parts))
        return "<dynamic>"
    return "<dynamic>"


def _kwarg_names(call):
    names = set()
    for kw in call.keywords:
        if kw.arg:
            names.add(kw.arg)
        else:  # **something — cannot resolve statically
            names.add("**")
    return names


def _dict_literal_keys(call):
    """Keys of dict literals passed positionally or by kw (specs/defaults forms),
    one level deep plus nested per-plot dicts."""
    keys = set()
    def collect(d):
        if isinstance(d, ast.Dict):
            for k, v in zip(d.keys, d.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
                collect(v)
    for a in list(call.args) + [kw.value for kw in call.keywords]:
        collect(a)
    return keys


class Census(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.rows = []           # dicts: path,line,category,symbol,detail
        self.func_stack = []
        self.wrapper_defs = {}   # name -> lineno (defs containing draw calls)

    # ---- function definitions: detect wrappers, keep stack ----
    def visit_FunctionDef(self, node):
        self.func_stack.append(node)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _mark_wrapper(self):
        if self.func_stack:
            fd = self.func_stack[-1]
            self.wrapper_defs.setdefault(fd.name, fd.lineno)

    def _row(self, line, category, symbol, detail=""):
        self.rows.append({
            "path": self.path, "line": line, "category": category,
            "symbol": symbol, "detail": detail,
        })

    def visit_Call(self, node):
        attr = _call_attr_name(node)
        if attr:
            recv = _receiver_repr(node)
            if attr in DRAW_SURFACES:
                cat = {"draw": "draw_call", "draw_batch": "draw_batch_call",
                       "draw_figures": "draw_figures_call"}[attr]
                if recv == "<dynamic>":
                    self._row(node.lineno, "dynamic_call", attr, recv)
                else:
                    self._row(node.lineno, cat, attr, f"recv={recv}")
                self._mark_wrapper()
                kw = _kwarg_names(node)
                dk = _dict_literal_keys(node)
                if "ax" in kw or "ax" in dk:
                    where = "kwarg" if "ax" in kw else "dict-literal"
                    self._row(node.lineno, "ax_usage", attr, where)
                mods = sorted((kw | dk) & set(MODIFIER_KWARGS) - {"ax"})
                if mods:
                    self._row(node.lineno, "modifier_usage", attr, ",".join(mods))
            elif attr.startswith(TYPED_PREFIXES):
                self._row(node.lineno, "typed_call", attr, f"recv={recv}")
                self._mark_wrapper()
            elif attr in MATERIALIZE_NAMES:
                self._row(node.lineno, "materialize_call", attr, f"recv={recv}")
            elif attr in RELEASE_NAMES:
                self._row(node.lineno, "release_call", attr, f"recv={recv}")
        # getattr(obj, 'draw')(…) → dynamic
        if (isinstance(node.func, ast.Call)
                and isinstance(node.func.func, ast.Name)
                and node.func.func.id == "getattr"):
            args = node.func.args
            if (len(args) >= 2 and isinstance(args[1], ast.Constant)
                    and str(args[1].value) in DRAW_SURFACES):
                self._row(node.lineno, "dynamic_call", str(args[1].value), "getattr")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for s in DRAW_SURFACES:
                if f".{s}(" in node.value:
                    self._row(getattr(node, "lineno", 0), "string_mention", s,
                              "string-literal/docstring")
                    break
        # no generic_visit needed for constants


def census_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return [{"path": path, "line": e.lineno or 0, "category": "parse_error",
                 "symbol": "", "detail": str(e)}], {}
    c = Census(path)
    c.visit(tree)
    for name, line in sorted(c.wrapper_defs.items()):
        c.rows.append({"path": path, "line": line, "category": "wrapper_def",
                       "symbol": name, "detail": "definition (not a call site)"})
    return c.rows, c.wrapper_defs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--out-prefix", default="census_draw_path")
    args = ap.parse_args(argv)

    files = []
    for t in args.targets:
        if os.path.isdir(t):
            for root, _dirs, names in os.walk(t):
                for n in sorted(names):
                    if n.endswith(".py"):
                        files.append(os.path.join(root, n))
        elif t.endswith(".py"):
            files.append(t)
    files = sorted(set(files))

    all_rows = []
    for f in files:
        rows, _w = census_file(f)
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: (r["path"], r["line"], r["category"], r["symbol"]))

    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "line", "category", "symbol", "detail"])
        w.writeheader()
        w.writerows(all_rows)

    totals, per_file = {}, {}
    for r in all_rows:
        totals[r["category"]] = totals.get(r["category"], 0) + 1
        pf = per_file.setdefault(r["path"], {})
        pf[r["category"]] = pf.get(r["category"], 0) + 1
    executable_total = sum(totals.get(k, 0) for k in
                           ("draw_call", "draw_batch_call", "draw_figures_call",
                            "typed_call"))
    summary = {
        "files_scanned": len(files),
        "category_totals": dict(sorted(totals.items())),
        "executable_draw_family_total": executable_total,
        "per_file": {k: dict(sorted(v.items())) for k, v in sorted(per_file.items())},
        "methodology": {
            "parser": "ast (comments never parsed; string/docstring examples "
                      "cannot produce Call nodes — counted as string_mention only)",
            "wrapper_defs": "listed, excluded from executable totals",
            "dynamic": "listed separately, excluded from executable totals",
            "sort_key": "(path, line, category, symbol) — deterministic",
        },
    }
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"census: {len(files)} files, {len(all_rows)} rows -> {csv_path}, {json_path}")
    for k, v in sorted(totals.items()):
        print(f"  {k:20s} {v}")
    print(f"  {'EXECUTABLE draw-family total':20s} {executable_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

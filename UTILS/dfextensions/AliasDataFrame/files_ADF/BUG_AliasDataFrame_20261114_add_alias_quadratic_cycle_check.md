# BUG_AliasDataFrame_20261114 — `add_alias` gets quadratically slower with the number of aliases

**Date:** 2026-11-14
**Reporter:** Claude2:dEdxAI (coder), for Marian Ivanov (Architect, dEdxAI, JIRA O2-6532)
**Owner:** AliasDataFrame
**Status:** OPEN. A fix is proposed below and tested on a copy; nothing is committed.
**Code:** miranov25/O2DPG, branch `feature/groupby-optimization`, commit `f248455` (10 Nov 2026), file `UTILS/dfextensions/AliasDataFrame/AliasDataFrame.py` (MD5 `2b01116110c1925032abd40fe5100bb0`). The line numbers below refer to this commit.

## Summary

Each `add_alias()` call runs a cycle check that rebuilds the whole alias dictionary once for every word in every alias expression. Defining one alias therefore costs time proportional to (number of aliases)² × (words per expression).

In a frame with many long aliases, `add_alias` looks frozen: 100% CPU and no output. Reading the dictionary once per check removes the problem. That is a three-line change, and it leaves the computed dependencies unchanged.

**Words used below:**
- **Alias:** a named ADF expression, stored in `_schema["columns"][name]["expr"]`.
- **Token:** a word found in an expression by `re.findall(r'\b\w+\b', ...)`.
- **N:** the number of aliases already defined in the frame.

## Impact

This was seen in the dEdxAI calibration session. The frame holds the Taylor regional fits: 21 fitted targets, each with FitFull, Fit0, Corr0 and MAD aliases, for several selection cuts. The FitFull aliases are 39-term expressions.

A loop that adds 4 aliases per target (84 `add_alias` calls) ran at 100% CPU with no output and did not finish. With the fix applied as a session patch, the same loop completed.

The cost grows with N², so every new fit makes the next `add_alias` slower.

## Cause

The call chain on every scalar `add_alias`:

```text
add_alias (l. 9308)
 └ _add_scalar_alias (l. 9452) ── calls self._check_for_cycles() at l. 9557
    └ _check_for_cycles (l. 9084)
       └ _topological_sort (l. 9954)
          └ _resolve_dependencies (l. 9888)
```

- **`aliases` is a property (l. 3320).** Each access builds a new `{name: expr}` dict from `_schema["columns"]` and wraps it in `_ReadOnlyAliasDict`, so one access costs O(N).
- **`_resolve_dependencies` reads it inside a nested loop.** It loops over all aliases, then over all tokens of each expression, and tests `if token in self.aliases:` (l. 9924) for each one. Every test rebuilds the dict.
- **The result:** one `add_alias` costs O(N × tokens × N). Building a frame alias by alias costs O(N³).

`_resolve_dependencies` is also called from `_invalidate_alias_cascade` (l. 10003, only when a materialized alias is redefined), `plot_alias_dependencies` (l. 9943) and `describe_aliases` (l. 10500). These calls get the same speed-up.

## Reproducer

Save this as `repro_add_alias_scaling.py` and run it as `python3 repro_add_alias_scaling.py /path/to/O2DPG/UTILS 200 400 800 1600`.

```python
"""Reproducer: time of ONE add_alias() versus the number of aliases already defined.
Usage: python3 repro_add_alias_scaling.py /path/to/O2DPG/UTILS N1 N2 ..."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from dfextensions.AliasDataFrame.AliasDataFrame import AliasDataFrame
# one 39-term expression, like a Taylor fit alias in dEdxAI (subframe coefficients x column products)
terms = "+".join(f"S.c{j}*(abs(fTgl)*mdEdxExp*fFt0OccN)" for j in range(39))
for n in map(int, sys.argv[2:]):
    adf = AliasDataFrame(pd.DataFrame({"k": np.arange(10), "fTgl": np.zeros(10, np.float32),
                                       "mdEdxExp": np.ones(10, np.float32), "fFt0OccN": np.ones(10, np.float32)}))
    adf.register_subframe("S", AliasDataFrame(pd.DataFrame({"k": np.arange(10), **{f"c{j}": np.ones(10) for j in range(39)}})),
                          index_columns=["k"])
    for i in range(n):                        # bulk-define n aliases without the per-call check
        adf._schema["columns"][f"a{i}"] = {"expr": terms}
    t0 = time.perf_counter()
    adf.add_alias("probe", terms, dtype=np.float32)
    print(f"aliases={n:5d}  one add_alias: {time.perf_counter() - t0:8.3f} s", flush=True)
```

These are the times for one `add_alias`, measured on the same 2-CPU machine:

| Aliases in frame (N) | `f248455` as is | With the fix |
|---:|---:|---:|
| 200 | 0.59 s | 0.018 s |
| 400 | 2.53 s | 0.034 s |
| 800 | 8.76 s | 0.075 s |
| 1600 | 34.7 s | 0.21 s |

Without the fix, doubling N multiplies the time by about 4. With the fix the time grows roughly linearly, from the remaining single pass over all expressions.

## Proposed fix

In `_resolve_dependencies`, the fix reads `self.aliases` once into a local dict and tests the tokens of each expression as a set. This is the patch against `f248455` (`git apply --check` passes):

```diff
--- a/UTILS/dfextensions/AliasDataFrame/AliasDataFrame.py
+++ b/UTILS/dfextensions/AliasDataFrame/AliasDataFrame.py
@@ -9902,7 +9902,11 @@
         if hasattr(self, '_subframes') and hasattr(self._subframes, 'subframes'):
             subframe_names = set(self._subframes.subframes.keys())
         
-        for name, expr in self.aliases.items():
+        # `self.aliases` is a property that rebuilds the full {name: expr} dict on
+        # every access. Read it ONCE: testing `token in self.aliases` per token made
+        # this function O(N_aliases^2 x tokens), and add_alias() calls it every time.
+        aliases = dict(self.aliases)
+        for name, expr in aliases.items():
             # Find all word tokens, but handle dotted expressions specially
             # First, remove subframe references like "subframe.column" from consideration
             expr_cleaned = expr
@@ -9910,8 +9914,8 @@
                 # Remove "subframe.anything" patterns
                 expr_cleaned = re.sub(rf'\b{sf_name}(?:\.\w+)+', '', expr_cleaned)
             
-            # Now find remaining tokens
-            tokens = re.findall(r'\b\w+\b', expr_cleaned)
+            # Now find remaining tokens (a set: repeated tokens are checked once)
+            tokens = set(re.findall(r'\b\w+\b', expr_cleaned))
             
             for token in tokens:
                 # Skip self-references (alias depending on itself is always wrong)
@@ -9921,7 +9925,7 @@
                 if token in subframe_names:
                     continue
                 # Only add if token is actually an alias
-                if token in self.aliases:
+                if token in aliases:
                     dependencies[name].add(token)
         
         return dependencies
```

**Behaviour is unchanged.** The returned map is a dict of sets, so testing each distinct token once gives the same sets. Nothing changes the schema between the start and the end of the function, so a snapshot taken at the start is exact.

## Validation done

1. **The dependency map is identical.** On a frame with a registered subframe and chained aliases (`a` uses the subframe; `b` uses `a`; `c2` uses `a`, `b`), the original and the fixed function return the same map: `{'b': {'a'}, 'c2': {'a', 'b'}}`.
2. **The ADF tests that cover dependencies give the same result.** I ran the test files that mention cycles, dependencies, invalidation or topological order: 54 files. `test_time_series_invariancetest.py` was left out because it needs the module `perfmonitor`, which is not installed here. Before and after the fix the counts are 2078 passed, 105 skipped, 22 xfailed, 2 xpassed, 86 failed and 8 errors. The lists of failing and erroring tests are identical. These failures already exist in this environment (no ROOT; for example the real-data harness and the polynomial-persistence tests), and the fix changes none of them.
3. **Not done:** the full suite with `run_tests.sh`. That is for the ADF team.

## Same pattern elsewhere (not fixed; for the ADF team to check)

These places also read the `aliases` property, which costs O(N) per access, inside a loop. Each one is cheaper than the cycle check, but with thousands of aliases or columns they add up:

| Line | Function | Loop |
|---:|---|---|
| 9134 | `validate_no_cycles` | `set(self.aliases.keys())` built once per alias |
| 11046 | `_get_alias_dependencies` | `token in self.aliases` per token |
| 11754 | `materialize_alias` | `token in self.aliases` per token, recursive |
| 12278 | `get_alias_series` | `token in self.aliases` per token |
| 12489 | `save` | `c not in self.aliases` per column |
| 18826 | `dematerialize` | `c in self.aliases` per column |
| 20098 | `_ensure_vector_kwargs_aliases` | `set(self.aliases.keys())` per expression |
| 4567 | `__getattr__` | one rebuild per attribute access |

## Options for a general fix (ADF team decision)

- **(a) A local snapshot in each hot loop,** as in the fix above. This is minimal and local, and it adds no new state.
- **(b) A cheap `aliases` property,** for example a cached read-only mapping that is rebuilt only when `_schema["columns"]` changes. The cache must be invalidated by every writer of the schema, and some writers assign directly (for example l. 9515 `self._schema["columns"][name] = spec`, and `update_schema`). A missed writer would give stale answers, so this needs care.
- **(c) An incremental cycle check in `add_alias`.** A new or redefined alias X can only close a cycle if X is reachable from its own alias dependencies. A search from those dependencies over a cached dependency map would replace re-reading every expression on every call.

## Proposed acceptance tests

- **Timing:** with N = 1600 aliases of 39 terms each, as in the reproducer, one `add_alias` takes less than 0.5 s, and going from N = 800 to N = 1600 at most about doubles the time.
- **Equivalence:** `_resolve_dependencies()` returns the same map before and after the change, on frames with subframes, chained aliases, repeated tokens in one expression, and self-named subframe columns.

## Workaround until the fix is released

This is a session-only patch, applied after import, that changes no file:

```python
import re
from collections import defaultdict
def _resolveDepsFast(self):
    aliases = dict(self.aliases)                       # built once, not once per token
    sfs = set(getattr(getattr(self, "_subframes", None), "subframes", {}).keys())
    deps = defaultdict(set)
    for name, expr in aliases.items():
        for sf in sfs:
            expr = re.sub(rf"\b{sf}(?:\.\w+)+", "", expr)
        for tok in set(re.findall(r"\b\w+\b", expr)):
            if tok != name and tok not in sfs and tok in aliases:
                deps[name].add(tok)
    return deps
type(adf)._resolve_dependencies = _resolveDepsFast
```

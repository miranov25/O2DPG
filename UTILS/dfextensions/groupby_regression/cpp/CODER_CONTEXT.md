# cpp/CODER_CONTEXT.md — Phase 13.18.GB Coder Context

**Purpose:** On-ramp document (≤ ~1 page) for any Coder resuming Phase 13.18.GB work mid-phase. Captures the canonical Layer A contract, key design decisions, forbidden headers, JSON fixture schema, and lessons recorded during implementation. Per proposal v1.1 §7 Turn 3 deliverable + P2-7 (Claude23).

**Reader expected to already know:** Proposal v1.1, FIXTURE_SPEC v1.0, PHASE_13_18_GBADF v0.3 brainstorming.

---

## 1. Canonical Layer A Constructor Contract

This is the single authoritative shape that **both** the ROOT glue (Layer B, Turn 6) and the ADF Python bridge (Phase 13.18.ADF / 13.19) target. If the shape changes, both consumers break.

```cpp
GroupByRegressionEvaluator(
    ModelSchema schema,         // group_columns, predictor_columns,
                                // targets, suffix, fit_intercept
    std::vector<SubframeRow> rows,  // one per populated bin
    MethodMode method,          // Lookup | Linear
    BoundsMode bounds);         // Nan | Clamp
```

Constructor does at load time:
1. **Infer `bin_centers`** — per-dim sorted distinct natural-label values from `rows`
2. **Build per-dim remap** — `natural_label → compact_idx` maps (exposed via `remap()` getter for ADF consumers)
3. **Allocate dense N-D coefficient arrays** — flat row-major linearization, one array per coefficient column name
4. **Populate `valid_mask`** — flat row-major bool, true at positions with a row present, false elsewhere
5. **Tolerate unknown-suffix columns** — `*_err_sw`, `*_rmse_sw`, `*_n_fitted_sw` accepted at load time, not exposed through `evaluate_*` (P1-δ)

## 2. Natural-Label vs Compact-Index — Responsibility Boundary

**CRITICAL for consumers** (per ADF review flag 3, preventing the Phase 13.18.ADF signal-loss incident):

> The C++ evaluator operates on **compact 0..N-1 integer indices** into `bin_centers`. Natural-label → compact-index remap at query time is the **consumer's responsibility.**

Examples:
- Natural label: `sector = 5` (a real sector ID in `{2, 5, 9}`)
- Compact index: `1` (position of 5 in sorted `bin_centers[sector_dim]`)

Consumer must call `ev.remap()[sector_dim].at(5)` → `1` before passing to `evaluate_lookup`. Shortcut for ADF bridge: **read from `ev.remap()` directly** rather than rebuilding. Single source of truth = no load-time vs query-time drift.

## 3. Safety Contract (Missing-Bin → NaN)

Per Phase 13.18.GBADF v0.3 §4.4 / v1.1 §3:

- Query at compact index with `valid_mask=false` → **NaN**, regardless of `bounds` mode
- Query out-of-grid + `bounds=Nan` → **NaN**
- Query out-of-grid + `bounds=Clamp` → snap to nearest in-grid index, THEN check `valid_mask`. If clamped cell has `valid_mask=false`, still NaN (F_23 adversarial fixture enforces this)
- **No silent neighbour fallback. No clamp-to-nearest-valid. NaN always.**

## 4. Forbidden Headers in Layer A (WASM-Safety)

Layer A sources (`gbe_kernel.*`, `json_reader.*`) must NOT `#include` any of:

- ROOT types: `TObject`, `TString`, `TTree`, `TFile`, `TBranch`, `TObjString`
- Filesystem / OS: `<filesystem>`, `dlfcn.h`, `unistd.h`, `sys/stat.h`, `sys/mman.h`
- Threading: `<thread>`, `<mutex>`, `<atomic>`

**Enforcement:** `make wasm_lint` grep-checks these patterns. Real emscripten compile at phase end is the acceptance gate (architect-executed).

**Exception:** `cli_runner.cpp` uses `<iostream>` at entry/exit (emscripten supports it); `cli_runner.cpp` is NOT subject to `wasm_lint` because it is a test binary, not the Layer A kernel itself. `cli_runner` is excluded from the WASM compile target.

## 5. JSON Fixture Schema (restricted parser target)

Per FIXTURE_SPEC v1.0 §4. Top-level keys (EXACTLY these, no more, no less):

```
{
  "fixture_id":       "<string>",
  "axis_values":      { ... },
  "input":            { ... },
  "expected_output":  { "<target>": [...values...] },
  "intermediates":    { ... },
  "metadata":         { ... }
}
```

`json_reader` rejects any top-level key outside this set at load time.

**NaN encoding:** JSON does not support NaN natively. The fixture format uses the **string `"NaN"`** (quoted) to represent NaN values in `expected_output`. The `cli_runner` emits `"NaN"` (quoted) in its stdout response. The Python test harness compares with explicit `isinstance(v, str) and v == "NaN"` check.

## 6. Public API Symmetry with Python (ADF Flag 2)

| Python evaluator | C++ Layer A |
|---|---|
| `evaluator.valid_mask()` (method) | `ev.valid_mask()` (method, returns `const std::vector<bool>&`) |
| `evaluator._bin_centers[col]` (internal) | `ev.bin_centers()` (public method, returns `const std::vector<std::vector<int64_t>>&`) |
| `evaluator.evaluate(positions, method=, bounds=)` | `ev.evaluate_lookup(position_idx, pred_vals)` / `ev.evaluate_linear(...)` in Turn 4 |

**Naming deliberately symmetric** for cross-language mental portability.

## 7. Build / Test Commands

```bash
cd cpp
make clean && make layer_a_lookup    # clean compile, strict flags
make wasm_lint                        # forbidden-headers grep check
pytest tests/test_layer_a_lookup.py -v  # 13 tests (12 fixtures + F_22 Safety)
```

Compile flags: `-std=c++17 -O2 -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion` (per P2-4).

## 8. Lessons Recorded

**From Turn 1 re-delivery cycle:** when a sanity-check arithmetic drifts from design intent, trace to root cause before shipping. The 14:10 fit_intercept axis-count imbalance was a symptom of 3 missing orthogonal cells, not a cosmetic mismatch. Same failure class as Phase 13.17.GB 528→529 inherited arithmetic.

**From Turn 2 deployment cycle:** when an `ImportError: No module named X` appears, **verify the `.py` file physically exists at the expected location before debugging import paths**. Takes 5 seconds; saved two lost round-trips. Related: **when a subproject (like `cpp/`) is not a Python package, `conftest.py` must be at the subproject root** (`cpp/conftest.py`), not inside `tests/`. Deliver explicit "place these files at these paths" checklists, never rely on implicit file-layout assumptions.

**From Turn 2→3 transition:** **always deliver a Deployment Checklist** explicitly mapping each artifact to its target path plus a single copy-paste verification command block. Review UI artifact listings show filenames, not repo paths.

**From Turn 3 implementation:** `-Wsign-conversion` strict flag caught a class of `static_cast<int>(size_t)` anti-pattern I was using reflexively. Lesson: let the type system guide indexing — use `std::size_t` for `std::vector` indexing, only cast at boundaries. Clean compile gate on `-Werror -Wconversion -Wsign-conversion` is worth the extra care.

## 9. Known Deferred Items

- **Linear method (Turn 4)** — `evaluate_linear` with 2^N corner renormalization on `valid_mask=false` corners
- **All-corners-invalid NaN test** (Turn 5, P1-β) — inline minimal fixture
- **Layer B ROOT glue** (Turn 6) — `TObjString` sidecar, `gInterpreter->Declare`, tabular `eval_on_tree`
- **`TECHNICAL_SUMMARY v3.5` doc of `__gbreg_schema` suffix** — ADF review flag for consumer namespace reservation
- **`json_reader.cpp` line overrun** — 250 lines vs the proposal's ~120 target, accepted with note in Turn 3 report

## 10. Reviewer Pair Rotation Reminder

| Turn | Main | Helping |
|---|---|---|
| 3 (THIS) | Claude20 | Claude23 |
| 4 | Claude20 | Claude21 |
| 5 | Claude20 | Claude23 |
| 6 | Claude20 | Claude21 |
| 7 | Claude20 | Claude21 *(swapped from v1.0)* |
| 8 | Claude20 | Claude21 |
| 9 | Claude20 | Claude23 / Claude21 commit-time |

Claude23 biased toward pure-C++ Layer A turns. Claude21 biased toward ROOT-dependent Layer B turns.

#!/bin/bash
# cpp/tests/run_cpp_tests.sh — Shell wrapper for C++ evaluator tests.
#
# Called from the top-level run_tests.sh to include C++ Layer A tests
# in the canonical test suite. Layer B (PyROOT) tests are run separately
# via `python3 tests/smoke_test_layer_b.py` on alma2.
#
# Usage:
#   cd cpp && bash tests/run_cpp_tests.sh
#   # or from repo root:
#   bash cpp/tests/run_cpp_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== C++ Evaluator Tests (Phase 13.18.GB) ==="
echo "  working directory: $CPP_DIR"

cd "$CPP_DIR"

# Step 1: compile Layer A
echo ""
echo "--- Compiling Layer A (cli_runner) ---"
make layer_a_lookup

# Step 2: WASM lint check
echo ""
echo "--- WASM lint ---"
make wasm_lint

# Step 3: run all Layer A pytest tests
echo ""
echo "--- Running pytest ---"
pytest tests/test_layer_a_lookup.py \
       tests/test_layer_a_linear.py \
       tests/test_layer_a_safety.py \
       tests/test_dfGB_roundtrip.py \
       -v

echo ""
echo "=== C++ Evaluator Tests: DONE ==="

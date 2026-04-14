# cpp/Makefile — Phase 13.18.GB Layer A + build infrastructure.
#
# Turn 3 targets:
#   layer_a_lookup — compiles cli_runner binary for subprocess tests
#   wasm_lint      — grep-check for forbidden headers in Layer A
#   clean          — remove build artifacts
#
# Layer B (ROOT-dependent) targets arrive in Turn 6.
#
# Compile flags per proposal v1.1 §7 Turn 3 + P2-4 (Claude21).

CXX      ?= g++
CXXSTD   ?= -std=c++17
# -ffp-contract=off: disable FMA contraction so float64 accumulation
# matches Python/numpy bit-exactly. Without this, g++ -O2 fuses
# `a*b + c*d` into fma(c,d, a*b), producing 1-ULP divergence on
# fixtures with multiple predictors (F_12, etc.). Required for
# Lookup integer-exact Safety contract per proposal v1.1 §6.3.
CXXFLAGS ?= $(CXXSTD) -O2 -ffp-contract=off \
            -Wall -Wextra -Werror \
            -Wshadow -Wconversion -Wsign-conversion
INCLUDES  = -Iinclude -Isrc

BUILD_DIR = build

LAYER_A_SRCS = src/gbe_kernel.cpp src/json_reader.cpp src/cli_runner.cpp
LAYER_A_OBJS = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(LAYER_A_SRCS))

LAYER_A_BIN = $(BUILD_DIR)/cli_runner

.PHONY: all layer_a_lookup wasm_lint clean

all: layer_a_lookup

layer_a_lookup: $(LAYER_A_BIN)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LAYER_A_BIN): $(LAYER_A_OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LAYER_A_OBJS) -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# -------- WASM safety grep check --------
#
# Layer A (gbe_kernel.cpp/.hpp, json_reader.cpp/.hpp) must not include
# any ROOT headers, filesystem access, threading, or dynamic loading.
# This grep check is a cheap dev-time gate; real emscripten compile at
# phase end is the acceptance gate (proposal v1.1 §9).
#
# Forbidden patterns: checked against Layer A sources only
# (cli_runner.cpp excluded — it uses <iostream> at entry/exit which
# emscripten supports, but the kernel itself must not).

WASM_LINT_SRCS = include/gbe_kernel.hpp src/gbe_kernel.cpp \
                 src/json_reader.hpp src/json_reader.cpp

WASM_FORBIDDEN_HEADERS = \
  TObject TString TTree TFile TBranch TObjString \
  dlfcn.h unistd.h sys/stat.h sys/mman.h \
  filesystem thread mutex atomic

wasm_lint:
	@echo "wasm_lint: scanning Layer A sources for forbidden headers"
	@fail=0; \
	for pat in $(WASM_FORBIDDEN_HEADERS); do \
	  if grep -nE "^\s*#\s*include\s*[<\"].*$$pat" $(WASM_LINT_SRCS) 2>/dev/null; then \
	    echo "WASM-LINT FAIL: forbidden header matching '$$pat' found above"; \
	    fail=1; \
	  fi; \
	done; \
	if [ $$fail -eq 0 ]; then echo "wasm_lint: OK"; exit 0; else exit 1; fi

clean:
	rm -rf $(BUILD_DIR)

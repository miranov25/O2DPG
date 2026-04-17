# cpp/Makefile — Phase 13.18.GB Layer A + Layer B build.
#
# Targets:
#   layer_a_lookup                    — Turn 3: cli_runner subprocess binary
#   wasm_lint                          — Turn 3: forbidden-header grep check
#   layer_b                            — Turn 6: shared library for PyROOT/ROOT
#   test                               — Turn 6: PyROOT smoke test
#   clean                              — remove build artifacts

CXX       ?= g++
CXXSTD    ?= -std=c++17

# Strict flags for Layer A (kernel + json_reader + cli_runner)
CXXFLAGS_STRICT ?= $(CXXSTD) -O2 -ffp-contract=off \
                   -Wall -Wextra -Werror \
                   -Wshadow -Wconversion -Wsign-conversion

# Relaxed flags for Layer B (ROOT-dependent sources)
CXXFLAGS_LAYERB ?= $(CXXSTD) -O2 -ffp-contract=off -fPIC -Wall

# -I. is needed so the rootcling-generated dictionary's
# `#include "include/gbe_root_io.h"` resolves from CWD.
INCLUDES = -I. -Iinclude -Isrc

BUILD_DIR = build

# -------- Layer A --------
LAYER_A_KERNEL_SRCS = src/gbe_kernel.cpp src/json_reader.cpp
LAYER_A_BIN_SRCS = $(LAYER_A_KERNEL_SRCS) src/cli_runner.cpp
LAYER_A_BIN_OBJS = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(LAYER_A_BIN_SRCS))
LAYER_A_BIN = $(BUILD_DIR)/cli_runner

# Layer A kernel objects with -fPIC for linking into the .so.
LAYER_A_PIC_OBJS = $(patsubst src/%.cpp,$(BUILD_DIR)/%.pic.o,$(LAYER_A_KERNEL_SRCS))

# -------- Layer B (ROOT) --------
ROOTCLING   = rootcling
ROOT_CFLAGS = $(shell root-config --cflags 2>/dev/null)
ROOT_LIBS   = $(shell root-config --libs   2>/dev/null)
ROOT_LIBDIR = $(shell root-config --libdir 2>/dev/null)
ROOT_INC_FLAGS = $(addprefix -I,$(subst :, ,$(ROOT_INCLUDE_PATH)))

LAYER_B_HEADER  = include/gbe_root_io.h
LAYER_B_SRC     = src/gbe_root_io.cxx
LAYER_B_OBJ     = $(BUILD_DIR)/gbe_root_io.o
LINKDEF         = LinkDef.h

DICT_SRC        = $(BUILD_DIR)/GBE_Dict.cxx
DICT_PCM        = $(BUILD_DIR)/GBE_Dict_rdict.pcm
LIB_TARGET      = libGroupByRegressionEvaluator.so

.PHONY: all layer_a_lookup wasm_lint layer_b test show-env clean check-root

all: layer_a_lookup

layer_a_lookup: $(LAYER_A_BIN)

# Layer A object compilation rule (strict flags).
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_STRICT) $(INCLUDES) -c $< -o $@

# Layer A kernel objects compiled with -fPIC for the .so.
$(BUILD_DIR)/%.pic.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_STRICT) -fPIC $(INCLUDES) -c $< -o $@

$(LAYER_A_BIN): $(LAYER_A_BIN_OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_STRICT) $(LAYER_A_BIN_OBJS) -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# -------- WASM safety grep check (Turn 3) --------
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

# -------- Layer B (Turn 6) --------

check-root:
	@if [ -z "$(ROOT_CFLAGS)" ]; then \
	  echo "ERROR: root-config not found in PATH. Source ROOT before building Layer B."; \
	  exit 1; \
	fi
	@echo "ROOT OK: $(shell root-config --version)"

show-env:
	@echo "CXX           = $(CXX)"
	@echo "ROOT version  = $(shell root-config --version 2>/dev/null || echo NOT FOUND)"
	@echo "ROOT_CFLAGS   = $(ROOT_CFLAGS)"
	@echo "ROOT_LIBS     = $(ROOT_LIBS)"
	@echo "ROOT_LIBDIR   = $(ROOT_LIBDIR)"
	@echo "ROOT_INCLUDE_PATH (count) = $(words $(subst :, ,$(ROOT_INCLUDE_PATH)))"

# Generate ROOT dictionary from headers + LinkDef.
$(DICT_SRC): $(LAYER_B_HEADER) include/gbe_kernel.hpp $(LINKDEF) | $(BUILD_DIR)
	@echo "Generating ROOT dictionary..."
	$(ROOTCLING) -f $@ $(INCLUDES) $(ROOT_INC_FLAGS) \
	    $(LAYER_B_HEADER) include/gbe_kernel.hpp $(LINKDEF)
	@if [ ! -f $(DICT_PCM) ]; then \
	  echo "ERROR: rootcling did not generate $(DICT_PCM)"; \
	  exit 1; \
	fi
	@echo "  Generated: $(DICT_SRC) + $(DICT_PCM)"

# Compile Layer B with relaxed flags (ROOT headers trip Wconversion).
$(LAYER_B_OBJ): $(LAYER_B_SRC) $(LAYER_B_HEADER) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_LAYERB) $(INCLUDES) $(ROOT_CFLAGS) $(ROOT_INC_FLAGS) \
	    -c $< -o $@

# Final shared library: Layer B obj + Layer A PIC objs + dictionary.
$(LIB_TARGET): check-root $(LAYER_B_OBJ) $(LAYER_A_PIC_OBJS) $(DICT_SRC)
	@echo "Building $(LIB_TARGET)..."
	$(CXX) $(CXXFLAGS_LAYERB) -shared $(INCLUDES) $(ROOT_CFLAGS) $(ROOT_INC_FLAGS) \
	    -o $@ \
	    $(DICT_SRC) $(LAYER_B_OBJ) $(LAYER_A_PIC_OBJS) \
	    -L$(ROOT_LIBDIR) $(ROOT_LIBS) \
	    -Wl,-rpath,$(ROOT_LIBDIR)
	@cp -f $(DICT_PCM) ./GBE_Dict_rdict.pcm
	@echo "Built: $(LIB_TARGET) + ./GBE_Dict_rdict.pcm"

# Phony alias for the library target.
layer_b: $(LIB_TARGET)

# PyROOT smoke test that the library loads and symbols are visible.
test: $(LIB_TARGET)
	@echo "Testing library load..."
	@python3 -c "import ROOT, sys; \
r = ROOT.gSystem.Load('./$(LIB_TARGET)'); \
print(f'  Load: {r}'); \
sys.exit(1) if r < 0 else None; \
print(f'  has_model(\"x\"): {ROOT.GBE.has_model(\"x\")}'); \
print(f'  list_models() len: {len(list(ROOT.GBE.list_models()))}'); \
print('All smoke tests passed')"

clean:
	rm -rf $(BUILD_DIR) $(LIB_TARGET) GBE_Dict_rdict.pcm

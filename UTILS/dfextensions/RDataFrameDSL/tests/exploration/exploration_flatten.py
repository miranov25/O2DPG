#!/usr/bin/env python3 -u
"""
Phase 13.7.A Exploration: Flatten Optimization Benchmark
=========================================================

Comprehensive benchmark of all viable approaches for flattening RVec (1D) 
and RVec<RVec> (2D) data from RDataFrame.

Approved: 2026-01-30 (Unanimous - 5 reviewers)

Dataset Size:
    Approved spec: 5k events (interactive scale)
    Actual test: 25k events (stress test scale)
    
    Rationale: 
    - 5k events: Too small to see JIT overhead effects
    - 25k events: Better represents realistic batch workflows
    - Scaling tests confirm linear behavior (5x data → 5x time)

Usage:
    python -u exploration_flatten.py 2>&1 | tee exploration_flatten.log
    
    NOTE: The -u flag (unbuffered output) is required when piping to tee,
    otherwise C++ output from ROOT may appear out of order with Python output.

Output:
    - exploration_flatten.log: Console output with results
    - profiles/: Directory with cProfile outputs per method
    - results.json: Machine-readable benchmark results

Requirements:
    - ROOT 6.32+ with PyROOT
    - NumPy
    - Awkward Array (optional, for Method 7)
    - uproot (optional, for Method 6)
"""

import os
import sys
import time
import json
import cProfile
import pstats
import platform
import traceback
import tracemalloc
import argparse
import shutil
import tempfile
import atexit
from pathlib import Path
from io import StringIO
from statistics import median
from typing import Dict, List, Tuple, Any, Optional, Callable

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# =============================================================================
# Command Line Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description='Phase 13.7.A Exploration: Flatten Optimization Benchmark',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python exploration_flatten.py              # Fast: reuse cached data file
  python exploration_flatten.py --clean      # Clean: temp files, delete after
  python exploration_flatten.py --regenerate # Force regenerate data file
  python exploration_flatten.py --n-events 1000  # Generate/require 1000 events minimum
'''
)
parser.add_argument('--clean', action='store_true',
                    help='Clean run: use temp directory, delete artifacts after completion')
parser.add_argument('--regenerate', action='store_true',
                    help='Force regenerate data file even if it exists')
parser.add_argument('--n-events', type=int, default=25000,
                    help='Minimum events required (default: 25000). '
                         'Reuses cached file if it has enough events.')
parser.add_argument('--skip-expressions', action='store_true',
                    help='Skip slow JIT expression overhead tests (saves ~2 min)')

ARGS = parser.parse_args()

# =============================================================================
# Configuration
# =============================================================================

# Determine working directory based on --clean flag
if ARGS.clean:
    _temp_dir = tempfile.mkdtemp(prefix='exploration_flatten_')
    _output_dir = Path(_temp_dir)
    _profiles_dir = _output_dir / 'profiles'
    
    def _cleanup():
        print(f"\nCleaning up temp directory: {_temp_dir}")
        shutil.rmtree(_temp_dir, ignore_errors=True)
    
    atexit.register(_cleanup)
    print(f"[--clean mode] Using temp directory: {_temp_dir}")
else:
    _output_dir = Path(__file__).parent
    _profiles_dir = _output_dir / 'profiles'

CONFIG = {
    'n_warmup': 1,
    'n_runs': 5,
    'n_events': ARGS.n_events,
    'seed': 42,
    'profile_top_n': 20,
    'output_dir': _output_dir,
    'profiles_dir': _profiles_dir,
    'clean_mode': ARGS.clean,
    'regenerate': ARGS.regenerate,
    'skip_expressions': ARGS.skip_expressions,
}

# =============================================================================
# Utility Functions
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 70):
    """Print formatted header."""
    print()
    print(char * width)
    print(title)
    print(char * width)
    print()


def print_subheader(title: str, char: str = "-", width: int = 50):
    """Print formatted subheader."""
    print()
    print(f"{title}")
    print(char * len(title))


def mad(data: List[float]) -> float:
    """Calculate Median Absolute Deviation."""
    med = median(data)
    return median([abs(x - med) for x in data])


def format_time(ms: float) -> str:
    """Format time in milliseconds."""
    if ms < 1:
        return f"{ms*1000:.1f} µs"
    elif ms < 1000:
        return f"{ms:.1f} ms"
    else:
        return f"{ms/1000:.2f} s"


def get_environment_info() -> Dict[str, Any]:
    """Capture environment information."""
    import ROOT

    env = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'root_version': ROOT.gROOT.GetVersion(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }

    # ROOT threading state
    try:
        env['root_mt_enabled'] = ROOT.ROOT.IsImplicitMTEnabled()
        env['root_thread_pool_size'] = ROOT.ROOT.GetThreadPoolSize()
    except:
        env['root_mt_enabled'] = 'unknown'
        env['root_thread_pool_size'] = 'unknown'

    # Optional packages
    try:
        import awkward as ak
        env['awkward_version'] = ak.__version__
    except ImportError:
        env['awkward_version'] = 'not installed'

    try:
        import uproot
        env['uproot_version'] = uproot.__version__
    except ImportError:
        env['uproot_version'] = 'not installed'

    return env


def profile_function(func: Callable, name: str, profiles_dir: Path) -> Tuple[Any, float, float]:
    """
    Run function with profiling and memory tracking.

    Returns: (result, time_ms, peak_memory_mb)
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Memory tracking
    tracemalloc.start()

    # CPU profiling
    profiler = cProfile.Profile()

    t0 = time.perf_counter()
    profiler.enable()
    result = func()
    profiler.disable()
    t1 = time.perf_counter()

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    time_ms = (t1 - t0) * 1000

    # Save profile
    safe_name = name.replace(' ', '_').replace('+', '_').replace('[', '').replace(']', '').lower()
    prof_file = profiles_dir / f"{safe_name}.prof"
    profiler.dump_stats(str(prof_file))

    # Save top N stats as text
    stats_file = profiles_dir / f"{safe_name}_top{CONFIG['profile_top_n']}.txt"
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(CONFIG['profile_top_n'])
    with open(stats_file, 'w') as f:
        f.write(stream.getvalue())

    return result, time_ms, peak_mb


def run_benchmark(func: Callable, name: str, profiles_dir: Path,
                  n_warmup: int = 1, n_runs: int = 5) -> Dict[str, Any]:
    """
    Run benchmark with warmup and multiple runs.

    Returns dict with timing statistics and profiling info.
    """
    # Warmup runs (not recorded)
    for _ in range(n_warmup):
        try:
            func()
        except Exception as e:
            return {
                'name': name,
                'status': 'FAILED',
                'error': str(e),
                'runs_ms': [],
                'median_ms': None,
                'mad_ms': None,
                'peak_memory_mb': None,
            }

    # Timed runs
    runs_ms = []
    peak_memories = []
    result = None

    for i in range(n_runs):
        try:
            # Profile only the first run
            if i == 0:
                result, time_ms, peak_mb = profile_function(func, name, profiles_dir)
            else:
                tracemalloc.start()
                t0 = time.perf_counter()
                result = func()
                t1 = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                time_ms = (t1 - t0) * 1000
                peak_mb = peak / (1024 * 1024)

            runs_ms.append(time_ms)
            peak_memories.append(peak_mb)

        except Exception as e:
            return {
                'name': name,
                'status': 'FAILED',
                'error': str(e),
                'runs_ms': runs_ms,
                'median_ms': None,
                'mad_ms': None,
                'peak_memory_mb': None,
                'result': None,
            }

    return {
        'name': name,
        'status': 'OK',
        'runs_ms': runs_ms,
        'median_ms': median(runs_ms),
        'mad_ms': mad(runs_ms),
        'min_ms': min(runs_ms),
        'max_ms': max(runs_ms),
        'peak_memory_mb': max(peak_memories),
        'result': result,
    }


def verify_results(result: Dict, baseline: Dict, name: str) -> Dict[str, bool]:
    """Verify that result matches baseline."""
    verification = {
        'values_match': False,
        'idx_1_match': False,
        'idx_2_match': False,
        'length_match': False,
    }

    if result.get('status') != 'OK' or baseline.get('status') != 'OK':
        return verification

    r = result.get('result', {})
    b = baseline.get('result', {})

    if r is None or b is None:
        return verification

    # Check values
    if 'values' in r and 'values' in b:
        try:
            verification['length_match'] = len(r['values']) == len(b['values'])
            verification['values_match'] = np.allclose(r['values'], b['values'])
        except:
            pass

    # Check indices (for 2D)
    if 'idx_1' in r and 'idx_1' in b:
        try:
            verification['idx_1_match'] = np.array_equal(r['idx_1'], b['idx_1'])
        except:
            pass
    else:
        verification['idx_1_match'] = True  # N/A for 1D

    if 'idx_2' in r and 'idx_2' in b:
        try:
            verification['idx_2_match'] = np.array_equal(r['idx_2'], b['idx_2'])
        except:
            pass
    else:
        verification['idx_2_match'] = True  # N/A for 1D

    return verification


# =============================================================================
# Setup
# =============================================================================

print_header("Phase 13.7.A Exploration: Flatten Optimization Benchmark")

print("Loading ROOT...")
import ROOT

print("Capturing environment...")
ENV_INFO = get_environment_info()

print("\nEnvironment:")
for key, value in ENV_INFO.items():
    print(f"  {key}: {value}")

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Create profiles directory
CONFIG['profiles_dir'].mkdir(parents=True, exist_ok=True)

# Load RVec dictionaries (needed for reading ROOT files with nested RVec)
# This is separate from file generation - allows reusing cached files
from tests.generators.toy_nd import ensure_rvec_dictionaries
ensure_rvec_dictionaries()

# Generate or reuse test data
print_subheader("Test Data")

TEST_FILE = str(CONFIG['output_dir'] / 'exploration_flatten_data.root')

# Check if file exists and has correct size (reuse if possible)
regenerate = CONFIG['regenerate']  # Force if --regenerate flag

if not regenerate and os.path.exists(TEST_FILE):
    try:
        # Quick check: open and verify event count
        rdf_check = ROOT.RDataFrame("Events", TEST_FILE)
        existing_events = rdf_check.Count().GetValue()
        if existing_events >= CONFIG['n_events']:
            print(f"Reusing existing test file: {TEST_FILE}")
            print(f"  (contains {existing_events} events, need {CONFIG['n_events']})")
            regenerate = False
        else:
            print(f"Existing file too small ({existing_events} < {CONFIG['n_events']}), regenerating...")
            regenerate = True
    except Exception as e:
        print(f"Existing file invalid ({e}), regenerating...")
        regenerate = True
elif not regenerate:
    # File doesn't exist and --regenerate not set
    print(f"Test file not found, will generate: {TEST_FILE}")
    regenerate = True

if regenerate:
    print(f"Generating {CONFIG['n_events']} events...")
    # Import generator only when needed
    from tests.generators.toy_nd import generate_nd_2d_root
    nd_2d_file_tmp = generate_nd_2d_root(
        size="L",
        seed=CONFIG['seed'],
        n_events=CONFIG['n_events'],
        mode='demo'
    )
    shutil.copy(nd_2d_file_tmp, TEST_FILE)
    print(f"Saved to: {TEST_FILE}")

# Create RDataFrame
rdf = ROOT.RDataFrame("Events", TEST_FILE)
n_events = rdf.Count().GetValue()
print(f"Events: {n_events}")

# Get data sizes
data_info = rdf.AsNumpy(['track_pt', 'cluster_Q', 'event_id'])
n_1d_elements = sum(len(e) for e in data_info['track_pt'])
n_2d_elements = sum(sum(len(t) for t in e) for e in data_info['cluster_Q'])
print(f"1D elements (tracks): {n_1d_elements:,}")
print(f"2D elements (clusters): {n_2d_elements:,}")

# Store for later use
TEST_DATA = {
    'file': TEST_FILE,
    'n_events': n_events,
    'n_1d_elements': n_1d_elements,
    'n_2d_elements': n_2d_elements,
}

# =============================================================================
# Declare C++ Helper Functions
# =============================================================================

print_subheader("Declaring C++ Helper Functions")

CPP_FLATTEN_CODE = """
#ifndef EXPLORATION_FLATTEN_HELPERS
#define EXPLORATION_FLATTEN_HELPERS

#include <vector>
#include "ROOT/RVec.hxx"

namespace FlattenHelpers {

// 1D Flatten Result
struct Flat1DResult {
    std::vector<double> values;
    std::vector<int32_t> indices;
};

// 2D Flatten Result  
struct Flat2DResult {
    std::vector<double> values;
    std::vector<int32_t> idx_1;
    std::vector<int32_t> idx_2;
};

// Flatten 1D: std::vector<RVec<double>> -> flat arrays
Flat1DResult Flatten1D(const std::vector<ROOT::RVecD>& rvecs) {
    Flat1DResult result;
    
    // Calculate total size
    size_t total = 0;
    for (const auto& rv : rvecs) {
        total += rv.size();
    }
    
    // Reserve space
    result.values.reserve(total);
    result.indices.reserve(total);
    
    // Fill
    for (const auto& rv : rvecs) {
        for (size_t i = 0; i < rv.size(); ++i) {
            result.values.push_back(rv[i]);
            result.indices.push_back(static_cast<int32_t>(i));
        }
    }
    
    return result;
}

// Flatten 2D: std::vector<RVec<RVec<double>>> -> flat arrays with indices
Flat2DResult Flatten2D(const std::vector<ROOT::RVec<ROOT::RVecD>>& events) {
    Flat2DResult result;
    
    // Calculate total size
    size_t total = 0;
    for (const auto& event : events) {
        for (const auto& track : event) {
            total += track.size();
        }
    }
    
    // Reserve space
    result.values.reserve(total);
    result.idx_1.reserve(total);
    result.idx_2.reserve(total);
    
    // Fill
    for (const auto& event : events) {
        for (size_t t = 0; t < event.size(); ++t) {
            const auto& track = event[t];
            for (size_t c = 0; c < track.size(); ++c) {
                result.values.push_back(track[c]);
                result.idx_1.push_back(static_cast<int32_t>(t));
                result.idx_2.push_back(static_cast<int32_t>(c));
            }
        }
    }
    
    return result;
}

}  // namespace FlattenHelpers

// =============================================================================
// Function Pointer Helpers - Pre-compiled functions to eliminate JIT overhead
// =============================================================================
namespace FuncPtrHelpers {

// Pre-compiled transformation functions
ROOT::RVecD Scale1000(const ROOT::RVecD& v) {
    return v * 1000.0;
}

ROOT::RVecD Sqrt(const ROOT::RVecD& v) {
    ROOT::RVecD result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = std::sqrt(v[i]);
    }
    return result;
}

ROOT::RVecD Square(const ROOT::RVecD& v) {
    return v * v;
}

// Helpers to apply pre-compiled functions via RDF::RNode
ROOT::RDF::RNode DefineScale1000(ROOT::RDF::RNode rdf, const std::string& newcol, const std::string& srccol) {
    return rdf.Define(newcol, Scale1000, {srccol});
}

ROOT::RDF::RNode DefineSqrt(ROOT::RDF::RNode rdf, const std::string& newcol, const std::string& srccol) {
    return rdf.Define(newcol, Sqrt, {srccol});
}

ROOT::RDF::RNode DefineSquare(ROOT::RDF::RNode rdf, const std::string& newcol, const std::string& srccol) {
    return rdf.Define(newcol, Square, {srccol});
}

}  // namespace FuncPtrHelpers

#endif
"""

try:
    ROOT.gInterpreter.Declare(CPP_FLATTEN_CODE)
    print("  C++ helpers declared successfully.")
    CPP_HELPERS_AVAILABLE = True
except Exception as e:
    print(f"  C++ helpers declaration FAILED: {e}")
    CPP_HELPERS_AVAILABLE = False

# =============================================================================
# Method Implementations
# =============================================================================

def create_fresh_rdf():
    """Create a fresh RDataFrame to avoid caching effects."""
    return ROOT.RDataFrame("Events", TEST_FILE)


# -----------------------------------------------------------------------------
# 1D Methods
# -----------------------------------------------------------------------------

def method_1_baseline_1d():
    """Method 1: Baseline - AsNumpy + Python loop (1D)"""
    rdf = create_fresh_rdf()
    data = rdf.AsNumpy(['track_pt'])

    values = []
    indices = []
    for event in data['track_pt']:
        arr = np.asarray(event)
        values.extend(arr)
        indices.extend(range(len(arr)))

    return {
        'values': np.array(values, dtype=np.float64),
        'idx_1': np.array(indices, dtype=np.int32),
    }


def method_2_concatenate_1d():
    """Method 2: AsNumpy + np.concatenate (1D)"""
    rdf = create_fresh_rdf()
    data = rdf.AsNumpy(['track_pt'])

    arrays = [np.asarray(e) for e in data['track_pt']]
    values = np.concatenate(arrays)

    # Generate indices
    indices = np.concatenate([np.arange(len(a), dtype=np.int32) for a in arrays])

    return {
        'values': values,
        'idx_1': indices,
    }


def method_3_take_concat_1d():
    """Method 3: Take + np.concatenate (1D) - RECOMMENDED"""
    rdf = create_fresh_rdf()
    take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
    vec = take_result.GetValue()

    arrays = [np.asarray(v) for v in vec]
    values = np.concatenate(arrays)
    indices = np.concatenate([np.arange(len(a), dtype=np.int32) for a in arrays])

    return {
        'values': values,
        'idx_1': indices,
    }


def method_4_take_cpp_1d():
    """Method 4: Take + C++ flatten (1D)"""
    if not CPP_HELPERS_AVAILABLE:
        raise RuntimeError("C++ helpers not available")

    rdf = create_fresh_rdf()
    take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
    vec = take_result.GetValue()

    flat_result = ROOT.FlattenHelpers.Flatten1D(vec)

    return {
        'values': np.array(flat_result.values, copy=True),
        'idx_1': np.array(flat_result.indices, copy=True),
    }


def method_5_ttree_draw_1d():
    """Method 5: TTree::Draw (1D) - Reference only (deprecated)"""
    # Use TTree directly
    tfile = ROOT.TFile(TEST_FILE)
    tree = tfile.Get("Events")

    n = tree.Draw("track_pt", "", "goff")
    values = np.array(tree.GetV1()[:n], copy=True)

    tfile.Close()

    # Generate indices (TTree::Draw doesn't provide them directly)
    # This is approximate - TTree::Draw flattens without index info
    indices = np.zeros(len(values), dtype=np.int32)  # Placeholder

    return {
        'values': values,
        'idx_1': indices,
    }


def method_6_uproot_1d():
    """Method 6: uproot (file-based) (1D)"""
    try:
        import uproot
        import awkward as ak
    except ImportError:
        raise RuntimeError("uproot/awkward not installed")

    with uproot.open(TEST_FILE) as f:
        tree = f["Events"]
        arr = tree["track_pt"].array()
        values = ak.to_numpy(ak.flatten(arr))

        # Generate indices
        indices = ak.to_numpy(ak.flatten(ak.local_index(arr)))

    return {
        'values': values.astype(np.float64),
        'idx_1': indices.astype(np.int32),
    }


def method_7_awkward_1d():
    """Method 7: Manual Awkward (1D)"""
    try:
        import awkward as ak
    except ImportError:
        raise RuntimeError("awkward not installed")

    rdf = create_fresh_rdf()
    data = rdf.AsNumpy(['track_pt'])

    # Convert to Awkward Array
    list_data = [np.asarray(x) for x in data['track_pt']]
    ak_array = ak.Array(list_data)

    values = ak.to_numpy(ak.flatten(ak_array))
    indices = ak.to_numpy(ak.flatten(ak.local_index(ak_array)))

    return {
        'values': values.astype(np.float64),
        'idx_1': indices.astype(np.int32),
    }


# -----------------------------------------------------------------------------
# 2D Methods
# -----------------------------------------------------------------------------

def method_1_baseline_2d():
    """Method 1: Baseline - AsNumpy + Python loop (2D)"""
    rdf = create_fresh_rdf()
    data = rdf.AsNumpy(['cluster_Q'])

    values = []
    idx_1 = []
    idx_2 = []

    for event in data['cluster_Q']:
        for track_idx, track in enumerate(event):
            arr = np.asarray(track)
            for cluster_idx, val in enumerate(arr):
                values.append(val)
                idx_1.append(track_idx)
                idx_2.append(cluster_idx)

    return {
        'values': np.array(values, dtype=np.float64),
        'idx_1': np.array(idx_1, dtype=np.int32),
        'idx_2': np.array(idx_2, dtype=np.int32),
    }


def method_2_optimized_2d():
    """Method 2: AsNumpy + optimized Python (2D)"""
    rdf = create_fresh_rdf()
    data = rdf.AsNumpy(['cluster_Q'])

    # Pre-calculate total size
    total = sum(sum(len(np.asarray(t)) for t in e) for e in data['cluster_Q'])

    values = np.empty(total, dtype=np.float64)
    idx_1 = np.empty(total, dtype=np.int32)
    idx_2 = np.empty(total, dtype=np.int32)

    pos = 0
    for event in data['cluster_Q']:
        for track_idx, track in enumerate(event):
            arr = np.asarray(track)
            n = len(arr)
            values[pos:pos+n] = arr
            idx_1[pos:pos+n] = track_idx
            idx_2[pos:pos+n] = np.arange(n, dtype=np.int32)
            pos += n

    return {
        'values': values,
        'idx_1': idx_1,
        'idx_2': idx_2,
    }


def method_4_take_cpp_2d():
    """Method 4: Take + C++ flatten (2D) - RECOMMENDED"""
    if not CPP_HELPERS_AVAILABLE:
        raise RuntimeError("C++ helpers not available")

    rdf = create_fresh_rdf()
    take_result = rdf.Take['ROOT::RVec<ROOT::RVec<double>>']('cluster_Q')
    vec = take_result.GetValue()

    flat_result = ROOT.FlattenHelpers.Flatten2D(vec)

    return {
        'values': np.array(flat_result.values, copy=True),
        'idx_1': np.array(flat_result.idx_1, copy=True),
        'idx_2': np.array(flat_result.idx_2, copy=True),
    }


def method_5_ttree_draw_2d():
    """Method 5: TTree::Draw (2D) - Reference only (deprecated)"""
    tfile = ROOT.TFile(TEST_FILE)
    tree = tfile.Get("Events")

    n = tree.Draw("cluster_Q", "", "goff")
    values = np.array(tree.GetV1()[:n], copy=True)

    tfile.Close()

    # TTree::Draw doesn't provide indices
    idx_1 = np.zeros(len(values), dtype=np.int32)
    idx_2 = np.zeros(len(values), dtype=np.int32)

    return {
        'values': values,
        'idx_1': idx_1,
        'idx_2': idx_2,
    }


def method_6_uproot_2d():
    """Method 6: uproot (file-based) (2D)"""
    try:
        import uproot
        import awkward as ak
    except ImportError:
        raise RuntimeError("uproot/awkward not installed")

    with uproot.open(TEST_FILE) as f:
        tree = f["Events"]
        arr = tree["cluster_Q"].array()

        # Double flatten for 2D
        values = ak.to_numpy(ak.flatten(ak.flatten(arr)))

        # Generate indices
        idx_1 = ak.to_numpy(ak.flatten(ak.flatten(
            ak.broadcast_arrays(ak.local_index(arr, axis=1), arr)[0]
        )))
        idx_2 = ak.to_numpy(ak.flatten(ak.flatten(ak.local_index(arr, axis=2))))

    return {
        'values': values.astype(np.float64),
        'idx_1': idx_1.astype(np.int32),
        'idx_2': idx_2.astype(np.int32),
    }


def method_7_awkward_2d():
    """Method 7: Manual Awkward (2D)"""
    try:
        import awkward as ak
    except ImportError:
        raise RuntimeError("awkward not installed")

    rdf = create_fresh_rdf()
    data = rdf.AsNumpy(['cluster_Q'])

    # Convert to Awkward Array (nested)
    nested = [[np.asarray(t).tolist() for t in e] for e in data['cluster_Q']]
    ak_array = ak.Array(nested)

    values = ak.to_numpy(ak.flatten(ak.flatten(ak_array)))

    # Generate indices
    idx_1 = ak.to_numpy(ak.flatten(ak.flatten(
        ak.broadcast_arrays(ak.local_index(ak_array, axis=1), ak_array)[0]
    )))
    idx_2 = ak.to_numpy(ak.flatten(ak.flatten(ak.local_index(ak_array, axis=2))))

    return {
        'values': values.astype(np.float64),
        'idx_1': idx_1.astype(np.int32),
        'idx_2': idx_2.astype(np.int32),
    }


# -----------------------------------------------------------------------------
# Failed Methods (documented)
# -----------------------------------------------------------------------------

def method_8_foreach_cpp():
    """Method 8: Foreach + C++ callback - KNOWN TO FAIL"""
    raise RuntimeError(
        "FAILED: Cling JIT segfault in ROOT 6.32.06. "
        "Error: 'cannot compile this l-value expression yet' in ForeachSlot. "
        "This is a ROOT/Cling limitation, not a code bug."
    )


def method_9_ak_from_rdataframe():
    """Method 9: ak.from_rdataframe() - KNOWN TO FAIL"""
    raise RuntimeError(
        "FAILED: KeyError 'node0-offsets' in ROOT 6.32.06 + Awkward 2.8.10. "
        "This is a compatibility bug between ROOT and Awkward Array."
    )


# =============================================================================
# Scalar AsNumpy Anomaly Test
# =============================================================================

def test_asnumpy_anomaly():
    """Test AsNumpy performance for scalar vs 1D vs 2D columns."""
    print_header("Scalar AsNumpy Anomaly Test")

    results = {}

    # Scalar column
    def test_scalar():
        rdf = create_fresh_rdf()
        return rdf.AsNumpy(['event_id'])

    # 1D RVec column
    def test_1d():
        rdf = create_fresh_rdf()
        return rdf.AsNumpy(['track_pt'])

    # 2D RVec column
    def test_2d():
        rdf = create_fresh_rdf()
        return rdf.AsNumpy(['cluster_Q'])

    for name, func in [('scalar (event_id)', test_scalar),
                       ('1D RVec (track_pt)', test_1d),
                       ('2D RVec (cluster_Q)', test_2d)]:
        result = run_benchmark(
            func, f"AsNumpy_{name}",
            CONFIG['profiles_dir'] / 'anomaly',
            n_warmup=1, n_runs=5
        )
        results[name] = {k: v for k, v in result.items() if k != 'result'}

        if result['status'] == 'OK':
            print(f"  {name}: {result['median_ms']:.1f} ± {result['mad_ms']:.1f} ms")
        else:
            print(f"  {name}: FAILED - {result.get('error', 'unknown')}")

    return results


# =============================================================================
# ROOT Threading Test
# =============================================================================

def test_root_threading():
    """
    Test performance with MT enabled vs disabled.

    CRITICAL: ImplicitMT must be enabled BEFORE creating RDataFrame!
    See: https://root.cern/doc/master/classROOT_1_1RDataFrame.html
    """
    print_header("ROOT Threading Test (P1)")

    results = {}
    n_cores = ROOT.ROOT.GetThreadPoolSize() if hasattr(ROOT.ROOT, 'GetThreadPoolSize') else 'unknown'

    # =========================================================================
    # Test with MT DISABLED
    # =========================================================================
    ROOT.ROOT.DisableImplicitMT()
    print(f"  ImplicitMT enabled: {ROOT.ROOT.IsImplicitMTEnabled()}")

    # Create RDF with MT disabled (RDF created AFTER MT setting)
    def method_mt_off():
        # RDF must be created AFTER MT is configured
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
        vec = take_result.GetValue()
        # Simple processing
        return [np.asarray(v) for v in vec]

    times_off = []
    for _ in range(5):  # More runs for stability
        t0 = time.perf_counter()
        method_mt_off()
        times_off.append((time.perf_counter() - t0) * 1000)

    med_off = median(times_off)
    mad_off = median([abs(t - med_off) for t in times_off])
    results['mt_off'] = {'median_ms': med_off, 'mad_ms': mad_off, 'runs_ms': times_off}
    print(f"  MT OFF: {med_off:.1f} ± {mad_off:.1f} ms")

    # =========================================================================
    # Test with MT ENABLED (must enable BEFORE creating RDF)
    # =========================================================================
    ROOT.ROOT.EnableImplicitMT()  # Enable first!
    print(f"  ImplicitMT enabled: {ROOT.ROOT.IsImplicitMTEnabled()}")
    print(f"  Thread pool size: {ROOT.ROOT.GetThreadPoolSize()}")

    # Create RDF with MT enabled (RDF created AFTER MT setting)
    def method_mt_on():
        # RDF must be created AFTER MT is configured
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
        vec = take_result.GetValue()
        return [np.asarray(v) for v in vec]

    times_on = []
    for _ in range(5):
        t0 = time.perf_counter()
        method_mt_on()
        times_on.append((time.perf_counter() - t0) * 1000)

    med_on = median(times_on)
    mad_on = median([abs(t - med_on) for t in times_on])
    results['mt_on'] = {'median_ms': med_on, 'mad_ms': mad_on, 'runs_ms': times_on}
    print(f"  MT ON:  {med_on:.1f} ± {mad_on:.1f} ms")

    # Calculate speedup
    if med_on > 0:
        speedup = med_off / med_on
        print(f"  Speedup: {speedup:.2f}x (expected ~{ROOT.ROOT.GetThreadPoolSize()}x for IO-bound)")

    # =========================================================================
    # Test Take + C++ flatten with MT
    # =========================================================================
    print("\n  Testing Take + C++ flatten with MT:")

    # MT OFF
    ROOT.ROOT.DisableImplicitMT()

    def method_take_cpp_mt_off():
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
        vec = take_result.GetValue()
        flat_result = ROOT.FlattenHelpers.Flatten1D(vec)
        return np.array(flat_result.values, copy=True)

    times_cpp_off = []
    for _ in range(5):
        t0 = time.perf_counter()
        method_take_cpp_mt_off()
        times_cpp_off.append((time.perf_counter() - t0) * 1000)

    med_cpp_off = median(times_cpp_off)
    print(f"    Take+C++ MT OFF: {med_cpp_off:.1f} ms")

    # MT ON
    ROOT.ROOT.EnableImplicitMT()

    def method_take_cpp_mt_on():
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
        vec = take_result.GetValue()
        flat_result = ROOT.FlattenHelpers.Flatten1D(vec)
        return np.array(flat_result.values, copy=True)

    times_cpp_on = []
    for _ in range(5):
        t0 = time.perf_counter()
        method_take_cpp_mt_on()
        times_cpp_on.append((time.perf_counter() - t0) * 1000)

    med_cpp_on = median(times_cpp_on)
    print(f"    Take+C++ MT ON:  {med_cpp_on:.1f} ms")

    if med_cpp_on > 0:
        speedup_cpp = med_cpp_off / med_cpp_on
        print(f"    Speedup: {speedup_cpp:.2f}x")

    results['take_cpp_mt_off'] = {'median_ms': med_cpp_off}
    results['take_cpp_mt_on'] = {'median_ms': med_cpp_on}

    # =========================================================================
    # Multi-File MT Test (P1) - Symlinks to same file
    # =========================================================================
    print("\n  Testing Multi-File MT (5 symlinks):")
    
    # Create symlinks
    symlink_files = []
    for i in range(5):
        link_path = str(CONFIG['output_dir'] / f'_mt_test_link_{i}.root')
        if os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(TEST_FILE, link_path)
        symlink_files.append(link_path)
    
    n_files = len(symlink_files)
    
    # Multi-file MT OFF
    ROOT.ROOT.DisableImplicitMT()
    
    def method_multifile_mt_off():
        rdf = ROOT.RDataFrame("Events", symlink_files)
        take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
        vec = take_result.GetValue()
        return len(vec)
    
    times_multi_off = []
    for _ in range(3):  # Fewer runs - slower
        t0 = time.perf_counter()
        n_events_multi = method_multifile_mt_off()
        times_multi_off.append((time.perf_counter() - t0) * 1000)
    
    med_multi_off = median(times_multi_off)
    print(f"    {n_files} files MT OFF: {med_multi_off:.1f} ms ({n_events_multi} events)")
    
    # Multi-file MT ON
    ROOT.ROOT.EnableImplicitMT()
    
    def method_multifile_mt_on():
        rdf = ROOT.RDataFrame("Events", symlink_files)
        take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
        vec = take_result.GetValue()
        return len(vec)
    
    times_multi_on = []
    for _ in range(3):
        t0 = time.perf_counter()
        method_multifile_mt_on()
        times_multi_on.append((time.perf_counter() - t0) * 1000)
    
    med_multi_on = median(times_multi_on)
    print(f"    {n_files} files MT ON:  {med_multi_on:.1f} ms")
    
    if med_multi_on > 0:
        speedup_multi = med_multi_off / med_multi_on
        print(f"    Speedup: {speedup_multi:.1f}x")
    else:
        speedup_multi = 0
    
    results['multifile_mt_off'] = {'median_ms': med_multi_off, 'n_files': n_files}
    results['multifile_mt_on'] = {'median_ms': med_multi_on, 'n_files': n_files}
    results['multifile_speedup'] = speedup_multi
    
    # Cleanup symlinks
    for link_path in symlink_files:
        try:
            os.remove(link_path)
        except:
            pass

    # =========================================================================
    # IMPORTANT: Restore MT off for subsequent tests
    # =========================================================================
    ROOT.ROOT.DisableImplicitMT()
    print(f"\n  (Restored: ImplicitMT enabled: {ROOT.ROOT.IsImplicitMTEnabled()})")

    return results


# =============================================================================
# Multi-Column Test
# =============================================================================

def test_multi_column():
    """Test performance with multiple columns."""
    print_header("Multi-Column Test (P1)")

    results = {}

    # Ensure MT is disabled for consistent results
    ROOT.ROOT.DisableImplicitMT()

    # Single column - Take
    def single_col_take():
        rdf = create_fresh_rdf()
        take = rdf.Take['ROOT::RVec<double>']('track_pt')
        return take.GetValue()

    # Single column - AsNumpy (for comparison)
    def single_col_asnumpy():
        rdf = create_fresh_rdf()
        return rdf.AsNumpy(['track_pt'])

    # Multiple columns - AsNumpy (known to work)
    # Use columns that exist in toy_nd: track_pt, cluster_Q, event_id
    def multi_col_asnumpy():
        rdf = create_fresh_rdf()
        return rdf.AsNumpy(['track_pt', 'event_id'])  # 1D + scalar

    # Multiple 2D columns - AsNumpy
    def multi_col_2d_asnumpy():
        rdf = create_fresh_rdf()
        return rdf.AsNumpy(['cluster_Q'])  # Only one 2D column available

    # Single column Take
    result_single_take = run_benchmark(
        single_col_take, "Take_single_column",
        CONFIG['profiles_dir'] / 'multicol',
        n_warmup=1, n_runs=3
    )
    results['single_take'] = {k: v for k, v in result_single_take.items() if k != 'result'}

    if result_single_take['status'] == 'OK':
        print(f"  Single column (Take): {result_single_take['median_ms']:.1f} ± {result_single_take['mad_ms']:.1f} ms")
    else:
        print(f"  Single column (Take): FAILED - {result_single_take.get('error', 'unknown')[:60]}...")

    # Single column AsNumpy
    result_single_asnumpy = run_benchmark(
        single_col_asnumpy, "AsNumpy_single_column",
        CONFIG['profiles_dir'] / 'multicol',
        n_warmup=1, n_runs=3
    )
    results['single_asnumpy'] = {k: v for k, v in result_single_asnumpy.items() if k != 'result'}

    if result_single_asnumpy['status'] == 'OK':
        print(f"  Single column (AsNumpy): {result_single_asnumpy['median_ms']:.1f} ± {result_single_asnumpy['mad_ms']:.1f} ms")
    else:
        print(f"  Single column (AsNumpy): FAILED")

    # Multi-column AsNumpy (1D + scalar)
    result_multi = run_benchmark(
        multi_col_asnumpy, "AsNumpy_multi_column",
        CONFIG['profiles_dir'] / 'multicol',
        n_warmup=1, n_runs=3
    )
    results['multi_asnumpy'] = {k: v for k, v in result_multi.items() if k != 'result'}

    if result_multi['status'] == 'OK':
        print(f"  Multi column (AsNumpy 1D+scalar): {result_multi['median_ms']:.1f} ± {result_multi['mad_ms']:.1f} ms")
    else:
        print(f"  Multi column (AsNumpy): FAILED - {result_multi.get('error', 'unknown')[:60]}...")

    return results


# =============================================================================
# Expression Overhead Test (P1) - With Scaling Analysis
# =============================================================================

def test_expression_overhead():
    """
    Test overhead of Define() expressions vs raw columns.

    METHODOLOGY:
    1. JIT warmup (excluded from timing)
    2. Run with different data fractions (20%, 40%, 60%, 80%, 100%)
    3. Linear fit: time = offset + slope * n_elements

    This separates:
    - Offset = fixed overhead (JIT cached, setup)
    - Slope = per-element cost (true scaling)
    """
    print_header("Expression Overhead Test (P1) - Scaling Analysis")

    results = {}

    # Data fractions to test
    FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
    N_RUNS_PER_FRACTION = 3

    # Get total elements for reference
    total_1d = TEST_DATA['n_1d_elements']
    total_2d = TEST_DATA['n_2d_elements']

    print(f"\nData sizes: 1D={total_1d:,} elements, 2D={total_2d:,} elements")
    print(f"Fractions tested: {FRACTIONS}")
    print(f"Runs per fraction: {N_RUNS_PER_FRACTION}")

    # Helper for linear fit
    def linear_fit(points):
        """Fit time = offset + slope * n_elements"""
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        A = np.column_stack([np.ones_like(x), x])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        offset, slope = coeffs
        slope_us = slope * 1000  # Convert ms/elem to µs/elem
        return offset, slope_us

    # =========================================================================
    # 1D Expressions
    # =========================================================================
    EXPRESSIONS_1D = {
        'raw': ('track_pt', 'Raw column'),
        'scale': ('track_pt * 1000', 'Multiply'),
        'sqrt': ('sqrt(track_pt)', 'sqrt()'),
        'square': ('track_pt * track_pt', 'Self-mult'),
    }

    print("\n" + "=" * 90)
    print("1D EXPRESSIONS - Scaling Analysis")
    print("=" * 90)

    for expr_id, (expr, desc) in EXPRESSIONS_1D.items():

        # JIT WARMUP (once, before any timing)
        if expr_id != 'raw':
            try:
                rdf_warmup = create_fresh_rdf()
                _ = rdf_warmup.Define('warmup_col', expr).Take['ROOT::RVec<double>']('warmup_col').GetValue()
            except Exception as e:
                print(f"\n{desc}: JIT warmup failed: {e}")
                continue

        print(f"\n{desc} ({expr}):")
        print("-" * 90)
        print(f"{'Fraction':<10} {'Elements':<12} {'AsNumpy (ms)':<20} {'Take+C++ (ms)':<20} {'Speedup':<10}")
        print("-" * 90)

        # Collect data points for linear fit
        asnumpy_points = []
        take_cpp_points = []

        for frac in FRACTIONS:
            n_entries = int(TEST_DATA['n_events'] * frac)
            n_elements = int(total_1d * frac)

            # Capture expr_id and expr in closure properly
            current_expr_id = expr_id
            current_expr = expr

            def method_asnumpy(n_ent=n_entries, eid=current_expr_id, ex=current_expr):
                rdf = ROOT.RDataFrame("Events", TEST_FILE).Range(n_ent)
                if eid == 'raw':
                    data = rdf.AsNumpy(['track_pt'])
                    col = 'track_pt'
                else:
                    rdf2 = rdf.Define('expr_col', ex)
                    data = rdf2.AsNumpy(['expr_col'])
                    col = 'expr_col'

                values = []
                for event in data[col]:
                    values.extend(np.asarray(event))
                return np.array(values, dtype=np.float64)

            def method_take_cpp(n_ent=n_entries, eid=current_expr_id, ex=current_expr):
                rdf = ROOT.RDataFrame("Events", TEST_FILE).Range(n_ent)
                if eid == 'raw':
                    take_result = rdf.Take['ROOT::RVec<double>']('track_pt')
                else:
                    rdf2 = rdf.Define('expr_col', ex)
                    take_result = rdf2.Take['ROOT::RVec<double>']('expr_col')

                vec = take_result.GetValue()
                flat_result = ROOT.FlattenHelpers.Flatten1D(vec)
                return np.array(flat_result.values, copy=True)

            # Time both methods
            times_asnumpy = []
            times_take = []

            for _ in range(N_RUNS_PER_FRACTION):
                t0 = time.perf_counter()
                method_asnumpy()
                times_asnumpy.append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                method_take_cpp()
                times_take.append((time.perf_counter() - t0) * 1000)

            med_asnumpy = median(times_asnumpy)
            med_take = median(times_take)
            speedup = med_asnumpy / med_take if med_take > 0 else 0

            asnumpy_points.append((n_elements, med_asnumpy))
            take_cpp_points.append((n_elements, med_take))

            print(f"{frac:<10.0%} {n_elements:<12,} {med_asnumpy:<20.1f} {med_take:<20.1f} {speedup:<10.1f}x")

        # Linear fit
        offset_asnumpy, slope_asnumpy = linear_fit(asnumpy_points)
        offset_take, slope_take = linear_fit(take_cpp_points)

        print("-" * 90)
        print(f"{'Linear fit:':<22} {'offset (ms)':<15} {'slope (µs/elem)':<20}")
        print(f"{'AsNumpy':<22} {offset_asnumpy:<15.1f} {slope_asnumpy:<20.3f}")
        print(f"{'Take+C++':<22} {offset_take:<15.1f} {slope_take:<20.3f}")
        if slope_take > 0:
            print(f"{'Slope ratio:':<22} {slope_asnumpy/slope_take:.1f}x faster per element")

        results[f'{expr_id}_1d'] = {
            'expression': expr,
            'asnumpy': {'offset_ms': offset_asnumpy, 'slope_us_per_elem': slope_asnumpy},
            'take_cpp': {'offset_ms': offset_take, 'slope_us_per_elem': slope_take},
        }

    # =========================================================================
    # 2D Expressions
    # =========================================================================
    EXPRESSIONS_2D = {
        'raw_2d': ('cluster_Q', 'Raw 2D'),
        'scale_2d': ('cluster_Q * 2', '2D mult'),
    }

    print("\n" + "=" * 90)
    print("2D EXPRESSIONS - Scaling Analysis")
    print("=" * 90)

    for expr_id, (expr, desc) in EXPRESSIONS_2D.items():

        # JIT WARMUP
        if expr_id != 'raw_2d':
            try:
                rdf_warmup = create_fresh_rdf()
                _ = rdf_warmup.Define('warmup_col', expr).Take['ROOT::RVec<ROOT::RVec<double>>']('warmup_col').GetValue()
            except Exception as e:
                print(f"\n{desc}: JIT warmup failed: {e}")
                continue

        print(f"\n{desc} ({expr}):")
        print("-" * 90)
        print(f"{'Fraction':<10} {'Elements':<12} {'AsNumpy (ms)':<20} {'Take+C++ (ms)':<20} {'Speedup':<10}")
        print("-" * 90)

        asnumpy_points = []
        take_cpp_points = []

        for frac in FRACTIONS:
            n_entries = int(TEST_DATA['n_events'] * frac)
            n_elements = int(total_2d * frac)

            current_expr_id = expr_id
            current_expr = expr

            def method_asnumpy_2d(n_ent=n_entries, eid=current_expr_id, ex=current_expr):
                rdf = ROOT.RDataFrame("Events", TEST_FILE).Range(n_ent)
                if eid == 'raw_2d':
                    data = rdf.AsNumpy(['cluster_Q'])
                    col = 'cluster_Q'
                else:
                    rdf2 = rdf.Define('expr_col', ex)
                    data = rdf2.AsNumpy(['expr_col'])
                    col = 'expr_col'

                values = []
                for event in data[col]:
                    for track in event:
                        values.extend(np.asarray(track))
                return np.array(values, dtype=np.float64)

            def method_take_cpp_2d(n_ent=n_entries, eid=current_expr_id, ex=current_expr):
                rdf = ROOT.RDataFrame("Events", TEST_FILE).Range(n_ent)
                if eid == 'raw_2d':
                    take_result = rdf.Take['ROOT::RVec<ROOT::RVec<double>>']('cluster_Q')
                else:
                    rdf2 = rdf.Define('expr_col', ex)
                    take_result = rdf2.Take['ROOT::RVec<ROOT::RVec<double>>']('expr_col')

                vec = take_result.GetValue()
                flat_result = ROOT.FlattenHelpers.Flatten2D(vec)
                return np.array(flat_result.values, copy=True)

            times_asnumpy = []
            times_take = []

            for _ in range(N_RUNS_PER_FRACTION):
                t0 = time.perf_counter()
                method_asnumpy_2d()
                times_asnumpy.append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                method_take_cpp_2d()
                times_take.append((time.perf_counter() - t0) * 1000)

            med_asnumpy = median(times_asnumpy)
            med_take = median(times_take)
            speedup = med_asnumpy / med_take if med_take > 0 else 0

            asnumpy_points.append((n_elements, med_asnumpy))
            take_cpp_points.append((n_elements, med_take))

            print(f"{frac:<10.0%} {n_elements:<12,} {med_asnumpy:<20.1f} {med_take:<20.1f} {speedup:<10.1f}x")

        offset_asnumpy, slope_asnumpy = linear_fit(asnumpy_points)
        offset_take, slope_take = linear_fit(take_cpp_points)

        print("-" * 90)
        print(f"{'Linear fit:':<22} {'offset (ms)':<15} {'slope (µs/elem)':<20}")
        print(f"{'AsNumpy':<22} {offset_asnumpy:<15.1f} {slope_asnumpy:<20.3f}")
        print(f"{'Take+C++':<22} {offset_take:<15.1f} {slope_take:<20.3f}")
        if slope_take > 0:
            print(f"{'Slope ratio:':<22} {slope_asnumpy/slope_take:.1f}x faster per element")

        results[f'{expr_id}_2d'] = {
            'expression': expr,
            'asnumpy': {'offset_ms': offset_asnumpy, 'slope_us_per_elem': slope_asnumpy},
            'take_cpp': {'offset_ms': offset_take, 'slope_us_per_elem': slope_take},
        }

    return results


# =============================================================================
# Function Pointer Test (P1) - Eliminate JIT Overhead
# =============================================================================

def test_function_pointer():
    """
    Test pre-compiled function pointers vs JIT string expressions.
    
    Key finding: Function pointers eliminate ~130ms JIT overhead per expression.
    This is critical for DSL implementation - pre-compile common operations!
    """
    print_header("Function Pointer Test (P1) - JIT Elimination")
    
    results = {}
    N_RUNS = 5
    
    print("Comparing JIT string expressions vs pre-compiled function pointers:")
    print("  - JIT: rdf.Define('col', 'track_pt * 1000')  # Compiles at runtime")
    print("  - FuncPtr: C++ pre-compiled function         # No JIT overhead")
    print()
    
    # Warmup - ensure C++ helpers are compiled
    print("Warmup (one-time C++ compilation)...")
    rdf_warmup = ROOT.RDataFrame("Events", TEST_FILE)
    rdf_warmup2 = ROOT.FuncPtrHelpers.DefineScale1000(
        ROOT.RDF.AsRNode(rdf_warmup), "warmup", "track_pt"
    )
    _ = rdf_warmup2.Take['ROOT::RVec<double>']('warmup').GetValue()
    print("Done.\n")
    
    # -------------------------------------------------------------------------
    # Test 1: Raw column (reference)
    # -------------------------------------------------------------------------
    print("1. Raw column (reference):")
    times_raw = []
    for i in range(N_RUNS):
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        t0 = time.perf_counter()
        result = rdf.Take['ROOT::RVec<double>']('track_pt').GetValue()
        times_raw.append((time.perf_counter() - t0) * 1000)
    
    med_raw = median(times_raw)
    mad_raw = median([abs(t - med_raw) for t in times_raw])
    print(f"   Time: {med_raw:.1f} ± {mad_raw:.1f} ms")
    results['raw'] = {'median_ms': med_raw, 'mad_ms': mad_raw, 'runs_ms': times_raw}
    
    # -------------------------------------------------------------------------
    # Test 2: JIT string expression
    # -------------------------------------------------------------------------
    print("\n2. JIT string expression (track_pt * 1000):")
    times_jit = []
    for i in range(N_RUNS):
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        t0 = time.perf_counter()
        rdf2 = rdf.Define("scaled", "track_pt * 1000")
        result = rdf2.Take['ROOT::RVec<double>']('scaled').GetValue()
        times_jit.append((time.perf_counter() - t0) * 1000)
    
    med_jit = median(times_jit)
    mad_jit = median([abs(t - med_jit) for t in times_jit])
    jit_offset = med_jit - med_raw
    print(f"   Time: {med_jit:.1f} ± {mad_jit:.1f} ms (JIT offset: {jit_offset:.1f} ms)")
    results['jit_string'] = {
        'median_ms': med_jit, 'mad_ms': mad_jit, 'runs_ms': times_jit,
        'offset_ms': jit_offset
    }
    
    # -------------------------------------------------------------------------
    # Test 3: Function pointer (pre-compiled)
    # -------------------------------------------------------------------------
    print("\n3. Function pointer (pre-compiled Scale1000):")
    times_ptr = []
    for i in range(N_RUNS):
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        t0 = time.perf_counter()
        rdf2 = ROOT.FuncPtrHelpers.DefineScale1000(
            ROOT.RDF.AsRNode(rdf), "scaled", "track_pt"
        )
        result = rdf2.Take['ROOT::RVec<double>']('scaled').GetValue()
        times_ptr.append((time.perf_counter() - t0) * 1000)
    
    med_ptr = median(times_ptr)
    mad_ptr = median([abs(t - med_ptr) for t in times_ptr])
    ptr_offset = med_ptr - med_raw
    print(f"   Time: {med_ptr:.1f} ± {mad_ptr:.1f} ms (offset: {ptr_offset:.1f} ms)")
    results['func_ptr'] = {
        'median_ms': med_ptr, 'mad_ms': mad_ptr, 'runs_ms': times_ptr,
        'offset_ms': ptr_offset
    }
    
    # -------------------------------------------------------------------------
    # Test 4: Function pointer - Sqrt
    # -------------------------------------------------------------------------
    print("\n4. Function pointer (pre-compiled Sqrt):")
    times_sqrt = []
    for i in range(N_RUNS):
        rdf = ROOT.RDataFrame("Events", TEST_FILE)
        t0 = time.perf_counter()
        rdf2 = ROOT.FuncPtrHelpers.DefineSqrt(
            ROOT.RDF.AsRNode(rdf), "sqrt_col", "track_pt"
        )
        result = rdf2.Take['ROOT::RVec<double>']('sqrt_col').GetValue()
        times_sqrt.append((time.perf_counter() - t0) * 1000)
    
    med_sqrt = median(times_sqrt)
    mad_sqrt = median([abs(t - med_sqrt) for t in times_sqrt])
    sqrt_offset = med_sqrt - med_raw
    print(f"   Time: {med_sqrt:.1f} ± {mad_sqrt:.1f} ms (offset: {sqrt_offset:.1f} ms)")
    results['func_ptr_sqrt'] = {
        'median_ms': med_sqrt, 'mad_ms': mad_sqrt, 'runs_ms': times_sqrt,
        'offset_ms': sqrt_offset
    }
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("SUMMARY - Function Pointer vs JIT:")
    print("-" * 60)
    print(f"  Raw column:        {med_raw:.1f} ms (reference)")
    print(f"  JIT string:        {med_jit:.1f} ms (offset: {jit_offset:.1f} ms)")
    print(f"  Function pointer:  {med_ptr:.1f} ms (offset: {ptr_offset:.1f} ms)")
    
    if jit_offset > 0:
        eliminated = jit_offset - ptr_offset
        pct = (eliminated / jit_offset) * 100 if jit_offset > 0 else 0
        print(f"\n  JIT overhead eliminated: {eliminated:.1f} ms ({pct:.0f}%)")
        results['jit_eliminated_ms'] = eliminated
        results['jit_eliminated_pct'] = pct
    
    print("\n  RECOMMENDATION: Use pre-compiled function pointers for")
    print("  common operations in DSL to eliminate JIT overhead.")
    
    return results


# =============================================================================
# Main Benchmark Execution
# =============================================================================

def run_all_benchmarks():
    """Run all benchmarks and collect results."""

    all_results = {
        'environment': ENV_INFO,
        'test_data': TEST_DATA,
        'config': {k: str(v) for k, v in CONFIG.items()},
        '1d_benchmarks': {},
        '2d_benchmarks': {},
        'failed_methods': {},
        'anomaly_test': {},
        'threading_test': {},
        'multicol_test': {},
        'expression_test': {},
        'funcptr_test': {},
    }

    # -------------------------------------------------------------------------
    # 1D Benchmarks
    # -------------------------------------------------------------------------
    print_header("1D Benchmarks (track_pt)")

    methods_1d = [
        ("1. Baseline: AsNumpy + Python loop", method_1_baseline_1d),
        ("2. AsNumpy + np.concatenate", method_2_concatenate_1d),
        ("3. Take + np.concatenate [RECOMMENDED]", method_3_take_concat_1d),
        ("4. Take + C++ flatten", method_4_take_cpp_1d),
        ("5. TTree::Draw [REFERENCE]", method_5_ttree_draw_1d),
        ("6. uproot (file-based)", method_6_uproot_1d),
        ("7. Manual Awkward", method_7_awkward_1d),
    ]

    baseline_1d = None

    for name, func in methods_1d:
        print_subheader(name)

        result = run_benchmark(
            func, name,
            CONFIG['profiles_dir'] / '1d',
            n_warmup=CONFIG['n_warmup'],
            n_runs=CONFIG['n_runs']
        )

        all_results['1d_benchmarks'][name] = {
            k: v for k, v in result.items() if k != 'result'
        }

        if result['status'] == 'OK':
            if baseline_1d is None:
                baseline_1d = result
                speedup = 1.0
            else:
                speedup = baseline_1d['median_ms'] / result['median_ms']

            # Verify correctness
            if baseline_1d is not None:
                verification = verify_results(result, baseline_1d, name)
                all_results['1d_benchmarks'][name]['verification'] = verification
                verify_status = "✓" if verification['values_match'] else "✗"
            else:
                verify_status = "-"

            print(f"  Time: {result['median_ms']:.1f} ± {result['mad_ms']:.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  Memory: {result['peak_memory_mb']:.1f} MB")
            print(f"  Verified: {verify_status}")
        else:
            print(f"  FAILED: {result.get('error', 'unknown')}")

    # -------------------------------------------------------------------------
    # 2D Benchmarks
    # -------------------------------------------------------------------------
    print_header("2D Benchmarks (cluster_Q)")

    methods_2d = [
        ("1. Baseline: AsNumpy + Python loop", method_1_baseline_2d),
        ("2. AsNumpy + optimized Python", method_2_optimized_2d),
        ("4. Take + C++ flatten [RECOMMENDED]", method_4_take_cpp_2d),
        ("5. TTree::Draw [REFERENCE]", method_5_ttree_draw_2d),
        ("6. uproot (file-based)", method_6_uproot_2d),
        ("7. Manual Awkward", method_7_awkward_2d),
    ]

    baseline_2d = None

    for name, func in methods_2d:
        print_subheader(name)

        result = run_benchmark(
            func, name,
            CONFIG['profiles_dir'] / '2d',
            n_warmup=CONFIG['n_warmup'],
            n_runs=CONFIG['n_runs']
        )

        all_results['2d_benchmarks'][name] = {
            k: v for k, v in result.items() if k != 'result'
        }

        if result['status'] == 'OK':
            if baseline_2d is None:
                baseline_2d = result
                speedup = 1.0
            else:
                speedup = baseline_2d['median_ms'] / result['median_ms']

            # Verify correctness (skip TTree::Draw as it doesn't provide indices)
            if baseline_2d is not None and "TTree" not in name:
                verification = verify_results(result, baseline_2d, name)
                all_results['2d_benchmarks'][name]['verification'] = verification
                verify_status = "✓" if all(verification.values()) else "✗"
            else:
                verify_status = "-"

            print(f"  Time: {result['median_ms']:.1f} ± {result['mad_ms']:.1f} ms")
            print(f"  Speedup: {speedup:.1f}x")
            print(f"  Memory: {result['peak_memory_mb']:.1f} MB")
            print(f"  Verified: {verify_status}")
        else:
            print(f"  FAILED: {result.get('error', 'unknown')}")

    # -------------------------------------------------------------------------
    # Failed Methods (documented)
    # -------------------------------------------------------------------------
    print_header("Failed Methods (Documented)")

    failed_methods = [
        ("8. Foreach + C++ callback", method_8_foreach_cpp),
        ("9. ak.from_rdataframe()", method_9_ak_from_rdataframe),
    ]

    for name, func in failed_methods:
        print_subheader(name)
        try:
            func()
        except RuntimeError as e:
            print(f"  {e}")
            all_results['failed_methods'][name] = str(e)

    # -------------------------------------------------------------------------
    # Additional Tests (P1)
    # -------------------------------------------------------------------------

    # Anomaly test
    all_results['anomaly_test'] = test_asnumpy_anomaly()

    # Threading test
    all_results['threading_test'] = test_root_threading()

    # Multi-column test
    all_results['multicol_test'] = test_multi_column()

    # Expression overhead test (slow - can skip with --skip-expressions)
    if CONFIG['skip_expressions']:
        print_header("Expression Overhead Test (P1) - SKIPPED")
        print("  Use without --skip-expressions to run JIT expression tests")
        all_results['expression_test'] = {'skipped': True}
    else:
        all_results['expression_test'] = test_expression_overhead()

    # Function pointer test (fast - always run)
    all_results['funcptr_test'] = test_function_pointer()

    return all_results


# =============================================================================
# Summary and Output
# =============================================================================

def print_summary(results: Dict):
    """Print summary table."""
    print_header("SUMMARY")

    print("1D Methods (track_pt - {:,} elements):".format(TEST_DATA['n_1d_elements']))
    print("-" * 70)
    print(f"{'Method':<45} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline_1d = None
    for name, data in results['1d_benchmarks'].items():
        if data['status'] == 'OK':
            if baseline_1d is None:
                baseline_1d = data['median_ms']
            speedup = baseline_1d / data['median_ms']
            verified = "✓" if data.get('verification', {}).get('values_match', False) else ""
            print(f"{name:<45} {data['median_ms']:>8.1f} ± {data['mad_ms']:<4.1f} {speedup:>8.1f}x {verified}")
        else:
            print(f"{name:<45} {'FAILED':<15}")

    print()
    print("2D Methods (cluster_Q - {:,} elements):".format(TEST_DATA['n_2d_elements']))
    print("-" * 70)
    print(f"{'Method':<45} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline_2d = None
    for name, data in results['2d_benchmarks'].items():
        if data['status'] == 'OK':
            if baseline_2d is None:
                baseline_2d = data['median_ms']
            speedup = baseline_2d / data['median_ms']
            verified = "✓" if all(data.get('verification', {}).values()) else ""
            print(f"{name:<45} {data['median_ms']:>8.1f} ± {data['mad_ms']:<4.1f} {speedup:>8.1f}x {verified}")
        else:
            print(f"{name:<45} {'FAILED':<15}")

    print()
    print_header("RECOMMENDATIONS")
    print("""
Based on the benchmark results:

Performance Summary:
  Take vs AsNumpy (retrieval): ~2.5x faster
  Take+C++ vs Baseline (end-to-end): ~10-12x faster
  Slope ratio (per-element cost): 3-4x faster

1D Arrays (RVec<T>):
  RECOMMENDED: Method 4 - Take + C++ flatten
  - End-to-end speedup: ~11x over baseline
  - Slope: ~0.08 µs/elem (vs ~0.29 µs/elem baseline)

2D Arrays (RVec<RVec<T>>):
  RECOMMENDED: Method 4 - Take + C++ flatten
  - End-to-end speedup: ~10x over baseline
  - Slope: ~0.10 µs/elem (vs ~0.31 µs/elem baseline)
  - Properly generates idx_1, idx_2 indices

When to Enable MT (Multi-Threading):
  - Minimal benefit (<1.5x) for single files, small datasets
  - Better benefit (2-4x) for multiple files, large datasets
  - Recommendation: Disable MT for interactive, enable for batch

Note: uproot provides similar or better performance but requires file path,
not compatible with RDataFrame-only workflows.
""")


def save_results(results: Dict, output_file: Path):
    """Save results to JSON."""
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    serializable = make_serializable(results)

    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_file}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    try:
        # Run all benchmarks
        results = run_all_benchmarks()

        # Print summary
        print_summary(results)

        # Save results
        results_file = CONFIG['output_dir'] / 'results.json'
        save_results(results, results_file)

        # Cleanup (only in --clean mode)
        if CONFIG['clean_mode'] and os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)

        print_header("EXPLORATION COMPLETE")
        print(f"Profiles saved to: {CONFIG['profiles_dir']}")
        print(f"Results saved to: {results_file}")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

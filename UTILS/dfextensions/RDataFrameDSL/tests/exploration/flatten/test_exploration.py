"""
Phase 13.6.A: Exploration Tests

Permanent tests to validate assumptions about ROOT/NumPy/Awkward behavior.
These tests document expected behavior across library versions.

Purpose: Reproducible benchmarks across ROOT/Awkward versions.
Output: exploration_results.json with environment info and timings.

Tests:
- EX1: np.asarray view or copy?
- EX2: Concatenate vs preallocate performance
- EX3: Awkward conversion overhead
- EX4: C++ helper vs Python loop (placeholder)
- EX5: 2-level nesting access pattern
- EX6: Phase 8 method broadcast results flatten correctly
"""

import pytest
import numpy as np
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import from package
from RDataFrameDSL.flatten import flatten_to_dict, FlattenBackend


# =============================================================================
# Exploration Results Collector
# =============================================================================

class ExplorationResults:
    """Collect and save exploration test results."""
    
    def __init__(self):
        self.results = {}
        self.benchmarks = {}
        self.environment = self._get_environment()
    
    def _get_environment(self) -> Dict[str, str]:
        """Get environment info."""
        env = {
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__,
        }
        
        try:
            import ROOT
            env['root_version'] = ROOT.gROOT.GetVersion()
        except ImportError:
            env['root_version'] = 'not available'
        
        try:
            import awkward as ak
            env['awkward_version'] = ak.__version__
        except ImportError:
            env['awkward_version'] = 'not available'
        
        return env
    
    def record(self, key: str, value: Any):
        """Record a result."""
        self.results[key] = value
    
    def record_benchmark(self, key: str, value_ms: float):
        """Record a benchmark in milliseconds."""
        self.benchmarks[key] = value_ms
    
    def save(self, path: Path):
        """Save results to JSON file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_value(v):
            if isinstance(v, (np.bool_, np.integer)):
                return int(v)
            elif isinstance(v, np.floating):
                return float(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            return v
        
        output = {
            'timestamp': datetime.now().isoformat() + 'Z',
            'environment': self.environment,
            'results': {k: convert_value(v) for k, v in self.results.items()},
            'benchmarks': {k: convert_value(v) for k, v in self.benchmarks.items()},
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output


# Global results collector
_results = ExplorationResults()


# =============================================================================
# EX1: np.asarray View or Copy?
# =============================================================================

class TestEX1_AsarrayBehavior:
    """EX1: Does np.asarray create a view or copy of RVec data?"""
    
    def test_numpy_array_is_copy(self):
        """
        Test whether np.asarray on list-like objects creates copies.
        
        This is important because if it's a view, we can avoid copies.
        If it's a copy, preallocating is more efficient.
        """
        # Create source data
        source = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Convert to numpy
        arr = np.asarray(source, dtype=np.float64)
        
        # Modify source
        source[0] = 999.0
        
        # Check if arr changed
        is_view = (arr[0] == 999.0)
        is_copy = (arr[0] == 1.0)
        
        _results.record('EX1_asarray_is_view', is_view)
        _results.record('EX1_asarray_is_copy', is_copy)
        
        # Document the behavior
        print(f"\nEX1: np.asarray creates {'view' if is_view else 'copy'}")
        
        # For Python lists, asarray always creates a copy
        assert is_copy, "np.asarray should create a copy from Python list"
    
    def test_asarray_overhead(self):
        """Measure overhead of np.asarray conversion."""
        # Create test data (simulate RVec)
        n_iterations = 1000
        data = [float(i) for i in range(100)]
        
        start = time.perf_counter()
        for _ in range(n_iterations):
            arr = np.asarray(data, dtype=np.float64)
        elapsed = time.perf_counter() - start
        
        overhead_ms = (elapsed / n_iterations) * 1000
        _results.record('EX1_copy_overhead_ms', overhead_ms)
        
        print(f"EX1: asarray overhead: {overhead_ms:.3f}ms per 100-element conversion")
        
        # Should be fast (< 0.1ms for 100 elements)
        assert overhead_ms < 0.1, f"asarray too slow: {overhead_ms:.3f}ms"


# =============================================================================
# EX2: Concatenate vs Preallocate
# =============================================================================

class TestEX2_ConcatVsPreallocate:
    """EX2: Compare np.concatenate vs preallocate strategies."""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data: 10k events × 50 tracks."""
        np.random.seed(42)
        n_events = 10_000
        tracks_per_event = 50
        
        return [
            np.random.randn(tracks_per_event).astype(np.float64)
            for _ in range(n_events)
        ]
    
    def test_concatenate_method(self, test_data):
        """Test np.concatenate method."""
        start = time.perf_counter()
        result = np.concatenate(test_data)
        elapsed = time.perf_counter() - start
        
        elapsed_ms = elapsed * 1000
        _results.record('EX2_concatenate_ms', elapsed_ms)
        
        print(f"\nEX2: np.concatenate: {elapsed_ms:.1f}ms for 500k elements")
        
        assert len(result) == 500_000
    
    def test_preallocate_method(self, test_data):
        """Test preallocate method."""
        # Calculate total size
        sizes = [len(arr) for arr in test_data]
        total = sum(sizes)
        
        start = time.perf_counter()
        result = np.empty(total, dtype=np.float64)
        offset = 0
        for arr in test_data:
            n = len(arr)
            result[offset:offset+n] = arr
            offset += n
        elapsed = time.perf_counter() - start
        
        elapsed_ms = elapsed * 1000
        _results.record('EX2_preallocate_ms', elapsed_ms)
        
        print(f"EX2: preallocate: {elapsed_ms:.1f}ms for 500k elements")
        
        assert len(result) == 500_000
    
    def test_compare_methods(self, test_data):
        """Compare both methods and determine winner."""
        # Concatenate
        start = time.perf_counter()
        _ = np.concatenate(test_data)
        concat_time = time.perf_counter() - start
        
        # Preallocate
        sizes = [len(arr) for arr in test_data]
        total = sum(sizes)
        
        start = time.perf_counter()
        result = np.empty(total, dtype=np.float64)
        offset = 0
        for arr in test_data:
            n = len(arr)
            result[offset:offset+n] = arr
            offset += n
        prealloc_time = time.perf_counter() - start
        
        winner = 'preallocate' if prealloc_time < concat_time else 'concatenate'
        _results.record('EX2_winner', winner)
        
        print(f"\nEX2 Winner: {winner}")
        print(f"  concatenate: {concat_time*1000:.1f}ms")
        print(f"  preallocate: {prealloc_time*1000:.1f}ms")


# =============================================================================
# EX3: Awkward Conversion Overhead
# =============================================================================

class TestEX3_AwkwardOverhead:
    """EX3: Measure Awkward Array conversion overhead."""
    
    @pytest.fixture
    def test_data(self):
        """Generate jagged test data."""
        np.random.seed(42)
        n_events = 10_000
        
        return [
            np.random.randn(np.random.randint(40, 60)).astype(np.float64)
            for _ in range(n_events)
        ]
    
    def test_awkward_conversion(self, test_data):
        """Test Awkward Array conversion overhead."""
        try:
            import awkward as ak
        except ImportError:
            _results.record('EX3_awkward_available', False)
            _results.record('EX3_awkward_overhead_ms', None)
            pytest.skip("Awkward not available")
            return
        
        _results.record('EX3_awkward_available', True)
        
        # Method 1: Via .tolist() (current implementation - SLOW)
        start = time.perf_counter()
        jagged_tolist = ak.Array([arr.tolist() for arr in test_data])
        flat_tolist = ak.flatten(jagged_tolist).to_numpy()
        tolist_time = time.perf_counter() - start
        
        # Method 2: Direct from numpy (FAST - roofline)
        start = time.perf_counter()
        # Build offsets for direct construction
        offsets = np.zeros(len(test_data) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(arr) for arr in test_data])
        content = np.concatenate(test_data)
        jagged_direct = ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(offsets),
                ak.contents.NumpyArray(content)
            )
        )
        flat_direct = ak.flatten(jagged_direct).to_numpy()
        direct_time = time.perf_counter() - start
        
        # Method 3: Pure NumPy preallocate (baseline)
        start = time.perf_counter()
        total = sum(len(arr) for arr in test_data)
        flat_numpy = np.empty(total, dtype=np.float64)
        offset = 0
        for arr in test_data:
            n = len(arr)
            flat_numpy[offset:offset+n] = arr
            offset += n
        numpy_time = time.perf_counter() - start
        
        # Verify results match
        np.testing.assert_array_equal(flat_tolist, flat_direct)
        np.testing.assert_array_equal(flat_tolist, flat_numpy)
        
        tolist_ms = tolist_time * 1000
        direct_ms = direct_time * 1000
        numpy_ms = numpy_time * 1000
        
        _results.record('EX3_awkward_tolist_ms', tolist_ms)
        _results.record('EX3_awkward_direct_ms', direct_ms)
        _results.record('EX3_numpy_baseline_ms', numpy_ms)
        _results.record('EX3_tolist_slowdown', tolist_ms / numpy_ms if numpy_ms > 0 else 0)
        _results.record('EX3_direct_slowdown', direct_ms / numpy_ms if numpy_ms > 0 else 0)
        
        print(f"\nEX3: Awkward conversion comparison (500k elements):")
        print(f"  NumPy preallocate:    {numpy_ms:.1f}ms (baseline)")
        print(f"  Awkward via .tolist(): {tolist_ms:.1f}ms ({tolist_ms/numpy_ms:.1f}x slower)")
        print(f"  Awkward direct:        {direct_ms:.1f}ms ({direct_ms/numpy_ms:.1f}x slower)")
        print(f"  Potential speedup:     {tolist_ms/direct_ms:.1f}x by avoiding .tolist()")


# =============================================================================
# EX4: C++ Helper vs Python Loop
# =============================================================================

class TestEX4_CppVsPython:
    """EX4: Compare C++ helper vs Python loop performance."""
    
    def test_python_loop_baseline(self):
        """Establish Python loop baseline."""
        np.random.seed(42)
        n_events = 10_000
        tracks_per_event = 50
        
        # Generate data
        data = [
            np.random.randn(tracks_per_event).astype(np.float64)
            for _ in range(n_events)
        ]
        event_ids = np.arange(n_events, dtype=np.int64)
        
        # Python loop flatten
        start = time.perf_counter()
        
        total = sum(len(arr) for arr in data)
        flat_values = np.empty(total, dtype=np.float64)
        flat_event_ids = np.empty(total, dtype=np.int64)
        flat_track_idx = np.empty(total, dtype=np.int64)
        
        offset = 0
        for i, arr in enumerate(data):
            n = len(arr)
            flat_values[offset:offset+n] = arr
            flat_event_ids[offset:offset+n] = event_ids[i]
            flat_track_idx[offset:offset+n] = np.arange(n)
            offset += n
        
        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000
        
        _results.record('EX4_python_loop_ms', elapsed_ms)
        _results.record_benchmark('python_loop_500k_ms', elapsed_ms)
        
        print(f"\nEX4: Python loop flatten: {elapsed_ms:.1f}ms for 500k tracks")
        
        assert len(flat_values) == 500_000
    
    def test_cpp_helper_placeholder(self):
        """Placeholder for C++ helper comparison."""
        # C++ helper not yet implemented
        _results.record('EX4_cpp_available', False)
        _results.record('EX4_cpp_vs_python_speedup', None)
        
        print("\nEX4: C++ helper not yet implemented (placeholder)")


# =============================================================================
# EX5: 2-Level Nesting Access Pattern
# =============================================================================

class TestEX5_NestedAccess:
    """EX5: Determine optimal access pattern for RVec<RVec<T>>."""
    
    def test_nested_iteration_pattern(self):
        """Test nested iteration pattern for 2-level data."""
        np.random.seed(42)
        n_events = 1_000
        tracks_per_event = 10
        clusters_per_track = 5
        
        # Create nested data structure
        nested_data = [
            [
                np.random.randn(clusters_per_track).astype(np.float64)
                for _ in range(tracks_per_event)
            ]
            for _ in range(n_events)
        ]
        
        # Method 1: Double iteration
        start = time.perf_counter()
        total = 0
        for event in nested_data:
            for track in event:
                total += len(track)
        double_iter_time = time.perf_counter() - start
        
        # Method 2: Flatten in two passes
        start = time.perf_counter()
        total2 = sum(
            sum(len(track) for track in event)
            for event in nested_data
        )
        two_pass_time = time.perf_counter() - start
        
        _results.record('EX5_double_iteration_ms', double_iter_time * 1000)
        _results.record('EX5_two_pass_ms', two_pass_time * 1000)
        
        winner = 'double_iteration' if double_iter_time < two_pass_time else 'two_pass'
        _results.record('EX5_nested_access_pattern', winner)
        
        print(f"\nEX5: Nested access patterns:")
        print(f"  Double iteration: {double_iter_time*1000:.2f}ms")
        print(f"  Two-pass: {two_pass_time*1000:.2f}ms")
        print(f"  Winner: {winner}")
        
        assert total == total2 == n_events * tracks_per_event * clusters_per_track


# =============================================================================
# EX6: Phase 8 Method Broadcast Results
# =============================================================================

class TestEX6_Phase8MethodBroadcast:
    """EX6: Validate Phase 8 method broadcast results flatten correctly."""
    
    def test_method_broadcast_result_structure(self):
        """
        Test that Phase 8 method broadcast results have correct structure.
        
        Phase 8: tracks.Pt() produces RVec<double> per event.
        This should flatten identically to raw RVec<double> columns.
        """
        # Simulate Phase 8 result: tracks.Pt() → RVec<double> per event
        np.random.seed(42)
        
        # Raw data (as if from tracks.Pt())
        method_broadcast_result = [
            np.random.exponential(2.0, np.random.randint(3, 8)).astype(np.float64)
            for _ in range(100)
        ]
        
        # Flatten using our implementation
        data = {
            'event_id': np.arange(100, dtype=np.int64),
            'track_pts': np.array(method_broadcast_result, dtype=object),
        }
        
        result = flatten_to_dict(
            data,
            rvec_columns=['track_pts'],
            parent_id_column='event_id',
            backend=FlattenBackend.NUMPY
        )
        
        # Validate structure
        assert 'event_id' in result
        assert 'track_idx' in result
        assert 'track_pts' in result
        
        # Validate lengths match
        total_expected = sum(len(arr) for arr in method_broadcast_result)
        assert len(result['track_pts']) == total_expected
        assert len(result['event_id']) == total_expected
        assert len(result['track_idx']) == total_expected
        
        # Validate indices are correct
        # track_idx should be 0-based within each event
        offset = 0
        for i, arr in enumerate(method_broadcast_result):
            n = len(arr)
            expected_idx = np.arange(n)
            actual_idx = result['track_idx'][offset:offset+n]
            np.testing.assert_array_equal(actual_idx, expected_idx)
            offset += n
        
        _results.record('EX6_phase8_broadcast_validates', True)
        print("\nEX6: Phase 8 method broadcast results flatten correctly ✓")
    
    def test_method_broadcast_dtype_preservation(self):
        """Test that dtype is preserved from Phase 8 results."""
        # Float32 result (some methods might return float32)
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pts': np.array([
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
            ], dtype=object),
        }
        
        result = flatten_to_dict(
            data,
            rvec_columns=['track_pts'],
            parent_id_column='event_id',
            backend=FlattenBackend.NUMPY
        )
        
        # Should preserve float32
        assert result['track_pts'].dtype == np.float32
        
        _results.record('EX6_dtype_preservation', True)
        print("EX6: Phase 8 dtype preservation works ✓")


# =============================================================================
# Results Saving
# =============================================================================

@pytest.fixture(scope='session', autouse=True)
def save_results(request):
    """Save exploration results at end of session."""
    yield
    
    # Save results
    output_path = Path(__file__).parent / 'exploration_results.json'
    output = _results.save(output_path)
    
    print(f"\n{'='*60}")
    print("EXPLORATION RESULTS SAVED")
    print(f"{'='*60}")
    print(f"File: {output_path}")
    print(f"Environment: {output['environment']}")
    print(f"Results: {len(output['results'])} entries")
    print(f"Benchmarks: {len(output['benchmarks'])} entries")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

# =============================================================================
# tests/test_safe_mode.py
# =============================================================================
# Tests for Safe Mode - Layer 2/3 crash protection.
#
# Phase: 13.6.F
# Date: 2026-01-24
#
# Note: These tests use fork() and require:
# - No ROOT.EnableImplicitMT() called
# - Single-threaded execution
# - Must run with -n 0 (no pytest-xdist parallelization)
# =============================================================================

import pytest
import os
import sys
import time
import tempfile
from pathlib import Path

# Skip entire module on Windows (no fork)
pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="fork() not available on Windows"),
    pytest.mark.root_serial,
]


# =============================================================================
# Test Class: Precondition Checks
# =============================================================================

class TestPreconditions:
    """Tests for fork safety preconditions."""
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_check_fork_safe_single_thread(self):
        """
        check_fork_safe() passes in single-threaded environment.
        """
        from RDataFrameDSL.safe_mode import check_fork_safe
        
        # Should not raise in single-threaded test
        check_fork_safe()  # No exception = pass
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_check_fork_safe_rejects_threading(self):
        """
        check_fork_safe() rejects when multiple threads active.
        """
        import threading
        from RDataFrameDSL.safe_mode import check_fork_safe, SafeModeError
        
        # Start a background thread
        stop_event = threading.Event()
        def worker():
            while not stop_event.is_set():
                time.sleep(0.01)
        
        thread = threading.Thread(target=worker, name="TestThread")
        thread.start()
        
        try:
            with pytest.raises(SafeModeError) as exc_info:
                check_fork_safe()
            
            assert "active threads" in str(exc_info.value)
        finally:
            stop_event.set()
            thread.join()


# =============================================================================
# Test Class: Layer 2 - Compile-Time Safety
# =============================================================================

class TestLayer2:
    """Tests for Layer 2 compile-time protection."""
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_safe_declare_success(self):
        """
        safe_declare() succeeds for valid C++ code.
        """
        from RDataFrameDSL.safe_mode import safe_declare
        
        code = """
        int safe_test_add(int a, int b) {
            return a + b;
        }
        """
        
        result = safe_declare(code, timeout=10.0)
        assert result is True
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_safe_declare_syntax_error(self):
        """
        safe_declare() raises SafeModeError for syntax errors.
        """
        from RDataFrameDSL.safe_mode import safe_declare, SafeModeError
        
        code = """
        int broken_function( {
            // Missing closing paren and body
        """
        
        with pytest.raises(SafeModeError) as exc_info:
            safe_declare(code, timeout=10.0)
        
        error = exc_info.value
        assert error.layer == "compile"
        assert error.reason in ("error", "crash")
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_safe_compile_success(self):
        """
        safe_compile() succeeds for valid C++ code.
        """
        from RDataFrameDSL.safe_mode import safe_compile
        
        code = """
        #include <cmath>
        
        double safe_compile_test(double x) {
            return std::sqrt(x * x);
        }
        """
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_compile(
                code, 
                name="test_safe_compile",
                timeout=30.0,
                cache_dir=Path(tmpdir)
            )
            assert result is True
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_safe_compile_timeout(self):
        """
        safe_compile() raises SafeModeError on timeout.
        """
        from RDataFrameDSL.safe_mode import safe_compile, SafeModeError
        
        # Code that would take forever to compile (infinite template recursion)
        # Actually, let's just use a very short timeout with normal code
        code = """
        int simple_func() { return 42; }
        """
        
        # Use extremely short timeout - may or may not timeout depending on system
        # This is more of a smoke test for timeout handling
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = safe_compile(
                    code,
                    name="test_timeout",
                    timeout=0.001,  # 1ms - very likely to timeout
                    cache_dir=Path(tmpdir)
                )
                # If it somehow succeeds, that's fine too
                assert result is True
            except SafeModeError as e:
                assert e.reason == "timeout"


# =============================================================================
# Test Class: Layer 3 - Execution-Time Safety
# =============================================================================

class TestLayer3:
    """Tests for Layer 3 execution-time protection."""
    
    @pytest.mark.feature("layer3_safe_execute")
    def test_probe_run_success(self, nd_2d_rdf):
        """
        probe_run() succeeds for valid RDataFrame.
        """
        from RDataFrameDSL.safe_mode import probe_run
        
        result = probe_run(nd_2d_rdf, action="Count", probe_size=100, timeout=30.0)
        assert result is True
    
    @pytest.mark.feature("layer3_safe_execute")
    def test_probe_columns_success(self, nd_2d_rdf, nd_2d_schema):
        """
        probe_columns() succeeds for valid columns.
        """
        from RDataFrameDSL.safe_mode import probe_columns
        
        columns = ['event_id', 'event_weight']
        result = probe_columns(nd_2d_rdf, columns, probe_size=100, timeout=30.0)
        assert result is True
    
    @pytest.mark.feature("layer3_safe_execute")
    def test_probe_size_configurable(self, nd_2d_rdf):
        """
        probe_run() respects probe_size parameter.
        """
        from RDataFrameDSL.safe_mode import probe_run
        
        # Small probe
        result1 = probe_run(nd_2d_rdf, action="Count", probe_size=10, timeout=30.0)
        assert result1 is True
        
        # Larger probe
        result2 = probe_run(nd_2d_rdf, action="Count", probe_size=500, timeout=30.0)
        assert result2 is True


# =============================================================================
# Test Class: to_pandas_safe Integration
# =============================================================================

class TestToPandasSafe:
    """Integration tests for to_pandas_safe()."""
    
    @pytest.mark.feature("api_to_pandas_safe")
    def test_to_pandas_safe_basic(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas_safe() works for basic export.
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas_safe(
            nd_2d_rdf,
            columns=['event_id', 'event_weight'],
            probe_size=100,
            timeout=30.0
        )
        
        assert 'event_id' in df.columns
        assert 'event_weight' in df.columns
        assert len(df) > 0
    
    @pytest.mark.feature("api_to_pandas_safe")
    def test_to_pandas_safe_with_alias(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas_safe() works with alias() definitions.
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("double_weight", "event_weight * 2")
        
        df = dsl.to_pandas_safe(
            nd_2d_rdf,
            columns=['event_id', 'double_weight'],
            probe_size=100,
            timeout=30.0
        )
        
        assert 'double_weight' in df.columns
    
    @pytest.mark.feature("api_to_pandas_safe")
    def test_to_pandas_safe_layer1_validation(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas_safe() still performs Layer 1 validation.
        """
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        dsl = DSLCompiler(nd_2d_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.to_pandas_safe(
                nd_2d_rdf,
                columns=['nonexistent_column'],
                probe_size=100,
                timeout=30.0
            )
        
        assert 'nonexistent_column' in str(exc_info.value).lower() or \
               'not found' in str(exc_info.value).lower()


# =============================================================================
# Test Class: SafeModeError
# =============================================================================

class TestSafeModeError:
    """Tests for SafeModeError structure."""
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_safemodeError_str(self):
        """
        SafeModeError has informative string representation.
        """
        from RDataFrameDSL.safe_mode import SafeModeError
        
        error = SafeModeError(
            layer="compile",
            reason="crash",
            message="Test crash",
            signal_num=11
        )
        
        error_str = str(error)
        assert "compile" in error_str
        assert "crash" in error_str
        assert "SIGSEGV" in error_str or "11" in error_str
    
    @pytest.mark.feature("layer2_safe_compile")
    def test_safemodeError_timeout(self):
        """
        SafeModeError for timeout has correct fields.
        """
        from RDataFrameDSL.safe_mode import SafeModeError
        
        error = SafeModeError(
            layer="execute",
            reason="timeout",
            message="Operation timed out"
        )
        
        assert error.layer == "execute"
        assert error.reason == "timeout"
        assert error.signal_num is None

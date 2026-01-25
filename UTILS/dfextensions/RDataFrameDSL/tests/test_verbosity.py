"""
Tests for Phase 13.6.G verbosity and logging system.

Tests:
- Verbosity bitmask constants
- Presets (VERBOSE_MINIMAL, VERBOSE_DEFAULT, VERBOSE_FULL)
- describe_structure() method
- Side-effect free contract
- Python logging integration
"""

import pytest
import logging


class TestVerbosityConstants:
    """Test verbosity bitmask constants and presets."""
    
    def test_bitmask_values(self):
        """Each flag has unique power-of-two value."""
        from RDataFrameDSL.verbosity import (
            VERBOSITY_BASIC, VERBOSITY_SCHEMA, VERBOSITY_ALIASES,
            VERBOSITY_ALIASES_FULL, VERBOSITY_DEFINES, VERBOSITY_COMPILED,
            VERBOSITY_SAFE_MODE, VERBOSITY_CACHE
        )
        
        flags = [
            VERBOSITY_BASIC, VERBOSITY_SCHEMA, VERBOSITY_ALIASES,
            VERBOSITY_ALIASES_FULL, VERBOSITY_DEFINES, VERBOSITY_COMPILED,
            VERBOSITY_SAFE_MODE, VERBOSITY_CACHE
        ]
        
        # Each flag is power of 2
        for flag in flags:
            assert flag & (flag - 1) == 0, f"Flag {flag} is not power of 2"
        
        # All flags unique
        assert len(flags) == len(set(flags))
    
    def test_bitmask_combinations(self):
        """Flags can be combined with OR."""
        from RDataFrameDSL.verbosity import (
            VERBOSITY_BASIC, VERBOSITY_SCHEMA, VERBOSITY_ALIASES
        )
        
        combined = VERBOSITY_BASIC | VERBOSITY_SCHEMA | VERBOSITY_ALIASES
        assert combined == 0x07
        
        # Test individual flags
        assert combined & VERBOSITY_BASIC
        assert combined & VERBOSITY_SCHEMA
        assert combined & VERBOSITY_ALIASES
    
    def test_presets(self):
        """Presets have expected values."""
        from RDataFrameDSL.verbosity import (
            VERBOSE_MINIMAL, VERBOSE_DEFAULT, VERBOSE_FULL, VERBOSE_ALL,
            VERBOSITY_BASIC
        )
        
        assert VERBOSE_MINIMAL == VERBOSITY_BASIC
        assert VERBOSE_FULL == 0xFF
        assert VERBOSE_ALL == VERBOSE_FULL
        
        # Default includes basic + schema + aliases + defines
        assert VERBOSE_DEFAULT & VERBOSITY_BASIC
    
    def test_describe_flags(self):
        """describe_flags() returns readable names."""
        from RDataFrameDSL.verbosity import (
            VERBOSITY_BASIC, VERBOSITY_SCHEMA, describe_flags
        )
        
        desc = describe_flags(VERBOSITY_BASIC | VERBOSITY_SCHEMA)
        assert 'VERBOSITY_BASIC' in desc
        assert 'VERBOSITY_SCHEMA' in desc


class TestDescribeStructure:
    """Test DSLCompiler.describe_structure() method."""
    
    def test_returns_string_by_default(self):
        """describe_structure() returns formatted string."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({'x': 'double', 'y': 'double'})
        result = dsl.describe_structure()
        
        assert isinstance(result, str)
        assert 'DSLCompiler Status' in result
        assert 'x' in result or 'Schema' in result
    
    def test_returns_dict_when_requested(self):
        """describe_structure(return_dict=True) returns dict."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({'x': 'double'})
        result = dsl.describe_structure(return_dict=True)
        
        assert isinstance(result, dict)
        assert 'schema' in result
        assert 'n_columns' in result
        assert result['n_columns'] == 1
    
    def test_shows_aliases(self):
        """describe_structure() includes alias pool."""
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.verbosity import VERBOSITY_ALIASES
        
        dsl = DSLCompiler({'x': 'double'})
        dsl.alias("y", "x * 2")
        
        result = dsl.describe_structure(VERBOSITY_ALIASES)
        assert 'y' in result or 'alias' in result.lower()
        
        # Dict form
        info = dsl.describe_structure(return_dict=True)
        assert 'y' in info['aliases']
    
    def test_verbosity_levels(self):
        """Different verbosity levels show different info."""
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.verbosity import (
            VERBOSE_MINIMAL, VERBOSE_DEFAULT, VERBOSE_FULL
        )
        
        dsl = DSLCompiler({'x': 'double', 'y': 'double'})
        
        minimal = dsl.describe_structure(VERBOSE_MINIMAL)
        default = dsl.describe_structure(VERBOSE_DEFAULT)
        full = dsl.describe_structure(VERBOSE_FULL)
        
        # More verbosity = more output
        assert len(minimal) <= len(default) <= len(full)


class TestDescribeStructureSideEffectFree:
    """
    CRITICAL: Verify describe_structure() is side-effect free.
    
    Contract from Phase 13.6.G Proposal §1.3:
    ❌ MUST NOT trigger compilation
    ❌ MUST NOT trigger type inference
    ❌ MUST NOT trigger execution
    ❌ MUST NOT mutate schema
    ❌ MUST NOT materialize aliases
    """
    
    def test_does_not_trigger_compilation(self):
        """describe_structure() with invalid alias must not raise."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({'x': 'double'})
        
        # Add invalid alias (references nonexistent column)
        dsl._aliases['bad'] = 'nonexistent + 1'
        
        # describe_structure() should NOT trigger compilation
        # So it should NOT raise an error
        result = dsl.describe_structure()
        assert isinstance(result, str)
        
        # Alias still in pool, not compiled
        assert 'bad' in dsl._aliases
        assert 'bad' not in dsl.schema
    
    def test_does_not_mutate_schema(self):
        """describe_structure() must not change schema."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({'x': 'double'})
        schema_before = dict(dsl.schema)
        
        dsl.alias("y", "x * 2")
        dsl.describe_structure()
        
        schema_after = dict(dsl.schema)
        assert schema_before == schema_after
    
    def test_does_not_materialize_aliases(self):
        """describe_structure() must not compile aliases."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({'x': 'double'})
        dsl.alias("y", "x * 2")
        dsl.alias("z", "y + 1")
        
        n_aliases_before = len(dsl._aliases)
        n_defined_before = len(dsl._defined_aliases)
        
        dsl.describe_structure()
        
        # Aliases still in pool
        assert len(dsl._aliases) == n_aliases_before
        assert len(dsl._defined_aliases) == n_defined_before


class TestLogging:
    """Test Python logging integration."""
    
    def test_logger_exists(self):
        """RDataFrameDSL logger is available."""
        logger = logging.getLogger("RDataFrameDSL")
        assert logger is not None
    
    def test_default_level_is_warning_or_higher(self):
        """Logger defaults to WARNING level (silent by default)."""
        logger = logging.getLogger("RDataFrameDSL")
        # Should be WARNING or higher (less verbose)
        assert logger.level >= logging.WARNING or logger.level == 0
    
    def test_verbose_parameter_works(self, caplog):
        """verbose=True enables DEBUG messages."""
        from RDataFrameDSL import DSLCompiler
        
        # Ensure logger propagates (required for caplog)
        logger = logging.getLogger("RDataFrameDSL")
        logger.propagate = True
        
        dsl = DSLCompiler({'x': 'double'})
        
        with caplog.at_level(logging.DEBUG, logger="RDataFrameDSL"):
            # Call with verbose=True
            dsl.define("y", "x * 2", verbose=True)
        
        # Should have logged something at DEBUG level
        debug_messages = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_messages) > 0, f"Expected DEBUG messages with verbose=True. Records: {[r.message for r in caplog.records]}"
    
    def test_logging_prefixes(self, caplog):
        """Log messages include structured prefixes."""
        from RDataFrameDSL import DSLCompiler
        
        # Ensure logger propagates (required for caplog)
        logger = logging.getLogger("RDataFrameDSL")
        logger.propagate = True
        
        dsl = DSLCompiler({'x': 'double'})
        
        with caplog.at_level(logging.DEBUG, logger="RDataFrameDSL"):
            dsl.define("y", "x * 2", verbose=True)
        
        # Check for structured prefixes
        messages = [r.message for r in caplog.records]
        has_prefix = any('[compile]' in m or '[alias]' in m for m in messages)
        assert has_prefix, f"Expected structured prefixes in: {messages}"


# Pytest markers
pytest.mark.feature("verbosity")
pytest.mark.phase("13.6.G")

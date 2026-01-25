"""
Invariance tests for alias order independence and syntax equivalence.

Phase 13.6.G: Alias pool order-independence vs define dependency-order.

Feature: alias_order_independence
Test IDs: INV-ALIAS-EQ-1 through INV-ALIAS-EQ-5

Core Invariant:
    alias() definitions in ANY order produce identical results when materialized.
    
Note: Aliases are materialized via to_pandas(columns=[...]) or draw(),
NOT via define("x", "alias_name").
"""
import pytest
import numpy as np


class TestInvarianceAliasEquivalence:
    """Invariance tests for alias order independence (INV-ALIAS-EQ-*)."""
    
    @pytest.fixture
    def compiler_and_data(self):
        """Set up DSL compiler and test data."""
        import ROOT
        from tests.generators.toy_nd import generate_nd_2d_root
        from RDataFrameDSL import DSLCompiler
        
        filename = generate_nd_2d_root(size='S', seed=42)
        rdf = ROOT.RDataFrame("Events", filename)
        
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'event_weight': 'double',
            'track_pt': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        return schema, rdf
    
    @pytest.mark.feature("alias_order_independence")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ALIAS_EQ_1_reverse_order_vs_define(self, compiler_and_data):
        """
        INV-ALIAS-EQ-1: Aliases in reverse dependency order == defines in correct order.
        
        Invariant: alias pool resolves references at compile time, not definition time.
        
        Alias order (reverse):
            total -> scaled -> first -> weight
            
        Define order (correct):
            first -> weight -> scaled -> total
        """
        from RDataFrameDSL import DSLCompiler
        schema, rdf = compiler_and_data
        
        # Method A: Aliases in REVERSE dependency order
        dsl_alias = DSLCompiler(schema)
        dsl_alias.alias("total_A", "scaled_A + first_A")         # Uses undefined!
        dsl_alias.alias("scaled_A", "Sum(track_pt) * weight_A")  # Uses undefined!
        dsl_alias.alias("first_A", "track_pt[0]")
        dsl_alias.alias("weight_A", "event_weight")
        
        # Materialize via to_pandas
        df_A = dsl_alias.to_pandas(rdf, columns=['total_A'])
        
        # Method D: Defines in CORRECT dependency order
        dsl_define = DSLCompiler(schema)
        dsl_define.define("first_D", "track_pt[0]")
        dsl_define.define("weight_D", "event_weight")
        dsl_define.define("scaled_D", "Sum(track_pt) * weight_D")
        dsl_define.define("total_D", "scaled_D + first_D")
        
        df_D = dsl_define.to_pandas(rdf, columns=['total_D'])
        
        # INVARIANCE CHECK
        result_A = df_A['total_A'].values
        result_D = df_D['total_D'].values
        
        assert len(result_A) == len(result_D), "Length mismatch"
        assert np.allclose(result_A, result_D, rtol=1e-10), \
            f"Alias vs Define mismatch: max_diff={np.max(np.abs(result_A - result_D))}"
    
    @pytest.mark.feature("alias_order_independence")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ALIAS_EQ_2_permutation_invariance(self, compiler_and_data):
        """
        INV-ALIAS-EQ-2: Any alias definition order produces identical results.
        
        Invariant: Alias resolution is order-independent.
        
        Tests 4 different orderings of the same 5 aliases.
        """
        from RDataFrameDSL import DSLCompiler
        schema, rdf = compiler_and_data
        
        def build_with_order(alias_order):
            """Build DSL with aliases in specified order."""
            dsl = DSLCompiler(schema)
            aliases = {
                'total': ("total", "scaled + first"),
                'scaled': ("scaled", "Sum(track_pt) * weight"),
                'first': ("first", "track_pt[0]"),
                'weight': ("weight", "event_weight"),
            }
            for key in alias_order:
                name, expr = aliases[key]
                dsl.alias(name, expr)
            return dsl
        
        orderings = [
            ['first', 'weight', 'scaled', 'total'],  # Forward
            ['total', 'scaled', 'first', 'weight'],  # Reverse
            ['weight', 'total', 'first', 'scaled'],  # Random1
            ['scaled', 'weight', 'total', 'first'],  # Random2
        ]
        
        results = []
        for order in orderings:
            dsl = build_with_order(order)
            df = dsl.to_pandas(rdf, columns=['total'])
            results.append(df['total'].values)
        
        # All must match the first
        reference = results[0]
        for i, res in enumerate(results[1:], 1):
            assert np.allclose(reference, res, rtol=1e-10), \
                f"Order {i} differs from reference: max_diff={np.max(np.abs(reference - res))}"
    
    @pytest.mark.feature("alias_order_independence")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ALIAS_EQ_3_inline_vs_decomposed(self, compiler_and_data):
        """
        INV-ALIAS-EQ-3: Inline define() == decomposed alias chain via to_pandas.
        
        Invariant: Breaking an expression into aliases doesn't change the result.
        
        Uses Sum(track_pt) * event_weight + track_pt[0] to test bounded indexing
        in a binary expression (requires proper ternary parenthesization).
        """
        from RDataFrameDSL import DSLCompiler
        schema, rdf = compiler_and_data
        
        # Method 1: Single inline expression via define
        dsl1 = DSLCompiler(schema)
        dsl1.define("result", "Sum(track_pt) * event_weight + track_pt[0]")
        df1 = dsl1.to_pandas(rdf, columns=['result'])
        
        # Method 2: Full decomposition with aliases
        dsl2 = DSLCompiler(schema)
        dsl2.alias("pt_sum", "Sum(track_pt)")
        dsl2.alias("weighted", "pt_sum * event_weight")
        dsl2.alias("first", "track_pt[0]")
        dsl2.alias("result", "weighted + first")
        df2 = dsl2.to_pandas(rdf, columns=['result'])
        
        # INVARIANCE CHECK
        r1 = df1['result'].values
        r2 = df2['result'].values
        
        assert np.allclose(r1, r2, rtol=1e-10), \
            f"Inline vs Decomposed mismatch: max_diff={np.max(np.abs(r1 - r2))}"
    
    @pytest.mark.feature("alias_order_independence")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ALIAS_EQ_4_commutativity(self, compiler_and_data):
        """
        INV-ALIAS-EQ-4: a * b == b * a for scalar × scalar reduction.
        
        Invariant: Multiplication commutativity holds.
        """
        from RDataFrameDSL import DSLCompiler
        schema, rdf = compiler_and_data
        
        # Method 1: Sum(...) * event_weight
        dsl1 = DSLCompiler(schema)
        dsl1.define("result", "Sum(track_pt) * event_weight + track_pt[0]")
        df1 = dsl1.to_pandas(rdf, columns=['result'])
        
        # Method 2: event_weight * Sum(...)
        dsl2 = DSLCompiler(schema)
        dsl2.define("result", "event_weight * Sum(track_pt) + track_pt[0]")
        df2 = dsl2.to_pandas(rdf, columns=['result'])
        
        r1 = df1['result'].values
        r2 = df2['result'].values
        
        assert np.allclose(r1, r2, rtol=1e-10), \
            f"Commutativity failed: max_diff={np.max(np.abs(r1 - r2))}"
    
    @pytest.mark.feature("alias_order_independence")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ALIAS_EQ_5_partial_alias(self, compiler_and_data):
        """
        INV-ALIAS-EQ-5: Partial alias (some inline, some aliased) == full inline.
        
        Invariant: Mixing alias and inline produces same result.
        """
        from RDataFrameDSL import DSLCompiler
        schema, rdf = compiler_and_data
        
        # Method 1: Fully inline via define
        dsl1 = DSLCompiler(schema)
        dsl1.define("result", "Sum(track_pt) * event_weight + track_pt[0]")
        df1 = dsl1.to_pandas(rdf, columns=['result'])
        
        # Method 2: Partial - alias for middle part, inline for rest
        dsl2 = DSLCompiler(schema)
        dsl2.alias("middle_sum", "Sum(track_pt)")
        dsl2.alias("result", "middle_sum * event_weight + track_pt[0]")
        df2 = dsl2.to_pandas(rdf, columns=['result'])
        
        r1 = df1['result'].values
        r2 = df2['result'].values
        
        assert np.allclose(r1, r2, rtol=1e-10), \
            f"Partial alias mismatch: max_diff={np.max(np.abs(r1 - r2))}"

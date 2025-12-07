"""
Tests for DFDraw class.
"""

import pytest
import pandas as pd
import numpy as np

from dfdraw import DFDraw


class TestDFDrawInit:
    """Test DFDraw initialization."""
    
    def test_init_with_dataframe(self):
        """Accept pandas DataFrame."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        plotter = DFDraw(df)
        assert len(plotter.df) == 3
    
    def test_init_with_dict(self):
        """Accept dict of arrays."""
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        plotter = DFDraw(data)
        assert isinstance(plotter.df, pd.DataFrame)
        assert list(plotter.df.columns) == ['x', 'y']
    
    def test_init_with_alias_dataframe_like(self):
        """Accept object with .df attribute."""
        class MockADF:
            def __init__(self):
                self.df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        
        adf = MockADF()
        plotter = DFDraw(adf)
        assert len(plotter.df) == 2
    
    def test_init_invalid_type_raises(self):
        """Reject invalid input types."""
        with pytest.raises(TypeError, match="Cannot create DFDraw"):
            DFDraw("invalid")


class TestExpressionParsing:
    """Test expression parsing."""
    
    def setup_method(self):
        self.df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        self.plotter = DFDraw(self.df)
    
    def test_parse_1d_expr(self):
        """Parse single variable expression."""
        y, x = self.plotter._parse_expr("x")
        assert y == "x"
        assert x is None
    
    def test_parse_2d_expr(self):
        """Parse y:x expression."""
        y, x = self.plotter._parse_expr("y:x")
        assert y == "y"
        assert x == "x"
    
    def test_parse_expr_with_spaces(self):
        """Handle spaces in expression."""
        y, x = self.plotter._parse_expr(" y : x ")
        assert y == "y"
        assert x == "x"
    
    def test_parse_invalid_expr_raises(self):
        """Reject invalid expressions."""
        with pytest.raises(ValueError, match="Invalid expression"):
            self.plotter._parse_expr("a:b:c:d")


class TestColumnEvaluation:
    """Test column/expression evaluation."""
    
    def setup_method(self):
        self.df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        self.plotter = DFDraw(self.df)
    
    def test_eval_direct_column(self):
        """Evaluate direct column name."""
        result = self.plotter._eval_column("x")
        assert list(result) == [1, 2, 3]
    
    def test_eval_computed_expression(self):
        """Evaluate computed expression."""
        result = self.plotter._eval_column("x + y")
        assert list(result) == [5, 7, 9]
    
    def test_eval_complex_expression(self):
        """Evaluate complex expression."""
        result = self.plotter._eval_column("x * 2 + y")
        assert list(result) == [6, 9, 12]
    
    def test_eval_invalid_raises(self):
        """Invalid expression should raise."""
        with pytest.raises(ValueError, match="Cannot evaluate"):
            self.plotter._eval_column("nonexistent_column")


class TestSelection:
    """Test selection/filtering."""
    
    def setup_method(self):
        self.df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'cat': ['a', 'a', 'b', 'b', 'b']
        })
        self.plotter = DFDraw(self.df)
    
    def test_selection_none(self):
        """None selection returns all data."""
        result = self.plotter._apply_selection(self.df, None)
        assert len(result) == 5
    
    def test_selection_string_query(self):
        """String selection uses pandas query."""
        result = self.plotter._apply_selection(self.df, "x > 2")
        assert len(result) == 3
        assert list(result['x']) == [3, 4, 5]
    
    def test_selection_callable(self):
        """Callable selection."""
        result = self.plotter._apply_selection(
            self.df, 
            lambda df: df['cat'] == 'a'
        )
        assert len(result) == 2
    
    def test_selection_boolean_mask(self):
        """Boolean mask selection."""
        mask = np.array([True, False, True, False, True])
        result = self.plotter._apply_selection(self.df, mask)
        assert len(result) == 3


class TestSampling:
    """Test random sampling."""
    
    def setup_method(self):
        self.df = pd.DataFrame({'x': range(1000)})
        self.plotter = DFDraw(self.df)
    
    def test_sampling_none_returns_all(self):
        """No sampling returns all data."""
        result = self.plotter._apply_sampling(self.df, None)
        assert len(result) == 1000
    
    def test_sampling_larger_than_data_returns_all(self):
        """Sampling larger than data returns all."""
        result = self.plotter._apply_sampling(self.df, 5000)
        assert len(result) == 1000
    
    def test_sampling_limits_size(self):
        """Sampling limits data size."""
        result = self.plotter._apply_sampling(self.df, 100)
        assert len(result) == 100
    
    def test_sampling_is_reproducible(self):
        """Sampling with same seed is reproducible."""
        result1 = self.plotter._apply_sampling(self.df, 50)
        result2 = self.plotter._apply_sampling(self.df, 50)
        assert list(result1.index) == list(result2.index)


class TestDrawDispatch:
    """Test draw() method dispatch."""
    
    def setup_method(self):
        self.df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        self.plotter = DFDraw(self.df)
    
    def test_draw_1d_dispatches_to_hist(self):
        """1D expression should dispatch to hist."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax, stats = self.plotter.draw("x")
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 3
        plt.close('all')
    
    def test_draw_2d_dispatches_to_scatter(self):
        """2D expression should dispatch to scatter."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax, stats = self.plotter.draw("y:x")
        assert isinstance(fig, plt.Figure)
        assert stats['n'] == 3
        plt.close('all')
    
    def test_draw_explicit_type_profile(self):
        """Explicit type=profile should dispatch correctly."""
        with pytest.raises(NotImplementedError, match="profile"):
            self.plotter.draw("y:x", type="profile")
    
    def test_draw_invalid_type_raises(self):
        """Invalid type should raise."""
        with pytest.raises(ValueError, match="Unknown plot type"):
            self.plotter.draw("y:x", type="invalid")

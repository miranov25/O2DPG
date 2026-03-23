"""
PolynomialSpec — N-dimensional polynomial specification for AliasDataFrame.

Phase 13.9.ADF: Polynomial Alias Support

Generates:
  - (key_name, expression) tuples for GroupBy Regression linear_columns
  - Numba JIT evaluators for alias registration (42× faster than eval)
  - Schema entries for JSON serialization and reconstruction

Architecture:
  - Standalone class, no imports from GroupBy Regression or dfdraw
  - Coefficients read from subframe via join indices (no materialization)
  - Generic N-dimensional (not hardcoded to 3D or 4D)

Usage:
    from AliasDataFrame.PolynomialSpec import PolynomialSpec

    spec = PolynomialSpec(['xM', 'driftM', 'dsecM', 'tgSlp'], (3, 3, 2, 1))
    linear_cols = spec.basis_expressions()  # 96 (key_name, expression) tuples
    # ... fit with GB Regression ...
    evaluator = spec.numba_evaluator(adf, 'PolyFit', coeff_cols)
    adf.register_function('polFit', evaluator)
    adf.add_alias('dy_corr', 'polFit(xM, driftM, dsecM, tgSlp)')
"""

import itertools
import numpy as np


class PolynomialSpec:
    """
    N-dimensional polynomial specification for AliasDataFrame.

    Generates basis expression strings (for fitting) and
    Numba JIT evaluators (for alias registration).

    Parameters
    ----------
    columns : list of str
        Column names for polynomial variables.
        Any number of dimensions supported.
    degrees : tuple of int
        Maximum degree per variable (e.g., (3, 3, 2, 1)).
        Must have same length as columns.
    term_filter : callable, optional
        Function(exponent_tuple) -> bool. Only terms where
        term_filter returns True are included.
        Example: lambda e: e[3] == 0  # only terms with tgSlp^0

    Examples
    --------
    >>> spec = PolynomialSpec(['xM', 'driftM', 'dsecM', 'tgSlp'], (3, 3, 2, 1))
    >>> spec.n_terms
    96
    >>> spec.basis_expressions()[:3]
    [('xM0_driftM0_dsecM0_tgSlp0', '1'), ('xM0_driftM0_dsecM0_tgSlp1', 'tgSlp'), ...]

    >>> # Sparse: only terms where tgSlp degree = 0
    >>> spec_spatial = PolynomialSpec(['xM', 'driftM', 'dsecM', 'tgSlp'], (3, 3, 2, 1),
    ...     term_filter=lambda e: e[3] == 0)
    >>> spec_spatial.n_terms
    48

    >>> # 2D case works identically
    >>> spec2d = PolynomialSpec(['x', 'y'], (2, 3))
    >>> spec2d.n_terms
    12
    """

    def __init__(self, columns, degrees, term_filter=None):
        self.columns = list(columns)
        self.degrees = tuple(degrees)
        self.term_filter = term_filter

        if len(columns) != len(degrees):
            raise ValueError(
                f"columns ({len(columns)}) and degrees ({len(degrees)}) "
                f"must have same length"
            )

        # Generate term exponents
        self._terms = self._generate_terms()

    def _generate_terms(self):
        """Generate list of exponent tuples, respecting term_filter."""
        ranges = [range(d + 1) for d in self.degrees]
        terms = []
        for exponents in itertools.product(*ranges):
            if self.term_filter is None or self.term_filter(exponents):
                terms.append(exponents)
        return terms

    @property
    def n_terms(self):
        """Number of basis terms."""
        return len(self._terms)

    @property
    def terms(self):
        """List of exponent tuples."""
        return list(self._terms)

    # =========================================================================
    # Basis Expression Generation (for GroupBy Regression)
    # =========================================================================

    def basis_expressions(self):
        """
        Generate (key_name, expression) tuples for GB Regression linear_columns.

        Key names follow deterministic naming convention:
            "<var0><order0>_<var1><order1>_..."

        Examples:
            ("xM0_driftM0_dsecM0_tgSlp0", "1")                    — constant
            ("xM1_driftM0_dsecM0_tgSlp0", "xM")                   — linear xM
            ("xM0_driftM0_dsecM0_tgSlp1", "tgSlp")                — linear tgSlp
            ("xM2_driftM1_dsecM0_tgSlp1", "xM**2*driftM*tgSlp")   — cross-term

        Returns
        -------
        list of tuple(str, str)
            [(key_name, expression), ...]
        """
        result = []
        for exponents in self._terms:
            # Build key name: "xM2_driftM1_dsecM0_tgSlp1"
            key_parts = [f"{col}{exp}" for col, exp in
                         zip(self.columns, exponents)]
            key_name = "_".join(key_parts)

            # Build expression: "xM**2*driftM*tgSlp"
            expr_parts = []
            for col, exp in zip(self.columns, exponents):
                if exp == 0:
                    continue
                elif exp == 1:
                    expr_parts.append(col)
                else:
                    expr_parts.append(f"{col}**{exp}")

            expression = "*".join(expr_parts) if expr_parts else "1"
            result.append((key_name, expression))

        return result

    # =========================================================================
    # Numba JIT Evaluator (for alias registration)
    # =========================================================================

    def numba_evaluator(self, adf, coefficients_subframe, coeff_select):
        """
        Generate Numba JIT evaluator that reads coefficients from subframe
        via join indices. No column materialization — coefficient matrix
        stays at subframe size (e.g., 36 rows × 96 cols = 27 KB).

        Proven: PoC test passed 2026-03-23 (9/9 rows correct, alias integration OK).

        Parameters
        ----------
        adf : AliasDataFrame
            Main ADF (needed for join index computation)
        coefficients_subframe : str
            Name of registered subframe containing coefficients
        coeff_select : list of str
            Coefficient column names in subframe, ordered to match terms.

        Returns
        -------
        callable
            Function(*column_arrays) -> np.ndarray
            Arguments are the base column arrays in self.columns order.

        Architecture
        ------------
        - join_indices: int64[N_main] — computed via _compute_join_indices (cached)
        - coeff_matrix: float64[N_subframe × N_terms] — tiny, fits in L1 cache
        - Powers precomputed per row: O(sum(degrees)) multiplications
        - Term evaluation: O(N_terms × N_dims) table lookups per row
        - No temporary arrays, no column materialization
        """
        from numba import njit, prange

        # Get subframe metadata
        index_cols = adf._subframes.get_entry(coefficients_subframe)['index']
        if isinstance(index_cols, str):
            index_cols = [index_cols]

        # Extract coefficient matrix (tiny: N_groups × N_terms)
        sf = adf.get_subframe(coefficients_subframe)

        if len(coeff_select) != len(self._terms):
            raise ValueError(
                f"coeff_select has {len(coeff_select)} columns, "
                f"but polynomial has {len(self._terms)} terms"
            )

        coeff_matrix = sf.df[coeff_select].values.astype(np.float64)

        # Build arrays for Numba (no Python objects in kernel)
        exponent_array = np.array(self._terms, dtype=np.int32)
        max_degrees = np.array(self.degrees, dtype=np.int32)
        n_dims = len(self.columns)
        n_terms = len(self._terms)
        sf_name = coefficients_subframe  # capture for closure
        max_deg_val = int(np.max(max_degrees))

        @njit(parallel=True, fastmath=True)
        def _eval_kernel(join_idx, coeff_mat, exponents, max_degs,
                         n_d, n_t, max_d, *columns):
            n = len(columns[0])
            result = np.empty(n, dtype=np.float64)
            for i in prange(n):
                idx = join_idx[i]
                if idx < 0:
                    result[i] = np.nan
                    continue
                # Precompute powers for all dimensions (generic N-dim)
                powers = np.empty((n_d, max_d + 1), dtype=np.float64)
                for d in range(n_d):
                    powers[d, 0] = 1.0
                    for p in range(1, max_degs[d] + 1):
                        powers[d, p] = powers[d, p - 1] * columns[d][i]
                # Evaluate polynomial via table lookup
                val = 0.0
                for t in range(n_t):
                    basis = 1.0
                    for d in range(n_d):
                        basis *= powers[d, exponents[t, d]]
                    val += coeff_mat[idx, t] * basis
                result[i] = val
            return result

        def evaluator(*args):
            """Evaluate polynomial. Args = column arrays in spec.columns order."""
            join_idx, _ = adf._compute_join_indices(sf_name, index_cols)
            return _eval_kernel(
                join_idx, coeff_matrix, exponent_array, max_degrees,
                n_dims, n_terms, max_deg_val,
                *[np.asarray(a, dtype=np.float64) for a in args]
            )

        return evaluator

    # =========================================================================
    # Schema Serialization
    # =========================================================================

    def to_schema(self):
        """
        Export as JSON-serializable dict.

        terms field is only included for sparse polynomials (when term_filter
        was used). Full polynomials can be reconstructed from columns + degrees.

        Returns
        -------
        dict
        """
        schema = {
            'type': 'polynomialND',
            'columns': self.columns,
            'degrees': list(self.degrees),
        }

        # Only include terms if sparse (subset of full product)
        full_product = list(itertools.product(*[range(d + 1) for d in self.degrees]))
        if len(self._terms) != len(full_product):
            schema['terms'] = [list(t) for t in self._terms]

        return schema

    @classmethod
    def from_schema(cls, schema):
        """
        Reconstruct from JSON schema.

        Parameters
        ----------
        schema : dict
            Output from to_schema()

        Returns
        -------
        PolynomialSpec
        """
        spec = cls(
            columns=schema['columns'],
            degrees=tuple(schema['degrees']),
        )
        # Override generated terms with explicit list (preserves sparse specs)
        if 'terms' in schema:
            spec._terms = [tuple(t) for t in schema['terms']]
        return spec

    # =========================================================================
    # ROOT C++ Expression Generation
    # =========================================================================

    def to_root_expression(self, coefficients):
        """
        Generate ROOT C++ expression string for TTree::Draw / TTreeFormula.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficient values matching terms order

        Returns
        -------
        str
            C++ expression (e.g., "0.0012 + (-0.0034)*xM + 0.0056*xM*xM + ...")
        """
        parts = []
        for coeff, exponents in zip(coefficients, self._terms):
            if abs(coeff) < 1e-15:
                continue
            term_parts = [f"({coeff:.10e})"]
            for col, exp in zip(self.columns, exponents):
                for _ in range(exp):
                    term_parts.append(col)
            parts.append("*".join(term_parts))
        return " + ".join(parts) if parts else "0"

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self):
        n = self.n_terms
        full_n = 1
        for d in self.degrees:
            full_n *= (d + 1)
        sparse = f", sparse={n}/{full_n}" if n != full_n else ""
        cols = ", ".join(self.columns)
        return f"PolynomialSpec([{cols}], {self.degrees}{sparse})"

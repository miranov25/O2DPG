"""
RDataFrameDSL Invariant Testing — Test Data Generator

Phase 13.2.1.DSL: Generate ROOT TTree with mathematical invariants.

This module generates test data where mathematical invariants hold by
construction. The DSL compiler tests verify these invariants are preserved
when generating C++ code for ROOT RDataFrame.

Usage:
    from .test_data_generator import generate_invariant_tree
    
    generate_invariant_tree("test_data.root", n_events=1000, seed=42)

The generated TTree contains:
    - Scalar columns (double, float, int, uint, bool) with arithmetic invariants
    - RVec columns with element-wise invariants
    - C-array branches (legacy TTree pattern) with indexing invariants
    - Edge case values (near-zero, large magnitude)
"""

import array
import numpy as np
from typing import Optional
from pathlib import Path

from .invariant_schema import (
    DEFAULT_N_EVENTS,
    DEFAULT_SEED,
    MAX_RVEC_LENGTH,
    MAX_CARRAY_LENGTH,
    RVEC_LENGTH_PATTERN,
    VALUE_RANGES,
    INVARIANTS,
)


def generate_invariant_tree(
    filepath: str,
    n_events: int = DEFAULT_N_EVENTS,
    seed: int = DEFAULT_SEED,
    tree_name: str = "invariants",
) -> str:
    """
    Generate a ROOT TTree with mathematical invariants.
    
    Parameters:
        filepath: Output ROOT file path
        n_events: Number of events to generate
        seed: Random seed for reproducibility
        tree_name: Name of the TTree
    
    Returns:
        Absolute path to generated file
    
    The generated tree contains columns where invariants hold by construction:
        - sum_ab = A + B
        - prod_ab = A * B
        - flag_and = flag_a && flag_b
        - vec_sum = vec_a + vec_b (element-wise)
        - picked = arr_d[idx]
        - etc.
    """
    import ROOT
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create file and tree
    filepath = str(Path(filepath).absolute())
    f = ROOT.TFile(filepath, "RECREATE")
    tree = ROOT.TTree(tree_name, "Invariant test data")
    
    # =========================================================================
    # SCALAR BRANCHES — Double precision
    # =========================================================================
    A = array.array('d', [0.0])
    B = array.array('d', [0.0])
    C = array.array('d', [0.0])
    sum_ab = array.array('d', [0.0])
    prod_ab = array.array('d', [0.0])
    A_positive = array.array('d', [0.0])
    prec_add_mul = array.array('d', [0.0])
    prec_mul_add = array.array('d', [0.0])
    
    tree.Branch("A", A, "A/D")
    tree.Branch("B", B, "B/D")
    tree.Branch("C", C, "C/D")
    tree.Branch("sum_ab", sum_ab, "sum_ab/D")
    tree.Branch("prod_ab", prod_ab, "prod_ab/D")
    tree.Branch("A_positive", A_positive, "A_positive/D")
    tree.Branch("prec_add_mul", prec_add_mul, "prec_add_mul/D")
    tree.Branch("prec_mul_add", prec_mul_add, "prec_mul_add/D")
    
    # =========================================================================
    # SCALAR BRANCHES — Single precision (float)
    # =========================================================================
    Af = array.array('f', [0.0])
    Bf = array.array('f', [0.0])
    sum_f = array.array('f', [0.0])
    
    tree.Branch("Af", Af, "Af/F")
    tree.Branch("Bf", Bf, "Bf/F")
    tree.Branch("sum_f", sum_f, "sum_f/F")
    
    # =========================================================================
    # SCALAR BRANCHES — Signed integers
    # =========================================================================
    Ai = array.array('i', [0])
    Bi = array.array('i', [0])
    sum_i = array.array('i', [0])
    
    tree.Branch("Ai", Ai, "Ai/I")
    tree.Branch("Bi", Bi, "Bi/I")
    tree.Branch("sum_i", sum_i, "sum_i/I")
    
    # =========================================================================
    # SCALAR BRANCHES — Unsigned integers
    # =========================================================================
    Au = array.array('I', [0])
    Bu = array.array('I', [0])
    sum_u = array.array('I', [0])
    
    tree.Branch("Au", Au, "Au/i")  # lowercase 'i' for unsigned in ROOT
    tree.Branch("Bu", Bu, "Bu/i")
    tree.Branch("sum_u", sum_u, "sum_u/i")
    
    # =========================================================================
    # BOOLEAN BRANCHES
    # =========================================================================
    flag_a = array.array('B', [0])  # 'B' = unsigned char for bool
    flag_b = array.array('B', [0])
    flag_c = array.array('B', [0])
    flag_and = array.array('B', [0])
    flag_or = array.array('B', [0])
    flag_not_a = array.array('B', [0])
    
    tree.Branch("flag_a", flag_a, "flag_a/O")  # 'O' = Bool_t in ROOT
    tree.Branch("flag_b", flag_b, "flag_b/O")
    tree.Branch("flag_c", flag_c, "flag_c/O")
    tree.Branch("flag_and", flag_and, "flag_and/O")
    tree.Branch("flag_or", flag_or, "flag_or/O")
    tree.Branch("flag_not_a", flag_not_a, "flag_not_a/O")
    
    # =========================================================================
    # EDGE CASE BRANCHES
    # =========================================================================
    near_zero = array.array('d', [0.0])
    large_val = array.array('d', [0.0])
    
    tree.Branch("near_zero", near_zero, "near_zero/D")
    tree.Branch("large_val", large_val, "large_val/D")
    
    # =========================================================================
    # RVec BRANCHES
    # =========================================================================
    vec_a = ROOT.std.vector('double')()
    vec_b = ROOT.std.vector('double')()
    vec_sum = ROOT.std.vector('double')()
    vec_len = array.array('i', [0])
    
    vec_i = ROOT.std.vector('int')()
    vec_i_doubled = ROOT.std.vector('int')()
    
    tree.Branch("vec_a", vec_a)
    tree.Branch("vec_b", vec_b)
    tree.Branch("vec_sum", vec_sum)
    tree.Branch("vec_len", vec_len, "vec_len/I")
    tree.Branch("vec_i", vec_i)
    tree.Branch("vec_i_doubled", vec_i_doubled)
    
    # =========================================================================
    # C-ARRAY BRANCHES (Legacy TTree pattern via leaflist)
    # =========================================================================
    n_arr = array.array('i', [0])
    arr_d = array.array('d', [0.0] * MAX_CARRAY_LENGTH)
    arr_i = array.array('i', [0] * MAX_CARRAY_LENGTH)
    arr_sum = array.array('d', [0.0] * MAX_CARRAY_LENGTH)
    idx = array.array('i', [0])
    picked = array.array('d', [0.0])
    
    tree.Branch("n_arr", n_arr, "n_arr/I")
    tree.Branch("arr_d", arr_d, "arr_d[n_arr]/D")
    tree.Branch("arr_i", arr_i, "arr_i[n_arr]/I")
    tree.Branch("arr_sum", arr_sum, "arr_sum[n_arr]/D")
    tree.Branch("idx", idx, "idx/I")
    tree.Branch("picked", picked, "picked/D")
    
    # =========================================================================
    # FILL EVENTS
    # =========================================================================
    n_lengths = len(RVEC_LENGTH_PATTERN)
    
    for i in range(n_events):
        # ---------------------------------------------------------------------
        # Double scalars with arithmetic invariants
        # ---------------------------------------------------------------------
        A[0] = np.random.uniform(-100, 100)
        B[0] = np.random.uniform(-100, 100)
        C[0] = np.random.uniform(-100, 100)
        
        sum_ab[0] = A[0] + B[0]           # Invariant: sum_ab - A == B
        prod_ab[0] = A[0] * B[0]          # Invariant: prod_ab / A == B (if A != 0)
        A_positive[0] = abs(A[0]) + 0.001 # Always > 0 for division tests
        
        prec_add_mul[0] = A[0] + (B[0] * C[0])  # Precedence: A + B * C
        prec_mul_add[0] = (A[0] + B[0]) * C[0]  # Precedence: (A + B) * C
        
        # ---------------------------------------------------------------------
        # Float scalars
        # ---------------------------------------------------------------------
        Af[0] = np.float32(np.random.uniform(-100, 100))
        Bf[0] = np.float32(np.random.uniform(-100, 100))
        sum_f[0] = Af[0] + Bf[0]
        
        # ---------------------------------------------------------------------
        # Integer scalars
        # ---------------------------------------------------------------------
        Ai[0] = np.random.randint(-1000, 1001)
        Bi[0] = np.random.randint(-1000, 1001)
        sum_i[0] = Ai[0] + Bi[0]
        
        # ---------------------------------------------------------------------
        # Unsigned integer scalars
        # ---------------------------------------------------------------------
        Au[0] = np.random.randint(0, 1001)
        Bu[0] = np.random.randint(0, 1001)
        sum_u[0] = Au[0] + Bu[0]
        
        # ---------------------------------------------------------------------
        # Boolean values with De Morgan invariants
        # ---------------------------------------------------------------------
        flag_a[0] = np.random.randint(0, 2)
        flag_b[0] = np.random.randint(0, 2)
        flag_c[0] = np.random.randint(0, 2)
        
        flag_and[0] = flag_a[0] and flag_b[0]  # Invariant: !(a&&b) == !a || !b
        flag_or[0] = flag_a[0] or flag_b[0]    # Invariant: !(a||b) == !a && !b
        flag_not_a[0] = not flag_a[0]          # Invariant: !!a == a
        
        # ---------------------------------------------------------------------
        # Edge case values
        # ---------------------------------------------------------------------
        near_zero[0] = np.random.uniform(-1e-15, 1e-15)
        large_val[0] = np.random.uniform(-1e10, 1e10)
        
        # ---------------------------------------------------------------------
        # RVec values with element-wise invariants
        # ---------------------------------------------------------------------
        # Variable length from pattern (includes 0, 1, and larger)
        length = RVEC_LENGTH_PATTERN[i % n_lengths]
        
        vec_a.clear()
        vec_b.clear()
        vec_sum.clear()
        vec_i.clear()
        vec_i_doubled.clear()
        
        for j in range(length):
            va = np.random.uniform(-100, 100)
            vb = np.random.uniform(-100, 100)
            vec_a.push_back(va)
            vec_b.push_back(vb)
            vec_sum.push_back(va + vb)  # Invariant: vec_sum[j] == vec_a[j] + vec_b[j]
            
            vi = np.random.randint(-100, 101)
            vec_i.push_back(vi)
            vec_i_doubled.push_back(vi * 2)  # Invariant: vec_i_doubled[j] == vec_i[j] * 2
        
        vec_len[0] = length  # Invariant: vec_len == Size(vec_a)
        
        # ---------------------------------------------------------------------
        # C-array values with indexing invariants
        # ---------------------------------------------------------------------
        # Variable length [1, MAX_CARRAY_LENGTH] (never 0 for valid indexing)
        arr_length = np.random.randint(1, MAX_CARRAY_LENGTH + 1)
        n_arr[0] = arr_length
        
        for j in range(arr_length):
            arr_d[j] = np.random.uniform(-100, 100)
            arr_i[j] = np.random.randint(-1000, 1001)
            arr_sum[j] = arr_d[j] + 1.0  # Invariant: arr_sum[j] == arr_d[j] + 1.0
        
        # Valid index into array
        idx[0] = np.random.randint(0, arr_length)
        picked[0] = arr_d[idx[0]]  # Invariant: picked == arr_d[idx]
        
        # Fill the tree
        tree.Fill()
    
    # Write and close
    tree.Write()
    f.Close()
    
    return filepath


def get_invariant_tree_info(filepath: str, tree_name: str = "invariants") -> dict:
    """
    Get information about a generated invariant tree.
    
    Parameters:
        filepath: Path to ROOT file
        tree_name: Name of the TTree
    
    Returns:
        Dict with tree information (n_events, branches, etc.)
    """
    import ROOT
    
    f = ROOT.TFile(filepath, "READ")
    tree = f.Get(tree_name)
    
    info = {
        "filepath": filepath,
        "tree_name": tree_name,
        "n_events": tree.GetEntries(),
        "branches": [b.GetName() for b in tree.GetListOfBranches()],
        "n_branches": tree.GetNbranches(),
    }
    
    f.Close()
    return info


# =============================================================================
# VERIFICATION HELPERS (for sanity tests)
# =============================================================================

def verify_invariants_python(filepath: str, tree_name: str = "invariants") -> dict:
    """
    Verify invariants using Python (uproot or PyROOT).
    
    This is the sanity check to ensure the generator produces correct data.
    
    Parameters:
        filepath: Path to ROOT file
        tree_name: Name of the TTree
    
    Returns:
        Dict with verification results:
            - passed: bool
            - checks: list of (name, passed, message)
    """
    import ROOT
    
    rdf = ROOT.RDataFrame(tree_name, filepath)
    n_events = rdf.Count().GetValue()
    
    checks = []
    all_passed = True
    
    # =========================================================================
    # Arithmetic invariants
    # =========================================================================
    
    # Check: sum_ab == A + B
    count = rdf.Filter("abs(sum_ab - (A + B)) > 1e-12").Count().GetValue()
    passed = count == 0
    checks.append(("sum_ab == A + B", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: prod_ab == A * B
    count = rdf.Filter("abs(prod_ab - (A * B)) > 1e-12").Count().GetValue()
    passed = count == 0
    checks.append(("prod_ab == A * B", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: A_positive > 0
    count = rdf.Filter("A_positive <= 0").Count().GetValue()
    passed = count == 0
    checks.append(("A_positive > 0", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: prec_add_mul == A + (B * C)
    count = rdf.Filter("abs(prec_add_mul - (A + B * C)) > 1e-10").Count().GetValue()
    passed = count == 0
    checks.append(("prec_add_mul == A + B * C", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: prec_mul_add == (A + B) * C
    count = rdf.Filter("abs(prec_mul_add - ((A + B) * C)) > 1e-12").Count().GetValue()
    passed = count == 0
    checks.append(("prec_mul_add == (A + B) * C", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # =========================================================================
    # Float invariants (with tolerance)
    # =========================================================================
    
    # Check: sum_f == Af + Bf (float tolerance)
    count = rdf.Filter("abs(sum_f - (Af + Bf)) > 1e-5").Count().GetValue()
    passed = count == 0
    checks.append(("sum_f == Af + Bf", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # =========================================================================
    # Integer invariants (exact)
    # =========================================================================
    
    # Check: sum_i == Ai + Bi
    count = rdf.Filter("sum_i != (Ai + Bi)").Count().GetValue()
    passed = count == 0
    checks.append(("sum_i == Ai + Bi", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: sum_u == Au + Bu
    count = rdf.Filter("sum_u != (Au + Bu)").Count().GetValue()
    passed = count == 0
    checks.append(("sum_u == Au + Bu", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # =========================================================================
    # Boolean invariants
    # =========================================================================
    
    # Check: flag_and == (flag_a && flag_b)
    count = rdf.Filter("flag_and != (flag_a && flag_b)").Count().GetValue()
    passed = count == 0
    checks.append(("flag_and == flag_a && flag_b", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: flag_or == (flag_a || flag_b)
    count = rdf.Filter("flag_or != (flag_a || flag_b)").Count().GetValue()
    passed = count == 0
    checks.append(("flag_or == flag_a || flag_b", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: flag_not_a == !flag_a
    count = rdf.Filter("flag_not_a != (!flag_a)").Count().GetValue()
    passed = count == 0
    checks.append(("flag_not_a == !flag_a", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # =========================================================================
    # RVec invariants
    # =========================================================================
    
    # Check: vec_len == Size(vec_a)
    count = rdf.Filter("vec_len != (int)vec_a.size()").Count().GetValue()
    passed = count == 0
    checks.append(("vec_len == Size(vec_a)", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: vec_a and vec_b same length
    count = rdf.Filter("vec_a.size() != vec_b.size()").Count().GetValue()
    passed = count == 0
    checks.append(("Size(vec_a) == Size(vec_b)", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # =========================================================================
    # C-array invariants
    # =========================================================================
    
    # Check: idx < n_arr (valid index)
    count = rdf.Filter("idx >= n_arr").Count().GetValue()
    passed = count == 0
    checks.append(("idx < n_arr", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    # Check: picked == arr_d[idx]
    count = rdf.Filter("abs(picked - arr_d[idx]) > 1e-12").Count().GetValue()
    passed = count == 0
    checks.append(("picked == arr_d[idx]", passed, f"{count} violations"))
    all_passed = all_passed and passed
    
    return {
        "passed": all_passed,
        "n_events": n_events,
        "n_checks": len(checks),
        "n_passed": sum(1 for _, p, _ in checks if p),
        "checks": checks,
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for generating test data."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ROOT TTree with mathematical invariants for DSL testing"
    )
    parser.add_argument(
        "-o", "--output",
        default="invariant_data.root",
        help="Output file path (default: invariant_data.root)"
    )
    parser.add_argument(
        "-n", "--n-events",
        type=int,
        default=DEFAULT_N_EVENTS,
        help=f"Number of events (default: {DEFAULT_N_EVENTS})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify invariants after generation"
    )
    
    args = parser.parse_args()
    
    print(f"Generating invariant test data...")
    print(f"  Output: {args.output}")
    print(f"  Events: {args.n_events}")
    print(f"  Seed: {args.seed}")
    
    filepath = generate_invariant_tree(
        args.output,
        n_events=args.n_events,
        seed=args.seed,
    )
    
    info = get_invariant_tree_info(filepath)
    print(f"\nGenerated:")
    print(f"  File: {info['filepath']}")
    print(f"  Events: {info['n_events']}")
    print(f"  Branches: {info['n_branches']}")
    
    if args.verify:
        print("\nVerifying invariants...")
        result = verify_invariants_python(filepath)
        print(f"  Checks: {result['n_passed']}/{result['n_checks']} passed")
        
        if not result['passed']:
            print("\n  FAILURES:")
            for name, passed, msg in result['checks']:
                if not passed:
                    print(f"    ✗ {name}: {msg}")
            return 1
        else:
            print("  ✓ All invariants verified")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
Example: DSL → Draw Integration
===============================

Phase 13.6.F demonstration of the complete workflow:
  1. Create RDataFrame from data
  2. Use DSLCompiler with alias() for deferred validation
  3. Use draw() with Layer 1 validation
  4. Use batch_draw() for multiple plots

This proves the TTree::Draw-like functionality works end-to-end.

Usage:
    python examples/04_dsl_draw.py
    
Requirements:
    - ROOT with RDataFrame
    - dfdraw (pip install dfdraw) for visualization
    - matplotlib
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_data(n_events=1000):
    """Create test RDataFrame with track-like data."""
    import ROOT
    
    # Create RDataFrame with simulated track data
    rdf = ROOT.RDataFrame(n_events)
    
    # Add event-level columns
    rdf = rdf.Define("event_id", "rdfentry_")
    rdf = rdf.Define("n_tracks", "gRandom->Poisson(5)")
    rdf = rdf.Define("event_weight", "gRandom->Gaus(1.0, 0.1)")
    
    # Add track-level columns (1D arrays)
    rdf = rdf.Define("track_pt", 
        "ROOT::RVecD v(n_tracks); for(auto& x:v) x=gRandom->Exp(2.0); return v;")
    rdf = rdf.Define("track_eta",
        "ROOT::RVecD v(n_tracks); for(auto& x:v) x=gRandom->Gaus(0,1.0); return v;")
    rdf = rdf.Define("track_phi",
        "ROOT::RVecD v(n_tracks); for(auto& x:v) x=gRandom->Uniform(-3.14159,3.14159); return v;")
    
    return rdf


def example_basic_draw():
    """
    Example 1: Basic draw() with validation.
    
    Demonstrates:
    - Layer 1 validation catches typos
    - Alias materialization
    - Single histogram creation
    """
    print("\n" + "="*60)
    print("Example 1: Basic draw() with alias")
    print("="*60)
    
    from RDataFrameDSL import DSLCompiler
    
    # Create test data
    rdf = create_test_data(1000)
    
    # Infer schema from RDataFrame
    dsl = DSLCompiler.from_rdf(rdf)
    print(f"Schema inferred: {list(dsl.schema.keys())}")
    
    # Define aliases (deferred validation - TTree::SetAlias style)
    dsl.alias("pt_gev", "track_pt")  # Already in GeV in our mock
    dsl.alias("high_pt", "track_pt > 2.0")
    dsl.alias("central", "abs(track_eta) < 1.0")
    
    print("Aliases defined (not yet validated):")
    print(f"  - pt_gev: track_pt")
    print(f"  - high_pt: track_pt > 2.0")
    print(f"  - central: abs(track_eta) < 1.0")
    
    # Draw histogram - this triggers alias materialization
    try:
        fig, ax, stats = dsl.draw("track_pt", rdf, bins=50)
        # stats structure depends on dfdraw version
        count = stats.get('count', stats.get('n', len(ax.patches) if hasattr(ax, 'patches') else 'N/A'))
        print(f"\n✓ Histogram created: {count} entries")
        fig.savefig("track_pt_histogram.png", dpi=150)
        print("  Saved to: track_pt_histogram.png")
    except ImportError:
        print("\n⚠ dfdraw not installed - skipping plot creation")
        print("  Install with: pip install dfdraw")
    
    # Demonstrate Layer 1 validation catching errors
    print("\nLayer 1 validation test:")
    from RDataFrameDSL.ir_errors import IRError
    try:
        dsl.draw("nonexistent_column", rdf)
    except IRError as e:
        print(f"  ✓ IRError caught: {str(e)[:50]}...")


def example_batch_draw():
    """
    Example 2: batch_draw() for multiple plots.
    
    Demonstrates:
    - Single data extraction for efficiency
    - All-or-nothing validation
    - Multiple plot types
    """
    print("\n" + "="*60)
    print("Example 2: batch_draw() for QA plots")
    print("="*60)
    
    from RDataFrameDSL import DSLCompiler
    
    rdf = create_test_data(2000)
    dsl = DSLCompiler.from_rdf(rdf)
    
    # Define computed columns
    dsl.define("pt_squared", "track_pt * track_pt")
    dsl.alias("eta_abs", "abs(track_eta)")
    
    # Batch plot specifications
    specs = {
        'pt_dist': {
            'expr': 'track_pt',
            'bins': 50,
        },
        'eta_dist': {
            'expr': 'track_eta', 
            'bins': 40,
        },
        'phi_dist': {
            'expr': 'track_phi',
            'bins': 60,
        },
        'eta_vs_phi': {
            'expr': 'track_eta:track_phi',
            'type': 'hist2d',
            'bins': [30, 30],
        },
    }
    
    print(f"Batch specs defined: {list(specs.keys())}")
    
    try:
        results = dsl.draw_batch(specs, rdf, save_dir='qa_plots')
        print(f"\n✓ Created {len(results)} plots:")
        for name, result in results.items():
            print(f"  - {name}: plot created")
        print("  Saved to: qa_plots/")
    except ImportError:
        print("\n⚠ dfdraw not installed - skipping batch plots")


def example_safe_mode():
    """
    Example 3: Safe mode with probe-run protection.
    
    Demonstrates:
    - to_pandas_safe() with crash protection
    - draw() with safe_mode=True
    """
    print("\n" + "="*60)
    print("Example 3: Safe mode (probe-run protection)")
    print("="*60)
    
    import sys
    if sys.platform == "win32":
        print("⚠ Safe mode requires fork() - not available on Windows")
        return
    
    from RDataFrameDSL import DSLCompiler
    
    rdf = create_test_data(500)
    dsl = DSLCompiler.from_rdf(rdf)
    
    # Safe export with probe-run
    print("to_pandas_safe() with probe_size=100...")
    df = dsl.to_pandas_safe(
        rdf,
        columns=['event_id', 'track_pt', 'track_eta'],
        probe_size=100
    )
    print(f"  ✓ Exported {len(df)} rows safely")
    print(f"  Columns: {list(df.columns)}")
    
    # Safe draw
    print("\ndraw() with safe_mode=True...")
    try:
        fig, ax, stats = dsl.draw(
            "track_pt", 
            rdf, 
            safe_mode=True,
            probe_size=100
        )
        print(f"  ✓ Safe draw completed")
    except ImportError:
        print("  ⚠ dfdraw not installed - but probe passed!")


def example_alias_chain():
    """
    Example 4: Complex alias chains.
    
    Demonstrates:
    - Alias dependency resolution
    - Type propagation
    - Pool-based compilation (only used aliases compiled)
    """
    print("\n" + "="*60)
    print("Example 4: Alias chains and selective compilation")
    print("="*60)
    
    from RDataFrameDSL import DSLCompiler
    
    rdf = create_test_data(500)
    dsl = DSLCompiler.from_rdf(rdf)
    
    # Define many aliases - only used ones will be compiled
    dsl.alias("pt", "track_pt")
    dsl.alias("pt_cut", "pt > 1.0")
    dsl.alias("eta", "track_eta")
    dsl.alias("eta_cut", "abs(eta) < 2.0")
    # NOTE: Use Python 'and', not C++ '&&'
    dsl.alias("good_track", "pt_cut and eta_cut")
    
    # Also define some unused aliases (will stay in pool)
    dsl.alias("unused_1", "track_phi * 2")
    dsl.alias("unused_2", "n_tracks + 1")
    
    print("Aliases in pool (before):", list(dsl._aliases.keys()))
    
    # Request only good_track - should compile: pt, pt_cut, eta, eta_cut, good_track
    df = dsl.to_pandas(rdf, columns=['event_id', 'good_track'])
    
    print("Aliases in pool (after):", list(dsl._aliases.keys()))
    print("Aliases compiled:", [k for k in ['pt', 'pt_cut', 'eta', 'eta_cut', 'good_track'] 
                                if k in dsl.schema])
    print(f"\n✓ Exported {len(df)} rows")
    print(f"  good_track True: {df['good_track'].sum()}")
    print(f"  good_track False: {(~df['good_track']).sum()}")


def main():
    """Run all examples."""
    print("="*60)
    print("RDataFrameDSL Phase 13.6.F - Draw API Examples")
    print("="*60)
    
    example_basic_draw()
    example_batch_draw()
    example_safe_mode()
    example_alias_chain()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()

"""
Benchmark Framework v1.0 — Visualization CLI Commands

Phase 12.14c.GB D3: CLI commands for benchmark visualization.

Commands:
    --history       Show benchmark history summary
    --history-stats Show noise statistics per benchmark
    --plot DIR      Generate trend plots to directory

Exit Codes:
    0 - Success
    1 - No data found
    2 - Partial success (some plots failed)
    3 - Dependency missing (matplotlib/pyyaml)

Key constraints (from review):
    P1-2: matplotlib lazy import + Agg backend + graceful error
    P1-5: CLI flags mutually exclusive
    P1-7: Missing-column graceful SKIP
    P1-8: Timestamp normalization
    P1-9: Exit code semantics documented
"""

import logging
from pathlib import Path
from typing import Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# MOCK ARGS FOR TESTING
# =============================================================================

class MockArgs:
    """Mock args object for testing CLI commands."""
    def __init__(self, **kwargs):
        self.subproject = kwargs.get('subproject', 'test')
        self.max_runs = kwargs.get('max_runs', None)
        self.baseline = kwargs.get('baseline', '7d')
        self.plot = kwargs.get('plot', None)
        self.history = kwargs.get('history', False)
        self.history_stats = kwargs.get('history_stats', False)


# =============================================================================
# --history COMMAND
# =============================================================================

def cmd_history(args: Any) -> int:
    """
    Handle --history command.
    
    Shows summary of benchmark history including:
    - Total results count
    - Unique benchmarks
    - Date range
    - Pass rate
    - Recent runs summary
    
    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments with subproject and max_runs
    
    Returns
    -------
    int
        Exit code: 0=success, 1=no data
    """
    from .benchmark_adf import load_benchmark_adf
    
    adf = load_benchmark_adf(args.subproject, max_runs=args.max_runs)
    df = adf.df
    
    if df.empty:
        print(f"\nNo benchmark history found for '{args.subproject}'")
        return 1
    
    # P1-8: Timestamp normalization
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Summary header
    print(f"\nBenchmark History: {args.subproject}")
    print("═" * 68)
    print(f"  Total results:      {len(df)}")
    print(f"  Unique benchmarks:  {df['benchmark_id'].nunique()}")
    
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        ts_min = df['timestamp'].min()
        ts_max = df['timestamp'].max()
        if hasattr(ts_min, 'strftime'):
            print(f"  Date range:         {ts_min.strftime('%Y-%m-%d %H:%M')} → {ts_max.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"  Date range:         {ts_min} → {ts_max}")
    
    if 'status' in df.columns:
        n_passed = (df['status'] == 'OK').sum()
        pct = 100 * n_passed / len(df) if len(df) > 0 else 0
        print(f"  Pass rate:          {n_passed}/{len(df)} ({pct:.1f}%)")
    
    # Recent runs (grouped by run_id)
    if 'run_id' in df.columns:
        print("\n  Recent Runs:")
        print("  " + "─" * 66)
        
        agg_dict = {}
        if 'timestamp' in df.columns:
            agg_dict['timestamp'] = 'first'
        if 'status' in df.columns:
            agg_dict['status'] = lambda x: (x == 'OK').sum()
        agg_dict['benchmark_id'] = 'count'
        if 'peak_rss_mb' in df.columns:
            agg_dict['peak_rss_mb'] = 'max'
        
        if agg_dict:
            recent = df.groupby('run_id').agg(agg_dict)
            if 'timestamp' in recent.columns:
                recent = recent.sort_values('timestamp', ascending=False)
            recent = recent.head(5)
            
            for run_id, row in recent.iterrows():
                parts = []
                
                if 'timestamp' in row.index and pd.notna(row['timestamp']):
                    ts = row['timestamp']
                    if hasattr(ts, 'strftime'):
                        parts.append(ts.strftime('%Y-%m-%dT%H:%M:%S'))
                    else:
                        parts.append(str(ts))
                
                if 'status' in row.index:
                    passed = int(row['status'])
                    total = int(row['benchmark_id'])
                    parts.append(f"{passed}/{total} passed")
                
                if 'peak_rss_mb' in row.index and pd.notna(row['peak_rss_mb']):
                    rss = row['peak_rss_mb']
                    parts.append(f"Peak RSS: {rss:.0f} MB")
                
                print(f"    {' | '.join(parts)}")
        
        print("  " + "─" * 66)
    
    return 0


# =============================================================================
# --history-stats COMMAND
# =============================================================================

def cmd_history_stats(args: Any) -> int:
    """
    Handle --history-stats command.
    
    Shows noise statistics per benchmark for alarm threshold tuning:
    - Mean execution time
    - Standard deviation
    - Coefficient of variation (CV%)
    - Sample count
    - High noise flag
    
    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments with subproject and baseline
    
    Returns
    -------
    int
        Exit code: 0=success, 1=no data
    """
    from .benchmark_adf import compute_benchmark_statistics
    
    stats = compute_benchmark_statistics(
        args.subproject,
        baseline=getattr(args, 'baseline', '7d'),
    )
    
    if stats.empty:
        print(f"\nNo statistics available for '{args.subproject}'")
        return 1
    
    baseline = getattr(args, 'baseline', '7d')
    print(f"\nBenchmark Statistics: {args.subproject} (baseline: {baseline})")
    print("═" * 68)
    print(f"  {'Benchmark':<35} {'Mean':>8} {'Std':>8} {'CV%':>7} {'N':>4}")
    print("  " + "─" * 66)
    
    for _, row in stats.iterrows():
        bid = str(row['benchmark_id'])[:35]
        
        mean_s = row.get('mean_time_s', 0)
        std_s = row.get('std_time_s', 0)
        cv = row.get('cv_pct', 0)
        n = int(row.get('n_samples', 0))
        high_noise = row.get('high_noise', False)
        
        # Format time (ms for sub-second, s for larger)
        if mean_s < 1:
            mean_str = f"{mean_s * 1000:>6.1f}ms"
            std_str = f"{std_s * 1000:>6.1f}ms"
        else:
            mean_str = f"{mean_s:>6.2f}s "
            std_str = f"{std_s:>6.2f}s "
        
        flag = "  ⚠ HIGH" if high_noise else ""
        
        print(f"  {bid:<35} {mean_str} {std_str} {cv:>6.1f}% {n:>4}{flag}")
    
    print("  " + "─" * 66)
    
    high_noise_count = stats['high_noise'].sum() if 'high_noise' in stats.columns else 0
    total = len(stats)
    print(f"\n  Summary: {high_noise_count}/{total} benchmarks have high noise (CV > 10%)")
    
    if high_noise_count > total / 2:
        print("\n  Recommendation: Sub-millisecond kernels have inherently high CV.")
        print("  Consider using median instead of mean for alarm thresholds.")
    
    return 0


# =============================================================================
# --plot COMMAND
# =============================================================================

def cmd_plot(args: Any) -> int:
    """
    Handle --plot command.
    
    Generates PNG trend plots to specified directory based on specs.
    
    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments with subproject, plot (directory), and max_runs
    
    Returns
    -------
    int
        Exit code: 0=success, 1=no data, 2=partial success, 3=dependency missing
    """
    # P1-2: matplotlib lazy import with graceful error
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for servers
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nError: matplotlib required for --plot")
        print("Install with: pip install matplotlib")
        return 3
    
    from .benchmark_adf import load_benchmark_adf
    from .specs import load_benchmark_specs, get_enabled_specs
    
    plot_dir = Path(args.plot)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots to: {plot_dir}")
    print("═" * 68)
    
    # Load data
    print("  Loading benchmark history...")
    try:
        adf = load_benchmark_adf(args.subproject, max_runs=getattr(args, 'max_runs', None))
        df = adf.df
    except Exception as e:
        print(f"  Error loading data: {e}")
        return 1
    
    if df.empty:
        print(f"  No data found for '{args.subproject}'")
        return 1
    
    # P1-8: Timestamp normalization
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    print(f"  Loaded {len(df)} results across {df['benchmark_id'].nunique()} benchmarks")
    print("\n  Generating plots:")
    
    # Load specs
    try:
        specs = get_enabled_specs()
    except Exception as e:
        print(f"  Error loading specs: {e}")
        return 3
    
    generated = 0
    skipped = 0
    failed = 0
    
    for spec in specs:
        name = spec['name']
        result = _generate_plot(df, spec, plot_dir, plt)
        
        if result == 'ok':
            print(f"    ✓ {name}.png")
            generated += 1
        elif result.startswith('skip:'):
            reason = result[5:]
            print(f"    ○ {name}.png — SKIP: {reason}")
            skipped += 1
        else:
            print(f"    ✗ {name}.png — ERROR: {result}")
            failed += 1
    
    print(f"\n  Generated {generated} plots to {plot_dir}")
    if skipped > 0:
        print(f"  Skipped {skipped} plots (missing columns or no data)")
    if failed > 0:
        print(f"  Failed {failed} plots")
    
    # Exit codes per spec
    if generated == 0:
        return 1  # No data
    elif failed > 0:
        return 2  # Partial success
    else:
        return 0  # Success


def _generate_plot(df: pd.DataFrame, spec: dict, output_dir: Path, plt) -> str:
    """
    Generate a single plot from spec.
    
    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data
    spec : dict
        Plot specification
    output_dir : Path
        Output directory for PNG
    plt : module
        matplotlib.pyplot module (passed to avoid reimport)
    
    Returns
    -------
    str
        'ok' on success, 'skip:reason' for graceful skip, or error message
    """
    name = spec['name']
    kind = spec.get('kind', 'scatter')
    x = spec.get('x')
    y = spec.get('y')
    groupby = spec.get('groupby')
    
    # P1-7: Missing column handling - skip with message
    if x and x not in df.columns:
        return f"skip:missing column '{x}'"
    if y and y not in df.columns:
        return f"skip:missing column '{y}'"
    if groupby and groupby not in df.columns:
        return f"skip:missing column '{groupby}'"
    
    try:
        # Apply filter if specified
        plot_df = df.copy()
        if 'filter' in spec:
            try:
                plot_df = plot_df.query(spec['filter'])
            except Exception as e:
                return f"skip:filter error: {e}"
        
        if plot_df.empty:
            return "skip:no data after filter"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if kind == 'hist':
            plot_df[x].hist(ax=ax, bins=spec.get('bins', 30))
            ax.set_xlabel(x)
            ax.set_ylabel('Count')
        
        elif kind in ('scatter', 'line'):
            if groupby:
                groups = plot_df.groupby(groupby)
                # P2: Legend cap for high cardinality (max 15)
                n_groups = len(groups)
                show_legend = n_groups <= 15
                
                for gname, group in groups:
                    if kind == 'scatter':
                        ax.scatter(group[x], group[y], label=gname, alpha=0.7, s=20)
                    else:
                        ax.plot(group[x], group[y], label=gname, marker='o', markersize=3)
                
                if show_legend:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                elif n_groups > 15:
                    # Add note when legend is hidden
                    ax.text(0.99, 0.01, f"({n_groups} groups, legend hidden)",
                            transform=ax.transAxes, ha='right', va='bottom',
                            fontsize=8, color='gray')
            else:
                if kind == 'scatter':
                    ax.scatter(plot_df[x], plot_df[y], alpha=0.7)
                else:
                    ax.plot(plot_df[x], plot_df[y], marker='o')
            
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        
        elif kind == 'bar':
            if groupby:
                plot_df.groupby(groupby)[y].mean().sort_values().plot(kind='bar', ax=ax)
            else:
                plot_df[y].plot(kind='bar', ax=ax)
            ax.set_ylabel(y)
        
        ax.set_title(spec.get('title', name))
        plt.tight_layout()
        
        output_path = output_dir / f"{name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return 'ok'
    
    except Exception as e:
        return str(e)


# =============================================================================
# CLI ARGUMENT INTEGRATION
# =============================================================================

def add_visualization_args(parser) -> None:
    """
    Add Phase 12.14c.GB visualization arguments to parser.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to
    """
    import argparse
    
    viz_group = parser.add_argument_group('Visualization (Phase 12.14c.GB)')
    
    # P1-5: Mutually exclusive group for viz commands
    viz_exclusive = viz_group.add_mutually_exclusive_group()
    
    viz_exclusive.add_argument(
        '--history',
        action='store_true',
        help='Show benchmark history summary',
    )
    
    viz_exclusive.add_argument(
        '--history-stats',
        action='store_true',
        help='Show noise statistics (mean, std, CV) per benchmark',
    )
    
    viz_exclusive.add_argument(
        '--plot',
        type=str,
        metavar='DIR',
        help='Generate trend plots to specified directory',
    )
    
    # Additional options for visualization commands
    viz_group.add_argument(
        '--baseline',
        type=str,
        default='7d',
        metavar='RANGE',
        help='Baseline range for statistics (default: 7d)',
    )
    
    viz_group.add_argument(
        '--max-runs',
        type=int,
        default=None,
        metavar='N',
        help='Maximum runs to load (default: all)',
    )


def handle_visualization_command(args) -> Optional[int]:
    """
    Handle visualization commands if present.
    
    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments
    
    Returns
    -------
    int or None
        Exit code if visualization command was handled, None otherwise
    """
    if getattr(args, 'history', False):
        return cmd_history(args)
    
    if getattr(args, 'history_stats', False):
        return cmd_history_stats(args)
    
    if getattr(args, 'plot', None):
        return cmd_plot(args)
    
    return None

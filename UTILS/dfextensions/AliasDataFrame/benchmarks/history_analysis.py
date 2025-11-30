#!/usr/bin/env python3
"""
history_analysis.py - Load benchmark history into DataFrames for analysis

Provides both long and wide format DataFrames for flexible querying.

Usage:
    from history_analysis import load_history_long, load_history_wide
    
    # Long format (one row per metric) - good for filtering/grouping
    df_long = load_history_long('results/history/')
    
    # Wide format (one row per run) - good for correlation analysis
    df_wide = load_history_wide('results/history/')

CLI:
    python history_analysis.py list results/history/
    python history_analysis.py show results/history/ --last 10
    python history_analysis.py export results/history/ --format wide -o history.csv
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


def load_history_to_records(history_dir):
    """
    Load all history JSON files into a list of records.
    
    Parameters
    ----------
    history_dir : str
        Path to history directory
        
    Returns
    -------
    list : List of dicts, one per JSON file
    """
    history_path = Path(history_dir)
    records = []
    
    for json_file in sorted(history_path.glob('benchmark_*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            data['_source_file'] = str(json_file.name)
            records.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    return records


def load_history_long(history_dir):
    """
    Load history into long-format DataFrame (one row per metric).
    
    Good for: filtering, grouping, time series of single metrics
    
    Columns:
        timestamp, commit, branch, dirty, host, source_file,
        benchmark, metric, value
    
    Parameters
    ----------
    history_dir : str
        Path to history directory
        
    Returns
    -------
    pd.DataFrame : Long-format DataFrame
    """
    records = load_history_to_records(history_dir)
    
    rows = []
    for record in records:
        # Extract common fields
        git = record.get('git', {})
        base_row = {
            'timestamp': record.get('created'),
            'commit': git.get('commit_short'),
            'commit_full': git.get('commit'),
            'branch': git.get('branch'),
            'dirty': git.get('dirty'),
            'commit_date': git.get('commit_date'),
            'commit_message': git.get('commit_message'),
            'host': record.get('host'),
            'python_version': record.get('python_version'),
            'source_file': record.get('_source_file'),
        }
        
        # Extract metrics from each benchmark
        for bench_name, bench_data in record.get('benchmarks', {}).items():
            # Add time_s as a metric
            if bench_data.get('time_s') is not None:
                row = base_row.copy()
                row['benchmark'] = bench_name
                row['metric'] = 'time_s'
                row['value'] = bench_data['time_s']
                rows.append(row)
            
            # Add all other metrics
            for metric_name, metric_value in bench_data.get('metrics', {}).items():
                if isinstance(metric_value, (int, float)):
                    row = base_row.copy()
                    row['benchmark'] = bench_name
                    row['metric'] = metric_name
                    row['value'] = metric_value
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns and len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def load_history_wide(history_dir):
    """
    Load history into wide-format DataFrame (one row per run).
    
    Good for: correlation analysis, comparing multiple metrics
    
    Columns:
        timestamp, commit, branch, dirty, host, source_file,
        <benchmark>_<metric> (one column per metric)
    
    Parameters
    ----------
    history_dir : str
        Path to history directory
        
    Returns
    -------
    pd.DataFrame : Wide-format DataFrame
    """
    records = load_history_to_records(history_dir)
    
    rows = []
    for record in records:
        git = record.get('git', {})
        row = {
            'timestamp': record.get('created'),
            'commit': git.get('commit_short'),
            'commit_full': git.get('commit'),
            'branch': git.get('branch'),
            'dirty': git.get('dirty'),
            'commit_date': git.get('commit_date'),
            'commit_message': git.get('commit_message'),
            'host': record.get('host'),
            'python_version': record.get('python_version'),
            'source_file': record.get('_source_file'),
        }
        
        # Flatten all metrics into columns
        for bench_name, bench_data in record.get('benchmarks', {}).items():
            # Clean benchmark name for column
            bench_short = bench_name.replace('benchmark_', '').replace('.py', '')
            
            # Add time_s
            if bench_data.get('time_s') is not None:
                row[f'{bench_short}_time_s'] = bench_data['time_s']
            
            # Add all metrics
            for metric_name, metric_value in bench_data.get('metrics', {}).items():
                if isinstance(metric_value, (int, float)):
                    row[f'{bench_short}_{metric_name}'] = metric_value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns and len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def list_metrics(df_long):
    """
    List all available metrics in a long-format DataFrame.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format DataFrame from load_history_long()
        
    Returns
    -------
    pd.DataFrame : Summary of available metrics
    """
    summary = df_long.groupby(['benchmark', 'metric']).agg(
        count=('value', 'count'),
        min=('value', 'min'),
        max=('value', 'max'),
        mean=('value', 'mean'),
    ).round(3)
    
    return summary


def filter_history(df_long, benchmark=None, metric=None, branch=None, 
                   dirty=None, since=None, until=None):
    """
    Filter long-format DataFrame with common criteria.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format DataFrame
    benchmark : str, optional
        Filter by benchmark name (substring match)
    metric : str, optional
        Filter by metric name (exact match)
    branch : str, optional
        Filter by git branch
    dirty : bool, optional
        Filter by dirty status
    since : str, optional
        Filter runs after this date (ISO format)
    until : str, optional
        Filter runs before this date (ISO format)
        
    Returns
    -------
    pd.DataFrame : Filtered DataFrame
    """
    df = df_long.copy()
    
    if benchmark:
        df = df[df['benchmark'].str.contains(benchmark, case=False)]
    if metric:
        df = df[df['metric'] == metric]
    if branch:
        df = df[df['branch'] == branch]
    if dirty is not None:
        df = df[df['dirty'] == dirty]
    if since:
        df = df[df['timestamp'] >= pd.to_datetime(since)]
    if until:
        df = df[df['timestamp'] <= pd.to_datetime(until)]
    
    return df


def get_metric_history(df_long, benchmark, metric):
    """
    Get time series of a specific metric.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format DataFrame
    benchmark : str
        Benchmark name (substring match)
    metric : str
        Metric name (exact match)
        
    Returns
    -------
    pd.DataFrame : Time series with timestamp, commit, value
    """
    df = df_long[
        (df_long['benchmark'].str.contains(benchmark, case=False)) &
        (df_long['metric'] == metric)
    ][['timestamp', 'commit', 'branch', 'dirty', 'value']].copy()
    
    return df.sort_values('timestamp').reset_index(drop=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze benchmark history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available metrics
    python history_analysis.py list results/history/
    
    # Show recent runs
    python history_analysis.py show results/history/ --last 10
    
    # Show specific metric history
    python history_analysis.py show results/history/ --metric direct_vs_safe_speedup
    
    # Export to CSV for external analysis
    python history_analysis.py export results/history/ --format wide --output history.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available metrics')
    list_parser.add_argument('history_dir', help='History directory')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show recent history')
    show_parser.add_argument('history_dir', help='History directory')
    show_parser.add_argument('--last', type=int, default=10, help='Show last N runs')
    show_parser.add_argument('--metric', help='Filter by metric name')
    show_parser.add_argument('--benchmark', help='Filter by benchmark name')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export history to file')
    export_parser.add_argument('history_dir', help='History directory')
    export_parser.add_argument('--format', choices=['long', 'wide'], default='long',
                              help='DataFrame format')
    export_parser.add_argument('--output', '-o', required=True, help='Output file (.csv)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        df = load_history_long(args.history_dir)
        if len(df) == 0:
            print("No history data found")
            return 1
        print(list_metrics(df).to_string())
        
    elif args.command == 'show':
        df = load_history_long(args.history_dir)
        if len(df) == 0:
            print("No history data found")
            return 1
            
        if args.metric:
            df = df[df['metric'] == args.metric]
        if args.benchmark:
            df = df[df['benchmark'].str.contains(args.benchmark)]
        
        # Get last N unique runs
        if 'timestamp' in df.columns:
            recent_timestamps = df['timestamp'].drop_duplicates().nlargest(args.last)
            df = df[df['timestamp'].isin(recent_timestamps)]
        
        # Select display columns
        display_cols = ['timestamp', 'commit', 'benchmark', 'metric', 'value']
        display_cols = [c for c in display_cols if c in df.columns]
        
        print(df[display_cols].to_string(index=False))
        
    elif args.command == 'export':
        if args.format == 'long':
            df = load_history_long(args.history_dir)
        else:
            df = load_history_wide(args.history_dir)
        
        if len(df) == 0:
            print("No history data found")
            return 1
        
        df.to_csv(args.output, index=False)
        print(f"Exported {len(df)} rows to {args.output}")
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

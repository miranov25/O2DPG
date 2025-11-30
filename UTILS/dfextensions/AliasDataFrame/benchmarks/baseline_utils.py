#!/usr/bin/env python3
"""
baseline_utils.py - Benchmark baseline management utilities

Provides tools for managing benchmark baselines:
- compare: Compare current results against stored baseline (detect regressions)
- merge: Merge individual benchmark results into unified baseline file

Usage:
    python baseline_utils.py compare results/ baseline.json --threshold 20
    python baseline_utils.py merge results/ baseline.json
    python baseline_utils.py --help

Part of the AliasDataFrame benchmark regression detection system.
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

# Schema version - increment when format changes
BASELINE_VERSION = 1


# =============================================================================
# GIT UTILITIES
# =============================================================================

def get_git_info():
    """
    Get current git commit info.
    
    Returns dict with commit hash, branch, dirty status, etc.
    Returns None values if git is not available or not in a repo.
    """
    import subprocess
    
    def run_git(args):
        try:
            result = subprocess.run(
                ['git'] + args,
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    # Check if we're in a git repo
    if run_git(['rev-parse', '--git-dir']) is None:
        return {
            'commit': None,
            'commit_short': None,
            'branch': None,
            'dirty': None,
            'commit_date': None,
            'commit_message': None,
        }
    
    # Get status to check dirty
    status_output = run_git(['status', '--porcelain'])
    is_dirty = status_output is not None and status_output != ''
    
    return {
        'commit': run_git(['rev-parse', 'HEAD']),
        'commit_short': run_git(['rev-parse', '--short', 'HEAD']),
        'branch': run_git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'dirty': is_dirty,
        'commit_date': run_git(['log', '-1', '--format=%ci']),
        'commit_message': run_git(['log', '-1', '--format=%s']),
    }


def archive_to_history(results_json, history_dir='results/history'):
    """
    Archive benchmark results to history directory with git info.
    
    Creates: history_dir/benchmark_YYYYMMDD_HHMMSS_COMMITHASH.json
    
    Parameters
    ----------
    results_json : str
        Path to the merged results JSON file
    history_dir : str
        Directory to store history files
        
    Returns
    -------
    Path : Path to the created history file
    """
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_json) as f:
        data = json.load(f)
    
    # Add git info
    data['git'] = get_git_info()
    
    # Create history filename
    # Extract timestamp from 'created' field or use current time
    created = data.get('created', datetime.now().isoformat())
    # Parse timestamp: "2025-11-30T09:03:17.481050" -> "20251130_090317"
    try:
        dt = datetime.fromisoformat(created.split('.')[0])
        timestamp = dt.strftime('%Y%m%d_%H%M%S')
    except (ValueError, AttributeError):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    commit_short = data['git'].get('commit_short') or 'nogit'
    history_name = f"benchmark_{timestamp}_{commit_short}.json"
    history_file = history_path / history_name
    
    # Save to history
    with open(history_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Archived to: {history_file}")
    return history_file


def cmd_archive(args):
    """Execute archive subcommand."""
    results_path = Path(args.results)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    try:
        history_file = archive_to_history(args.results, args.history_dir)
        print(f"✓ History archived: {history_file}")
        return 0
    except Exception as e:
        print(f"Error archiving results: {e}")
        return 1


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def load_json(filepath):
    """Load and parse JSON file."""
    with open(filepath) as f:
        return json.load(f)


def format_value(val, metric_name):
    """Format metric value for display."""
    if val is None:
        return "N/A"
    
    metric_lower = metric_name.lower()
    
    # Time metrics
    if metric_name == "time" or metric_name.endswith("_s") or 'time' in metric_lower:
        if val < 0.1:
            return f"{val*1000:.0f}ms"
        return f"{val:.2f}s"
    
    # Rate metrics
    if 'per_second' in metric_lower or 'rate' in metric_lower:
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if val >= 1000:
            return f"{val/1000:.0f}k"
        return f"{val:.0f}"
    
    # Speedup metrics
    if 'speedup' in metric_lower:
        return f"{val:.2f}x"
    
    # Default
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


# =============================================================================
# COMPARE FUNCTIONALITY
# =============================================================================

def find_latest_result(results_dir):
    """Find most recent benchmark result file."""
    results_path = Path(results_dir)
    
    # Look for merged results first, then individual benchmarks
    patterns = ["benchmark_merged_*.json", "benchmark_performance_*.json"]
    
    for pattern in patterns:
        json_files = sorted(results_path.glob(pattern), reverse=True)
        if json_files:
            return json_files[0]
    
    # Fallback: any benchmark JSON
    json_files = sorted(results_path.glob("benchmark_*.json"), reverse=True)
    return json_files[0] if json_files else None


def validate_baseline(baseline):
    """
    Validate baseline file format and version.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(baseline, dict):
        return False, "Baseline must be a JSON object"
    
    version = baseline.get("version")
    if version is None:
        return False, "Baseline missing 'version' field"
    
    if version != BASELINE_VERSION:
        return False, f"Baseline version mismatch (got {version}, expected {BASELINE_VERSION})"
    
    if "benchmarks" not in baseline:
        return False, "Baseline missing 'benchmarks' field"
    
    return True, None


def compare_metrics(current, baseline, threshold_pct):
    """
    Compare current metrics against baseline.
    
    Args:
        current: Current benchmark results dict
        baseline: Baseline results dict
        threshold_pct: Regression threshold percentage
    
    Returns:
        dict with 'passed', 'regressions', 'improvements', 'missing', 'details'
    """
    results = {
        'passed': True,
        'regressions': [],
        'improvements': [],
        'missing': [],
        'details': {},
        'threshold_pct': threshold_pct,
    }
    
    for bench_name, bench_baseline in baseline.get('benchmarks', {}).items():
        bench_current = current.get('benchmarks', {}).get(bench_name, {})
        
        if not bench_current:
            results['missing'].append(bench_name)
            results['details'][bench_name] = {'status': 'MISSING'}
            continue
        
        bench_details = {'metrics': {}, 'status': 'OK'}
        has_regression = False
        
        # Compare overall time
        baseline_time = bench_baseline.get('time_s')
        current_time = bench_current.get('time_s')
        
        if baseline_time and current_time and baseline_time > 0:
            pct_change = (current_time - baseline_time) / baseline_time * 100
            status = 'OK'
            
            if pct_change > threshold_pct:
                status = 'REGRESSION'
                has_regression = True
                results['regressions'].append({
                    'benchmark': bench_name,
                    'metric': 'time',
                    'baseline': baseline_time,
                    'current': current_time,
                    'change_pct': pct_change
                })
                results['passed'] = False
            elif pct_change < -threshold_pct:
                status = 'IMPROVED'
                results['improvements'].append({
                    'benchmark': bench_name,
                    'metric': 'time',
                    'baseline': baseline_time,
                    'current': current_time,
                    'change_pct': pct_change
                })
            
            bench_details['time'] = {
                'baseline': baseline_time,
                'current': current_time,
                'change_pct': pct_change,
                'status': status
            }
        
        # Compare sub-metrics
        baseline_metrics = bench_baseline.get('metrics', {})
        current_metrics = bench_current.get('metrics', {})
        
        for metric_name, baseline_val in baseline_metrics.items():
            current_val = current_metrics.get(metric_name)
            
            if current_val is None:
                continue
            
            # Guard against division by zero
            if baseline_val == 0:
                continue
            
            # Determine if higher is better or lower is better
            higher_is_better = any(kw in metric_name.lower() for kw in 
                                   ['per_second', 'speedup', 'throughput', 'rate'])
            
            pct_change = (current_val - baseline_val) / baseline_val * 100
            
            if higher_is_better:
                is_regression = pct_change < -threshold_pct
                is_improvement = pct_change > threshold_pct
            else:
                is_regression = pct_change > threshold_pct
                is_improvement = pct_change < -threshold_pct
            
            status = 'OK'
            if is_regression:
                status = 'REGRESSION'
                has_regression = True
                results['regressions'].append({
                    'benchmark': bench_name,
                    'metric': metric_name,
                    'baseline': baseline_val,
                    'current': current_val,
                    'change_pct': pct_change
                })
                results['passed'] = False
            elif is_improvement:
                status = 'IMPROVED'
                results['improvements'].append({
                    'benchmark': bench_name,
                    'metric': metric_name,
                    'baseline': baseline_val,
                    'current': current_val,
                    'change_pct': pct_change
                })
            
            bench_details['metrics'][metric_name] = {
                'baseline': baseline_val,
                'current': current_val,
                'change_pct': pct_change,
                'status': status
            }
        
        if has_regression:
            bench_details['status'] = 'REGRESSION'
        
        results['details'][bench_name] = bench_details
    
    return results


def print_comparison(results, baseline_info, threshold_pct):
    """Print formatted comparison results."""
    print()
    print("=" * 65)
    print("BENCHMARK COMPARISON")
    print("=" * 65)
    
    created = baseline_info.get('created', 'unknown')
    host = baseline_info.get('host', 'unknown')
    print(f"Baseline: {created}")
    print(f"Host:     {host}")
    print(f"Threshold: {threshold_pct}%")
    print()
    
    for bench_name, details in results['details'].items():
        status_icon = {'OK': '✓', 'REGRESSION': '⚠️', 'IMPROVED': '⬆', 'MISSING': '?'
                      }.get(details.get('status', 'OK'), ' ')
        
        print(f"{status_icon} {bench_name}:")
        
        if details.get('status') == 'MISSING':
            print("    (not in current results - SKIPPED)")
            print()
            continue
        
        if 'time' in details:
            t = details['time']
            status_str = {'OK': '✓ OK', 'REGRESSION': '⚠️ REGRESSION', 'IMPROVED': '⬆ IMPROVED'
                         }.get(t['status'], t['status'])
            print(f"    time: {format_value(t['baseline'], 'time')} → "
                  f"{format_value(t['current'], 'time')} ({t['change_pct']:+.1f}%) {status_str}")
        
        for metric_name, m in details.get('metrics', {}).items():
            status_str = {'OK': '✓', 'REGRESSION': '⚠️ REGRESSION', 'IMPROVED': '⬆'
                         }.get(m['status'], m['status'])
            print(f"    {metric_name}: {format_value(m['baseline'], metric_name)} → "
                  f"{format_value(m['current'], metric_name)} ({m['change_pct']:+.1f}%) {status_str}")
        print()
    
    if results['missing']:
        print("-" * 65)
        print(f"⚠️  Missing benchmarks ({len(results['missing'])}):")
        for name in results['missing']:
            print(f"    - {name}")
        print()
    
    print("=" * 65)
    
    if results['regressions']:
        print("\033[31m")
        print(f"⚠️  REGRESSIONS DETECTED: {len(results['regressions'])} metrics exceeded threshold")
        print("\033[0m")
        for reg in results['regressions']:
            print(f"    - {reg['benchmark']}: {reg['metric']} ({reg['change_pct']:+.1f}%)")
        print()
        print("Action: Investigate performance before committing.")
    
    if results['improvements']:
        print(f"\033[32m✓ {len(results['improvements'])} metrics improved significantly\033[0m")
    
    print()
    print("=" * 65)
    if results['passed']:
        print(f"\033[32m✓ All benchmarks within threshold ({threshold_pct}%)\033[0m")
    else:
        n_reg = len(results['regressions'])
        print(f"\033[31m✗ {n_reg} regression{'s' if n_reg > 1 else ''} detected\033[0m")
    print("=" * 65)


def cmd_compare(args):
    """Execute compare subcommand."""
    # Load baseline
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        return 2
    
    try:
        baseline = load_json(args.baseline)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in baseline file: {e}")
        return 2
    
    is_valid, error_msg = validate_baseline(baseline)
    if not is_valid:
        print(f"Error: {error_msg}")
        return 2
    
    # Load current results
    current_path = Path(args.current)
    if args.latest or current_path.is_dir():
        current_file = find_latest_result(args.current)
        if not current_file:
            print(f"Error: No benchmark results found in {args.current}")
            return 2
        if not args.quiet:
            print(f"Using latest result: {current_file.name}")
    else:
        current_file = current_path
        if not current_file.exists():
            print(f"Error: Results file not found: {args.current}")
            return 2
    
    try:
        current = load_json(current_file)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in results file: {e}")
        return 2
    
    # Compare
    results = compare_metrics(current, baseline, args.threshold)
    
    if not args.quiet:
        print_comparison(results, baseline, args.threshold)
    
    # Export if requested
    if args.json:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'baseline_file': str(args.baseline),
            'current_file': str(current_file),
            'threshold_pct': args.threshold,
            **results
        }
        with open(args.json, 'w') as f:
            json.dump(export_data, f, indent=2)
        if not args.quiet:
            print(f"\nComparison exported to: {args.json}")
    
    if args.strict and not results['passed']:
        return 1
    return 0


# =============================================================================
# MERGE FUNCTIONALITY
# =============================================================================

def find_benchmark_files(results_dir, timestamp=None):
    """Find benchmark result files in directory."""
    results_path = Path(results_dir)
    
    # All recognized benchmarks
    benchmark_names = [
        'benchmark_performance',
        'benchmark_materialize_aliases',  # NEW
        'benchmark_read_tree', 
        'benchmark_subframe',
        'benchmark_parallel'
    ]
    
    found = {}
    for bench_name in benchmark_names:
        if timestamp:
            pattern = f"{bench_name}_{timestamp}.json"
            matches = list(results_path.glob(pattern))
        else:
            pattern = f"{bench_name}_*.json"
            matches = sorted(results_path.glob(pattern), reverse=True)
        
        if matches:
            found[bench_name] = matches[0]
    
    return found


def extract_metrics(bench_name, data):
    """Extract standardized metrics from benchmark-specific JSON format."""
    result = {'time_s': None, 'metrics': {}}
    
    if bench_name == 'benchmark_performance':
        result['time_s'] = data.get('elapsed_s') or data.get('total_time_s')
        for test in data.get('tests', []):
            test_name = test.get('name', 'unknown')
            if 'elapsed_s' in test:
                result['metrics'][f"{test_name}_time_s"] = test['elapsed_s']
            if 'rows_per_sec' in test:
                result['metrics'][f"{test_name}_rows_per_sec"] = test['rows_per_sec']
        if 'all_passed' in data:
            result['metrics']['all_passed'] = 1 if data['all_passed'] else 0
    
    elif bench_name == 'benchmark_materialize_aliases':
        # NEW: Extract metrics from materialize_aliases benchmark
        result['time_s'] = data.get('time_s') or data.get('total_time_s')
        
        # Extract per-scenario metrics
        for scenario in data.get('scenarios', []):
            name = scenario.get('name', 'unknown')
            if 'time_s' in scenario:
                result['metrics'][f"{name}_time_s"] = scenario['time_s']
            if 'rows_per_sec' in scenario:
                result['metrics'][f"{name}_rows_per_sec"] = scenario['rows_per_sec']
            if 'memory_mb' in scenario:
                result['metrics'][f"{name}_memory_mb"] = scenario['memory_mb']
        
        # Extract speedup metrics
        if 'direct_vs_safe_speedup' in data.get('metrics', {}):
            result['metrics']['direct_vs_safe_speedup'] = data['metrics']['direct_vs_safe_speedup']
        if 'safe_vs_simple_ratio' in data.get('metrics', {}):
            result['metrics']['safe_vs_simple_ratio'] = data['metrics']['safe_vs_simple_ratio']
        
        # Alternative: metrics might be at top level
        for key in ['direct_vs_safe_speedup', 'safe_vs_simple_ratio', 'missing_pct']:
            if key in data:
                result['metrics'][key] = data[key]
    
    elif bench_name == 'benchmark_read_tree':
        result['time_s'] = data.get('elapsed_s') or data.get('total_time_s')
        if 'rows_per_sec' in data:
            result['metrics']['rows_per_second'] = data['rows_per_sec']
        if 'rows' in data:
            result['metrics']['rows'] = data['rows']
        for key, val in data.items():
            if key.startswith('workers_') and isinstance(val, dict):
                if 'elapsed_s' in val:
                    result['metrics'][f"{key}_time_s"] = val['elapsed_s']
    
    elif bench_name == 'benchmark_subframe':
        result['time_s'] = data.get('elapsed_s') or data.get('total_time_s')
        for test in data.get('tests', []):
            test_name = test.get('name', 'unknown').replace(' ', '_').lower()
            if 'elapsed_s' in test:
                result['metrics'][f"{test_name}_time_s"] = test['elapsed_s']
        for key in ['join_time_s', 'auto_alias_time_s', 'materialize_time_s']:
            if key in data:
                result['metrics'][key] = data[key]
    
    elif bench_name == 'benchmark_parallel':
        result['time_s'] = data.get('elapsed_s') or data.get('total_time_s')
        for entry in data.get('scaling', []):
            workers = entry.get('workers', 0)
            if 'speedup' in entry:
                result['metrics'][f"speedup_{workers}_workers"] = entry['speedup']
            if 'time_s' in entry:
                result['metrics'][f"time_{workers}_workers"] = entry['time_s']
        if 'best_speedup' in data:
            result['metrics']['best_speedup'] = data['best_speedup']
        if 'optimal_workers' in data:
            result['metrics']['optimal_workers'] = data['optimal_workers']
    
    else:
        # Generic fallback for unknown benchmarks
        result['time_s'] = data.get('elapsed_s') or data.get('total_time_s') or data.get('time_s')
        for key, val in data.items():
            if isinstance(val, (int, float)) and key not in ['elapsed_s', 'time_s']:
                result['metrics'][key] = val
    
    return result


def merge_results(results_dir, timestamp=None):
    """Merge all benchmark results into a unified structure."""
    files = find_benchmark_files(results_dir, timestamp)
    
    if not files:
        raise ValueError(f"No benchmark result files found in {results_dir}")
    
    baseline = {
        'version': BASELINE_VERSION,
        'created': datetime.now().isoformat(),
        'host': platform.node(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'platform': platform.platform(),
        'benchmarks': {}
    }
    
    print(f"Merging {len(files)} benchmark results:")
    
    for bench_name, filepath in sorted(files.items()):
        print(f"  - {bench_name}: {filepath.name}")
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            metrics = extract_metrics(bench_name, data)
            key = f"{bench_name}.py"
            baseline['benchmarks'][key] = metrics
            
        except Exception as e:
            print(f"    WARNING: Failed to parse {filepath}: {e}")
            continue
    
    return baseline


def cmd_merge(args):
    """Execute merge subcommand."""
    if not Path(args.results_dir).is_dir():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    try:
        baseline = merge_results(args.results_dir, args.timestamp)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print()
    print(f"Merged baseline:")
    print(f"  Version: {baseline['version']}")
    print(f"  Host: {baseline['host']}")
    print(f"  Python: {baseline['python_version']}")
    print(f"  Benchmarks: {len(baseline['benchmarks'])}")
    
    for bench_name, data in baseline['benchmarks'].items():
        n_metrics = len(data.get('metrics', {}))
        time_s = data.get('time_s')
        time_str = f"{time_s:.2f}s" if time_s else "N/A"
        print(f"    - {bench_name}: {time_str}, {n_metrics} metrics")
    
    if args.dry_run:
        print()
        print("Dry run - not writing file")
        print()
        print(json.dumps(baseline, indent=2))
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print()
        print(f"✓ Baseline saved to: {args.output}")
    
    return 0


# =============================================================================
# DIFF FUNCTIONALITY
# =============================================================================

def compare_history_files(file_a, file_b, threshold_pct=10.0):
    """
    Compare two history/benchmark JSON files and show differences.
    
    Parameters
    ----------
    file_a : str
        Path to first JSON file (typically older/baseline)
    file_b : str
        Path to second JSON file (typically newer/current)
    threshold_pct : float
        Highlight changes larger than this percentage
        
    Returns
    -------
    dict : Comparison results with changes for each metric
    """
    a = load_json(file_a)
    b = load_json(file_b)
    
    results = {
        'file_a': str(file_a),
        'file_b': str(file_b),
        'commit_a': a.get('git', {}).get('commit_short', 'N/A'),
        'commit_b': b.get('git', {}).get('commit_short', 'N/A'),
        'timestamp_a': a.get('created', 'N/A'),
        'timestamp_b': b.get('created', 'N/A'),
        'changes': [],
        'regressions': [],
        'improvements': [],
    }
    
    # Get all benchmarks from both files
    benchmarks_a = a.get('benchmarks', {})
    benchmarks_b = b.get('benchmarks', {})
    all_benchmarks = set(benchmarks_a.keys()) | set(benchmarks_b.keys())
    
    for bench_name in sorted(all_benchmarks):
        metrics_a = benchmarks_a.get(bench_name, {}).get('metrics', {})
        metrics_b = benchmarks_b.get(bench_name, {}).get('metrics', {})
        
        # Also compare time_s
        time_a = benchmarks_a.get(bench_name, {}).get('time_s')
        time_b = benchmarks_b.get(bench_name, {}).get('time_s')
        if time_a and time_b:
            metrics_a = dict(metrics_a)  # Copy to avoid modifying original
            metrics_b = dict(metrics_b)
            metrics_a['time_s'] = time_a
            metrics_b['time_s'] = time_b
        
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
        
        for metric_name in sorted(all_metrics):
            val_a = metrics_a.get(metric_name)
            val_b = metrics_b.get(metric_name)
            
            if val_a is None or val_b is None:
                continue
            if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
                continue
            if val_a == 0:
                continue
                
            change_pct = (val_b - val_a) / abs(val_a) * 100
            
            change = {
                'benchmark': bench_name,
                'metric': metric_name,
                'value_a': val_a,
                'value_b': val_b,
                'change_pct': change_pct,
            }
            
            results['changes'].append(change)
            
            # Classify as regression or improvement
            # For time metrics, increase is regression
            # For speedup metrics, decrease is regression
            is_time_metric = 'time' in metric_name.lower() or metric_name.endswith('_s')
            
            if is_time_metric:
                if change_pct > threshold_pct:
                    results['regressions'].append(change)
                elif change_pct < -threshold_pct:
                    results['improvements'].append(change)
            else:
                if change_pct < -threshold_pct:
                    results['regressions'].append(change)
                elif change_pct > threshold_pct:
                    results['improvements'].append(change)
    
    return results


def print_diff_report(results):
    """Print formatted diff report."""
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"  A: {results['commit_a']} ({results['timestamp_a'][:19] if len(results['timestamp_a']) > 19 else results['timestamp_a']})")
    print(f"  B: {results['commit_b']} ({results['timestamp_b'][:19] if len(results['timestamp_b']) > 19 else results['timestamp_b']})")
    print()
    
    if results['regressions']:
        print(f"⚠️  REGRESSIONS ({len(results['regressions'])}):")
        for r in results['regressions']:
            print(f"    {r['benchmark']}: {r['metric']}")
            print(f"      {r['value_a']:.3f} → {r['value_b']:.3f} ({r['change_pct']:+.1f}%)")
        print()
    
    if results['improvements']:
        print(f"✓ IMPROVEMENTS ({len(results['improvements'])}):")
        for r in results['improvements']:
            print(f"    {r['benchmark']}: {r['metric']}")
            print(f"      {r['value_a']:.3f} → {r['value_b']:.3f} ({r['change_pct']:+.1f}%)")
        print()
    
    print(f"{'='*70}")
    print("ALL CHANGES:")
    print(f"{'='*70}")
    print(f"{'Benchmark':<35} {'Metric':<25} {'A':>10} {'B':>10} {'Change':>10}")
    print("-" * 95)
    
    for c in results['changes']:
        bench_short = c['benchmark'].replace('benchmark_', '').replace('.py', '')[:33]
        metric_short = c['metric'][:23]
        print(f"{bench_short:<35} {metric_short:<25} {c['value_a']:>10.3f} {c['value_b']:>10.3f} {c['change_pct']:>+9.1f}%")
    
    print(f"{'='*70}")


def cmd_diff(args):
    """Execute diff subcommand."""
    import glob
    
    # Handle glob patterns
    files_a = glob.glob(args.file_a)
    files_b = glob.glob(args.file_b)
    
    if not files_a:
        print(f"Error: No files matching: {args.file_a}")
        return 1
    if not files_b:
        print(f"Error: No files matching: {args.file_b}")
        return 1
    
    # Use most recent if multiple matches
    file_a = sorted(files_a)[-1]
    file_b = sorted(files_b)[-1]
    
    results = compare_history_files(file_a, file_b, args.threshold)
    print_diff_report(results)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nExported to: {args.json}")
    
    # Exit code
    if args.strict and results['regressions']:
        return 1
    return 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark baseline management utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare latest results against baseline
    python baseline_utils.py compare results/ baseline.json --latest
    
    # Compare with custom threshold, fail on regression
    python baseline_utils.py compare results/ baseline.json --threshold 15 --strict
    
    # Merge results into new baseline
    python baseline_utils.py merge results/ baseline.json
    
    # Preview merge without writing
    python baseline_utils.py merge results/ baseline.json --dry-run
    
    # Archive results to history with git info
    python baseline_utils.py archive results/benchmark_merged.json --history-dir results/history
    
    # Compare two history files (supports glob patterns)
    python baseline_utils.py diff 'results/history/*_f9df9cf*' 'results/history/*_18caba7*'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Compare subcommand
    compare_parser = subparsers.add_parser('compare', 
        help='Compare current results against baseline')
    compare_parser.add_argument('current', 
        help='Current results JSON file or results directory')
    compare_parser.add_argument('baseline', 
        help='Baseline JSON file')
    compare_parser.add_argument('--threshold', type=float, default=20.0,
        help='Regression threshold percentage (default: 20)')
    compare_parser.add_argument('--latest', action='store_true',
        help='Use latest result from directory')
    compare_parser.add_argument('--json', 
        help='Export comparison to JSON file')
    compare_parser.add_argument('--strict', action='store_true',
        help='Exit with code 1 on regression')
    compare_parser.add_argument('--quiet', action='store_true',
        help='Minimal output (exit code only)')
    
    # Merge subcommand
    merge_parser = subparsers.add_parser('merge', 
        help='Merge benchmark results into unified baseline')
    merge_parser.add_argument('results_dir', 
        help='Directory containing benchmark result JSONs')
    merge_parser.add_argument('output', 
        help='Output baseline JSON file')
    merge_parser.add_argument('--timestamp', 
        help='Specific timestamp to merge (e.g., 20251127_221916)')
    merge_parser.add_argument('--dry-run', action='store_true',
        help='Preview merge without writing file')
    
    # Archive subcommand
    archive_parser = subparsers.add_parser('archive',
        help='Archive results to history with git info')
    archive_parser.add_argument('results',
        help='Merged results JSON file to archive')
    archive_parser.add_argument('--history-dir', default='results/history',
        help='History directory (default: results/history)')
    
    # Diff subcommand
    diff_parser = subparsers.add_parser('diff',
        help='Compare two benchmark results')
    diff_parser.add_argument('file_a',
        help='First JSON file (baseline/older), supports glob patterns')
    diff_parser.add_argument('file_b', 
        help='Second JSON file (current/newer), supports glob patterns')
    diff_parser.add_argument('--threshold', type=float, default=10.0,
        help='Change threshold for flagging (default: 10%%)')
    diff_parser.add_argument('--json',
        help='Export comparison to JSON file')
    diff_parser.add_argument('--strict', action='store_true',
        help='Exit with code 1 if regressions detected')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'merge':
        return cmd_merge(args)
    elif args.command == 'archive':
        return cmd_archive(args)
    elif args.command == 'diff':
        return cmd_diff(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

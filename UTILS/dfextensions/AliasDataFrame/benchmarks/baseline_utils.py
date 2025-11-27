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
    
    benchmark_names = [
        'benchmark_performance',
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
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'merge':
        return cmd_merge(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

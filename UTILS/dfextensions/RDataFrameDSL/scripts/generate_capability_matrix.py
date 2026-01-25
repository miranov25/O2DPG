#!/usr/bin/env python3
"""
Generate CAPABILITY_MATRIX.md from test annotations.

Usage:
    python scripts/generate_capability_matrix.py [--output docs/CAPABILITY_MATRIX.md]
    python scripts/generate_capability_matrix.py --dry-run
    
Reads:
    - tests/feature_taxonomy.py (FEATURE_TAXONOMY, KNOWN_LIMITATIONS)
    - pytest results (pass/fail/xfail status via pytest-json-report)
    - @pytest.mark.feature markers from test files
    - Existing docs/CAPABILITY_MATRIX.md (preserves MANUAL sections)

Outputs:
    - docs/CAPABILITY_MATRIX.md (with AUTO-GENERATED and MANUAL sections)

Prerequisites:
    pip install pytest-json-report
    # Or add to requirements-dev.txt

Features:
    - Parses @pytest.mark.feature markers to map features to tests
    - Preserves human-edited MANUAL sections across regenerations
    - Computes status from actual test outcomes
    - Handles all pytest outcomes: passed, failed, xfailed, xpassed, skipped, error

Phase: 13.6.F
Date: 2026-01-23
Version: 1.3 (run only feature-marked tests for speed)
"""

import subprocess
import json
import tempfile
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import re

# =============================================================================
# PATH SETUP
# =============================================================================

# Determine project root
# Works whether script is in project root or scripts/ subdirectory
SCRIPT_DIR = Path(__file__).parent.resolve()
if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    # Script is in project root (e.g., during development)
    PROJECT_ROOT = SCRIPT_DIR

# Add both project root and tests/ to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


# =============================================================================
# FEATURE MARKER PARSING
# =============================================================================

def parse_feature_markers_from_files(test_pattern: str) -> Dict[str, List[str]]:
    """
    Parse @pytest.mark.feature("feature_id") markers directly from test files.
    
    Returns dict mapping feature_id -> [list of test nodeids]
    
    This is the KEY function that maps features to tests based on markers.
    
    Phase 13.6.F fix: Skip markers inside triple-quoted strings (docstrings/comments).
    """
    feature_tests = {}
    
    # Expand test pattern
    test_files = glob.glob(str(PROJECT_ROOT / test_pattern))
    
    for filepath in test_files:
        filepath = Path(filepath)
        relative_path = filepath.relative_to(PROJECT_ROOT)
        
        with open(filepath) as f:
            content = f.read()
        
        lines = content.split('\n')
        current_class = None
        pending_features = []  # Can have multiple markers before one test
        in_triple_quote = False  # Track if we're inside triple-quoted string
        triple_quote_char = None  # Track which quote type (''' or """)
        
        for i, line in enumerate(lines):
            # Check for triple quote transitions
            # Count occurrences of ''' and """ in the line
            for quote in ['"""', "'''"]:
                count = line.count(quote)
                if count > 0:
                    if not in_triple_quote:
                        # Entering triple quote
                        in_triple_quote = True
                        triple_quote_char = quote
                        # If odd count, we end inside; if even, we exit
                        if count % 2 == 0:
                            in_triple_quote = False
                            triple_quote_char = None
                    elif quote == triple_quote_char:
                        # Exiting triple quote (or re-entering)
                        if count % 2 == 1:
                            in_triple_quote = False
                            triple_quote_char = None
                    break  # Only process one quote type per line
            
            # Skip processing if inside triple-quoted string
            if in_triple_quote:
                continue
            
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            
            # Track class context
            class_match = re.match(r'^class (\w+)', line)
            if class_match:
                current_class = class_match.group(1)
                pending_features = []  # Reset on new class
                continue
            
            # Look for feature marker - must start with @ (decorator)
            # This ensures we only match actual decorators, not examples in strings
            feature_match = re.match(r'\s*@pytest\.mark\.feature\(["\'](\w+)["\']\)', line)
            if feature_match:
                pending_features.append(feature_match.group(1))
                continue
            
            # Look for test definition
            test_match = re.match(r'\s*def (test_\w+)\(', line)
            if test_match:
                test_name = test_match.group(1)
                
                # Build nodeid using BASENAME only (matches pytest json-report format)
                # pytest uses: "test_file.py::Class::test" not "tests/test_file.py::..."
                basename = filepath.name
                if current_class:
                    nodeid = f"{basename}::{current_class}::{test_name}"
                else:
                    nodeid = f"{basename}::{test_name}"
                
                # Add to feature mapping for all pending features
                for feature_id in pending_features:
                    if feature_id not in feature_tests:
                        feature_tests[feature_id] = []
                    feature_tests[feature_id].append(nodeid)
                
                pending_features = []  # Reset after def
            
            # Reset pending features on non-decorator, non-def lines
            # (but keep them for stacked decorators)
            stripped = line.strip()
            if stripped and not stripped.startswith('@') and not stripped.startswith('def '):
                if not stripped.startswith('#') and not stripped.startswith('"""'):
                    pending_features = []
    
    return feature_tests


# =============================================================================
# PYTEST EXECUTION
# =============================================================================

def run_pytest_for_outcomes(test_pattern: str = "tests/test_*.py") -> Dict[str, str]:
    """
    Run pytest to get actual pass/fail/xfail status.
    
    Returns dict mapping nodeid -> outcome (passed/failed/xfailed/xpassed/skipped/error)
    
    Uses tempfile to avoid path collisions (P2 fix).
    
    Phase 13.6.F: Only runs tests with @pytest.mark.feature marker for speed.
    """
    # Use tempfile for report path (avoids /tmp/ collision issues)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_path = Path(f.name)
    
    try:
        # Expand glob pattern BEFORE passing to subprocess
        # (subprocess doesn't do shell glob expansion)
        expanded_files = glob.glob(str(PROJECT_ROOT / test_pattern))
        if not expanded_files:
            print(f"Warning: No files match pattern {test_pattern}")
            return {}
        
        # Build command with expanded file list
        # Phase 13.6.F: Add -m "feature" to only run feature-marked tests
        # Phase 13.6.F: Add -n 0 to disable parallel execution (ROOT stability)
        cmd = [
            sys.executable, "-m", "pytest",
            *expanded_files,  # Expanded file paths
            "-m", "feature",  # Only run tests with @pytest.mark.feature
            "-n", "0",        # No parallelization (ROOT interpreter stability)
            "-q",
            "--json-report",
            f"--json-report-file={report_path}",
            "--tb=no",  # No tracebacks for speed
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on test failures
            cwd=PROJECT_ROOT,
        )
        
        if not report_path.exists():
            print(f"Warning: pytest report not generated at {report_path}")
            print(f"stdout: {result.stdout[:500] if result.stdout else 'empty'}")
            print(f"stderr: {result.stderr[:500] if result.stderr else 'empty'}")
            return {}
        
        with open(report_path) as f:
            report = json.load(f)
        
        # Return dict keyed by nodeid (full path::test_name)
        outcomes = {}
        for test in report.get("tests", []):
            nodeid = test.get("nodeid", "")
            outcome = test.get("outcome", "unknown")
            outcomes[nodeid] = outcome
        
        return outcomes
    
    finally:
        # Cleanup temp file
        if report_path.exists():
            report_path.unlink()


# =============================================================================
# STATUS COMPUTATION
# =============================================================================

def compute_status(
    feature_id: str,
    feature_test_nodeids: List[str],
    test_outcomes: Dict[str, str],
    limitations: Dict,
    taxonomy: Dict,
) -> Tuple[str, int, List[str]]:
    """
    Compute feature status from test outcomes.
    
    Returns (status_string, test_count, failed_tests)
    
    Status derivation (priority order - highest to lowest):
    1. failed/error → 🧨 Broken (even if limitation exists)
    2. xpassed → 🧨 Broken (unexpected pass = implementation changed)
    3. xfailed with limitation → limitation status (usually ⚠️ Partial)
    4. xfailed without limitation → ❌ Not Implemented
    5. unknown/skipped only → ❓ Unknown
    6. all passed → ✅ Working
    7. no tests → ❌ Not Implemented
    """
    if not feature_test_nodeids:
        return "❌ Not Implemented", 0, []
    
    # Collect outcomes for this feature's tests
    outcomes = []
    failed_tests = []
    for nodeid in feature_test_nodeids:
        # Normalize nodeid: extract just filename::class::test
        # Handle cases like "tests/test_foo.py::Class::test" vs "test_foo.py::Class::test"
        nodeid_normalized = nodeid
        if nodeid.startswith("tests/"):
            nodeid_normalized = nodeid[6:]  # Remove "tests/" prefix
        
        # Try multiple matching strategies
        outcome = None
        matched_nodeid = nodeid
        
        # Strategy 1: Exact match
        outcome = test_outcomes.get(nodeid)
        if outcome:
            matched_nodeid = nodeid
        
        # Strategy 2: With tests/ prefix
        if outcome is None:
            outcome = test_outcomes.get(f"tests/{nodeid}")
            if outcome:
                matched_nodeid = f"tests/{nodeid}"
        
        # Strategy 3: Without tests/ prefix (if nodeid has it)
        if outcome is None and nodeid.startswith("tests/"):
            outcome = test_outcomes.get(nodeid_normalized)
            if outcome:
                matched_nodeid = nodeid_normalized
        
        # Strategy 4: Substring match on normalized nodeid
        if outcome is None:
            for key, val in test_outcomes.items():
                # Normalize the key too
                key_normalized = key[6:] if key.startswith("tests/") else key
                if nodeid_normalized == key_normalized:
                    outcome = val
                    matched_nodeid = key
                    break
        
        # Strategy 5: Fuzzy match - nodeid contained in key or vice versa
        if outcome is None:
            for key, val in test_outcomes.items():
                if nodeid_normalized in key or key.endswith(nodeid_normalized):
                    outcome = val
                    matched_nodeid = key
                    break
        
        outcomes.append(outcome or "unknown")
        
        # Track failed tests
        if outcome in ("failed", "error", "xpassed"):
            failed_tests.append(matched_nodeid)
    
    test_count = len(feature_test_nodeids)
    
    # Priority 1: Any failed/error = broken
    if "failed" in outcomes or "error" in outcomes:
        return "🧨 Broken", test_count, failed_tests
    
    # Priority 2: xpassed = broken (unexpected pass)
    if "xpassed" in outcomes:
        return "🧨 Broken", test_count, failed_tests
    
    # Priority 3-4: xfailed handling
    if "xfailed" in outcomes:
        # Check if limitation exists for this feature
        feature_limitation = taxonomy.get(feature_id, {}).get("limitation")
        if feature_limitation and feature_limitation in limitations:
            lim_status = limitations[feature_limitation].get("status", "⚠️ Partial")
            return lim_status, test_count, []
        return "❌ Not Implemented", test_count, []
    
    # Priority 5: Only unknown/skipped
    known_outcomes = [o for o in outcomes if o not in ("unknown", "skipped")]
    if not known_outcomes:
        return "❓ Unknown", test_count, []
    
    # Priority 6: All passed
    if all(o == "passed" for o in known_outcomes):
        return "✅ Working", test_count, []
    
    # Fallback
    return "❓ Unknown", test_count, []


# =============================================================================
# MATRIX SECTIONS
# =============================================================================

def read_existing_matrix(path: Path) -> Optional[str]:
    """Read existing matrix file if it exists."""
    if path.exists():
        return path.read_text()
    return None


def extract_manual_section(content: str) -> Optional[str]:
    """Extract the MANUAL section from existing matrix content."""
    if not content:
        return None
    
    # Look for MANUAL section marker
    manual_marker = "<!-- MANUAL:"
    if manual_marker not in content:
        return None
    
    # Extract everything after the manual marker
    idx = content.index(manual_marker)
    return content[idx:]


def generate_auto_section(
    taxonomy: Dict,
    limitations: Dict,
    test_outcomes: Dict[str, str],
    feature_tests: Dict[str, List[str]],
) -> List[str]:
    """Generate the AUTO section of the matrix."""
    lines = []
    
    # Header
    lines.extend([
        "# Capability Matrix",
        "",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "**Generated By:** `scripts/generate_capability_matrix.py`",
        f"**Phase:** 13.6.F",
        "",
        "> **Note:** Quick Status and Examples Index are auto-generated.",
        "> Known Limitations and For Reviewers sections are human-maintained.",
        "",
        "---",
        "",
        "<!-- AUTO-GENERATED: Do not edit below this line until END AUTO-GENERATED -->",
        "",
    ])
    
    # Quick Status Table
    lines.append("## Quick Status")
    lines.append("")
    lines.append("| Feature | Status | Proof | Tests | Notes |")
    lines.append("|---------|--------|-------|-------|-------|")
    
    working = 0
    partial = 0
    broken = 0
    
    # Track all broken tests by feature
    broken_features = {}  # feature_name -> [failed_tests]
    
    for feature_id, feature_info in taxonomy.items():
        name = feature_info.get("name", feature_id)
        proof = feature_info.get("proof", "None")
        proof_display = f"`{proof}`" if proof else "None"
        
        # Get tests from parsed markers (not from taxonomy)
        feature_test_list = feature_tests.get(feature_id, [])
        
        # Compute status (now returns failed_tests too)
        status, test_count, failed_tests = compute_status(
            feature_id, feature_test_list, test_outcomes, limitations, taxonomy
        )
        
        # Count for summary
        if "Working" in status:
            working += 1
        elif "Partial" in status:
            partial += 1
        elif "Broken" in status:
            broken += 1
            if failed_tests:
                broken_features[name] = failed_tests
        
        # Notes column - show failed count if broken
        notes = ""
        if feature_info.get("limitation"):
            notes = f"See L{feature_info['limitation'][1:]}"
        if failed_tests:
            notes = f"{len(failed_tests)} failed"
        
        lines.append(f"| {name} | {status} | {proof_display} | {test_count} tests | {notes} |")
    
    total = len(taxonomy)
    lines.append("")
    lines.append(f"**Summary:** {working}/{total} working, {partial} partial, {broken} broken")
    lines.append("")
    
    # Examples Index
    lines.append("## Examples Index")
    lines.append("")
    lines.append("| Example | Features Demonstrated |")
    lines.append("|---------|----------------------|")
    
    # Collect examples
    example_features = {}
    for feature_id, feature_info in taxonomy.items():
        proof = feature_info.get("proof")
        if proof:
            if proof not in example_features:
                example_features[proof] = []
            example_features[proof].append(feature_info.get("name", feature_id))
    
    for example, features in sorted(example_features.items()):
        lines.append(f"| `{example}` | {', '.join(features)} |")
    
    lines.append("")
    
    # Test Coverage Details
    lines.append("## Test Coverage Details")
    lines.append("")
    lines.append("Tests per feature (for traceability). Approval logic: Feature = ✅ Working iff **ALL** tests pass.")
    lines.append("")
    
    for feature_id, feature_info in taxonomy.items():
        name = feature_info.get("name", feature_id)
        feature_test_list = feature_tests.get(feature_id, [])
        test_count = len(feature_test_list)
        
        lines.append("<details>")
        lines.append(f"<summary><strong>{name}</strong> ({test_count} tests)</summary>")
        lines.append("")
        
        if feature_test_list:
            for nodeid in sorted(feature_test_list):
                # Normalize nodeid for matching
                nodeid_normalized = nodeid[6:] if nodeid.startswith("tests/") else nodeid
                
                # Try multiple matching strategies
                outcome = test_outcomes.get(nodeid)
                if outcome is None:
                    outcome = test_outcomes.get(f"tests/{nodeid}")
                if outcome is None:
                    outcome = test_outcomes.get(nodeid_normalized)
                if outcome is None:
                    # Fuzzy match
                    for key, val in test_outcomes.items():
                        key_normalized = key[6:] if key.startswith("tests/") else key
                        if nodeid_normalized == key_normalized:
                            outcome = val
                            break
                if outcome is None:
                    for key, val in test_outcomes.items():
                        if nodeid_normalized in key or key.endswith(nodeid_normalized):
                            outcome = val
                            break
                
                # Status emoji based on outcome
                if outcome == "passed":
                    status_emoji = "✅"
                elif outcome == "failed":
                    status_emoji = "❌"
                elif outcome == "error":
                    status_emoji = "💥"
                elif outcome == "skipped":
                    status_emoji = "⏭️"
                elif outcome == "xfailed":
                    status_emoji = "⚠️"
                elif outcome == "xpassed":
                    status_emoji = "🔄"
                else:
                    status_emoji = "❓"
                
                lines.append(f"- {status_emoji} `{nodeid}`")
        else:
            lines.append("*No tests with @pytest.mark.feature marker*")
        
        lines.append("")
        lines.append("</details>")
        lines.append("")
    
    # Broken Tests Section (if any)
    if broken_features:
        lines.append("## ⚠️ Broken Tests")
        lines.append("")
        lines.append("The following tests are failing and need attention:")
        lines.append("")
        
        for feature_name, failed_tests in sorted(broken_features.items()):
            lines.append(f"### {feature_name} ({len(failed_tests)} failed)")
            lines.append("")
            for test in sorted(failed_tests):
                lines.append(f"- `{test}`")
            lines.append("")
    
    # End auto-generated marker
    lines.append("<!-- END AUTO-GENERATED -->")
    
    return lines


def generate_default_manual_section(limitations: Dict) -> List[str]:
    """Generate default MANUAL section from limitations."""
    lines = [
        "",
        "---",
        "",
        "<!-- MANUAL: Human-maintained sections below. Do not auto-generate. -->",
        "",
        "## Known Limitations",
        "",
    ]
    
    for lim_id, lim_info in limitations.items():
        lines.append(f"### {lim_id}: {lim_info.get('name', 'Unknown')}")
        lines.append("")
        lines.append(f"**Status:** {lim_info.get('status', '⚠️ Partial')}")
        
        if lim_info.get("description"):
            lines.append(f"**Symptom:** {lim_info['description']}")
        if lim_info.get("workaround"):
            lines.append(f"**Workaround:** {lim_info['workaround']}")
        if lim_info.get("bug_report"):
            lines.append(f"**Bug report:** `{lim_info['bug_report']}`")
        if lim_info.get("resolution"):
            lines.append(f"**Resolution:** {lim_info['resolution']}")
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## For Reviewers: Verification Gate",
        "",
        "Before accepting any claim that a feature is broken:",
        "",
        "- [ ] Check Quick Status above — is feature marked \"✅ Working\"?",
        "- [ ] Check if proof example exists and run it",
        "- [ ] Verify test uses correct fixture/data",
        "- [ ] Check Known Limitations section above",
        "",
        "> **Reference:** MTTU_Reviewer.md v1.7 § Verification Gate [MUST]",
        "",
        "**If Matrix shows \"✅ Working\" but test fails:**",
        "1. First assume **test is wrong** (wrong fixture, wrong data)",
        "2. Run the proof example from Quick Status",
        "3. If example works → fix the test",
        "4. If example fails → update Quick Status to 🧨 Broken",
        "",
        "---",
        "",
        "*This matrix is the single source of truth for feature status.*",
        "*Update it whenever feature status changes.*",
        "",
    ])
    
    return lines


def generate_matrix(
    taxonomy: Dict,
    limitations: Dict,
    test_outcomes: Dict[str, str],
    feature_tests: Dict[str, List[str]],
    existing_content: Optional[str] = None,
) -> str:
    """
    Generate complete capability matrix, preserving MANUAL sections.
    
    If existing_content is provided, extracts and preserves the MANUAL section.
    Otherwise, generates default MANUAL section from KNOWN_LIMITATIONS.
    """
    # Generate AUTO section (always regenerated)
    auto_lines = generate_auto_section(taxonomy, limitations, test_outcomes, feature_tests)
    
    # Try to preserve existing MANUAL section
    preserved_manual = extract_manual_section(existing_content) if existing_content else None
    
    if preserved_manual:
        # Splice preserved manual content
        lines = auto_lines + ["---", "", preserved_manual.strip()]
    else:
        # First-time generation: create default MANUAL section
        manual_lines = generate_default_manual_section(limitations)
        lines = auto_lines + manual_lines
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate CAPABILITY_MATRIX.md from test annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate matrix (default location)
    python scripts/generate_capability_matrix.py
    
    # Dry run (print to stdout)
    python scripts/generate_capability_matrix.py --dry-run
    
    # Custom output location
    python scripts/generate_capability_matrix.py --output custom/path.md
    
    # Skip pytest run (use for testing script changes)
    python scripts/generate_capability_matrix.py --skip-tests

Prerequisites:
    pip install pytest-json-report
    # Or add to requirements-dev.txt
        """
    )
    parser.add_argument(
        "--output", 
        default="docs/CAPABILITY_MATRIX.md",
        help="Output file path (default: docs/CAPABILITY_MATRIX.md)"
    )
    parser.add_argument(
        "--test-pattern",
        default="tests/test_*.py",  # Phase 13.6.F: Changed default to all tests
        help="Pytest test pattern (default: tests/test_*.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print to stdout instead of writing file"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest (use empty outcomes)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--from-reports",
        nargs="+",
        metavar="JSON_FILE",
        help="Read test outcomes from existing JSON report files (from pytest-json-report)"
    )
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Import taxonomy (try multiple paths for robustness)
    # -------------------------------------------------------------------------
    FEATURE_TAXONOMY = None
    KNOWN_LIMITATIONS = None
    
    import_attempts = [
        "feature_taxonomy",        # When tests/ is in sys.path
        "tests.feature_taxonomy",  # When project root is in sys.path
    ]
    
    for module_name in import_attempts:
        try:
            import importlib
            module = importlib.import_module(module_name)
            FEATURE_TAXONOMY = getattr(module, "FEATURE_TAXONOMY", None)
            KNOWN_LIMITATIONS = getattr(module, "KNOWN_LIMITATIONS", {})
            if FEATURE_TAXONOMY is not None:
                if args.verbose:
                    print(f"Loaded taxonomy from: {module_name}")
                break
        except ImportError:
            continue
    
    if FEATURE_TAXONOMY is None:
        print("Error: Cannot import feature taxonomy")
        print("Tried: " + ", ".join(import_attempts))
        print("Make sure tests/feature_taxonomy.py exists")
        sys.exit(1)
    
    if args.verbose:
        print(f"Loaded {len(FEATURE_TAXONOMY)} features, {len(KNOWN_LIMITATIONS)} limitations")
    
    # -------------------------------------------------------------------------
    # Parse feature markers from test files
    # -------------------------------------------------------------------------
    print(f"Parsing @pytest.mark.feature markers from {args.test_pattern}...")
    feature_tests = parse_feature_markers_from_files(args.test_pattern)
    print(f"Found {sum(len(v) for v in feature_tests.values())} test-feature mappings across {len(feature_tests)} features")
    
    if args.verbose:
        for fid, tests in sorted(feature_tests.items()):
            print(f"  {fid}: {len(tests)} tests")
    
    # -------------------------------------------------------------------------
    # Collect test outcomes (from reports or by running pytest)
    # -------------------------------------------------------------------------
    if args.from_reports:
        # Read outcomes from existing JSON report files
        print(f"Reading test outcomes from {len(args.from_reports)} report file(s)...")
        test_outcomes = {}
        for report_path in args.from_reports:
            report_path = Path(report_path)
            if not report_path.exists():
                print(f"  Warning: Report not found: {report_path}")
                continue
            try:
                with open(report_path) as f:
                    report = json.load(f)
                count = 0
                for test in report.get("tests", []):
                    nodeid = test.get("nodeid", "")
                    outcome = test.get("outcome", "unknown")
                    
                    # Normalize nodeid: ensure it has tests/ prefix for consistency
                    # Serial runs from tests/ dir so nodeids lack prefix
                    # Parallel runs from project root so nodeids have tests/ prefix
                    if not nodeid.startswith("tests/"):
                        nodeid_normalized = f"tests/{nodeid}"
                    else:
                        nodeid_normalized = nodeid
                    
                    # Store both forms for flexible matching
                    test_outcomes[nodeid] = outcome
                    test_outcomes[nodeid_normalized] = outcome
                    count += 1
                print(f"  Loaded {count} outcomes from {report_path.name}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: Failed to parse {report_path}: {e}")
        print(f"Total: {len(test_outcomes)} test outcomes loaded (with normalized duplicates)")
    elif args.skip_tests:
        print("Skipping pytest run (--skip-tests)")
        test_outcomes = {}
    else:
        print(f"Running pytest (only @pytest.mark.feature tests) to collect outcomes...")
        test_outcomes = run_pytest_for_outcomes(args.test_pattern)
        print(f"Collected outcomes for {len(test_outcomes)} tests")
    
    if args.verbose:
        # Show outcome distribution
        outcome_counts = {}
        for outcome in test_outcomes.values():
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        print(f"Outcomes: {outcome_counts}")
    
    # -------------------------------------------------------------------------
    # Read existing matrix (for manual section preservation)
    # -------------------------------------------------------------------------
    output_path = Path(args.output)
    existing_content = read_existing_matrix(output_path)
    
    if existing_content:
        print(f"Found existing matrix at {output_path} — preserving MANUAL sections")
    else:
        print(f"No existing matrix found — will create with default MANUAL section")
    
    # -------------------------------------------------------------------------
    # Generate matrix
    # -------------------------------------------------------------------------
    print(f"Generating capability matrix...")
    matrix = generate_matrix(
        FEATURE_TAXONOMY, 
        KNOWN_LIMITATIONS, 
        test_outcomes, 
        feature_tests,
        existing_content
    )
    
    # Output
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN OUTPUT:")
        print("=" * 60 + "\n")
        print(matrix)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(matrix)
        print(f"✅ Generated {args.output}")
        
        # Verify manual section preserved
        if existing_content:
            new_content = output_path.read_text()
            if "<!-- MANUAL:" in new_content:
                print("✅ MANUAL section preserved")
            else:
                print("⚠️ Warning: MANUAL section may not have been preserved")
        
        # Generate debug log with detailed test outcomes
        debug_log_path = output_path.parent / "capability_matrix_debug.log"
        debug_lines = [
            f"Capability Matrix Debug Log",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"=" * 70,
            f"TEST OUTCOMES BY FEATURE",
            f"=" * 70,
            f"",
        ]
        
        for feature_id, feature_info in FEATURE_TAXONOMY.items():
            name = feature_info.get("name", feature_id)
            feature_test_list = feature_tests.get(feature_id, [])
            
            # Compute status
            status, test_count, failed_tests = compute_status(
                feature_id, feature_test_list, test_outcomes, KNOWN_LIMITATIONS, FEATURE_TAXONOMY
            )
            
            debug_lines.append(f"Feature: {name}")
            debug_lines.append(f"  ID: {feature_id}")
            debug_lines.append(f"  Status: {status}")
            debug_lines.append(f"  Tests: {test_count}")
            
            if failed_tests:
                debug_lines.append(f"  FAILED TESTS:")
                for t in failed_tests:
                    debug_lines.append(f"    ❌ {t}")
            
            # Show all test outcomes for this feature
            if feature_test_list:
                debug_lines.append(f"  All test outcomes:")
                for nodeid in sorted(feature_test_list):
                    outcome = test_outcomes.get(nodeid)
                    if outcome is None:
                        outcome = test_outcomes.get(f"tests/{nodeid}")
                    if outcome is None:
                        for key, val in test_outcomes.items():
                            if nodeid in key or key.endswith(nodeid):
                                outcome = val
                                break
                    outcome = outcome or "unknown"
                    marker = "✅" if outcome == "passed" else "❌" if outcome == "failed" else "❓"
                    debug_lines.append(f"    {marker} [{outcome:8}] {nodeid}")
            
            debug_lines.append("")
        
        # Summary of all failures
        debug_lines.append("=" * 70)
        debug_lines.append("ALL FAILED TESTS")
        debug_lines.append("=" * 70)
        debug_lines.append("")
        
        for nodeid, outcome in sorted(test_outcomes.items()):
            if outcome in ("failed", "error"):
                debug_lines.append(f"❌ {nodeid}")
        
        debug_log_path.write_text("\n".join(debug_lines))
        print(f"✅ Generated debug log: {debug_log_path}")


if __name__ == "__main__":
    main()

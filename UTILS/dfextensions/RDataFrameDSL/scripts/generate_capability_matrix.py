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

Phase: 13.6.B.fix
Date: 2026-01-18
Version: 1.2 (fixed glob expansion in subprocess)
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

# Determine project root (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

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
        
        for i, line in enumerate(lines):
            # Track class context
            class_match = re.match(r'^class (\w+)', line)
            if class_match:
                current_class = class_match.group(1)
                pending_features = []  # Reset on new class
                continue
            
            # Look for feature marker
            feature_match = re.search(r'@pytest\.mark\.feature\(["\'](\w+)["\']\)', line)
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

def run_pytest_for_outcomes(test_pattern: str = "tests/test_invariance_*.py") -> Dict[str, str]:
    """
    Run pytest to get actual pass/fail/xfail status.
    
    Returns dict mapping nodeid -> outcome (passed/failed/xfailed/xpassed/skipped/error)
    
    Uses tempfile to avoid path collisions (P2 fix).
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
        cmd = [
            sys.executable, "-m", "pytest",
            *expanded_files,  # Expanded file paths
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
) -> Tuple[str, int]:
    """
    Compute feature status from test outcomes.
    
    Returns (status_string, test_count)
    
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
        return "❌ Not Implemented", 0
    
    # Collect outcomes for all tests in this feature
    has_failed = False
    has_error = False
    has_xfailed = False
    has_xpassed = False
    has_passed = False
    has_skipped = False
    has_unknown = False
    test_count = 0
    
    for test_nodeid in feature_test_nodeids:
        outcome = test_outcomes.get(test_nodeid, "unknown")
        test_count += 1
        
        if outcome == "failed":
            has_failed = True
        elif outcome == "error":
            has_error = True
        elif outcome == "xfailed":
            has_xfailed = True
        elif outcome == "xpassed":
            has_xpassed = True
        elif outcome == "passed":
            has_passed = True
        elif outcome == "skipped":
            has_skipped = True
        else:
            has_unknown = True
    
    # Priority 1: Failures and errors always escalate to Broken
    if has_failed or has_error:
        return "🧨 Broken", test_count
    
    # Priority 2: Unexpected pass (xpassed) = implementation changed unexpectedly
    if has_xpassed:
        return "🧨 Broken", test_count
    
    # Priority 3: Expected failures (xfailed)
    if has_xfailed:
        # Check if feature has explicit limitation in taxonomy
        feature_info = taxonomy.get(feature_id, {})
        if feature_info.get("limitation"):
            lim_id = feature_info["limitation"]
            if lim_id in limitations:
                return limitations[lim_id]["status"], test_count
        # xfail without limitation = partial (tests exist but don't pass)
        return "⚠️ Partial", test_count
    
    # Priority 4: Unknown/skipped only (no real evidence)
    if (has_unknown or has_skipped) and not has_passed:
        return "❓ Unknown", test_count
    
    # Priority 5: At least some passed
    if has_passed:
        return "✅ Working", test_count
    
    # Fallback: no tests matched
    return "❓ Unknown", test_count


# =============================================================================
# MANUAL SECTION PRESERVATION
# =============================================================================

def extract_manual_section(existing_content: str) -> Optional[str]:
    """
    Extract the MANUAL section from existing CAPABILITY_MATRIX.md.
    
    Preserves everything after the MANUAL marker, including:
    - Known Limitations (human-curated notes)
    - For Reviewers section
    - Any other human additions
    
    Returns None if no existing content or no MANUAL marker.
    """
    if not existing_content:
        return None
    
    # Look for the MANUAL marker
    manual_marker = "<!-- MANUAL: Human-maintained sections below"
    
    if manual_marker not in existing_content:
        return None
    
    # Extract everything from the marker onwards
    parts = existing_content.split(manual_marker, 1)
    if len(parts) < 2:
        return None
    
    # Return the marker + content
    return manual_marker + parts[1]


def read_existing_matrix(output_path: Path) -> Optional[str]:
    """Read existing CAPABILITY_MATRIX.md if it exists."""
    if output_path.exists():
        return output_path.read_text()
    return None


# =============================================================================
# MATRIX GENERATION
# =============================================================================

def generate_auto_section(
    taxonomy: Dict,
    limitations: Dict,
    test_outcomes: Dict[str, str],
    feature_tests: Dict[str, List[str]],
) -> List[str]:
    """
    Generate the AUTO-GENERATED section of the matrix.
    
    Uses feature_tests mapping from marker parsing, NOT taxonomy["tests"].
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    lines = [
        "# Capability Matrix",
        "",
        f"**Last Updated:** {timestamp}",
        "**Generated By:** `scripts/generate_capability_matrix.py`",
        "**Phase:** 13.6.B.fix",
        "",
        "> **Note:** Quick Status and Examples Index are auto-generated.",
        "> Known Limitations and For Reviewers sections are human-maintained.",
        "",
        "---",
        "",
        "<!-- AUTO-GENERATED: Do not edit below this line until END AUTO-GENERATED -->",
        "",
        "## Quick Status",
        "",
        "| Feature | Status | Proof | Tests | Notes |",
        "|---------|--------|-------|-------|-------|",
    ]
    
    # Track summary stats
    working = 0
    partial = 0
    broken = 0
    
    for feature_id, feature in taxonomy.items():
        # Get tests for this feature from parsed markers
        tests = feature_tests.get(feature_id, [])
        
        # Compute status from test outcomes
        status, test_count = compute_status(
            feature_id, tests, test_outcomes, limitations, taxonomy
        )
        
        # Update counters
        if "✅" in status:
            working += 1
        elif "⚠️" in status:
            partial += 1
        elif "🧨" in status:
            broken += 1
        
        # Format row
        name = feature.get("name", feature_id)
        proof = feature.get("proof", "—")
        if proof != "—":
            proof = f"`{proof}`"
        
        notes = feature.get("notes", "")
        if feature.get("limitation"):
            lim_id = feature["limitation"]
            notes = f"⚠️ {lim_id}: {limitations.get(lim_id, {}).get('name', 'Unknown')}"
        
        # Truncate long notes
        if len(notes) > 50:
            notes = notes[:47] + "..."
        
        lines.append(f"| {name} | {status} | {proof} | {test_count} tests | {notes} |")
    
    total = len(taxonomy)
    lines.extend([
        "",
        f"**Summary:** {working}/{total} working, {partial} partial, {broken} broken",
        "",
    ])
    
    # Examples Index
    lines.extend([
        "## Examples Index",
        "",
        "| Example | Features Demonstrated |",
        "|---------|----------------------|",
    ])
    
    # Collect examples -> features mapping
    example_features = {}
    for feature_id, feature in taxonomy.items():
        proof = feature.get("proof")
        if proof and proof != "—":
            if proof not in example_features:
                example_features[proof] = []
            example_features[proof].append(feature.get("name", feature_id))
    
    for example, features in sorted(example_features.items()):
        lines.append(f"| `{example}` | {', '.join(features)} |")
    
    lines.extend([
        "",
        "<!-- END AUTO-GENERATED -->",
        "",
    ])
    
    return lines


def generate_default_manual_section(limitations: Dict) -> List[str]:
    """
    Generate default MANUAL section for first-time creation.
    
    This is only used when no existing CAPABILITY_MATRIX.md exists.
    After first generation, this section is preserved across regenerations.
    """
    lines = [
        "---",
        "",
        "<!-- MANUAL: Human-maintained sections below. Do not auto-generate. -->",
        "",
        "## Known Limitations",
        "",
    ]
    
    # Initial population from KNOWN_LIMITATIONS
    for lim_id, lim in limitations.items():
        lines.extend([
            f"### {lim_id}: {lim['name']}",
            "",
            f"**Status:** {lim['status']}",
            f"**Symptom:** {lim['description']}",
            f"**Workaround:** {lim['workaround']}",
            f"**Tests affected:** {len(lim.get('tests_affected', []))} tests",
            f"**Bug report:** `{lim['bug_report']}`",
            f"**Resolution:** {lim['resolution']}",
            "",
        ])
    
    # Verification Gate section (required per Organization-structure v1.12)
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
        default="tests/test_invariance_*.py",
        help="Pytest test pattern (default: tests/test_invariance_*.py)"
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
    # Run pytest to collect outcomes
    # -------------------------------------------------------------------------
    if args.skip_tests:
        print("Skipping pytest run (--skip-tests)")
        test_outcomes = {}
    else:
        print(f"Running pytest to collect test outcomes...")
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


if __name__ == "__main__":
    main()

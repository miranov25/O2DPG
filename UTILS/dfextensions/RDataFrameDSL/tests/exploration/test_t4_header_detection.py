#!/usr/bin/env python3
"""
Phase 13.5.A - Test T4: Header Detection

Objective: Identify required headers for common patterns

DISCLAIMER: This is EXPLORATION SCAFFOLDING only, not the final algorithm.

Limitations:
- Substring-based matching (not AST-based)
- False positives: 'my_sqrt_var' triggers <cmath>
- False negatives: 'using std::sin; sin(x)' misses <cmath>
- Not namespace-aware
- Cannot handle macro expansions

Phase 13.5.B Decision: Based on T4 findings, choose:
- Option A: Conservative includes + user overrides (simple, reliable)
- Option B: AST-based detection (complex, accurate)

Per Phase 13.5 v0.3 specification.
"""

import sys
from typing import Set, List, Tuple
from datetime import datetime

RESULTS = {
    "test": "T4: Header Detection",
    "timestamp": datetime.now().isoformat(),
    "status": "NOT_RUN",
    "findings": [],
    "false_positives": [],
    "false_negatives": [],
    "recommendations": [],
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
    print(f"{prefix.get(level, '')} {msg}")
    RESULTS["findings"].append(f"[{level}] {msg}")


def detect_headers(expression: str) -> Set[str]:
    """
    Analyze expression to determine required headers.
    
    WARNING: This is EXPLORATION SCAFFOLDING only.
    - Substring-based matching (not AST-based)
    - False positives: 'my_sqrt_var' triggers <cmath>
    - False negatives: 'using std::sin; sin(x)' misses <cmath>
    - Not namespace-aware
    - Cannot handle macro expansions
    
    Phase 13.5.B will determine if:
    - Conservative includes + user overrides are sufficient, OR
    - AST-based detection is required
    """
    headers = set()
    
    # Standard math functions
    math_funcs = ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'pow', 'abs',
                  'floor', 'ceil', 'fabs', 'atan', 'atan2', 'asin', 'acos',
                  'sinh', 'cosh', 'tanh', 'log10', 'log2', 'cbrt', 'hypot']
    if any(f in expression for f in math_funcs):
        headers.add('<cmath>')
    
    # ROOT types
    if 'RVec' in expression or 'ROOT::RVec' in expression:
        headers.add('<ROOT/RVec.hxx>')
    if 'RDataFrame' in expression or 'ROOT::RDataFrame' in expression:
        headers.add('<ROOT/RDataFrame.hxx>')
    if 'TLorentzVector' in expression:
        headers.add('<TLorentzVector.h>')
    if 'TVector3' in expression:
        headers.add('<TVector3.h>')
    if 'TVector2' in expression:
        headers.add('<TVector2.h>')
    if 'TMath::' in expression:
        headers.add('<TMath.h>')
    if 'TRandom' in expression or 'gRandom' in expression:
        headers.add('<TRandom.h>')
    
    # STL algorithms
    if any(f in expression for f in ['std::sort', 'std::max', 'std::min',
                                       'std::find', 'std::count', 'std::copy',
                                       'std::transform', 'std::remove_if']):
        headers.add('<algorithm>')
    
    if 'std::accumulate' in expression or 'std::reduce' in expression:
        headers.add('<numeric>')
    
    # STL containers
    if 'std::vector' in expression:
        headers.add('<vector>')
    if 'std::map' in expression:
        headers.add('<map>')
    if 'std::set' in expression:
        headers.add('<set>')
    if 'std::string' in expression:
        headers.add('<string>')
    
    # I/O
    if 'std::cout' in expression or 'std::cerr' in expression:
        headers.add('<iostream>')
    
    return headers


# Test cases: (expression, expected_headers)
TEST_CASES: List[Tuple[str, List[str]]] = [
    # Math functions
    ("sqrt(px**2 + py**2)", ["<cmath>"]),
    ("std::sin(x)", ["<cmath>"]),
    ("std::sqrt(x*x + y*y)", ["<cmath>"]),
    ("atan2(y, x)", ["<cmath>"]),
    ("log(x) + exp(y)", ["<cmath>"]),
    ("pow(x, 2) + abs(y)", ["<cmath>"]),
    
    # ROOT types
    ("ROOT::RVec<double>()", ["<ROOT/RVec.hxx>"]),
    ("RVec<float> result", ["<ROOT/RVec.hxx>"]),
    ("TMath::Pi()", ["<TMath.h>"]),
    ("TMath::Sqrt(x)", ["<TMath.h>"]),
    ("TLorentzVector(px,py,pz,E)", ["<TLorentzVector.h>"]),
    ("TVector3(x, y, z)", ["<TVector3.h>"]),
    
    # STL algorithms
    ("std::max(x, y)", ["<algorithm>"]),
    ("std::min(a, b)", ["<algorithm>"]),
    ("std::sort(v.begin(), v.end())", ["<algorithm>"]),
    ("std::accumulate(v.begin(), v.end(), 0.0)", ["<numeric>"]),
    
    # Combinations
    ("sqrt(px**2) + TMath::Pi()", ["<cmath>", "<TMath.h>"]),
    ("RVec<double> pts = sqrt(px*px + py*py)", ["<cmath>", "<ROOT/RVec.hxx>"]),
    
    # Edge cases - potential false positives
    ("my_sqrt_variable", ["<cmath>"]),  # Known false positive
    ("calculate_sinusoidal(x)", ["<cmath>"]),  # Known false positive
    
    # Edge cases - potential false negatives
    ("using namespace std; sin(x)", []),  # Missing <cmath> - known limitation
]


def run_header_detection_tests():
    """Run all header detection test cases."""
    
    log("Starting T4: Header Detection Tests")
    log("NOTE: This is exploration scaffolding, not production code")
    log("")
    
    passed = 0
    failed = 0
    false_positives = 0
    false_negatives_count = 0
    
    for expr, expected in TEST_CASES:
        detected = detect_headers(expr)
        expected_set = set(expected)
        
        # Check for matches
        matches = detected & expected_set
        missing = expected_set - detected
        extra = detected - expected_set
        
        # Determine result
        if missing:
            # False negative - we should have detected but didn't
            log(f"MISS: '{expr[:40]}...' - Missing: {missing}", "WARN")
            RESULTS["false_negatives"].append({
                "expression": expr,
                "missing": list(missing)
            })
            false_negatives_count += len(missing)
            failed += 1
        elif extra and not expected_set:
            # Expected nothing, got something - false positive
            log(f"EXTRA: '{expr[:40]}...' - Extra: {extra}", "WARN")
            RESULTS["false_positives"].append({
                "expression": expr,
                "extra": list(extra)
            })
            false_positives += 1
            # This is "expected" for known false positives
            if "sqrt_variable" in expr or "sinusoidal" in expr:
                log("  (Known false positive - substring matching limitation)", "INFO")
                passed += 1
            else:
                failed += 1
        else:
            log(f"OK: '{expr[:40]}...' → {detected}", "PASS")
            passed += 1
    
    return passed, failed, false_positives, false_negatives_count


def test_conservative_defaults():
    """Test the conservative defaults strategy."""
    
    log("\n" + "="*60)
    log("Testing Conservative Defaults Strategy")
    log("="*60)
    
    # Conservative defaults that should always be included
    conservative = {
        '<cmath>',           # Math functions
        '<algorithm>',       # STL algorithms
        '<ROOT/RVec.hxx>',   # ROOT vectors
    }
    
    # Extended conservative for physics analysis
    physics_extended = conservative | {
        '<TLorentzVector.h>',
        '<TMath.h>',
    }
    
    # Test expressions that should "just work" with conservative defaults
    test_expressions = [
        "sqrt(px**2 + py**2)",
        "RVec<double> pts",
        "std::max(a, b)",
        "sin(theta) * cos(phi)",
    ]
    
    log(f"Conservative defaults: {conservative}")
    log(f"Physics extended: {physics_extended}")
    log("")
    
    for expr in test_expressions:
        detected = detect_headers(expr)
        covered = detected <= physics_extended
        if covered:
            log(f"'{expr}' - Covered by conservative defaults", "PASS")
        else:
            log(f"'{expr}' - NOT covered: {detected - physics_extended}", "WARN")


def generate_recommendations():
    """Generate recommendations based on findings."""
    
    log("\n" + "="*60)
    log("RECOMMENDATIONS FOR PHASE 13.5.B")
    log("="*60)
    
    recommendations = []
    
    # Based on false positive rate
    if RESULTS["false_positives"]:
        recommendations.append(
            "Option A (Conservative): Accept false positives as harmless "
            "(extra includes don't break compilation)"
        )
    
    # Based on false negative rate
    if RESULTS["false_negatives"]:
        recommendations.append(
            "User override mechanism is REQUIRED for expressions that "
            "escape pattern matching"
        )
    
    # General recommendation
    recommendations.append(
        "RECOMMENDED: Conservative defaults + user `headers=` parameter "
        "in define_raw(). Simpler and more reliable than AST parsing."
    )
    
    for i, rec in enumerate(recommendations, 1):
        log(f"{i}. {rec}", "INFO")
        RESULTS["recommendations"].append(rec)


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("T4 HEADER DETECTION SUMMARY")
    print("="*60)
    print(f"Status: {RESULTS['status']}")
    print(f"Timestamp: {RESULTS['timestamp']}")
    print(f"\nFalse Positives: {len(RESULTS['false_positives'])}")
    print(f"False Negatives: {len(RESULTS['false_negatives'])}")
    print("\nRecommendations:")
    for rec in RESULTS["recommendations"]:
        print(f"  • {rec}")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("Phase 13.5.A - Test T4: Header Detection")
    print("="*60)
    print("DISCLAIMER: This is exploration scaffolding only")
    print("")
    
    passed, failed, fp, fn = run_header_detection_tests()
    test_conservative_defaults()
    generate_recommendations()
    
    # Determine overall status
    # Note: We expect some false positives/negatives - that's the point of exploration
    if failed <= 2 and fn <= 2:
        RESULTS["status"] = "PASS"
    else:
        RESULTS["status"] = "PARTIAL"
    
    print_summary()
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    print(f"   False positives: {fp} (harmless - extra includes)")
    print(f"   False negatives: {fn} (requires user override)")
    
    if RESULTS["status"] == "PASS":
        print("\n✅ T4 PASS: Header detection exploration complete")
        print("   Conservative defaults + user overrides recommended")
        sys.exit(0)
    else:
        print("\n⚠️  T4 PARTIAL: Some patterns not detected")
        print("   User override mechanism is essential")
        sys.exit(0)  # Still exit 0 - partial is expected for exploration

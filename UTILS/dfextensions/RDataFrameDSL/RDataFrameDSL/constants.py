"""
Constants shared across RDataFrameDSL modules.

This module centralizes function mappings, type definitions, and other
constants used by ir_builder.py and backend_cpp.py to avoid duplication.

Phase 9 Refactoring: Extracted from ir_builder.py and backend_cpp.py
"""

from typing import Dict, List, Set

__all__ = [
    'KNOWN_FUNCTIONS',
    'MATH_FUNCTIONS',
    'TMATH_FUNCTIONS',
    'REDUCTION_FUNCTIONS',
    'RVEC_METHODS',
    'RVEC_AGGREGATION_METHODS',
    'FUNCTION_HEADERS',
    'FUNCTION_CPP_NAMES',
    'CLASS_HEADERS',
    'SCALAR_TYPES',
    # Phase 11.1: Namespace support
    'KNOWN_NAMESPACES',
    'NAMESPACE_HEADERS',
    'NAMESPACE_FUNCTION_TYPES',
    # Phase 11.1b: Scalar-to-vector broadcasting
    'VECTORIZED_NAMESPACES',
]


# =============================================================================
# Known Functions (for IR Builder validation)
# =============================================================================

MATH_FUNCTIONS: Set[str] = {
    # Standard math
    'sqrt', 'abs', 'fabs',
    'sin', 'cos', 'tan',
    'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh',
    'exp', 'exp2', 'log', 'log10', 'log2',
    'pow', 'floor', 'ceil', 'round', 'trunc',
    'fmod', 'hypot',
    'min', 'max',
}

TMATH_FUNCTIONS: Set[str] = {
    'TMath.Gaus', 'TMath.Landau',
    'TMath.Sqrt', 'TMath.Abs',
    'TMath.Sin', 'TMath.Cos', 'TMath.Tan',
    'TMath.Log', 'TMath.Exp', 'TMath.Power',
    'TMath.Pi', 'TMath.E', 'TMath.TwoPi',
    'TMath.PiOver2', 'TMath.PiOver4',
    'TMath.DegToRad', 'TMath.RadToDeg',
    'TMath.ATan2', 'TMath.Hypot',
    'TMath.Sign', 'TMath.Min', 'TMath.Max', 'TMath.Range',
}

# =============================================================================
# RVec Reduction Functions (Phase 10.5)
# =============================================================================

REDUCTION_FUNCTIONS: Set[str] = {
    'Sum', 'Mean', 'Max', 'Min',
    'Any', 'All',
    'StdDev', 'Var',
}

# Combined set for validation
KNOWN_FUNCTIONS: Set[str] = MATH_FUNCTIONS | TMATH_FUNCTIONS | REDUCTION_FUNCTIONS


# =============================================================================
# RVec Methods
# =============================================================================

RVEC_METHODS: Dict[str, str] = {
    'size': 'size_t',
    'empty': 'bool',
    'at': 'element',  # Returns element type
}

# RVec aggregation methods (lowercase, method-style) -> function name
RVEC_AGGREGATION_METHODS: Dict[str, str] = {
    'sum': 'Sum',
    'mean': 'Mean',
    'max': 'Max',
    'min': 'Min',
    'any': 'Any',
    'all': 'All',
    'std': 'StdDev',
    'var': 'Var',
}


# =============================================================================
# Scalar Type Names (for broadcast detection)
# =============================================================================

SCALAR_TYPES: Set[str] = {
    # C++ fundamental types
    'double', 'float',
    'int', 'long', 'short', 'char',
    'unsigned int', 'unsigned long', 'unsigned short', 'unsigned char',
    'long long', 'unsigned long long',
    'bool',
    'size_t', 'ptrdiff_t',
    # ROOT typedefs
    'Double_t', 'Float_t',
    'Int_t', 'Long_t', 'Short_t', 'Char_t',
    'UInt_t', 'ULong_t', 'UShort_t', 'UChar_t',
    'Long64_t', 'ULong64_t',
    'Bool_t',
    'Size_t',
}


# =============================================================================
# Header Registry (for C++ code generation)
# =============================================================================

FUNCTION_HEADERS: Dict[str, List[str]] = {
    # Standard math (std:: functions from <cmath>)
    "sqrt": ["<cmath>"],
    "sin": ["<cmath>"],
    "cos": ["<cmath>"],
    "tan": ["<cmath>"],
    "asin": ["<cmath>"],
    "acos": ["<cmath>"],
    "atan": ["<cmath>"],
    "atan2": ["<cmath>"],
    "sinh": ["<cmath>"],
    "cosh": ["<cmath>"],
    "tanh": ["<cmath>"],
    "log": ["<cmath>"],
    "log10": ["<cmath>"],
    "log2": ["<cmath>"],
    "exp": ["<cmath>"],
    "exp2": ["<cmath>"],
    "pow": ["<cmath>"],
    "abs": ["<cmath>"],
    "fabs": ["<cmath>"],
    "floor": ["<cmath>"],
    "ceil": ["<cmath>"],
    "round": ["<cmath>"],
    "trunc": ["<cmath>"],
    "fmod": ["<cmath>"],
    "hypot": ["<cmath>"],
    
    # TMath functions
    "TMath::Gaus": ["<TMath.h>"],
    "TMath::Landau": ["<TMath.h>"],
    "TMath::Sqrt": ["<TMath.h>"],
    "TMath::Abs": ["<TMath.h>"],
    "TMath::Sin": ["<TMath.h>"],
    "TMath::Cos": ["<TMath.h>"],
    "TMath::Tan": ["<TMath.h>"],
    "TMath::Log": ["<TMath.h>"],
    "TMath::Exp": ["<TMath.h>"],
    "TMath::Power": ["<TMath.h>"],
    "TMath::Pi": ["<TMath.h>"],
    "TMath::E": ["<TMath.h>"],
    "TMath::TwoPi": ["<TMath.h>"],
    "TMath::PiOver2": ["<TMath.h>"],
    "TMath::PiOver4": ["<TMath.h>"],
    "TMath::DegToRad": ["<TMath.h>"],
    "TMath::RadToDeg": ["<TMath.h>"],
    "TMath::ATan2": ["<TMath.h>"],
    "TMath::Hypot": ["<TMath.h>"],
    "TMath::Sign": ["<TMath.h>"],
    "TMath::Min": ["<TMath.h>"],
    "TMath::Max": ["<TMath.h>"],
    "TMath::Range": ["<TMath.h>"],
    
    # RVec reduction functions (Phase 10.5)
    "Sum": ["<ROOT/RVec.hxx>"],
    "Mean": ["<ROOT/RVec.hxx>"],
    "Max": ["<ROOT/RVec.hxx>"],
    "Min": ["<ROOT/RVec.hxx>"],
    "Any": ["<ROOT/RVec.hxx>"],
    "All": ["<ROOT/RVec.hxx>"],
    "StdDev": ["<ROOT/RVec.hxx>"],
    "Var": ["<ROOT/RVec.hxx>"],
}

# Mapping from Python/DSL function names to C++ equivalents
FUNCTION_CPP_NAMES: Dict[str, str] = {
    # Standard math -> std:: versions
    "sqrt": "std::sqrt",
    "sin": "std::sin",
    "cos": "std::cos",
    "tan": "std::tan",
    "asin": "std::asin",
    "acos": "std::acos",
    "atan": "std::atan",
    "atan2": "std::atan2",
    "sinh": "std::sinh",
    "cosh": "std::cosh",
    "tanh": "std::tanh",
    "log": "std::log",
    "log10": "std::log10",
    "log2": "std::log2",
    "exp": "std::exp",
    "exp2": "std::exp2",
    "pow": "std::pow",
    "abs": "std::abs",
    "fabs": "std::fabs",
    "floor": "std::floor",
    "ceil": "std::ceil",
    "round": "std::round",
    "trunc": "std::trunc",
    "fmod": "std::fmod",
    "hypot": "std::hypot",
    "min": "std::min",
    "max": "std::max",
    
    # TMath functions (keep as-is with :: separator)
    "TMath.Gaus": "TMath::Gaus",
    "TMath.Landau": "TMath::Landau",
    "TMath.Sqrt": "TMath::Sqrt",
    "TMath.Abs": "TMath::Abs",
    "TMath.Sin": "TMath::Sin",
    "TMath.Cos": "TMath::Cos",
    "TMath.Tan": "TMath::Tan",
    "TMath.Log": "TMath::Log",
    "TMath.Exp": "TMath::Exp",
    "TMath.Power": "TMath::Power",
    "TMath.Pi": "TMath::Pi",
    "TMath.E": "TMath::E",
    "TMath.TwoPi": "TMath::TwoPi",
    "TMath.PiOver2": "TMath::PiOver2",
    "TMath.PiOver4": "TMath::PiOver4",
    "TMath.DegToRad": "TMath::DegToRad",
    "TMath.RadToDeg": "TMath::RadToDeg",
    "TMath.ATan2": "TMath::ATan2",
    "TMath.Hypot": "TMath::Hypot",
    "TMath.Sign": "TMath::Sign",
    "TMath.Min": "TMath::Min",
    "TMath.Max": "TMath::Max",
    "TMath.Range": "TMath::Range",
    
    # RVec reduction functions (Phase 10.5)
    "Sum": "ROOT::VecOps::Sum",
    "Mean": "ROOT::VecOps::Mean",
    "Max": "ROOT::VecOps::Max",
    "Min": "ROOT::VecOps::Min",
    "Any": "ROOT::VecOps::Any",
    "All": "ROOT::VecOps::All",
    "StdDev": "ROOT::VecOps::StdDev",
    "Var": "ROOT::VecOps::Var",
}


# =============================================================================
# Class Headers (for ROOT classes)
# =============================================================================

CLASS_HEADERS: Dict[str, List[str]] = {
    # ROOT core classes
    "TLorentzVector": ["<TLorentzVector.h>"],
    "TVector3": ["<TVector3.h>"],
    "TVector2": ["<TVector2.h>"],
    "TParticle": ["<TParticle.h>"],
    "TString": ["<TString.h>"],
    
    # ROOT RVec (always needed for RVec operations)
    "RVec": ["<ROOT/RVec.hxx>"],
    "ROOT::RVec": ["<ROOT/RVec.hxx>"],
    "ROOT::VecOps::RVec": ["<ROOT/RVec.hxx>"],
}


# =============================================================================
# Reflection Headers (for private member access)
# =============================================================================

REFLECTION_HEADERS: List[str] = [
    "<TClass.h>",
    "<TDataMember.h>",
]


# =============================================================================
# Phase 11.1: Namespace Support
# =============================================================================

# Known ROOT/C++ namespaces (builtins)
KNOWN_NAMESPACES: Set[str] = {
    "TMath",
    "ROOT",
    "ROOT.Math",
    "ROOT.Math.VectorUtil",
    "ROOT.VecOps",
    "std",
}

# Headers required for namespaces
NAMESPACE_HEADERS: Dict[str, str] = {
    "TMath": "<TMath.h>",
    "ROOT.Math": "<Math/Vector4D.h>",
    "ROOT.Math.VectorUtil": "<Math/VectorUtil.h>",
    "ROOT.VecOps": "<ROOT/RVec.hxx>",
    "std": "<cmath>",
}

# Return types for common namespace functions (explicit, no guessing)
NAMESPACE_FUNCTION_TYPES: Dict[str, Dict[str, str]] = {
    "TMath": {
        "Pi": "double",
        "E": "double",
        "Sin": "double",
        "Cos": "double",
        "Tan": "double",
        "ASin": "double",
        "ACos": "double",
        "ATan": "double",
        "ATan2": "double",
        "Sqrt": "double",
        "Exp": "double",
        "Log": "double",
        "Log10": "double",
        "Abs": "double",
        "Power": "double",
        "Min": "double",
        "Max": "double",
        "Sign": "double",
        "Gaus": "double",
        "BreitWigner": "double",
        "Landau": "double",
        "TwoPi": "double",
        "PiOver2": "double",
        "PiOver4": "double",
        "DegToRad": "double",
        "RadToDeg": "double",
        "Hypot": "double",
        "Range": "double",
        "Floor": "double",
        "Ceil": "double",
        "Nint": "int",
    },
    "ROOT.Math.VectorUtil": {
        "DeltaPhi": "double",
        "DeltaR": "double",
        "CosTheta": "double",
        "Angle": "double",
        "InvariantMass": "double",
    },
}


# =============================================================================
# Phase 11.1b: Scalar-to-Vector Broadcasting
# =============================================================================

# Namespaces that already support vectorized operations (don't wrap in loop)
# All other namespaces are treated as scalar-only by default
VECTORIZED_NAMESPACES: Set[str] = {
    "ROOT.VecOps",      # Already handles RVec natively
    "ROOT::VecOps",     # C++ notation variant
    "std",              # std:: math functions use ADL to find ROOT::VecOps versions
}

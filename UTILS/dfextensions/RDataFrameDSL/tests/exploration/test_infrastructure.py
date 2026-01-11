#!/usr/bin/env python3
"""
Phase 13.5.B0 Extended Tests — Common Infrastructure

Provides:
- setup_test_env(): Local workspace (fixes T7d /tmp issue)
- Observation tracking and reporting
- Mock DSL for testing (from T11-T14)
- Common test utilities

Per Phase 13.5.B0 v7 specification.
All tests use local workspace, NOT /tmp (macOS SIP restrictions).
"""

import os
import sys
import tempfile
import hashlib
import re
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

# =============================================================================
# Test Environment Setup (Fixes T7d /tmp Issue)
# =============================================================================

_WORKSPACE: Optional[Path] = None

def setup_test_env(prefix: str = "dsl_test_") -> Path:
    """
    Create local workspace for test artifacts.
    
    Addresses T7d /tmp issue on macOS (SIP restrictions).
    Uses current working directory, not /tmp.
    
    Returns:
        Path to workspace directory
    """
    global _WORKSPACE
    
    if _WORKSPACE is not None and _WORKSPACE.exists():
        return _WORKSPACE
    
    # Use local directory, NOT /tmp
    base_dir = Path.cwd() / "build_tests"
    base_dir.mkdir(exist_ok=True)
    
    # Create unique workspace for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = base_dir / f"{prefix}{timestamp}"
    workspace.mkdir(exist_ok=True)
    
    # Set environment variables for ROOT
    os.environ["ROOT_INCLUDE_PATH"] = str(workspace)
    
    # Add to LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on macOS)
    lib_path_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    existing = os.environ.get(lib_path_var, "")
    os.environ[lib_path_var] = f"{workspace}:{existing}" if existing else str(workspace)
    
    _WORKSPACE = workspace
    return workspace


def cleanup_test_env():
    """Clean up test workspace after tests complete."""
    global _WORKSPACE
    if _WORKSPACE is not None and _WORKSPACE.exists():
        try:
            shutil.rmtree(_WORKSPACE)
        except Exception as e:
            print(f"Warning: Could not clean up {_WORKSPACE}: {e}")
        _WORKSPACE = None


def get_workspace() -> Path:
    """Get current workspace, creating if needed."""
    global _WORKSPACE
    if _WORKSPACE is None:
        return setup_test_env()
    return _WORKSPACE


# =============================================================================
# Observation Tracking
# =============================================================================

@dataclass
class TestObservation:
    """Single observation from a test."""
    key: str
    value: Any
    implication: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestResult:
    """Complete result from a test."""
    test_id: str
    title: str
    status: str = "NOT_RUN"  # PASSED, FAILED, PARTIAL, BLOCKED
    hypothesis: str = ""
    observations: List[TestObservation] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: Dict[str, str] = field(default_factory=dict)
    
    def observe(self, key: str, value: Any, implication: str = ""):
        """Record an observation."""
        self.observations.append(TestObservation(key, value, implication))
        print(f"🔍 OBSERVATION: {key} = {value}")
        if implication:
            print(f"   → Implication: {implication}")
    
    def log(self, msg: str, level: str = "INFO"):
        """Log a finding."""
        prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}
        print(f"{prefix.get(level, '')} {msg}")
        self.findings.append(f"[{level}] {msg}")
    
    def set_environment(self):
        """Capture environment info."""
        try:
            import ROOT
            self.environment["root_version"] = ROOT.gROOT.GetVersion()
        except:
            self.environment["root_version"] = "N/A"
        
        self.environment["python_version"] = sys.version.split()[0]
        self.environment["platform"] = sys.platform
        self.environment["workspace"] = str(get_workspace())
    
    def to_markdown(self) -> str:
        """Generate markdown observation report."""
        lines = [
            f"## Test {self.test_id}: {self.title}",
            "",
            "### Date",
            self.timestamp,
            "",
            "### Environment",
            f"- ROOT Version: {self.environment.get('root_version', 'N/A')}",
            f"- Python Version: {self.environment.get('python_version', 'N/A')}",
            f"- Platform: {self.environment.get('platform', 'N/A')}",
            f"- Workspace: {self.environment.get('workspace', 'N/A')}",
            "",
            "### Hypothesis",
            self.hypothesis,
            "",
            "### Key Observations",
        ]
        
        for obs in self.observations:
            lines.append(f"- **{obs.key}:** `{obs.value}`")
            if obs.implication:
                lines.append(f"  - Implication: {obs.implication}")
        
        lines.extend([
            "",
            "### Status",
            f"**{self.status}**",
            "",
        ])
        
        if self.action_items:
            lines.append("### Action Items")
            for item in self.action_items:
                lines.append(f"- [ ] {item}")
        
        return "\n".join(lines)


# =============================================================================
# Mock DSL Compiler (Extended from T11-T14)
# =============================================================================

HASH_LENGTH = 16

@dataclass
class RegisteredFunction:
    """Internal representation of a registered function."""
    name: str
    cpp_name: str
    hash: str
    params: List[Tuple[str, str]]
    return_type: str
    body: str
    full_cpp: str
    headers: Set[str]
    pragmas: Set[str]
    declared: bool
    timestamp: float


class MockDSLCompiler:
    """
    Mock implementation of register_function_cpp for testing.
    Implements v7 specification for validation.
    """
    
    HEADER_MAP = {
        'sqrt': '<cmath>', 'sin': '<cmath>', 'cos': '<cmath>',
        'tan': '<cmath>', 'atan2': '<cmath>', 'exp': '<cmath>',
        'log': '<cmath>', 'pow': '<cmath>', 'abs': '<cmath>',
        'atanh': '<cmath>', 'acos': '<cmath>', 'asin': '<cmath>',
        'Sum': '<ROOT/RVec.hxx>', 'Mean': '<ROOT/RVec.hxx>',
        'RVec': '<ROOT/RVec.hxx>',
    }
    
    # Schema version for cache invalidation (T17)
    HASH_SCHEMA_VERSION = 1
    
    def __init__(self, schema: Optional[Dict[str, str]] = None, 
                 validation_mode: str = "direct"):
        self._schema = schema or {}
        self._functions: Dict[str, RegisteredFunction] = {}
        self._by_name: Dict[str, List[str]] = {}
        self._validation_mode = validation_mode
        self._root_available = self._check_root()
        self._declared_cpp_names: Set[str] = set()
        self._applied = False
    
    def _check_root(self) -> bool:
        try:
            import ROOT
            return True
        except ImportError:
            return False
    
    def register_function_cpp(
        self,
        code: str,
        headers: Optional[List[str]] = None,
        pragmas: Optional[List[str]] = None,
    ) -> 'MockDSLCompiler':
        """Register a C++ function using natural syntax."""
        
        # Parse function signature
        parsed = self._parse_function(code)
        if not parsed:
            raise ValueError(f"Could not parse function signature from:\n{code}")
        
        name, params, return_type, body = parsed
        
        # Auto-detect headers if not provided
        if headers is None:
            headers = list(self._detect_headers(body, return_type))
        
        # Generate hash (excludes function name per v7)
        func_hash = self._generate_hash(params, return_type, body, set(headers))
        
        # Create C++ name: dsl_<name>_<hash16>
        cpp_name = f"dsl_{name}_{func_hash}"
        
        # Build full C++ code
        param_str = ", ".join(f"{ptype} {pname}" for pname, ptype in params)
        full_cpp = f"{return_type} {cpp_name}({param_str}) {{\n    {body}\n}}"
        
        # Add headers
        header_code = "\n".join(f"#include {h}" for h in sorted(headers))
        if header_code:
            full_cpp = header_code + "\n\n" + full_cpp
        
        # Validate with ROOT if available
        declared = False
        if self._root_available and self._validation_mode != "skip":
            declared = self._validate_and_declare(cpp_name, full_cpp)
        else:
            declared = True
        
        # Create registered function
        func = RegisteredFunction(
            name=name,
            cpp_name=cpp_name,
            hash=func_hash,
            params=params,
            return_type=return_type,
            body=body,
            full_cpp=full_cpp,
            headers=set(headers),
            pragmas=set(pragmas or []),
            declared=declared,
            timestamp=time.time(),
        )
        
        # Register
        self._functions[cpp_name] = func
        if name not in self._by_name:
            self._by_name[name] = []
        self._by_name[name].append(cpp_name)
        
        return self
    
    def register_functions_cpp(
        self,
        code: str,
        headers: Optional[List[str]] = None,
        pragmas: Optional[List[str]] = None,
    ) -> 'MockDSLCompiler':
        """Register multiple functions from code block."""
        # Split by function boundaries
        pattern = r'((?:[\w:<>]+\s+)+\w+\s*\([^)]*\)\s*\{[^}]+\})'
        functions = re.findall(pattern, code, re.DOTALL)
        
        for func_code in functions:
            func_code = func_code.strip()
            if func_code:
                self.register_function_cpp(func_code, headers, pragmas)
        
        return self
    
    def _parse_function(self, code: str) -> Optional[Tuple]:
        """Parse C++ function: returns (name, params, return_type, body)"""
        # Normalize whitespace but preserve structure
        code = code.strip()
        
        # Remove comments for parsing
        code_no_comments = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_comments, flags=re.DOTALL)
        
        # Pattern: return_type name(params) { body }
        # Handles: const Type&, RVec<T>, namespaces
        pattern = r'([\w:<>&\s]+?)\s+(\w+)\s*\(([^)]*)\)\s*\{(.+)\}'
        match = re.search(pattern, code_no_comments, re.DOTALL)
        
        if not match:
            return None
        
        return_type = ' '.join(match.group(1).split())
        name = match.group(2).strip()
        params_str = match.group(3).strip()
        body = match.group(4).strip()
        
        # Parse parameters
        params = []
        if params_str:
            # Handle complex types like "const RVec<double>&"
            param_pattern = r'([\w:<>&\s]+?)\s+(\w+)\s*(?:,|$)'
            for pmatch in re.finditer(param_pattern, params_str + ','):
                ptype = ' '.join(pmatch.group(1).split())
                pname = pmatch.group(2).strip()
                params.append((pname, ptype))
        
        return name, params, return_type, body
    
    def _generate_hash(
        self,
        params: List[Tuple[str, str]],
        return_type: str,
        body: str,
        headers: Set[str],
    ) -> str:
        """Generate deterministic hash for function (v7: excludes name)."""
        # Normalize body (T17: whitespace invariant)
        # 1. Collapse all whitespace to single spaces
        normalized_body = ' '.join(body.split())
        # 2. Normalize spacing around operators and punctuation
        # Remove spaces around operators: x * 2 -> x*2
        for op in ['*', '+', '-', '/', '%', '=', '<', '>', '&', '|', '^', '!', '~']:
            normalized_body = normalized_body.replace(f' {op} ', op)
            normalized_body = normalized_body.replace(f' {op}', op)
            normalized_body = normalized_body.replace(f'{op} ', op)
        # Normalize semicolons and braces
        normalized_body = normalized_body.replace(' ;', ';').replace('; ', ';')
        normalized_body = normalized_body.replace(' {', '{').replace('{ ', '{')
        normalized_body = normalized_body.replace(' }', '}').replace('} ', '}')
        
        # Canonicalize types
        param_types = sorted([self._canonicalize_type(t) for _, t in params])
        ret_type = self._canonicalize_type(return_type)
        
        # Build hash input (T17: order invariant for headers)
        hash_input = "|".join([
            f"v{self.HASH_SCHEMA_VERSION}",  # Schema version salt
            normalized_body,
            ",".join(param_types),
            ret_type,
            ",".join(sorted(headers)),
        ])
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:HASH_LENGTH]
    
    def _canonicalize_type(self, type_str: str) -> str:
        """Normalize C++ type strings."""
        type_str = ' '.join(type_str.split())
        # Expand RVec to full name
        if 'RVec<' in type_str and 'ROOT::VecOps::RVec' not in type_str:
            type_str = re.sub(r'\bRVec<', 'ROOT::VecOps::RVec<', type_str)
        # Normalize spacing around templates
        type_str = re.sub(r'<\s+', '<', type_str)
        type_str = re.sub(r'\s+>', '>', type_str)
        type_str = re.sub(r'\s*&', '&', type_str)
        return type_str
    
    def _detect_headers(self, body: str, return_type: str) -> Set[str]:
        """Auto-detect required headers."""
        headers = set()
        combined = body + " " + return_type
        for func, header in self.HEADER_MAP.items():
            if func in combined:
                headers.add(header)
        return headers
    
    def _validate_and_declare(self, cpp_name: str, cpp_code: str) -> bool:
        """Validate and declare with ROOT (T23: idempotent)."""
        # Skip if already declared (T23: idempotency)
        if cpp_name in self._declared_cpp_names:
            return True
        
        import ROOT
        
        if self._validation_mode == "subprocess":
            # T28: Safe mode - validate in subprocess
            return self._validate_subprocess(cpp_code)
        
        # Direct validation
        result = ROOT.gInterpreter.Declare(cpp_code)
        if result:
            self._declared_cpp_names.add(cpp_name)
        return result
    
    def _validate_subprocess(self, cpp_code: str) -> bool:
        """Validate in subprocess (T28: safe mode)."""
        import subprocess
        
        test_script = f'''
import ROOT
code = """{cpp_code}"""
result = ROOT.gInterpreter.Declare(code)
print("SUCCESS" if result else "FAILURE")
'''
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return "SUCCESS" in result.stdout
        except Exception:
            return False
    
    def apply(self, rdf):
        """Apply DSL to RDataFrame (T23: idempotent)."""
        # Mark as applied - functions should already be declared
        self._applied = True
        return rdf
    
    def get_function(self, name: str) -> Optional[RegisteredFunction]:
        """Get most recent function by name."""
        cpp_names = self._by_name.get(name, [])
        if not cpp_names:
            return None
        funcs = [self._functions[cn] for cn in cpp_names]
        return max(funcs, key=lambda f: f.timestamp)
    
    def get_all_functions(self, name: str) -> List[RegisteredFunction]:
        """Get all functions with given name."""
        cpp_names = self._by_name.get(name, [])
        return [self._functions[cn] for cn in cpp_names]
    
    def list_functions(self) -> List[str]:
        """List all function names."""
        return list(self._by_name.keys())
    
    def export_macro(
        self,
        filepath: str,
        namespace: str = "dsl",
        clean_names: bool = True,
        compile: bool = False,
    ) -> Dict:
        """Export functions to C++ macro."""
        workspace = get_workspace()
        
        # Use workspace for output
        if not os.path.isabs(filepath):
            filepath = str(workspace / filepath)
        
        # Collect all headers
        all_headers = set()
        for func in self._functions.values():
            all_headers.update(func.headers)
        
        # Build macro code
        lines = []
        
        # Headers
        for h in sorted(all_headers):
            lines.append(f"#include {h}")
        lines.append("")
        
        # Namespace
        lines.append(f"namespace {namespace} {{")
        lines.append("")
        
        # Functions (only most recent per name)
        exported = 0
        for name in self._by_name:
            func = self.get_function(name)
            if func:
                param_str = ", ".join(f"{ptype} {pname}" for pname, ptype in func.params)
                func_name = name if clean_names else func.cpp_name
                lines.append(f"{func.return_type} {func_name}({param_str}) {{")
                lines.append(f"    {func.body}")
                lines.append("}")
                lines.append("")
                exported += 1
        
        lines.append(f"}} // namespace {namespace}")
        
        # Write file
        macro_code = "\n".join(lines)
        with open(filepath, 'w') as f:
            f.write(macro_code)
        
        result = {
            'filepath': filepath,
            'namespace': namespace,
            'functions_exported': exported,
            'compiled': False,
            'so_path': None,
        }
        
        # Compile if requested
        if compile and self._root_available:
            import ROOT
            ROOT.gROOT.ProcessLine(f'.L {filepath}+')
            so_path = filepath.replace('.C', '_C.so')
            if os.path.exists(so_path):
                result['compiled'] = True
                result['so_path'] = so_path
        
        return result


# =============================================================================
# Test Utilities
# =============================================================================

def wait_for_timestamp_change(seconds: float = 1.1):
    """
    Wait for filesystem timestamp to change.
    
    T18/T25: Filesystem timestamp resolution can cause false negatives.
    Most filesystems have 1-second resolution.
    """
    time.sleep(seconds)


def create_test_macro(name: str, code: str, workspace: Optional[Path] = None) -> str:
    """
    Create a test macro file in workspace.
    
    Args:
        name: Macro name (without .C extension)
        code: C++ code
        workspace: Directory to create file in (default: test workspace)
    
    Returns:
        Full path to created file
    """
    if workspace is None:
        workspace = get_workspace()
    
    filepath = workspace / f"{name}.C"
    with open(filepath, 'w') as f:
        f.write(code)
    
    return str(filepath)


def get_root_version() -> str:
    """Get ROOT version string."""
    try:
        import ROOT
        return ROOT.gROOT.GetVersion()
    except:
        return "N/A"


def check_root_available() -> bool:
    """Check if ROOT is available."""
    try:
        import ROOT
        return True
    except ImportError:
        return False


# =============================================================================
# Entry Point for Testing Infrastructure
# =============================================================================

if __name__ == "__main__":
    print("Phase 13.5.B0 Extended Tests — Infrastructure")
    print("=" * 60)
    
    # Test workspace setup
    workspace = setup_test_env()
    print(f"✅ Workspace created: {workspace}")
    
    # Test ROOT availability
    if check_root_available():
        print(f"✅ ROOT available: {get_root_version()}")
    else:
        print("⚠️  ROOT not available")
    
    # Test MockDSLCompiler
    dsl = MockDSLCompiler()
    dsl.register_function_cpp('''
        double test_pt(double px, double py) {
            return sqrt(px*px + py*py);
        }
    ''')
    
    func = dsl.get_function("test_pt")
    if func:
        print(f"✅ Mock DSL works: {func.cpp_name}")
    
    print("\n✅ Infrastructure ready for T15-T32")

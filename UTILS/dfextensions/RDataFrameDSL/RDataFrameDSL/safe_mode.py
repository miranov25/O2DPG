# =============================================================================
# RDataFrameDSL/safe_mode.py
# =============================================================================
# Safe Mode Implementation - Crash Protection for ROOT Operations
#
# Phase: 13.6.F
# Date: 2026-01-24
#
# Provides:
# - Layer 2: Compile-time protection (fork-probe for JIT compilation)
# - Layer 3: Execution-time protection (probe-run for RDataFrame actions)
#
# Mechanism:
# - Fork child process to attempt potentially crashing operation
# - Parent waits with timeout
# - If child crashes (signal) or fails, parent reports error safely
# - If child succeeds, parent proceeds with actual operation
# =============================================================================

import os
import sys
import signal
import time
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Any, Callable

from .ir_errors import IRError, IRErrorKind


# =============================================================================
# SafeModeError - Structured error for safe mode failures
# =============================================================================

@dataclass
class SafeModeError(Exception):
    """
    Structured error for safe mode failures.
    
    Attributes:
        layer: Which layer failed ("compile" or "execute")
        reason: Why it failed ("crash", "timeout", "precondition", "error")
        message: Human-readable description
        signal_num: Signal number if crashed (e.g., 6 for SIGABRT, 11 for SIGSEGV)
        exit_code: Exit code if exited normally with error
        details: Additional details (e.g., stderr output)
    """
    layer: str
    reason: str
    message: str
    signal_num: Optional[int] = None
    exit_code: Optional[int] = None
    details: Optional[str] = None
    
    def __str__(self):
        parts = [f"SafeModeError [{self.layer}:{self.reason}]: {self.message}"]
        if self.signal_num is not None:
            sig_name = _signal_name(self.signal_num)
            parts.append(f"  Signal: {self.signal_num} ({sig_name})")
        if self.exit_code is not None:
            parts.append(f"  Exit code: {self.exit_code}")
        if self.details:
            parts.append(f"  Details: {self.details}")
        return "\n".join(parts)


def _signal_name(signum: int) -> str:
    """Get human-readable signal name."""
    signal_names = {
        6: "SIGABRT",
        9: "SIGKILL",
        11: "SIGSEGV",
        15: "SIGTERM",
    }
    return signal_names.get(signum, f"SIG{signum}")


# =============================================================================
# Precondition Checks
# =============================================================================

def check_fork_safe() -> None:
    """
    Verify that fork is safe to use.
    
    Raises:
        SafeModeError: If fork would be unsafe
        
    Checks:
        - ROOT implicit MT not enabled (would corrupt after fork)
        - No additional Python threads active
    """
    # Check 1: ROOT implicit MT
    try:
        import ROOT
        if ROOT.IsImplicitMTEnabled():
            raise SafeModeError(
                layer="precondition",
                reason="precondition",
                message="Cannot use safe mode after ROOT.EnableImplicitMT(). "
                        "Call safe mode methods before enabling multi-threading."
            )
    except ImportError:
        pass  # ROOT not available, skip check
    
    # Check 2: Python threading
    active_threads = threading.active_count()
    if active_threads > 1:
        thread_names = [t.name for t in threading.enumerate()]
        raise SafeModeError(
            layer="precondition",
            reason="precondition",
            message=f"Cannot use safe mode with {active_threads} active threads. "
                    f"Active threads: {thread_names}"
        )


# =============================================================================
# Layer 2: Compile-Time Safe Mode
# =============================================================================

def safe_compile(
    code: str,
    name: str,
    timeout: float = 30.0,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Safely compile C++ code using fork-probe.
    
    Layer 2: Protects main process from JIT compilation crashes.
    
    Args:
        code: C++ code to compile
        name: Name for the compiled artifact
        timeout: Maximum time to wait for compilation (seconds)
        cache_dir: Directory to cache compiled artifacts
        
    Returns:
        True if compilation succeeded
        
    Raises:
        SafeModeError: If compilation crashed or failed
    """
    check_fork_safe()
    
    # Prepare cache directory
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "rdataframe_dsl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Write code to file
    code_path = cache_dir / f"{name}.C"
    code_path.write_text(code)
    
    # Fork and probe
    pid = os.fork()
    
    if pid == 0:
        # === CHILD PROCESS ===
        try:
            import ROOT
            # Attempt compilation
            result = ROOT.gSystem.CompileMacro(str(code_path), "k")
            os._exit(0 if result else 1)
        except Exception as e:
            # Write error to temp file for parent
            error_path = cache_dir / f"{name}.error"
            error_path.write_text(str(e))
            os._exit(2)
    else:
        # === PARENT PROCESS ===
        return _wait_for_child(
            pid=pid,
            timeout=timeout,
            layer="compile",
            operation=f"compiling {name}"
        )


def safe_declare(
    code: str,
    timeout: float = 10.0,
) -> bool:
    """
    Safely declare C++ code using fork-probe.
    
    Layer 2: Protects main process from gInterpreter.Declare() crashes.
    
    Args:
        code: C++ code to declare
        timeout: Maximum time to wait (seconds)
        
    Returns:
        True if declaration succeeded
        
    Raises:
        SafeModeError: If declaration crashed or failed
    """
    check_fork_safe()
    
    pid = os.fork()
    
    if pid == 0:
        # === CHILD PROCESS ===
        try:
            import ROOT
            result = ROOT.gInterpreter.Declare(code)
            os._exit(0 if result else 1)
        except Exception:
            os._exit(2)
    else:
        # === PARENT PROCESS ===
        return _wait_for_child(
            pid=pid,
            timeout=timeout,
            layer="compile",
            operation="declaring C++ code"
        )


# =============================================================================
# Layer 3: Execution-Time Safe Mode (Probe-Run)
# =============================================================================

def probe_run(
    rdf,
    action: str,
    probe_size: int = 1000,
    timeout: float = 30.0,
) -> bool:
    """
    Run a small probe to detect execution crashes.
    
    Layer 3: Protects main process from RDataFrame action crashes.
    
    Args:
        rdf: RDataFrame instance
        action: Action to probe ("Count", "AsNumpy", etc.)
        probe_size: Number of entries to test (default 1000)
        timeout: Maximum time to wait (seconds)
        
    Returns:
        True if probe succeeded
        
    Raises:
        SafeModeError: If probe crashed or failed
    """
    check_fork_safe()
    
    pid = os.fork()
    
    if pid == 0:
        # === CHILD PROCESS ===
        try:
            # Create probe RDF with limited entries
            probe_rdf = rdf.Range(probe_size)
            
            # Execute action
            if action == "Count":
                probe_rdf.Count().GetValue()
            elif action == "AsNumpy":
                # Get first column for probe
                cols = list(probe_rdf.GetColumnNames())
                if cols:
                    probe_rdf.AsNumpy([str(cols[0])])
            elif action == "Sum":
                cols = list(probe_rdf.GetColumnNames())
                if cols:
                    probe_rdf.Sum(str(cols[0])).GetValue()
            else:
                # Generic: try to call as method
                getattr(probe_rdf, action)()
            
            os._exit(0)
        except Exception:
            os._exit(1)
    else:
        # === PARENT PROCESS ===
        return _wait_for_child(
            pid=pid,
            timeout=timeout,
            layer="execute",
            operation=f"probe-run {action}"
        )


def probe_columns(
    rdf,
    columns: List[str],
    probe_size: int = 1000,
    timeout: float = 30.0,
) -> bool:
    """
    Probe specific columns for execution safety.
    
    Layer 3: Tests AsNumpy on specific columns before full execution.
    
    Args:
        rdf: RDataFrame instance
        columns: Columns to probe
        probe_size: Number of entries to test
        timeout: Maximum time to wait (seconds)
        
    Returns:
        True if probe succeeded
        
    Raises:
        SafeModeError: If probe crashed or failed
    """
    check_fork_safe()
    
    pid = os.fork()
    
    if pid == 0:
        # === CHILD PROCESS ===
        try:
            probe_rdf = rdf.Range(probe_size)
            probe_rdf.AsNumpy(columns)
            os._exit(0)
        except Exception:
            os._exit(1)
    else:
        # === PARENT PROCESS ===
        return _wait_for_child(
            pid=pid,
            timeout=timeout,
            layer="execute",
            operation=f"probe columns {columns}"
        )


# =============================================================================
# Internal: Wait for Child Process
# =============================================================================

def _wait_for_child(
    pid: int,
    timeout: float,
    layer: str,
    operation: str,
) -> bool:
    """
    Wait for child process with timeout and crash detection.
    
    Args:
        pid: Child process ID
        timeout: Maximum wait time (seconds)
        layer: Layer name for error reporting
        operation: Operation description for error reporting
        
    Returns:
        True if child succeeded
        
    Raises:
        SafeModeError: If child crashed, timed out, or failed
    """
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            pid_result, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            # Child already reaped
            return True
        
        if pid_result != 0:
            # Child finished
            if os.WIFSIGNALED(status):
                # Child was killed by signal (CRASH)
                signum = os.WTERMSIG(status)
                raise SafeModeError(
                    layer=layer,
                    reason="crash",
                    message=f"Operation crashed during {operation}",
                    signal_num=signum,
                )
            
            if os.WIFEXITED(status):
                exit_code = os.WEXITSTATUS(status)
                if exit_code == 0:
                    return True
                else:
                    raise SafeModeError(
                        layer=layer,
                        reason="error",
                        message=f"Operation failed during {operation}",
                        exit_code=exit_code,
                    )
        
        # Still running, wait a bit
        time.sleep(0.05)
    
    # Timeout - kill child
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)  # Reap zombie
    except (ProcessLookupError, ChildProcessError):
        pass
    
    raise SafeModeError(
        layer=layer,
        reason="timeout",
        message=f"Operation timed out during {operation} (limit: {timeout}s)",
    )


# =============================================================================
# Convenience: Combined Safe Operations
# =============================================================================

def safe_to_numpy(
    rdf,
    columns: List[str],
    probe_size: int = 1000,
    timeout: float = 60.0,
) -> dict:
    """
    Safely execute AsNumpy with probe-run protection.
    
    Combines Layer 3 probe-run with actual execution.
    
    Args:
        rdf: RDataFrame instance
        columns: Columns to export
        probe_size: Probe size for safety check
        timeout: Timeout for probe (full execution has no timeout)
        
    Returns:
        Dict of column arrays (same as rdf.AsNumpy())
        
    Raises:
        SafeModeError: If probe failed
    """
    # First, probe
    probe_columns(rdf, columns, probe_size=probe_size, timeout=timeout)
    
    # Probe passed, execute full operation
    return rdf.AsNumpy(columns)

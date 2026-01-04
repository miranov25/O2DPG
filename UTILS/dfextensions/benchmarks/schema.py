"""
Benchmark Framework v1.0 — Schema Definitions

JSON schema for benchmark results with validation.

Phase 12.10.BF: Standardized benchmark storage format.
Phase 12.11: Added ProfileInfo, BackendInfo for CPU profiling integration.
Phase 12.14b.GB: Added NumpyEncoder for np.bool_ serialization fix.
Phase 12.14b.GB-addendum: Added wall_time_s, profile_path; spec-driven IDs.

Key fields:
- env_id: Environment fingerprint for baseline filtering
- run_mode: "gate" (normal) or "profile" (with tracemalloc)
- peak_rss_mb: Process-wide peak RSS (not per-benchmark isolated)
- profile: Optional CPU profile metadata (Phase 12.11)
- backend: Optional backend selection info (Phase 12.11)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Any, Union, List, Dict
import json
import numpy as np  # Phase 12.14b.GB: For NumpyEncoder
import os
import platform
import subprocess
import sys
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

SCHEMA_VERSION = 1
RUNNER_VERSION = "1.0.0"

DEFAULT_WARMUP_RUNS = 2
DEFAULT_N_RUNS = 3
DEFAULT_TIME_THRESHOLD = 0.10
DEFAULT_MEMORY_THRESHOLD = 0.15  # Higher than time due to RSS variance
DEFAULT_TOP_N = 10


# =============================================================================
# JSON ENCODER (Phase 12.14b.GB)
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy types.
    
    Phase 12.14b.GB: Fixes serialization of np.bool_, np.integer, np.floating
    which are returned by benchmark gate comparisons.
    """
    
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_git_info() -> dict:
    """
    Extract current git state.
    
    Returns dict with: commit, branch, dirty
    Handles missing git gracefully.
    """
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        dirty = subprocess.call(
            ['git', 'diff', '--quiet'],
            stderr=subprocess.DEVNULL
        ) != 0
        
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": True,
        }


def get_tool_versions() -> dict:
    """Get versions of key dependencies."""
    versions = {}
    
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        versions["numpy"] = "not installed"
    
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except ImportError:
        versions["pandas"] = "not installed"
    
    try:
        import pyarrow
        versions["pyarrow"] = pyarrow.__version__
    except ImportError:
        versions["pyarrow"] = "not installed"
    
    try:
        import numba
        versions["numba"] = numba.__version__
    except ImportError:
        versions["numba"] = "not installed"
    
    return versions


def get_cpu_model() -> str:
    """Get CPU model string (best effort)."""
    system = platform.system()
    
    if system == "Darwin":
        try:
            result = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return result
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    elif system == "Linux":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':')[1].strip()
        except (FileNotFoundError, PermissionError):
            pass
    
    return platform.processor() or "unknown"


def build_env_id(
    plat: str,
    cpu_model: str,
    python_version: str,
    numpy_version: str,
    numba_version: str,
) -> str:
    """
    Build environment fingerprint for baseline filtering.
    
    Format: {platform}_{cpu_model}_{python}_{numpy}_{numba}
    Sanitizes special characters for filesystem/query compatibility.
    """
    # Sanitize CPU model (remove spaces, special chars)
    cpu_clean = cpu_model.replace(' ', '_').replace('(', '').replace(')', '')
    cpu_clean = cpu_clean.replace('@', '').replace(',', '')
    # Truncate if too long
    if len(cpu_clean) > 30:
        cpu_clean = cpu_clean[:30]
    
    return f"{plat}_{cpu_clean}_{python_version}_np{numpy_version}_nb{numba_version}"


# =============================================================================
# PHASE 12.11: PROFILE AND BACKEND INFO
# =============================================================================

@dataclass
class ProfileInfo:
    """
    Profile information for a benchmark result.
    
    Phase 12.11: CPU and memory profiling metadata.
    All fields optional for backward compatibility with Phase 12.10.
    
    Note on timing:
        - wall_time_s is the consistent timing metric (always measured)
        - cpu_total_time_s is cProfile's total_tt (only when CPU profiling enabled)
    """
    # CPU profiling status
    cpu_enabled: bool = False
    
    # Timing (wall_time is always available)
    wall_time_s: Optional[float] = None
    cpu_total_time_s: Optional[float] = None
    
    # Profile artifact paths (relative to run directory)
    cpu_prof_path: Optional[str] = None
    cpu_txt_path: Optional[str] = None
    cpu_json_path: Optional[str] = None
    
    # Profile summary
    cpu_total_calls: Optional[int] = None
    cpu_top_functions: Optional[List[Dict]] = None
    cpu_sort_key: str = "cumulative"
    
    # Memory profiling
    tracemalloc_peak_mb: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict, removing None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class BackendInfo:
    """
    Backend selection information.
    
    Phase 12.11: Track which backend was ACTUALLY used (not inferred).
    Critical for verifying Numba bypass fix is active.
    """
    selected_backend: str  # "numba" or "sequential"
    n_jobs: int
    n_chunks: Optional[int] = None
    numba_available: bool = False
    numba_version: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict, removing None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BenchmarkParams:
    """Parameters for a single benchmark run."""
    n_rows: int
    n_groups: int
    n_fits: int = 6
    n_jobs: int = 1
    n_chunks: Optional[int] = None
    parallel_backend: str = "numba"
    
    def __post_init__(self):
        if self.n_chunks is None:
            self.n_chunks = max(1, self.n_jobs)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """
    Result of a single benchmark execution.
    
    Phase 12.11: Added optional profile and backend fields.
    Phase 12.14b.GB-addendum: Added wall_time_s for observability, profile_path for cProfile.
    """
    id: str
    name: str
    scenario: str
    params: dict
    time_s: float              # Kernel-only timing (authoritative for perf comparison)
    time_std_s: float
    n_runs: int
    peak_rss_mb: float
    wall_time_s: Optional[float] = None  # Fallback to time_s for backward compat
    status: str = "OK"
    bench_version: int = 1
    peak_tracemalloc_mb: Optional[float] = None
    throughput_rows_per_sec: Optional[float] = None
    memory_top_allocations: Optional[list] = None
    cpu_top_functions: Optional[list] = None
    error_message: Optional[str] = None
    
    # Phase 12.11: Profile and backend info (optional for backward compat)
    profile: Optional[ProfileInfo] = None
    backend: Optional[BackendInfo] = None
    
    # Phase 12.14b.GB-addendum: cProfile storage path
    profile_path: Optional[str] = None
    # Phase 12.14c.GB D6: Explanation when profile is skipped
    profile_note: Optional[str] = None
    
    def __post_init__(self):
        """Backward compat: fallback wall_time_s to time_s for old data."""
        if self.wall_time_s is None:
            self.wall_time_s = self.time_s
    
    @classmethod
    def from_timing(
        cls,
        name: str,
        scenario: str,
        params: dict,
        times: list[float],
        peak_rss_mb: float,
        wall_time_s: Optional[float] = None,
        uses_n_jobs: bool = True,
        **kwargs,
    ) -> "BenchmarkResult":
        """
        Create BenchmarkResult from timing measurements.
        
        Phase 12.14b.GB-addendum:
        - Added wall_time_s for observability (wrapper time)
        - Added uses_n_jobs for ID hygiene (omit n_jobs suffix when False)
        """
        import numpy as np
        
        n_jobs = params.get("n_jobs", 1)
        
        # Phase 12.14b.GB-addendum: ID hygiene - only include n_jobs if benchmark uses it
        if uses_n_jobs:
            bench_id = f"{name}:{scenario}:n_jobs={n_jobs}"
        else:
            bench_id = f"{name}:{scenario}"
        
        mean_time = float(np.mean(times))
        std_time = float(np.std(times))
        n_rows = params.get("n_rows", 0)
        
        throughput = n_rows / mean_time if mean_time > 0 else 0.0
        
        # Phase 12.14b.GB-addendum: wall_time_s defaults to mean_time if not provided
        if wall_time_s is None:
            wall_time_s = mean_time
        
        return cls(
            id=bench_id,
            name=name,
            scenario=scenario,
            params=params,
            time_s=mean_time,
            wall_time_s=wall_time_s,
            time_std_s=std_time,
            n_runs=len(times),
            peak_rss_mb=peak_rss_mb,
            throughput_rows_per_sec=throughput,
            **kwargs,
        )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Remove None values for cleaner JSON
        result = {}
        for k, v in d.items():
            if v is None:
                continue
            if k == 'profile' and isinstance(v, dict):
                # Remove None values from nested profile dict
                result[k] = {pk: pv for pk, pv in v.items() if pv is not None}
            elif k == 'backend' and isinstance(v, dict):
                # Remove None values from nested backend dict
                result[k] = {bk: bv for bk, bv in v.items() if bv is not None}
            else:
                result[k] = v
        return result


@dataclass
class RunMeta:
    """Metadata for a benchmark run."""
    schema_version: int = SCHEMA_VERSION
    runner_version: str = RUNNER_VERSION
    run_id: str = ""
    run_mode: str = "gate"  # "gate" or "profile"
    suite: str = "quick"    # "quick" or "release"
    timestamp: str = ""
    commit: str = ""
    branch: str = ""
    dirty: bool = False
    env_id: str = ""
    hostname: str = ""
    platform: str = ""
    cpu_model: str = ""
    cpu_count: int = 0
    python_version: str = ""
    tool_versions: dict = field(default_factory=dict)
    subproject: str = ""
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    n_runs: int = DEFAULT_N_RUNS
    
    # Phase 12.11: Profile configuration
    profile_top_n: int = DEFAULT_TOP_N
    
    @classmethod
    def create(
        cls,
        subproject: str,
        run_mode: str = "gate",
        suite: str = "quick",
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
        n_runs: int = DEFAULT_N_RUNS,
        profile_top_n: int = DEFAULT_TOP_N,
    ) -> "RunMeta":
        """Create RunMeta with auto-detected system info."""
        git = get_git_info()
        tool_versions = get_tool_versions()
        
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        plat = platform.system()
        cpu_model = get_cpu_model()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        env_id = build_env_id(
            plat=plat,
            cpu_model=cpu_model,
            python_version=python_version,
            numpy_version=tool_versions.get("numpy", "unknown"),
            numba_version=tool_versions.get("numba", "unknown"),
        )
        
        # run_id: timestamp_commit_pid
        ts_compact = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        run_id = f"{ts_compact}_{git['commit']}_{os.getpid()}"
        
        return cls(
            run_id=run_id,
            run_mode=run_mode,
            suite=suite,
            timestamp=timestamp_str,
            commit=git["commit"],
            branch=git["branch"],
            dirty=git["dirty"],
            env_id=env_id,
            hostname=platform.node(),
            platform=plat,
            cpu_model=cpu_model,
            cpu_count=os.cpu_count() or 1,
            python_version=python_version,
            tool_versions=tool_versions,
            subproject=subproject,
            warmup_runs=warmup_runs,
            n_runs=n_runs,
            profile_top_n=profile_top_n,
        )
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSummary:
    """Summary statistics for a benchmark run."""
    total_time_s: float = 0.0
    peak_rss_mb: float = 0.0
    n_benchmarks: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    n_regressions: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alarm:
    """Regression alarm."""
    benchmark: str
    scenario: str
    param_n_jobs: int
    metric: str
    current: float
    baseline: float
    baseline_type: str
    change_pct: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkRun:
    """Complete benchmark run with all results."""
    meta: RunMeta
    summary: RunSummary
    benchmarks: list[BenchmarkResult]
    alarms: list[Alarm]
    
    def to_dict(self) -> dict:
        return {
            "meta": self.meta.to_dict(),
            "summary": self.summary.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "alarms": [a.to_dict() for a in self.alarms],
        }
    
    def save(self, path: Union[Path, str]):
        """Save to JSON file. Phase 12.14b.GB: Uses NumpyEncoder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, cls=NumpyEncoder))
    
    @classmethod
    def load(cls, path: Union[Path, str]) -> "BenchmarkRun":
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        
        meta = RunMeta(**data["meta"])
        summary = RunSummary(**data["summary"])
        
        benchmarks = []
        for b in data["benchmarks"]:
            # Handle optional profile and backend fields (Phase 12.11)
            profile_data = b.pop("profile", None)
            backend_data = b.pop("backend", None)
            
            profile = ProfileInfo(**profile_data) if profile_data else None
            backend = BackendInfo(**backend_data) if backend_data else None
            
            benchmarks.append(BenchmarkResult(**b, profile=profile, backend=backend))
        
        alarms = []
        for a in data.get("alarms", []):
            alarms.append(Alarm(**a))
        
        return cls(
            meta=meta,
            summary=summary,
            benchmarks=benchmarks,
            alarms=alarms,
        )
    
    def update_summary(self):
        """Recalculate summary from benchmarks."""
        self.summary.n_benchmarks = len(self.benchmarks)
        self.summary.n_passed = sum(1 for b in self.benchmarks if b.status == "OK")
        self.summary.n_failed = sum(1 for b in self.benchmarks if b.status == "FAILED")
        self.summary.n_skipped = sum(1 for b in self.benchmarks if b.status == "SKIPPED")
        self.summary.n_regressions = len(self.alarms)
        
        if self.benchmarks:
            self.summary.peak_rss_mb = max(b.peak_rss_mb for b in self.benchmarks)


# =============================================================================
# STORAGE PATH UTILITIES
# =============================================================================

def get_benchmark_prefix() -> Path:
    """Get base benchmark storage directory."""
    if "BENCHMARK_PREFIX" in os.environ:
        return Path(os.environ["BENCHMARK_PREFIX"])
    return Path("./benchmarks")


def get_run_output_dir(subproject: str, timestamp: str) -> Path:
    """
    Get output directory for a benchmark run.
    
    Format: $BENCHMARK_PREFIX/<timestamp>/<subproject>/
    """
    # Parse timestamp to folder name (YYYY-MM-DDTHH-MM-SS)
    # Input: "2024-12-20T14:30:00.123Z"
    # Output folder: "2024-12-20T14-30-00"
    ts_folder = timestamp[:19].replace(":", "-")
    
    base = get_benchmark_prefix()
    return base / ts_folder / subproject


def get_results_path(subproject: str, timestamp: str) -> Path:
    """Get path for results.json file."""
    return get_run_output_dir(subproject, timestamp) / "results.json"


def get_alarms_path(subproject: str, timestamp: str) -> Path:
    """Get path for alarms.json file."""
    return get_run_output_dir(subproject, timestamp) / "alarms.json"


def get_profiles_dir(subproject: str, timestamp: str) -> Path:
    """Get path for profiles directory (Phase 12.11)."""
    return get_run_output_dir(subproject, timestamp) / "profiles"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_run(run: BenchmarkRun) -> list[str]:
    """
    Validate a BenchmarkRun.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Check schema version
    if run.meta.schema_version != SCHEMA_VERSION:
        errors.append(f"Unsupported schema version: {run.meta.schema_version}")
    
    # Check required fields
    if not run.meta.run_id:
        errors.append("Missing run_id")
    if not run.meta.timestamp:
        errors.append("Missing timestamp")
    if not run.meta.subproject:
        errors.append("Missing subproject")
    if not run.meta.env_id:
        errors.append("Missing env_id")
    
    # Check run_mode
    if run.meta.run_mode not in ("gate", "profile"):
        errors.append(f"Invalid run_mode: {run.meta.run_mode}")
    
    # Check suite
    if run.meta.suite not in ("quick", "release"):
        errors.append(f"Invalid suite: {run.meta.suite}")
    
    # Check benchmarks
    for b in run.benchmarks:
        if not b.id:
            errors.append(f"Benchmark missing id: {b.name}")
        if b.time_s < 0:
            errors.append(f"Negative time for {b.id}: {b.time_s}")
        if b.peak_rss_mb < 0:
            errors.append(f"Negative RSS for {b.id}: {b.peak_rss_mb}")
        if b.status not in ("OK", "FAILED", "SKIPPED"):
            errors.append(f"Invalid status for {b.id}: {b.status}")
    
    return errors


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "RUNNER_VERSION",
    "DEFAULT_WARMUP_RUNS",
    "DEFAULT_N_RUNS",
    "DEFAULT_TIME_THRESHOLD",
    "DEFAULT_MEMORY_THRESHOLD",
    "DEFAULT_TOP_N",
    # JSON Encoder (Phase 12.14b.GB)
    "NumpyEncoder",
    # Helpers
    "get_git_info",
    "get_tool_versions",
    "get_cpu_model",
    "build_env_id",
    # Phase 12.11: Profile and Backend
    "ProfileInfo",
    "BackendInfo",
    # Dataclasses
    "BenchmarkParams",
    "BenchmarkResult",
    "RunMeta",
    "RunSummary",
    "Alarm",
    "BenchmarkRun",
    # Storage
    "get_benchmark_prefix",
    "get_run_output_dir",
    "get_results_path",
    "get_alarms_path",
    "get_profiles_dir",
    # Validation
    "validate_run",
]

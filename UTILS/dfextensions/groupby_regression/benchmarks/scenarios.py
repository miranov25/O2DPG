"""
Benchmark Scenarios for groupby_regression

Phase 12.10.BF: Standardized scenario definitions.

Scenarios use exponential scaling (×2) for rows and groups:

| Scenario | n_rows | n_groups | n_fits | Target Time (n_jobs=1) |
|----------|--------|----------|--------|------------------------|
| S1       | 50K    | 500      | 6      | ~0.5s                  |
| S2       | 100K   | 1K       | 6      | ~1s                    |
| S3       | 200K   | 2K       | 6      | ~2s                    |
| S4       | 400K   | 4K       | 6      | ~4s                    |
| S5       | 25M    | 192K     | 6      | ~60s (release only)    |

S5 matches production calibration workload (ALICE TPC).

n_fits is fixed at 6 to match production use case.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class Scenario:
    """Benchmark scenario definition."""
    name: str
    n_rows: int
    n_groups: int
    n_fits: int = 6
    description: str = ""
    
    @property
    def rows_per_group(self) -> int:
        return self.n_rows // self.n_groups


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {
    "S1": Scenario(
        name="S1",
        n_rows=50_000,
        n_groups=500,
        n_fits=6,
        description="Tiny: Quick smoke test",
    ),
    "S2": Scenario(
        name="S2",
        n_rows=100_000,
        n_groups=1_000,
        n_fits=6,
        description="Small: Unit test level",
    ),
    "S3": Scenario(
        name="S3",
        n_rows=200_000,
        n_groups=2_000,
        n_fits=6,
        description="Medium: Integration test",
    ),
    "S4": Scenario(
        name="S4",
        n_rows=400_000,
        n_groups=4_000,
        n_fits=6,
        description="Large: Performance test",
    ),
    "S5": Scenario(
        name="S5",
        n_rows=25_000_000,
        n_groups=192_000,
        n_fits=6,
        description="Calibration: Production-like (ALICE TPC)",
    ),
}

# Suite definitions
QUICK_SCENARIOS = ["S1", "S2", "S3", "S4"]
RELEASE_SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]


def get_scenarios(suite: str = "quick") -> list[Scenario]:
    """
    Get scenarios for a suite.
    
    Parameters:
        suite: "quick" or "release"
    
    Returns:
        List of Scenario objects
    """
    if suite == "release":
        names = RELEASE_SCENARIOS
    else:
        names = QUICK_SCENARIOS
    
    return [SCENARIOS[name] for name in names]


def get_scenario(name: str) -> Scenario:
    """Get a single scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def create_test_data(
    scenario: Scenario,
    seed: int = 42,
    n_linear: int = 2,
    add_nans: bool = False,
    nan_fraction: float = 0.01,
) -> pd.DataFrame:
    """
    Create synthetic test data for a scenario.
    
    Parameters:
        scenario: Scenario definition
        seed: Random seed for reproducibility
        n_linear: Number of linear predictor columns
        add_nans: Add NaN values for robustness testing
        nan_fraction: Fraction of NaN values if add_nans=True
    
    Returns:
        DataFrame with columns:
            - group: Group identifier (0 to n_groups-1)
            - x1, x2, ...: Linear predictors
            - y1, y2, ...: Target variables (one per fit)
            - w: Weights (positive)
    """
    np.random.seed(seed)
    
    n = scenario.n_rows
    g = scenario.n_groups
    n_fits = scenario.n_fits
    
    # Group column (even distribution)
    rows_per_group = n // g
    group = np.repeat(np.arange(g), rows_per_group)
    
    # Handle remainder
    if len(group) < n:
        remainder = n - len(group)
        group = np.concatenate([group, np.zeros(remainder, dtype=int)])
    
    # Create DataFrame
    df = pd.DataFrame({"group": group})
    
    # Linear predictors
    for i in range(n_linear):
        col_name = f"x{i+1}"
        df[col_name] = np.random.randn(n)
    
    # Target variables (correlated with predictors + noise)
    for i in range(n_fits):
        col_name = f"y{i+1}"
        # y = sum(x_j) + noise
        signal = sum(df[f"x{j+1}"] for j in range(n_linear))
        noise = np.random.randn(n) * 0.5
        df[col_name] = signal + noise
    
    # Weights (positive)
    df["w"] = np.abs(np.random.randn(n)) + 0.1
    
    # Add NaNs if requested
    if add_nans:
        n_nans = int(n * nan_fraction)
        for col in [f"y{i+1}" for i in range(n_fits)]:
            nan_indices = np.random.choice(n, n_nans, replace=False)
            df.loc[nan_indices, col] = np.nan
    
    return df


def create_minimal_warmup_data(n_groups: int = 10, rows_per_group: int = 20) -> pd.DataFrame:
    """
    Create minimal data for JIT warmup.
    
    Uses tiny dataset to trigger Numba compilation quickly.
    """
    np.random.seed(0)
    n = n_groups * rows_per_group
    
    return pd.DataFrame({
        "group": np.repeat(np.arange(n_groups), rows_per_group),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "y1": np.random.randn(n),
        "w": np.abs(np.random.randn(n)) + 0.1,
    })


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Scenario",
    "SCENARIOS",
    "QUICK_SCENARIOS",
    "RELEASE_SCENARIOS",
    "get_scenarios",
    "get_scenario",
    "create_test_data",
    "create_minimal_warmup_data",
]

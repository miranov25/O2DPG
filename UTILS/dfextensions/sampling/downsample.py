"""
dfextensions/sampling/downsample.py

Stratified downsampling utilities for DataFrames.

Ported from:
  - distortionMLFit.py  (downsampleDF, lines 116-147)
  - pidSkimmedFit.py    (downsampleDFTrigger, lines 224-261)

Phase 13.10.DF v3.0 — Downsampling Utility and Weight Integration
"""

from typing import List, Union, Dict
import numpy as np
import pandas as pd


def downsampleDF(
    df: pd.DataFrame,
    frac: float,
    stratify: Union[str, List[str]],
    random_state: int,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
) -> pd.DataFrame:
    """
    Downsample a DataFrame with inverse-group-size weighting.

    Groups rows by ``stratify`` columns, assigns sampling weights inversely
    proportional to group size, normalizes to a probability distribution,
    and samples ``frac * len(df)`` rows without replacement.

    This is the generic algorithm from distortionMLFit.py (lines 116-147),
    extended with ``random_state``, type hints, and input protection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    frac : float
        Fraction of rows to retain (0 < frac <= 1).
    stratify : str or list of str
        Column(s) defining groups for stratified sampling.
    random_state : int
        Random seed for reproducibility. Required.
    keep_weights : bool, default True
        If True, include the sampling-weight column in the output.
    weight_dtype : np.dtype, default np.float32
        Data type for the weight column.
    weight_column : str, default 'weight'
        Name of the weight column added to the output.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with ``n = int(len(df) * frac)`` rows.
        If *keep_weights* is True, contains a ``weight_column`` with the
        normalised sampling weights (sum = 1).

    Raises
    ------
    ValueError
        If ``frac`` is not in (0, 1], if ``stratify`` columns are missing,
        or if ``weight_column`` already exists in *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'g': ['A']*900 + ['B']*100, 'x': np.random.randn(1000)})
    >>> out = downsampleDF(df, frac=0.1, stratify='g', random_state=42)
    >>> len(out)
    100
    """
    # --- input validation ---------------------------------------------------
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    if isinstance(stratify, str):
        stratify = [stratify]

    missing = [c for c in stratify if c not in df.columns]
    if missing:
        raise ValueError(f"stratify columns not in DataFrame: {missing}")

    if weight_column in df.columns:
        raise ValueError(
            f"weight_column '{weight_column}' already exists in DataFrame"
        )

    # --- algorithm (unchanged from distortionMLFit.py) ----------------------
    # Step 1: group sizes
    group_sizes = df.groupby(stratify).size()

    # Step 2: inverse-size weights, normalised to sum=1
    weights = 1 / group_sizes
    weights /= weights.sum()

    # Step 3: merge weights onto every row
    temp_df = df.merge(
        weights.reset_index(name=weight_column), on=stratify, how="left"
    )
    temp_df[weight_column] = temp_df[weight_column].astype(weight_dtype)

    # Step 4: sample
    n_samples = int(len(df) * frac)
    downsampled_df = temp_df.sample(
        n=n_samples, weights=weight_column, replace=False,
        random_state=random_state,
    )

    if not keep_weights:
        downsampled_df = downsampled_df.drop(columns=[weight_column])

    return downsampled_df


def downsampleDFTrigger(
    df: pd.DataFrame,
    triggers: List[Dict],
    random_state: int,
    weight_dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """
    Multi-trigger bitmask downsampling.

    Each trigger independently samples rows according to its own stratification
    and fraction.  A combined bitmask records which trigger(s) selected each
    row.  Only rows selected by at least one trigger are returned.

    Ported from pidSkimmedFit.py (lines 224-261), with the parallel-list API
    replaced by a list-of-dicts API.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    triggers : list of dict
        Each dict must contain:

        - ``'name'``     : str — trigger label (used for weight column names)
        - ``'stratify'`` : str or list of str — grouping columns
        - ``'frac'``     : float — fraction to sample (0 < frac <= 1)
    random_state : int
        Base random seed.  Each trigger uses ``random_state + i``.
    weight_dtype : np.dtype, default np.float32
        Data type for per-trigger weight columns.

    Returns
    -------
    pd.DataFrame
        Rows selected by at least one trigger.  Contains:

        - ``weight_{name}`` columns (one per trigger)
        - ``combined_trigger`` column (uint16 bitmask)

    Raises
    ------
    ValueError
        If any trigger dict is missing required keys, if more than 16
        triggers are requested, or if ``frac`` is invalid.

    Examples
    --------
    >>> triggers = [
    ...     {'name': 'flat_pt', 'stratify': ['pt_bin'], 'frac': 0.1},
    ...     {'name': 'flat_eta', 'stratify': ['eta_bin'], 'frac': 0.1},
    ... ]
    >>> out = downsampleDFTrigger(df, triggers, random_state=42)
    """
    # --- input validation ---------------------------------------------------
    if len(triggers) > 16:
        raise ValueError(
            f"Maximum 16 triggers supported (uint16 bitmask), got {len(triggers)}"
        )
    required_keys = {"name", "stratify", "frac"}
    for i, t in enumerate(triggers):
        missing_keys = required_keys - set(t.keys())
        if missing_keys:
            raise ValueError(
                f"Trigger {i} missing required keys: {missing_keys}"
            )
        if not (0 < t["frac"] <= 1):
            raise ValueError(
                f"Trigger '{t['name']}': frac must be in (0, 1], got {t['frac']}"
            )

    # --- algorithm (unchanged from pidSkimmedFit.py) ------------------------
    result = df.copy()
    combined_trigger = np.zeros(len(result), dtype=int)
    trigger_bitmask = 1  # start with 0x1

    for i, trigger in enumerate(triggers):
        name = trigger["name"]
        stratify = trigger["stratify"]
        frac = trigger["frac"]

        if isinstance(stratify, str):
            stratify = [stratify]

        missing = [c for c in stratify if c not in result.columns]
        if missing:
            raise ValueError(
                f"Trigger '{name}': stratify columns not in DataFrame: {missing}"
            )

        # Calculate group sizes and weights
        group_sizes = result.groupby(stratify).size()
        weights = 1 / group_sizes
        weights /= weights.sum()
        weights = weights.astype(weight_dtype)

        # Merge weights into the DataFrame
        weight_column = f"weight_{name}"
        result = result.merge(
            weights.reset_index(name=weight_column), on=stratify, how="left"
        )

        # Sample indices based on weights and fraction
        n_samples = int(len(result) * frac)
        sampled_indices = result.sample(
            n=n_samples, weights=weight_column, replace=False,
            random_state=random_state + i,
        ).index

        # Update the combined trigger bitmask
        combined_trigger[sampled_indices] |= trigger_bitmask

        # Shift the bitmask for the next trigger
        trigger_bitmask <<= 1

    # Add combined trigger column
    result["combined_trigger"] = combined_trigger.astype(np.uint16)

    # Filter: keep only rows selected by at least one trigger
    return result[result["combined_trigger"] > 0]

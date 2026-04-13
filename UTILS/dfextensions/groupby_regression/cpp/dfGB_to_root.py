#!/usr/bin/env python3
"""Phase 13.18.GB Turn 2 — Python helper to dump a trained dfGB DataFrame
into a ROOT .root file consumable by the C++ evaluator (Layer B).

The output .root file contains:
  - A TTree named by tree_name (default "dfGB") with one branch per
    dfGB column (index columns + coefficient columns + error columns).
    One TTree entry per dfGB row.
  - A TObjString named <tree_name>__gbreg_schema holding a small JSON
    blob with the schema fields the Layer B Option A load path reads
    (group_columns, predictor_columns, targets, suffix, fit_intercept).

The sidecar naming convention <tree_name>__gbreg_schema is the Phase
13.18.GB demo-level default per proposal v1.1 §13 A-2. ADF can adopt or
propose an alternative in Phase 13.19.

Dependencies
------------
uproot (pip install uproot)

This module does NOT require ROOT/PyROOT. uproot writes the .root file
in a format ROOT reads natively.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uproot


SCHEMA_SIDECAR_SUFFIX = "__gbreg_schema"


def dfGB_to_root(
    dfGB: pd.DataFrame,
    output_path: str | Path,
    tree_name: str = "dfGB",
    *,
    group_columns: list[str],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    overwrite: bool = True,
) -> Path:
    """Dump a trained dfGB to a .root file + schema sidecar.

    Parameters
    ----------
    dfGB : pd.DataFrame
        The trained model DataFrame. One row per populated bin.
        Must contain all group_columns and every expected coefficient
        column derived from (targets, predictor_columns, suffix,
        fit_intercept).
    output_path : str | Path
        Destination .root file.
    tree_name : str
        Name of the TTree inside the file. Default "dfGB".
    group_columns : list[str]
        Group (index) column names; must be present in dfGB.
    predictor_columns : list[str]
        Predictor column names driving slope coefficient column naming.
    targets : list[str]
        Target column names driving the <target>_intercept<suffix> and
        <target>_slope_<pred><suffix> column naming.
    suffix : str
        Coefficient column suffix (e.g. "_fit", "_sw").
    fit_intercept : bool
        Whether the model has an intercept column per target.
    overwrite : bool
        If True and output_path exists, it is overwritten.

    Returns
    -------
    Path
        The output path.

    Raises
    ------
    ValueError
        If dfGB is missing any required column, or if any group or
        coefficient column has incompatible dtype for TTree storage.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate required columns are present
    required = list(group_columns)
    for t in targets:
        if fit_intercept:
            required.append(f"{t}_intercept{suffix}")
        for pred in predictor_columns:
            required.append(f"{t}_slope_{pred}{suffix}")
    missing = [c for c in required if c not in dfGB.columns]
    if missing:
        raise ValueError(
            f"dfGB missing required columns: {missing}. "
            f"Available columns: {list(dfGB.columns)}")

    # Build a branches dict for uproot: column name -> numpy array.
    # Keep group_columns as int64 and coefficient columns as float64.
    # Unknown-suffix columns (e.g. "_err_sw", "_rmse_sw", "_n_fitted_sw")
    # are PRESERVED per P1-δ: load-time the C++ side tolerates them
    # without schema error.
    branches: dict[str, np.ndarray] = {}
    for col in dfGB.columns:
        arr = dfGB[col].to_numpy()
        if col in group_columns:
            # Enforce integer dtype for group columns
            if not np.issubdtype(arr.dtype, np.integer):
                try:
                    arr = arr.astype(np.int64)
                except Exception as e:
                    raise ValueError(
                        f"group column {col!r} not integer-castable: {e}")
            else:
                arr = arr.astype(np.int64)
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64)
        else:
            # Non-numeric columns not supported (strings, objects etc.)
            raise ValueError(
                f"column {col!r} has unsupported dtype {arr.dtype} "
                f"for TTree storage")
        branches[col] = arr

    # Schema sidecar JSON
    schema = {
        "group_columns": list(group_columns),
        "predictor_columns": list(predictor_columns),
        "targets": list(targets),
        "suffix": suffix,
        "fit_intercept": bool(fit_intercept),
    }
    schema_json = json.dumps(schema, sort_keys=True)
    sidecar_name = f"{tree_name}{SCHEMA_SIDECAR_SUFFIX}"

    # Write the file
    with uproot.recreate(output_path) as f:
        f[tree_name] = branches
        # uproot writes strings as TObjString compatible records
        f[sidecar_name] = schema_json

    return output_path


def root_to_dfGB(
    input_path: str | Path,
    tree_name: str = "dfGB",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inverse helper — read a .root file produced by dfGB_to_root.

    Returns
    -------
    (df, schema) : tuple
        df : pd.DataFrame restored from the TTree
        schema : dict parsed from the sidecar TObjString

    Raises
    ------
    KeyError if the tree or sidecar is not present.
    """
    input_path = Path(input_path)
    sidecar_name = f"{tree_name}{SCHEMA_SIDECAR_SUFFIX}"
    with uproot.open(input_path) as f:
        if tree_name not in f:
            raise KeyError(f"tree {tree_name!r} not in {input_path}")
        tree = f[tree_name]
        # uproot-5 returns a structured ndarray or a dict depending on
        # version; handle both paths robustly.
        arrs = tree.arrays(library="np")
        if isinstance(arrs, dict):
            df = pd.DataFrame({k: arrs[k] for k in arrs})
        elif hasattr(arrs, "dtype") and arrs.dtype.names:
            # structured ndarray
            df = pd.DataFrame({name: arrs[name] for name in arrs.dtype.names})
        else:
            raise RuntimeError(
                f"unexpected uproot arrays return type: {type(arrs)}")

        # Schema sidecar: uproot returns the stored string
        if sidecar_name not in f:
            raise KeyError(f"schema sidecar {sidecar_name!r} not in "
                           f"{input_path}")
        schema_raw = f[sidecar_name]
        # uproot returns stored strings as Python str directly when read
        if isinstance(schema_raw, str):
            schema = json.loads(schema_raw)
        else:
            # Some uproot versions wrap; try .tostring() / .fString
            schema_str = getattr(schema_raw, "fString", None) or str(schema_raw)
            schema = json.loads(schema_str)
    return df, schema

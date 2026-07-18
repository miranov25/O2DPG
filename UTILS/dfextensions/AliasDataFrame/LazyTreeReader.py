"""
LazyTreeReader - On-demand branch loading from ROOT files.

Phase 7.1: Explicit branches only
Phase 7.2+: Auto-detection integration
Phase 7.4+: Chain mode
"""

import uproot
import pandas as pd
import numpy as np
from typing import List, Set, Optional


class LazyTreeReader:
    """
    Manages lazy loading of branches from ROOT TTree.
    
    Reads TTree metadata immediately, but loads branch data
    only when requested via ensure_branches().
    
    Attributes:
        file_path: Path to ROOT file
        tree_name: Name of TTree
        available_branches: All branches in the TTree (from metadata)
        loaded_branches: Branches currently loaded into DataFrame
        num_entries: Number of entries in the TTree
    
    Example:
        reader = LazyTreeReader('data.root', 'tree')
        print(reader.available_branches)  # All branches
        df = reader.ensure_branches(['x', 'y'], pd.DataFrame())
        print(reader.loaded_branches)  # {'x', 'y'}
    """
    
    def __init__(self, file_path: str, tree_name: str):
        """
        Initialize reader and load TTree metadata.
        
        Args:
            file_path: Path to ROOT file
            tree_name: Name of TTree in file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If tree not found in file
        """
        self.file_path = file_path
        self.tree_name = tree_name
        self._file = None
        self._tree = None
        self.loaded_branches: Set[str] = set()
        
        # Load metadata immediately
        self._open_file()
        self.available_branches: Set[str] = set(self._tree.keys())
        self.num_entries: int = self._tree.num_entries

        # Phase 13.59.ADF (BUG_20260613): recover ADF metadata (subframes, indices,
        # aliases) on construction, following AD-3 read precedence
        # (ROOT UserInfo -> uproot UserInfo -> standalone key -> names). Self-contained
        # here (Option a); read_tree_lazy consumes self.adf_metadata to register lazy
        # subframes. Additive: does not touch the branch-data path (_open_file/load_branches).
        self.adf_metadata: Optional[dict] = None
        try:
            from adf_metadata_compat import read_adf_metadata
            self.adf_metadata = read_adf_metadata(self.file_path, self.tree_name)
        except Exception as e:
            import warnings
            warnings.warn(
                f"LazyTreeReader: could not recover ADF metadata for "
                f"'{self.tree_name}' in {self.file_path}: {e}"
            )
    
    @property
    def entries(self) -> int:
        """Total entries in tree (alias for num_entries)."""
        return self.num_entries
    
    def _open_file(self):
        """Open file and tree if not already open."""
        if self._file is None:
            self._file = uproot.open(self.file_path)
            self._tree = self._file[self.tree_name]
    
    def load_branches(self, names: List[str], existing_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load branches from ROOT file.
        
        Returns NEW data only (doesn't merge with existing_df parameter,
        which is kept for API compatibility but ignored).
        
        Parameters
        ----------
        names : List[str]
            Branch names to load
        existing_df : pd.DataFrame, optional
            Ignored. Kept for API compatibility with chain reader.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with requested branches (new data only)
            
        Raises
        ------
        ValueError
            If any branch doesn't exist in TTree
        """
        if not names:
            return pd.DataFrame()
        
        # Validate branches exist
        names_set = set(names)
        missing = names_set - self.available_branches
        if missing:
            raise ValueError(
                f"Branches not found in TTree: {sorted(missing)}. "
                f"Available: {sorted(self.available_branches)}"
            )
        
        # Find branches that need loading
        to_load = names_set - self.loaded_branches
        if not to_load:
            return pd.DataFrame()  # Already loaded, nothing new
        
        # Load from ROOT file
        self._open_file()
        new_data = self._tree.arrays(
            filter_name=list(to_load),
            library='pd'
        )

        # Some uproot/awkward version combinations return an awkward Array even for flat
        # branches (instead of a DataFrame). Coerce to a DataFrame so downstream .copy()
        # and merge work. No-op when uproot already returns a DataFrame.
        if not isinstance(new_data, pd.DataFrame):
            import awkward as ak
            new_data = ak.to_dataframe(new_data)

        # Copy to avoid modifying uproot's internal buffer
        new_data = new_data.copy()
        
        # Track loaded branches
        self.loaded_branches.update(to_load)
        
        return new_data

    def release_branches(self, names):
        """PHASE_13_68_ADF: forget the given physical branches so a later
        access re-reads them from file. Mutates ``loaded_branches`` in place;
        names that are not loaded are ignored (idempotent)."""
        if isinstance(names, str):
            names = [names]
        self.loaded_branches.difference_update(names)

    def is_scalar_branch(self, branch_name):
        """PHASE_13_75_ADF: shape classification for struct auto-detection.

        Returns True  -> one-value-per-entry scalar (AsDtype, non-object)
                False -> jagged/var-length, strings, objects, containers
                None  -> UNKNOWN (branch absent, or interpretation unavailable)
        The caller (detect_structs) must treat None/False as NOT auto-registrable
        (C1: unknown is never scalar).
        """
        try:
            tree = self._tree
        except AttributeError:
            return None
        try:
            if branch_name not in self.available_branches:
                return None
            interp = tree[branch_name].interpretation
        except Exception:
            return None
        cls = type(interp).__name__
        if cls == "AsDtype":
            return True          # one plain value per entry
        return False             # AsJagged / AsStrings / AsObjects / AsGroup / ...

    def ensure_branches(self, names: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure specified branches are loaded into DataFrame.
        
        Legacy method that loads and merges. For new code, prefer
        load_branches() + external merge.
        
        Args:
            names: Branch names to load
            df: Current DataFrame to extend
            
        Returns:
            DataFrame with requested branches added
            
        Raises:
            ValueError: If any branch doesn't exist in TTree
        """
        if not names:
            return df
        
        # Load new data
        new_data = self.load_branches(names)
        
        if new_data is None or len(new_data) == 0:
            return df  # Nothing new loaded
        
        # Merge with existing DataFrame
        if df is None or len(df) == 0:
            return new_data
        else:
            # Add columns to existing DataFrame
            for col in new_data.columns:
                df[col] = new_data[col].values
            return df
    
    def is_loaded(self, name: str) -> bool:
        """Check if a branch is currently loaded."""
        return name in self.loaded_branches
    
    def get_branch_dtype(self, name: str) -> str:
        """
        Get dtype of a branch from TTree metadata.
        
        Args:
            name: Branch name
            
        Returns:
            String representation of dtype
            
        Raises:
            ValueError: If branch not in TTree
        """
        if name not in self.available_branches:
            raise ValueError(f"Branch '{name}' not in TTree")
        self._open_file()
        return str(self._tree[name].interpretation)

    def estimate_memory(self, branches: List[str] = None) -> dict:
        """
        Estimate memory for loading branches (single-tree lazy).

        Phase 13.58.ADF (D3): mirrors LazyChainReader.estimate_memory but uses the real
        per-branch dtype item size from the TTree interpretation (not a float32 constant),
        so the estimate matches the eager `sum(df[col].nbytes)` exactly (AC-5, tolerance 0)
        for the same branches on deterministic data. Removes the single-tree-lazy
        AttributeError (ADF.estimate_memory previously delegated to a method that only
        existed on the chain reader).

        Parameters
        ----------
        branches : List[str], optional
            Branches to estimate. None = all available.

        Returns
        -------
        dict
            'bytes': int, 'human': str, 'branches': int, 'entries': int,
            'warning': str or None
        """
        self._open_file()
        if branches is None:
            branches = list(self.available_branches)
        total = 0
        for b in branches:
            if b not in self.available_branches:
                continue
            total += self._branch_itemsize(b) * self.num_entries

        warning = None
        if total > 32 * 1024**3:
            warning = "Estimated memory exceeds 32 GB - consider chunked loading"
        elif total > 16 * 1024**3:
            warning = "Estimated memory exceeds 16 GB"

        return {
            'bytes': total,
            'human': self._format_bytes(total),
            'branches': len(branches),
            'entries': self.num_entries,
            'warning': warning,
        }

    def _branch_itemsize(self, branch: str) -> int:
        """Bytes-per-entry for a branch, from the real TTree interpretation dtype.

        Falls back to 4 bytes only if the dtype cannot be determined (e.g. an unusual
        interpretation); flat numeric branches resolve exactly.
        """
        try:
            dt = np.dtype(self._tree[branch].interpretation.numpy_dtype)
            return dt.base.itemsize if dt.subdtype is not None else dt.itemsize
        except Exception:
            return 4

    @staticmethod
    def _format_bytes(n: int) -> str:
        """Format bytes as a human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    def close(self):
        """Close file handle."""
        if self._file is not None:
            # uproot files don't have explicit close, but we can release references
            self._file = None
            self._tree = None
    
    def __del__(self):
        """Cleanup file handle on deletion."""
        self.close()
    
    def __repr__(self):
        return (
            f"LazyTreeReader('{self.file_path}', '{self.tree_name}', "
            f"available={len(self.available_branches)}, "
            f"loaded={len(self.loaded_branches)})"
        )

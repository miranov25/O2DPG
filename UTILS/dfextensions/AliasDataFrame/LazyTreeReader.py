"""
LazyTreeReader - On-demand branch loading from ROOT files.

Phase 7.1: Explicit branches only
Phase 7.2+: Auto-detection integration
Phase 7.4+: Chain mode
"""

import uproot
import pandas as pd
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

"""
Lazy chain reader for multiple ROOT files.

Phase 7.4: Implements composition pattern per architecture review §A1.

Key Design Decisions (from reviewer feedback):
- Reader returns new data; ADF handles merge (Arrow-compatible pattern)
- Single pd.concat with ignore_index=True (no incremental concat)
- LRU cache for file handles (K=8 default)
- __file_idx__ is added to data but NOT reported in available_branches
"""

import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
import glob

import numpy as np
import pandas as pd

from LazyTreeReader import LazyTreeReader
from exceptions import BranchNotFoundError, ChainValidationError


class LazyChainReader:
    """
    Manages lazy loading across multiple ROOT files.
    
    Uses composition pattern: wraps List[LazyTreeReader] with LRU caching.
    
    Parameters
    ----------
    files : List[dict]
        List of file specifications: [{'path': str, 'tree': str}, ...]
    validation : str, default 'first'
        Branch validation mode:
        - 'first': Use first file as reference, warn on differences.
          Missing branches in later files are filled with NaN.
          Extra branches in later files are ignored.
        - 'strict': Error if any file differs from first file
        - 'intersection': Only branches present in ALL files
        - 'union': All branches from any file; missing filled with NaN
    max_open_files : int, default 8
        Maximum number of file handles to keep open (LRU cache)
    add_file_index : bool, default False
        Add '__file_idx__' column tracking source file.
        Note: This column is NOT included in available_branches.
        
    Attributes
    ----------
    available_branches : Set[str]
        Branches available for loading (based on validation mode).
        Does NOT include '__file_idx__' even when add_file_index=True.
    loaded_branches : Set[str]
        Branches currently loaded (may include '__file_idx__')
    entries : int
        Total entries across all files
    file_count : int
        Number of files in chain
    entry_offsets : List[int]
        Cumulative entry offsets for global→local mapping
    """
    
    DEFAULT_MAX_OPEN_FILES = 8
    
    def __init__(self, 
                 files: List[dict],
                 validation: str = 'first',
                 max_open_files: int = None,
                 add_file_index: bool = False):
        
        if not files:
            raise ValueError("Cannot create chain with empty file list")
        
        self._files = files
        self._validation = validation
        self._max_open = max_open_files or self.DEFAULT_MAX_OPEN_FILES
        self._add_file_index = add_file_index
        
        # LRU cache for readers (OrderedDict for LRU behavior)
        self._readers: OrderedDict[int, LazyTreeReader] = OrderedDict()
        
        # State tracking
        self._entry_offsets: List[int] = []
        self._total_entries: int = 0
        self._available_branches: Set[str] = set()
        self._loaded_branches: Set[str] = set()
        self._reference_branches: Set[str] = set()  # From first file
        
        # Per-file branch sets (needed for union/first modes)
        self._file_branches: List[Set[str]] = []
        # PHASE_13_67_ADF: per-file ADF metadata (UserInfo), stable, captured here so
        # read_chain_lazy needs no LRU-cached readers and no duplicate file opens (D5).
        self._file_metadata: List = []
        
        # Initialize chain (validate, compute offsets)
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Validate files and compute entry offsets."""
        cumulative = 0
        all_branch_sets = []
        
        for idx, file_spec in enumerate(self._files):
            # Get metadata from each file (uses LRU cache)
            reader = self._get_reader(idx)
            branches = reader.available_branches
            entries = reader.entries
            # PHASE_13_67_ADF (#1/D5): capture this file's ADF metadata now.
            self._file_metadata.append(getattr(reader, 'adf_metadata', None))
            
            # Store entry count in file spec
            file_spec['entries'] = entries
            
            # Track offsets
            self._entry_offsets.append(cumulative)
            cumulative += entries
            
            # Collect branch sets for validation
            all_branch_sets.append(branches)
            self._file_branches.append(branches.copy())
            
            if idx == 0:
                self._reference_branches = branches.copy()
        
        self._total_entries = cumulative
        
        # Validate and determine available branches
        self._validate_branches(all_branch_sets)
    
    def _validate_branches(self, all_branch_sets: List[Set[str]]):
        """Validate branch consistency across files."""
        reference = self._reference_branches
        
        if self._validation == 'strict':
            for idx, branches in enumerate(all_branch_sets):
                if branches != reference:
                    raise ChainValidationError(
                        idx, self._files[idx]['path'], reference, branches
                    )
            self._available_branches = reference.copy()
        
        elif self._validation == 'first':
            for idx, branches in enumerate(all_branch_sets[1:], start=1):
                if branches != reference:
                    missing = reference - branches
                    extra = branches - reference
                    warnings.warn(
                        f"Branch mismatch in file {idx} ({self._files[idx]['path']}). "
                        f"Missing: {sorted(missing)}, Extra: {sorted(extra)}. "
                        f"Using first file as reference. "
                        f"Missing branches will be filled with NaN.",
                        UserWarning
                    )
            self._available_branches = reference.copy()
        
        elif self._validation == 'intersection':
            self._available_branches = set.intersection(*all_branch_sets)
            if self._available_branches != reference:
                removed = reference - self._available_branches
                warnings.warn(
                    f"Intersection mode: {len(removed)} branches not in all files: "
                    f"{sorted(removed)[:5]}{'...' if len(removed) > 5 else ''}",
                    UserWarning
                )
        
        elif self._validation == 'union':
            self._available_branches = set.union(*all_branch_sets)
            if self._available_branches != reference:
                added = self._available_branches - reference
                warnings.warn(
                    f"Union mode: {len(added)} branches not in first file will have NaN "
                    f"for entries from that file: {sorted(added)[:5]}{'...' if len(added) > 5 else ''}",
                    UserWarning
                )
        
        else:
            raise ValueError(f"Unknown validation mode: {self._validation}")
    
    def _get_reader(self, file_idx: int) -> LazyTreeReader:
        """Get or create reader with LRU eviction."""
        if file_idx in self._readers:
            # Move to end (most recently used)
            self._readers.move_to_end(file_idx)
            return self._readers[file_idx]
        
        # Evict oldest if at capacity
        while len(self._readers) >= self._max_open:
            oldest_idx, oldest_reader = self._readers.popitem(last=False)
            oldest_reader.close()
        
        # Create new reader
        file_spec = self._files[file_idx]
        reader = LazyTreeReader(file_spec['path'], file_spec['tree'])
        self._readers[file_idx] = reader
        return reader
    
    @property
    def available_branches(self) -> Set[str]:
        """
        Branches available for loading.
        
        Note: Does NOT include '__file_idx__' even when add_file_index=True.
        The file index column is metadata added during loading, not a branch.
        """
        return self._available_branches.copy()
    
    @property
    def loaded_branches(self) -> Set[str]:
        """Branches currently loaded (may include '__file_idx__')."""
        return self._loaded_branches.copy()
    
    @property
    def entries(self) -> int:
        """Total entries across all files."""
        return self._total_entries
    
    @property
    def file_count(self) -> int:
        """Number of files in chain."""
        return len(self._files)
    
    @property
    def entry_offsets(self) -> List[int]:
        """Cumulative entry offsets."""
        return self._entry_offsets.copy()
    
    def get_file_for_entry(self, global_entry: int) -> int:
        """Map global entry index to file index."""
        if global_entry < 0 or global_entry >= self._total_entries:
            raise IndexError(f"Entry {global_entry} out of range [0, {self._total_entries})")
        # NumPy for entry mapping (Arrow-compatible pattern)
        return int(np.searchsorted(self._entry_offsets, global_entry, side='right') - 1)
    
    def load_branches(self, names: List[str]) -> pd.DataFrame:
        """
        Load branches from all files with single concatenation.
        
        Note: This method returns NEW data only. The caller (AliasDataFrame)
        is responsible for merging with existing data. This separation
        enables future Arrow backend migration.
        
        Parameters
        ----------
        names : List[str]
            Branch names to load
            
        Returns
        -------
        pd.DataFrame
            Combined data from all files (new branches only).
            Index is reset to 0..N-1 (ignore_index=True applied).
        """
        names_set = set(names)
        to_load = names_set - self._loaded_branches
        
        if not to_load:
            return pd.DataFrame()  # Nothing new to load
        
        # Validate requested branches exist
        missing = to_load - self._available_branches
        if missing:
            raise BranchNotFoundError(missing, self._available_branches)
        
        # Load from all files - collect first, concat once (CRITICAL pattern)
        dfs = []
        for idx in range(len(self._files)):
            reader = self._get_reader(idx)
            
            # Get branches available in this specific file
            file_branches = self._file_branches[idx]
            loadable = to_load & file_branches
            
            if loadable:
                # Load branches that exist in this file
                file_df = reader.load_branches(list(loadable), None)
            else:
                # File doesn't have any of these branches - create empty frame
                file_df = pd.DataFrame(index=range(self._files[idx]['entries']))
            
            # Ensure we have the right number of rows
            expected_rows = self._files[idx]['entries']
            if len(file_df) == 0:
                file_df = pd.DataFrame(index=range(expected_rows))
            
            # Fill missing branches with NaN (for first/union modes)
            for branch in to_load - file_branches:
                file_df[branch] = np.nan
            
            # Add file index column if requested
            if self._add_file_index:
                file_df['__file_idx__'] = idx
            
            dfs.append(file_df)
        
        # Single concatenation with ignore_index=True (CRITICAL per architecture review)
        new_data = pd.concat(dfs, ignore_index=True)
        
        # Verify row count
        assert len(new_data) == self._total_entries, \
            f"Row count mismatch: got {len(new_data)}, expected {self._total_entries}"
        
        # Update loaded tracking
        self._loaded_branches.update(to_load)
        if self._add_file_index:
            self._loaded_branches.add('__file_idx__')
        
        # Return new data - DO NOT merge here (ADF handles merge)
        return new_data

    def release_branches(self, names):
        """PHASE_13_68_ADF: forget the given physical branches so a later access
        re-reads them from all chain files. Mutates the internal ``_loaded_branches``
        set directly — NOT the copy returned by the ``loaded_branches`` property, a
        write to which would silently no-op. Names not loaded are ignored.

        Also clears the released names from every currently-cached per-file
        ``LazyTreeReader`` in the LRU: those readers each track their own loaded
        set, and a re-read after release would otherwise be suppressed. LRU-evicted
        readers reset on recreation, so clearing the cached ones is sufficient."""
        if isinstance(names, str):
            names = [names]
        names = list(names)
        self._loaded_branches.difference_update(names)
        for reader in self._readers.values():
            reader.release_branches(names)

    def estimate_memory(self, branches: List[str] = None) -> dict:
        """
        Estimate memory for loading branches.
        
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
        if branches is None:
            branches = list(self._available_branches)
        
        # TODO: Use real dtype info from first file's TTree metadata.
        # Currently assuming 4 bytes per entry (float32) as conservative estimate.
        bytes_per_entry = 4
        total = len(branches) * bytes_per_entry * self._total_entries
        
        warning = None
        if total > 32 * 1024**3:
            warning = "Estimated memory exceeds 32 GB - consider chunked loading"
        elif total > 16 * 1024**3:
            warning = "Estimated memory exceeds 16 GB"
        
        return {
            'bytes': total,
            'human': self._format_bytes(total),
            'branches': len(branches),
            'entries': self._total_entries,
            'warning': warning
        }
    
    @staticmethod
    def _format_bytes(n: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"
    
    def close(self):
        """Close all cached file handles."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
    
    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def __repr__(self):
        return (
            f"LazyChainReader(files={self.file_count}, "
            f"entries={self.entries:,}, "
            f"branches={len(self._available_branches)}, "
            f"loaded={len(self._loaded_branches)})"
        )

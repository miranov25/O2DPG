"""
Phase 13.4.DSL - C-Array Detector (D2 + D3)

Auto-detect C-array branches from TTree/RDataFrame:
- D2: Parse branch titles for dimensions
- D3: Discover counter branches for variable-length arrays

Usage:
    detector = CArrayDetector(tree_or_rdf)
    schema = detector.detect()
    # schema["mat"] = CArrayInfo(shape=(3,4), dtype="float", fixed=True)
    # schema["arr"] = CArrayInfo(shape=("n",), dtype="float", fixed=False, counter="n")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import re


@dataclass
class CArrayInfo:
    """Information about a detected C-array branch."""
    
    name: str
    shape: Tuple[Union[int, str], ...]  # int for fixed, str for counter variable
    dtype: str  # "float", "double", "int", etc.
    fixed: bool  # True if all dimensions are fixed
    counter: Optional[str] = None  # Counter branch name for variable-length
    total_size: Optional[int] = None  # Total elements if fixed
    
    def __post_init__(self):
        """Calculate total size for fixed arrays."""
        if self.fixed and all(isinstance(d, int) for d in self.shape):
            self.total_size = 1
            for d in self.shape:
                self.total_size *= d
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)
    
    @property
    def is_variable(self) -> bool:
        """True if array has variable length."""
        return not self.fixed
    
    def get_strides(self) -> List[int]:
        """
        Get strides for row-major indexing (fixed arrays only).
        
        For shape (3, 4, 5):
        - stride[0] = 4 * 5 = 20
        - stride[1] = 5
        - stride[2] = 1
        """
        if not self.fixed:
            raise ValueError("Cannot compute strides for variable-length array")
        
        strides = []
        for i in range(len(self.shape)):
            stride = 1
            for j in range(i + 1, len(self.shape)):
                stride *= self.shape[j]
            strides.append(stride)
        return strides


@dataclass
class CArraySchema:
    """Schema containing all detected C-arrays."""
    
    arrays: Dict[str, CArrayInfo] = field(default_factory=dict)
    counters: Dict[str, str] = field(default_factory=dict)  # array_name -> counter_name
    
    def __getitem__(self, name: str) -> CArrayInfo:
        return self.arrays[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self.arrays
    
    def get(self, name: str, default=None) -> Optional[CArrayInfo]:
        return self.arrays.get(name, default)
    
    def add(self, info: CArrayInfo):
        """Add a C-array to the schema."""
        self.arrays[info.name] = info
        if info.counter:
            self.counters[info.name] = info.counter
    
    def is_carray(self, name: str) -> bool:
        """Check if a branch name is a C-array."""
        return name in self.arrays
    
    def list_arrays(self) -> List[str]:
        """List all C-array names."""
        return list(self.arrays.keys())
    
    def list_fixed(self) -> List[str]:
        """List fixed-size arrays."""
        return [name for name, info in self.arrays.items() if info.fixed]
    
    def list_variable(self) -> List[str]:
        """List variable-length arrays."""
        return [name for name, info in self.arrays.items() if info.is_variable]


class BranchTitleParser:
    """
    Parse TTree branch titles to extract C-array information.
    
    Supported formats:
    - "arr[10]/F"         → 1D fixed, size 10, float
    - "mat[3][4]/F"       → 2D fixed, shape (3,4), float
    - "tensor[2][3][4]/D" → 3D fixed, shape (2,3,4), double
    - "arr[n]/F"          → 1D variable, counter "n", float
    - "mat[n][3]/F"       → 2D hybrid, first dim variable, second fixed
    """
    
    # Pattern for branch title: name[dim1][dim2]...[dimN]/TYPE
    # Dimensions can be integers or variable names
    TITLE_PATTERN = re.compile(
        r'^(\w+)'                    # Branch name
        r'((?:\[\w+\])+)'            # One or more [dim] 
        r'/([FISDC]|[BILSO])$'       # Type specifier
    )
    
    # Pattern for individual dimension
    DIM_PATTERN = re.compile(r'\[(\w+)\]')
    
    # ROOT type codes to dtype names
    TYPE_MAP = {
        'F': 'float',
        'D': 'double',
        'I': 'int',
        'S': 'short',
        'L': 'long',
        'B': 'char',
        'O': 'bool',
        'C': 'char',  # Character string (special case)
    }
    
    @classmethod
    def parse(cls, title: str, branch_name: str = None) -> Optional[CArrayInfo]:
        """
        Parse a branch title and return CArrayInfo if it's a C-array.
        
        Args:
            title: Branch title (e.g., "arr[10]/F")
            branch_name: Override name (uses name from title if None)
        
        Returns:
            CArrayInfo if valid C-array, None otherwise
        """
        match = cls.TITLE_PATTERN.match(title.strip())
        if not match:
            return None
        
        name = branch_name or match.group(1)
        dims_str = match.group(2)
        type_code = match.group(3)
        
        # Parse dimensions
        dims = cls.DIM_PATTERN.findall(dims_str)
        if not dims:
            return None
        
        # Convert dimensions to int or keep as string (counter variable)
        shape = []
        counter = None
        fixed = True
        
        for dim in dims:
            if dim.isdigit():
                shape.append(int(dim))
            else:
                # Variable dimension - first one is typically the counter
                shape.append(dim)
                if counter is None:
                    counter = dim
                fixed = False
        
        return CArrayInfo(
            name=name,
            shape=tuple(shape),
            dtype=cls.TYPE_MAP.get(type_code, 'unknown'),
            fixed=fixed,
            counter=counter
        )


class CArrayDetector:
    """
    Detect C-array branches from TTree or RDataFrame.
    
    Usage:
        # From TTree
        detector = CArrayDetector.from_tree(tree)
        schema = detector.detect()
        
        # From RDataFrame (limited - no branch title access)
        detector = CArrayDetector.from_rdf(rdf)
        schema = detector.detect()
    """
    
    def __init__(self):
        self._tree = None
        self._rdf = None
        self._schema = CArraySchema()
    
    @classmethod
    def from_tree(cls, tree) -> 'CArrayDetector':
        """Create detector from ROOT TTree."""
        detector = cls()
        detector._tree = tree
        return detector
    
    @classmethod
    def from_rdf(cls, rdf, tree=None) -> 'CArrayDetector':
        """
        Create detector from RDataFrame.
        
        Note: RDataFrame doesn't expose branch titles directly.
        If tree is provided, use it for detection.
        Otherwise, detection is limited.
        """
        detector = cls()
        detector._rdf = rdf
        detector._tree = tree
        return detector
    
    @classmethod
    def from_file(cls, filename: str, tree_name: str = "tree") -> 'CArrayDetector':
        """Create detector from ROOT file."""
        import ROOT
        f = ROOT.TFile.Open(filename)
        tree = f.Get(tree_name)
        detector = cls.from_tree(tree)
        detector._file = f  # Keep file open
        return detector
    
    def detect(self) -> CArraySchema:
        """
        Detect all C-array branches and return schema.
        
        Returns:
            CArraySchema with all detected C-arrays
        """
        if self._tree is not None:
            self._detect_from_tree()
        elif self._rdf is not None:
            self._detect_from_rdf()
        
        return self._schema
    
    def _detect_from_tree(self):
        """Detect C-arrays from TTree branches."""
        tree = self._tree
        
        # Iterate over all branches
        for branch in tree.GetListOfBranches():
            branch_name = branch.GetName()
            branch_title = branch.GetTitle()
            
            # Try to parse as C-array
            info = BranchTitleParser.parse(branch_title, branch_name)
            if info:
                # Use GetLeafCount for more accurate counter detection (D3)
                self._detect_counter(branch_name, info)
                self._schema.add(info)
    
    def _detect_counter(self, branch_name: str, info: CArrayInfo):
        """
        Use TLeaf::GetLeafCount() to find counter branch (D3).
        
        This is more reliable than parsing the title.
        """
        tree = self._tree
        leaf = tree.GetLeaf(branch_name)
        
        if leaf:
            counter_leaf = leaf.GetLeafCount()
            if counter_leaf:
                counter_name = counter_leaf.GetName()
                # Update info with accurate counter
                info.counter = counter_name
                info.fixed = False
    
    def _detect_from_rdf(self):
        """
        Detect C-arrays from RDataFrame column types.
        
        Limited detection: can only detect RVec columns,
        cannot determine original dimensions.
        """
        rdf = self._rdf
        
        # Get column names
        col_names = [str(c) for c in rdf.GetColumnNames()]
        
        for col_name in col_names:
            col_type = rdf.GetColumnType(col_name)
            
            # Check if it's an RVec (indicates possible C-array)
            if 'RVec' in col_type:
                # Extract element type
                dtype = self._extract_rvec_dtype(col_type)
                
                # We can't determine dimensions from RDF alone
                # Mark as 1D variable (conservative)
                info = CArrayInfo(
                    name=col_name,
                    shape=(-1,),  # Unknown size
                    dtype=dtype,
                    fixed=False,
                    counter=None
                )
                self._schema.add(info)
    
    def _extract_rvec_dtype(self, col_type: str) -> str:
        """Extract element type from RVec<T> type string."""
        # Pattern: ROOT::VecOps::RVec<Float_t> or RVec<float>
        match = re.search(r'RVec<(\w+)>', col_type)
        if match:
            root_type = match.group(1)
            # Map ROOT types to simple names
            type_map = {
                'Float_t': 'float',
                'Double_t': 'double',
                'Int_t': 'int',
                'Short_t': 'short',
                'Long_t': 'long',
                'Bool_t': 'bool',
                'float': 'float',
                'double': 'double',
                'int': 'int',
            }
            return type_map.get(root_type, root_type.lower())
        return 'unknown'


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_carrays(source, tree_name: str = "tree") -> CArraySchema:
    """
    Convenience function to detect C-arrays from various sources.
    
    Args:
        source: TTree, RDataFrame, filename, or (filename, tree_name) tuple
        tree_name: Tree name if source is filename
    
    Returns:
        CArraySchema with detected arrays
    """
    if isinstance(source, str):
        # Filename
        detector = CArrayDetector.from_file(source, tree_name)
    elif isinstance(source, tuple):
        # (filename, tree_name)
        detector = CArrayDetector.from_file(source[0], source[1])
    elif hasattr(source, 'GetListOfBranches'):
        # TTree
        detector = CArrayDetector.from_tree(source)
    elif hasattr(source, 'GetColumnNames'):
        # RDataFrame
        detector = CArrayDetector.from_rdf(source)
    else:
        raise TypeError(f"Cannot detect C-arrays from {type(source)}")
    
    return detector.detect()


def print_schema(schema: CArraySchema):
    """Print schema in human-readable format."""
    print("=" * 60)
    print("C-Array Schema")
    print("=" * 60)
    
    if not schema.arrays:
        print("  (no C-arrays detected)")
        return
    
    for name, info in schema.arrays.items():
        shape_str = "×".join(str(d) for d in info.shape)
        fixed_str = "fixed" if info.fixed else f"variable (counter: {info.counter})"
        print(f"  {name}: {info.dtype}[{shape_str}] ({fixed_str})")
    
    print("=" * 60)

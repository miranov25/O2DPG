"""
AliasDataFrameRDF - RDataFrame Integration for AliasDataFrame

Converts AliasDataFrame's flexible alias system to RDataFrame's
ordered Define() calls. Supports friend trees for subframe joins.

This is a research prototype for RNTuple migration.
RDataFrame requires Define() calls in dependency order (no AST resolution).

Note: TTree::Draw is deprecated; RDataFrame will be the only interface for RNTuple.
"""

import ast
import re
from typing import List, Dict, Optional, Any, Set, Tuple


# =============================================================================
# Expression Conversion (AST-based)
# =============================================================================

class CppExprConverter(ast.NodeVisitor):
    """
    Convert Python expression AST to C++ string.
    
    Handles:
    - Power operator: x**2 → pow(x, 2)
    - Numpy functions: np.sqrt → sqrt
    - Boolean: ~ → !
    - Constants: np.pi → M_PI, True → true
    """
    
    NUMPY_TO_CPP = {
        'sqrt': 'sqrt', 'abs': 'abs', 'exp': 'exp',
        'log': 'log', 'sin': 'sin', 'cos': 'cos',
        'tan': 'tan', 'arctan': 'atan', 'arctan2': 'atan2',
        'pi': 'M_PI',
    }
    
    BUILTIN_TO_CPP = {
        'abs': 'abs',
        'True': 'true',
        'False': 'false',
    }
    
    def __init__(self, subframe_alias_map: Dict[str, str] = None):
        """
        Parameters
        ----------
        subframe_alias_map : dict, optional
            Mapping of subframe columns to aliases (e.g., {'T.mP3': 'T_mP3'})
            Used when RDataFrame doesn't support dot notation in jitted expressions.
        """
        self.subframe_alias_map = subframe_alias_map or {}
    
    def convert(self, expr: str) -> str:
        """
        Convert Python expression to C++.
        
        Parameters
        ----------
        expr : str
            Python/numpy expression (e.g., 'np.sqrt(x**2 + y**2)')
            
        Returns
        -------
        str
            C++ expression (e.g., 'sqrt(pow(x, 2) + pow(y, 2))')
        """
        # Pre-process: remove 'np.' prefix for cleaner parsing
        expr_clean = self._preprocess_numpy(expr)
        
        try:
            tree = ast.parse(expr_clean, mode='eval')
            result = self.visit(tree.body)
        except SyntaxError:
            # Fallback to simple string replacements
            result = self._fallback_convert(expr)
        
        # Apply subframe aliasing if needed (T.mP3 → T_mP3)
        for original, alias in self.subframe_alias_map.items():
            result = result.replace(original, alias)
        
        return result
    
    def _preprocess_numpy(self, expr: str) -> str:
        """Replace np.func with __np_func__ for AST parsing."""
        # np.sqrt(x) → __np_sqrt__(x)
        expr = re.sub(r'\bnp\.(\w+)\b', r'__np_\1__', expr)
        return expr
    
    def _fallback_convert(self, expr: str) -> str:
        """Simple regex-based conversion for unparseable expressions."""
        result = expr
        
        # np.func → func
        for np_name, cpp_name in self.NUMPY_TO_CPP.items():
            result = re.sub(rf'\bnp\.{np_name}\b', cpp_name, result)
        
        # True/False → true/false
        result = re.sub(r'\bTrue\b', 'true', result)
        result = re.sub(r'\bFalse\b', 'false', result)
        
        # ~ → ! (logical not)
        result = re.sub(r'~\s*(\w+)', r'!(\1)', result)
        result = re.sub(r'~\s*\(', r'!(', result)
        
        return result
    
    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Handle binary operations, especially ** → pow()."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Pow):
            return f"pow({left}, {right})"
        elif isinstance(node.op, ast.Add):
            return f"({left} + {right})"
        elif isinstance(node.op, ast.Sub):
            return f"({left} - {right})"
        elif isinstance(node.op, ast.Mult):
            return f"({left} * {right})"
        elif isinstance(node.op, ast.Div):
            return f"({left} / {right})"
        elif isinstance(node.op, ast.Mod):
            return f"({left} % {right})"
        elif isinstance(node.op, ast.BitAnd):
            return f"({left} & {right})"  # Keep as bitwise
        elif isinstance(node.op, ast.BitOr):
            return f"({left} | {right})"  # Keep as bitwise
        elif isinstance(node.op, ast.BitXor):
            return f"({left} ^ {right})"
        else:
            return f"({left} ?? {right})"  # Unknown operator
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Handle unary operations, especially ~ → !."""
        operand = self.visit(node.operand)
        
        if isinstance(node.op, ast.Invert):  # ~
            return f"!({operand})"  # Convert to logical NOT
        elif isinstance(node.op, ast.Not):  # not
            return f"!({operand})"
        elif isinstance(node.op, ast.USub):  # -
            return f"(-{operand})"
        elif isinstance(node.op, ast.UAdd):  # +
            return f"(+{operand})"
        else:
            return operand
    
    def visit_Compare(self, node: ast.Compare) -> str:
        """Handle comparison operators."""
        result = self.visit(node.left)
        
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            
            if isinstance(op, ast.Lt):
                result = f"({result} < {right})"
            elif isinstance(op, ast.LtE):
                result = f"({result} <= {right})"
            elif isinstance(op, ast.Gt):
                result = f"({result} > {right})"
            elif isinstance(op, ast.GtE):
                result = f"({result} >= {right})"
            elif isinstance(op, ast.Eq):
                result = f"({result} == {right})"
            elif isinstance(op, ast.NotEq):
                result = f"({result} != {right})"
            else:
                result = f"({result} ?? {right})"
        
        return result
    
    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """Handle 'and' / 'or' → '&&' / '||'."""
        values = [self.visit(v) for v in node.values]
        
        if isinstance(node.op, ast.And):
            return "(" + " && ".join(values) + ")"
        elif isinstance(node.op, ast.Or):
            return "(" + " || ".join(values) + ")"
        else:
            return "(" + " ?? ".join(values) + ")"
    
    def visit_Call(self, node: ast.Call) -> str:
        """Handle function calls, especially numpy functions."""
        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)
        
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # np.sqrt → __np_sqrt__ after preprocessing
            func_name = node.func.attr
        else:
            func_name = self.visit(node.func)
        
        # Handle __np_func__ from preprocessing
        if func_name.startswith('__np_') and func_name.endswith('__'):
            np_func = func_name[5:-2]  # Extract 'sqrt' from '__np_sqrt__'
            cpp_func = self.NUMPY_TO_CPP.get(np_func, np_func)
            return f"{cpp_func}({args_str})"
        
        # Handle builtin functions
        cpp_func = self.BUILTIN_TO_CPP.get(func_name, func_name)
        return f"{cpp_func}({args_str})"
    
    def visit_Name(self, node: ast.Name) -> str:
        """Handle variable names and constants."""
        name = node.id
        
        # Handle __np_pi__ → M_PI
        if name == '__np_pi__':
            return 'M_PI'
        
        # Handle True/False
        if name in self.BUILTIN_TO_CPP:
            return self.BUILTIN_TO_CPP[name]
        
        return name
    
    def visit_Constant(self, node: ast.Constant) -> str:
        """Handle literal constants."""
        if isinstance(node.value, bool):
            return 'true' if node.value else 'false'
        elif isinstance(node.value, (int, float)):
            return str(node.value)
        elif isinstance(node.value, str):
            return f'"{node.value}"'
        else:
            return str(node.value)
    
    def visit_Num(self, node) -> str:
        """Handle numeric literals (Python 3.7 compatibility)."""
        return str(node.n)
    
    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Handle attribute access like T.mP3."""
        value = self.visit(node.value)
        return f"{value}.{node.attr}"
    
    def visit_Subscript(self, node: ast.Subscript) -> str:
        """Handle array indexing."""
        value = self.visit(node.value)
        if isinstance(node.slice, ast.Index):  # Python 3.8-
            idx = self.visit(node.slice.value)
        else:  # Python 3.9+
            idx = self.visit(node.slice)
        return f"{value}[{idx}]"
    
    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Handle ternary: a if cond else b → cond ? a : b."""
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"({test} ? {body} : {orelse})"
    
    def generic_visit(self, node) -> str:
        """Fallback for unhandled node types."""
        return f"/* UNHANDLED: {type(node).__name__} */"


def to_cpp_expr(expr: str, subframe_alias_map: Dict[str, str] = None) -> str:
    """
    Convert pandas/numpy expression to C++ for RDataFrame.
    
    Parameters
    ----------
    expr : str
        Original expression (e.g., 'np.sqrt(x**2 + y**2)')
    subframe_alias_map : dict, optional
        Mapping of subframe columns to aliases (e.g., {'T.mP3': 'T_mP3'})
        
    Returns
    -------
    str
        C++ expression (e.g., 'sqrt(pow(x, 2) + pow(y, 2))')
    """
    converter = CppExprConverter(subframe_alias_map)
    return converter.convert(expr)


# =============================================================================
# Dependency Extraction
# =============================================================================

def extract_dependencies(expr: str, known_names: Set[str] = None) -> List[str]:
    """
    Extract variable dependencies from expression.
    
    Parameters
    ----------
    expr : str
        Expression to parse
    known_names : set, optional
        Set of known column/alias names to filter against
        
    Returns
    -------
    list of str
        List of dependencies found in expression
    """
    # Find all identifiers (including T.mP3 style)
    # Pattern: word characters, optionally followed by .word
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'
    
    candidates = set(re.findall(pattern, expr))
    
    # Remove known functions and keywords
    functions = {
        'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'pow',
        'atan', 'atan2', 'arctan', 'arctan2',
        'np', 'True', 'False', 'true', 'false', 'M_PI',
        'and', 'or', 'not', 'if', 'else',
    }
    candidates -= functions
    
    if known_names is not None:
        candidates &= known_names
    
    return sorted(candidates)


# =============================================================================
# Dependency Resolution
# =============================================================================

def get_ordered_defines(
    aliases: List[str] = None,
    aDF: Any = None,
    schema: Dict = None,
    tree: Any = None
) -> List[Dict]:
    """
    Extract ordered definitions for RDataFrame.
    
    Parameters
    ----------
    aliases : list of str
        Target aliases to resolve (with all dependencies)
    aDF : AliasDataFrame, optional
        Source for schema and dependency resolution
    schema : dict, optional
        Direct schema dict (alternative to aDF)
    tree : ROOT.TTree, optional
        For checking available branches (optional validation)
        
    Returns
    -------
    list of dict
        Ordered list ready for sequential Define() calls.
        Each dict has: name, expr, deps, cpp_expr
        
    Raises
    ------
    ValueError
        If circular dependency detected
    """
    # Get all aliases - prefer aDF.aliases property which handles schema properly
    if aDF is not None and hasattr(aDF, 'aliases'):
        # AliasDataFrame stores aliases in _schema["columns"] with "expr" key
        # The .aliases property returns {name: expr} dict
        all_aliases = aDF.aliases
    elif schema is not None:
        # Fallback: try 'aliases' key or extract from 'columns'
        if 'aliases' in schema:
            all_aliases = schema['aliases']
        elif 'columns' in schema:
            # Extract aliases from columns (entries with 'expr' key)
            all_aliases = {
                k: v.get('expr', v) if isinstance(v, dict) else v
                for k, v in schema['columns'].items()
                if isinstance(v, dict) and 'expr' in v
            }
        else:
            all_aliases = {}
    else:
        raise ValueError("Must provide either aDF or schema")
    
    # If specific aliases requested, use them; otherwise all
    if aliases is None:
        aliases = list(all_aliases.keys())
    
    # Try to use ADF's dependency resolver (uses networkx)
    if aDF is not None and hasattr(aDF, '_resolve_dependencies'):
        try:
            ordered = aDF._resolve_dependencies(aliases)
        except Exception as e:
            # Fallback to local implementation
            ordered = _local_topological_sort(aliases, all_aliases)
    else:
        ordered = _local_topological_sort(aliases, all_aliases)
    
    # Optional: validate leaf dependencies against tree
    if tree is not None:
        _validate_leaf_deps(ordered, all_aliases, tree)
    
    # Build result list
    result = []
    for name in ordered:
        expr = all_aliases.get(name, '')
        deps = extract_dependencies(expr, set(all_aliases.keys()))
        cpp_expr = to_cpp_expr(expr)
        
        result.append({
            'name': name,
            'expr': expr,
            'deps': deps,
            'cpp_expr': cpp_expr
        })
    
    return result


def _local_topological_sort(targets: List[str], all_aliases: Dict) -> List[str]:
    """
    Topological sort with cycle detection (Kahn's algorithm).
    
    Raises ValueError if circular dependency detected.
    """
    # Build dependency graph for all aliases
    deps_graph = {}
    for name, info in all_aliases.items():
        expr = info.get('expr', '') if isinstance(info, dict) else str(info)
        deps = extract_dependencies(expr, set(all_aliases.keys()))
        deps_graph[name] = deps
    
    # Find all needed aliases (targets + their dependencies)
    needed = set()
    stack = list(targets)
    while stack:
        node = stack.pop()
        if node in needed or node not in deps_graph:
            continue
        needed.add(node)
        stack.extend(deps_graph.get(node, []))
    
    # Kahn's algorithm
    in_degree = {n: 0 for n in needed}
    for n in needed:
        for dep in deps_graph.get(n, []):
            if dep in needed:
                in_degree[n] += 1
    
    queue = [n for n in needed if in_degree[n] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        for n in needed:
            if node in deps_graph.get(n, []):
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    queue.append(n)
    
    # Check for cycles
    if len(result) != len(needed):
        missing = needed - set(result)
        raise ValueError(f"Circular dependency detected involving: {missing}")
    
    return result


def _validate_leaf_deps(ordered: List[str], all_aliases: Dict, tree) -> None:
    """Validate that leaf dependencies exist in tree."""
    # Get all alias names
    alias_names = set(all_aliases.keys())
    
    # Get tree branches
    branches = set()
    for branch in tree.GetListOfBranches():
        branches.add(branch.GetName())
    
    # Also check friends
    if tree.GetListOfFriends():
        for friend in tree.GetListOfFriends():
            friend_tree = friend.GetTree()
            friend_alias = friend.GetName()
            for branch in friend_tree.GetListOfBranches():
                branches.add(f"{friend_alias}.{branch.GetName()}")
    
    # Check each alias's dependencies
    for name in ordered:
        info = all_aliases.get(name, {})
        expr = info.get('expr', '') if isinstance(info, dict) else str(info)
        deps = extract_dependencies(expr)
        
        for dep in deps:
            # Skip if it's another alias
            if dep in alias_names:
                continue
            # Check if it's a tree branch
            if dep not in branches:
                print(f"Warning: '{name}' depends on '{dep}' which is not in tree")


# =============================================================================
# Tree Setup (Mirrors AliasDataFrameTree.C)
# =============================================================================

def setup_tree_with_friends(
    filename: str, 
    treename: str, 
    schema: Dict = None
) -> Tuple[Any, Any]:
    """
    Load tree with subframes as indexed friends.
    Python equivalent of LoadADFTree() from AliasDataFrameTree.C.
    
    Parameters
    ----------
    filename : str
        ROOT file path
    treename : str
        Main tree name
    schema : dict, optional
        Schema with subframe index definitions
        
    Returns
    -------
    tuple
        (tree, file_handle) - keep file_handle alive!
    """
    import ROOT
    
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise IOError(f"Cannot open file: {filename}")
    
    tree = f.Get(treename)
    if not tree:
        raise ValueError(f"Tree '{treename}' not found in {filename}")
    
    subframes = schema.get('subframes', {}) if schema else {}
    
    for sf_name, sf_info in subframes.items():
        sf_tree_name = f"{treename}__subframe__{sf_name}"
        sf_tree = f.Get(sf_tree_name)
        
        if not sf_tree:
            print(f"Warning: Subframe '{sf_name}' not found")
            continue
        
        # Get index columns - schema uses 'index' key
        index_cols = sf_info.get('index', sf_info.get('index_columns', []))
        
        if len(index_cols) == 0:
            print(f"Warning: Subframe '{sf_name}' has no index columns")
            continue
        elif len(index_cols) == 1:
            sf_tree.BuildIndex(index_cols[0])
        elif len(index_cols) == 2:
            sf_tree.BuildIndex(index_cols[0], index_cols[1])
        else:
            # Composite index - need __adf_key__ column
            key_branch = f"__adf_key_{sf_name}__"
            if sf_tree.GetBranch(key_branch):
                sf_tree.BuildIndex(key_branch)
            else:
                print(f"Warning: {sf_name} has {len(index_cols)} keys but no composite key branch")
                continue
        
        tree.AddFriend(sf_tree, sf_name)
        print(f"  Added friend: {sf_name} ({sf_tree.GetEntries()} entries)")
    
    return tree, f


# =============================================================================
# Code Generation
# =============================================================================

def generate_rdf_code(
    defines: List[Dict],
    tree_setup: str = None,
    include_mt: bool = False
) -> str:
    """
    Generate complete RDataFrame C++ code.
    
    Parameters
    ----------
    defines : list of dict
        Output from get_ordered_defines()
    tree_setup : str, optional
        Custom tree loading code
    include_mt : bool
        Include ROOT::EnableImplicitMT()
        
    Returns
    -------
    str
        Complete C++ macro
    """
    lines = [
        '// Auto-generated by AliasDataFrameRDF',
        '#include <ROOT/RDataFrame.hxx>',
        '#include <TFile.h>',
        '#include <TTree.h>',
        '#include <cmath>',
        '',
        'void analyze(const char* filename = "data.root") {',
    ]
    
    if include_mt:
        lines.append('    ROOT::EnableImplicitMT();')
        lines.append('')
    
    if tree_setup:
        lines.append(tree_setup)
    else:
        lines.extend([
            '    TFile* f = TFile::Open(filename);',
            '    TTree* tree = f->Get<TTree>("tree");',
            '    ROOT::RDataFrame df(*tree);',
        ])
    
    lines.append('')
    lines.append('    // Defines in dependency order')
    
    if defines:
        lines.append('    auto df_final = df')
        for i, d in enumerate(defines):
            semicolon = ';' if i == len(defines) - 1 else ''
            # Escape quotes in expression
            cpp_expr = d['cpp_expr'].replace('"', '\\"')
            lines.append(f'        .Define("{d["name"]}", "{cpp_expr}"){semicolon}')
    else:
        lines.append('    auto df_final = df;')
    
    lines.extend([
        '',
        '    // Ready for analysis',
        '    // auto h = df_final.Histo1D("column_name");',
        '    // h->Draw();',
        '}',
    ])
    
    return '\n'.join(lines)

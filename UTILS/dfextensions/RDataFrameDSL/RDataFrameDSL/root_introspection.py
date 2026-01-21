# ROOT Introspection Utility for DSL Method Discovery
# Phase: 13.6.D+
# Purpose: Automatically discover class methods using ROOT's reflection

"""
ROOT Method Introspection for DSL Type Inference

This module provides utilities to automatically discover method signatures
from custom classes using ROOT's TClass reflection system.

Usage:
    from root_introspection import discover_class_methods, register_class_schema
    
    # Discover all methods from ToyTrack
    methods = discover_class_methods('ToyTrack')
    # Returns: {'Pt': 'double', 'Eta': 'double', 'clusters': 'ROOT::VecOps::RVec<ToyCluster>'}
    
    # Register for DSL use
    register_class_schema('ToyTrack', dsl_compiler)
"""

import ROOT
from typing import Dict, Optional, List, Tuple
import re


# =============================================================================
# Type Mapping: ROOT → C++/DSL Types
# =============================================================================

ROOT_TO_CPP_TYPE = {
    # Basic types
    'double': 'double',
    'float': 'float',
    'int': 'int',
    'long': 'long',
    'bool': 'bool',
    'char': 'char',
    'short': 'short',
    'unsigned int': 'unsigned int',
    'unsigned long': 'unsigned long',
    
    # ROOT types
    'Double_t': 'double',
    'Float_t': 'float',
    'Int_t': 'int',
    'Long64_t': 'long',
    'Bool_t': 'bool',
    'Char_t': 'char',
    
    # Known physics types
    'TLorentzVector': 'TLorentzVector',
    'TVector3': 'TVector3',
    'TVector2': 'TVector2',
}


def normalize_type(root_type: str) -> str:
    """
    Normalize ROOT type string to C++ type.
    
    Args:
        root_type: Type string from ROOT (e.g., "const double&", "ROOT::VecOps::RVec<int>")
    
    Returns:
        Normalized C++ type (e.g., "double", "RVec<int>")
    
    Examples:
        >>> normalize_type("const double&")
        'double'
        >>> normalize_type("ROOT::VecOps::RVec<ToyCluster>")
        'RVec<ToyCluster>'
    """
    # Remove const, &, *, whitespace
    clean = root_type.replace('const', '').replace('&', '').replace('*', '').strip()
    
    # Simplify ROOT namespace prefixes
    # Order matters: do ROOT::VecOps::RVec first, then ROOT::RVec
    if 'ROOT::VecOps::RVec' in clean:
        clean = clean.replace('ROOT::VecOps::RVec', 'RVec')
    elif 'ROOT::RVec' in clean:
        clean = clean.replace('ROOT::RVec', 'RVec')
    
    # Map known ROOT types
    if clean in ROOT_TO_CPP_TYPE:
        return ROOT_TO_CPP_TYPE[clean]
    
    return clean


# =============================================================================
# Method Discovery
# =============================================================================

def discover_class_methods(class_name: str, 
                           include_inherited: bool = False,
                           verbose: bool = False) -> Dict[str, str]:
    """
    Discover all public methods of a class using ROOT reflection.
    
    Args:
        class_name: Name of the class (e.g., 'ToyTrack', 'TLorentzVector')
        include_inherited: Include methods from base classes
        verbose: Print discovery process
    
    Returns:
        Dict mapping method_name → return_type
        
    Example:
        >>> methods = discover_class_methods('TLorentzVector')
        >>> methods['Pt']
        'double'
        >>> methods['Vect']
        'TVector3'
    """
    # Get TClass for this type
    tclass = ROOT.TClass.GetClass(class_name)
    
    if not tclass:
        raise ValueError(f"Class '{class_name}' not found in ROOT dictionary. "
                        f"Did you forget to register pragmas?")
    
    methods = {}
    
    # Get list of all methods
    method_list = tclass.GetListOfMethods()
    
    for method in method_list:
        method_name = method.GetName()
        
        # Skip constructors, destructors, operators
        if method_name == class_name:  # Constructor
            continue
        if method_name.startswith('~'):  # Destructor
            continue
        if method_name.startswith('operator'):  # Operators
            continue
        if method_name.startswith('_'):  # Private/internal
            continue
        
        # Get return type
        return_type = method.GetReturnTypeName()
        
        # Normalize type
        cpp_type = normalize_type(return_type)
        
        # Check if method takes no arguments (for now, skip methods with args)
        n_args = method.GetNargs()
        if n_args > 0:
            if verbose:
                print(f"  Skipping {method_name}(...) - has {n_args} arguments")
            continue
        
        # Store
        methods[method_name] = cpp_type
        
        if verbose:
            print(f"  {method_name}() → {cpp_type}")
    
    return methods


def discover_class_data_members(class_name: str, 
                                verbose: bool = False) -> Dict[str, str]:
    """
    Discover all public data members of a class.
    
    Args:
        class_name: Name of the class
        verbose: Print discovery process
    
    Returns:
        Dict mapping member_name → type
        
    Example:
        >>> members = discover_class_data_members('ToyCluster')
        >>> members['fQ']
        'double'
    """
    tclass = ROOT.TClass.GetClass(class_name)
    
    if not tclass:
        raise ValueError(f"Class '{class_name}' not found")
    
    members = {}
    
    # Get data members
    member_list = tclass.GetListOfDataMembers()
    
    for member in member_list:
        member_name = member.GetName()
        
        # Get type
        type_name = member.GetTypeName()
        cpp_type = normalize_type(type_name)
        
        members[member_name] = cpp_type
        
        if verbose:
            print(f"  {member_name}: {cpp_type}")
    
    return members


# =============================================================================
# Schema Generation
# =============================================================================

def generate_class_schema(class_name: str, 
                          include_methods: bool = True,
                          include_members: bool = False,
                          verbose: bool = False) -> Dict[str, str]:
    """
    Generate complete schema for a class with methods and members.
    
    Args:
        class_name: Name of the class
        include_methods: Include method signatures
        include_members: Include data members
        verbose: Print discovery process
    
    Returns:
        Dict with '_methods' and optionally '_members' keys
        
    Example:
        >>> schema = generate_class_schema('ToyTrack')
        >>> schema['_methods']['Pt']
        'double'
    """
    schema = {}
    
    if include_methods:
        if verbose:
            print(f"Discovering methods for {class_name}:")
        methods = discover_class_methods(class_name, verbose=verbose)
        schema['_methods'] = {class_name: methods}
    
    if include_members:
        if verbose:
            print(f"Discovering members for {class_name}:")
        members = discover_class_data_members(class_name, verbose=verbose)
        schema['_members'] = {class_name: members}
    
    return schema


def generate_schema_with_pragmas(schema: Dict[str, str], 
                                 auto_detect: bool = True,
                                 verbose: bool = False) -> Dict[str, any]:
    """
    Enhance schema with automatically detected pragmas and method signatures.
    
    Args:
        schema: Original schema (e.g., {'tracks': 'RVec<ToyTrack>'})
        auto_detect: Automatically detect custom types
        verbose: Print discovery process
    
    Returns:
        Enhanced schema with '_pragmas' and '_methods' keys
        
    Example:
        >>> original = {'tracks': 'RVec<ToyTrack>'}
        >>> enhanced = generate_schema_with_pragmas(original)
        >>> enhanced['_pragmas']
        ['#pragma link C++ class ToyTrack+;', ...]
        >>> enhanced['_methods']['ToyTrack']['Pt']
        'double'
    """
    enhanced = dict(schema)
    
    if not auto_detect:
        return enhanced
    
    # Find all custom types in schema
    custom_types = set()
    
    for column, dtype in schema.items():
        if column.startswith('_'):
            continue
        
        # Extract type from RVec<Type>
        types = extract_custom_types(dtype)
        custom_types.update(types)
    
    if verbose:
        print(f"Found custom types: {custom_types}")
    
    # Generate pragmas
    pragmas = []
    all_methods = {}
    
    # Iterate over a list copy to avoid "Set changed size during iteration" error
    for custom_type in list(custom_types):
        # Check if ROOT knows this type
        tclass = ROOT.TClass.GetClass(custom_type)
        
        if not tclass:
            if verbose:
                print(f"  Warning: {custom_type} not in ROOT dictionary")
            continue
        
        # Generate pragmas
        pragmas.append(f'#pragma link C++ class {custom_type}+;')
        pragmas.append(f'#pragma link C++ class ROOT::VecOps::RVec<{custom_type}>+;')
        
        # Discover methods
        if verbose:
            print(f"Discovering methods for {custom_type}:")
        methods = discover_class_methods(custom_type, verbose=verbose)
        all_methods[custom_type] = methods
        
        # Check for nested RVec members (need additional pragmas)
        members = discover_class_data_members(custom_type, verbose=False)
        for member_name, member_type in members.items():
            if 'RVec<' in member_type:
                # Extract nested type
                nested = extract_custom_types(member_type)
                for nested_type in nested:
                    if nested_type not in custom_types:
                        pragmas.append(f'#pragma link C++ class {nested_type}+;')
                        pragmas.append(f'#pragma link C++ class ROOT::VecOps::RVec<{nested_type}>+;')
                        custom_types.add(nested_type)
    
    # Add to schema
    if pragmas:
        enhanced['_pragmas'] = pragmas
    
    if all_methods:
        enhanced['_methods'] = all_methods
    
    return enhanced


def extract_custom_types(dtype: str) -> List[str]:
    """
    Extract custom type names from schema type string.
    
    Examples:
        >>> extract_custom_types("RVec<ToyTrack>")
        ['ToyTrack']
        >>> extract_custom_types("RVec<RVec<ToyCluster>>")
        ['ToyCluster']
    """
    # Known primitives
    primitives = {
        'int', 'long', 'float', 'double', 'bool', 'char', 'short',
        'Int_t', 'Long64_t', 'Float_t', 'Double_t', 'Bool_t', 'Char_t'
    }
    
    # Strip RVec wrappers
    inner = dtype
    while 'RVec<' in inner or 'ROOT::VecOps::RVec<' in inner:
        # Find innermost type
        match = re.search(r'RVec<([^<>]+)>', inner)
        if match:
            inner = match.group(1)
        else:
            break
    
    inner = inner.strip()
    
    # Check if primitive
    if inner in primitives:
        return []
    
    # Check if known ROOT type
    if inner.startswith('T') and len(inner) > 1:  # TLorentzVector, TVector3, etc.
        # These are known ROOT types, might not need pragmas
        # But return anyway for completeness
        pass
    
    return [inner]


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    # Example: Discover TLorentzVector methods
    print("=== TLorentzVector Methods ===")
    methods = discover_class_methods('TLorentzVector', verbose=True)
    
    print("\n=== Auto-Generate Schema ===")
    original_schema = {
        'event_id': 'long',
        'tracks': 'RVec<ToyTrack>',
    }
    
    enhanced = generate_schema_with_pragmas(original_schema, verbose=True)
    
    print("\nEnhanced Schema:")
    print(f"  Pragmas: {len(enhanced.get('_pragmas', []))}")
    print(f"  Methods discovered: {list(enhanced.get('_methods', {}).keys())}")

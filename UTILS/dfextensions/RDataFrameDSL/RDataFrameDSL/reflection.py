"""
Class reflection for C++ objects using ROOT's TClass API.

This module provides ReflectionCache which resolves method and property types
for C++ objects. It uses a two-tier resolution strategy:

1. TClass reflection (PRIMARY) - Query ROOT TClass for method/property info
2. Schema override (FALLBACK) - Only when reflection fails or user overrides

Usage:
    cache = ReflectionCache()
    method_info = cache.resolve_method("TLorentzVector", "Pt")
    print(method_info.return_type)  # "Double_t"

For classes without dictionaries, provide schema:
    schema = {
        "methods": {
            "MyClass::getX": {"return_type": "float", "args": [], "is_const": True}
        },
        "properties": {
            "MyClass::mValue": {"type": "double"}
        }
    }
    cache = ReflectionCache(schema=schema)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .ir_types import IRType, IRTypeKind, cpp_type_to_ir
from .ir_errors import IRError, IRErrorKind

__all__ = [
    'ReflectionCache',
    'MethodInfo',
    'PropertyInfo',
]


# =============================================================================
# Info Dataclasses
# =============================================================================

@dataclass
class MethodInfo:
    """
    Information about a C++ class method.
    
    Attributes:
        class_name: Fully qualified class name
        method_name: Method name
        return_type: C++ return type string
        arg_types: List of argument type strings (for overload resolution)
        is_const: Whether method is const
        source: Where info came from ("tclass" or "schema")
    """
    class_name: str
    method_name: str
    return_type: str
    arg_types: List[str] = field(default_factory=list)
    is_const: bool = False
    source: str = "tclass"  # "tclass" or "schema"
    
    @classmethod
    def from_schema(cls, class_name: str, method_name: str,
                    schema_entry: dict) -> 'MethodInfo':
        """Create MethodInfo from schema entry."""
        return cls(
            class_name=class_name,
            method_name=method_name,
            return_type=schema_entry.get("return_type", "Unknown"),
            arg_types=schema_entry.get("args", []),
            is_const=schema_entry.get("is_const", False),
            source="schema"
        )
    
    def get_ir_type(self) -> IRType:
        """Convert return_type to IRType."""
        return cpp_type_to_ir(self.return_type)
    
    def __repr__(self) -> str:
        args_str = ", ".join(self.arg_types) if self.arg_types else ""
        const_str = " const" if self.is_const else ""
        return (f"MethodInfo({self.class_name}::{self.method_name}({args_str}) "
                f"-> {self.return_type}{const_str}, source={self.source})")


@dataclass
class PropertyInfo:
    """
    Information about a C++ class property/data member.
    
    Attributes:
        class_name: Fully qualified class name
        property_name: Property/member name
        property_type: C++ type string
        source: Where info came from ("tclass" or "schema")
    """
    class_name: str
    property_name: str
    property_type: str
    source: str = "tclass"  # "tclass" or "schema"
    
    @classmethod
    def from_schema(cls, class_name: str, property_name: str,
                    schema_entry: dict) -> 'PropertyInfo':
        """Create PropertyInfo from schema entry."""
        return cls(
            class_name=class_name,
            property_name=property_name,
            property_type=schema_entry.get("type", "Unknown"),
            source="schema"
        )
    
    def get_ir_type(self) -> IRType:
        """Convert property_type to IRType."""
        return cpp_type_to_ir(self.property_type)
    
    def __repr__(self) -> str:
        return (f"PropertyInfo({self.class_name}::{self.property_name}: "
                f"{self.property_type}, source={self.source})")


# =============================================================================
# Reflection Cache
# =============================================================================

class ReflectionCache:
    """
    Caches C++ reflection results with TClass primary, schema fallback.
    
    The cache uses a two-tier resolution strategy:
    1. TClass reflection (PRIMARY) - Uses ROOT's TClass API
    2. Schema override (FALLBACK) - For classes without dictionaries
    
    Example:
        >>> cache = ReflectionCache()
        >>> info = cache.resolve_method("TLorentzVector", "Pt")
        >>> print(info.return_type)
        'Double_t'
    """
    
    def __init__(self, schema: dict = None):
        """
        Initialize reflection cache.
        
        Args:
            schema: Optional schema dict for fallback resolution
        """
        self.schema = schema or {}
        self._method_cache: Dict[Tuple[str, str, Tuple[str, ...]], MethodInfo] = {}
        self._property_cache: Dict[Tuple[str, str], PropertyInfo] = {}
        self._class_cache: Dict[str, Any] = {}  # Cache TClass lookups
        self._root_available: Optional[bool] = None
    
    def _check_root(self) -> bool:
        """Check if ROOT is available."""
        if self._root_available is None:
            try:
                import ROOT
                self._root_available = True
            except ImportError:
                self._root_available = False
        return self._root_available
    
    def _get_tclass(self, class_name: str):
        """Get TClass for class_name, with caching."""
        if class_name in self._class_cache:
            return self._class_cache[class_name]
        
        if not self._check_root():
            self._class_cache[class_name] = None
            return None
        
        import ROOT
        tclass = ROOT.TClass.GetClass(class_name)
        
        # ROOT null pointers can be truthy in Python, need robust validation
        if tclass is None:
            self._class_cache[class_name] = None
            return None
        
        # Try to verify the class is actually valid/loaded
        try:
            # Check if bool conversion returns False (null pointer in cppyy)
            if not bool(tclass):
                self._class_cache[class_name] = None
                return None
            
            # Check if class is actually loaded (has dictionary)
            # IsLoaded() returns true if dictionary exists
            if hasattr(tclass, 'IsLoaded') and not tclass.IsLoaded():
                self._class_cache[class_name] = None
                return None
            
            # Also check HasDictionary as backup
            if hasattr(tclass, 'HasDictionary') and not tclass.HasDictionary():
                self._class_cache[class_name] = None
                return None
                
        except Exception:
            self._class_cache[class_name] = None
            return None
        
        self._class_cache[class_name] = tclass
        return tclass
    
    def resolve_method(self, class_name: str, method_name: str,
                       arg_types: List[str] = None) -> MethodInfo:
        """
        Resolve method info. TClass first, schema fallback.
        
        Args:
            class_name: C++ class name
            method_name: Method name to resolve
            arg_types: Argument types for overload resolution (optional)
            
        Returns:
            MethodInfo with method details
            
        Raises:
            IRError: If method cannot be resolved
        """
        cache_key = (class_name, method_name, tuple(arg_types or []))
        if cache_key in self._method_cache:
            return self._method_cache[cache_key]
        
        # Step 1: Try TClass reflection first
        tclass = self._get_tclass(class_name)
        
        if tclass:
            info = self._resolve_method_from_tclass(tclass, class_name, 
                                                     method_name, arg_types)
            if info:
                self._method_cache[cache_key] = info
                return info
        
        # Step 2: If reflection fails, try schema
        info = self._resolve_method_from_schema(class_name, method_name)
        if info:
            self._method_cache[cache_key] = info
            return info
        
        # Step 3: Neither worked - raise appropriate error
        self._raise_method_not_found(tclass, class_name, method_name)
    
    def _resolve_method_from_tclass(self, tclass, class_name: str,
                                     method_name: str,
                                     arg_types: List[str] = None) -> Optional[MethodInfo]:
        """Try to resolve method from TClass."""
        method = None
        
        try:
            # If arg_types provided, try to find specific overload
            if arg_types:
                proto = ", ".join(arg_types)
                method = tclass.GetMethodWithPrototype(method_name, proto)
            
            # Otherwise get any matching method using GetMethodAny
            # (GetMethod requires 2 args: name and params)
            if method is None:
                method = tclass.GetMethodAny(method_name)
            
            if method is None:
                return None
            
            # Extract method info
            return_type = method.GetReturnTypeName()
            
            # Check if method is const
            is_const = False
            try:
                prop = method.Property()
                # ROOT.kIsConstMethod = 0x00000040
                is_const = bool(prop & 0x00000040)
            except:
                pass
            
            return MethodInfo(
                class_name=class_name,
                method_name=method_name,
                return_type=return_type,
                arg_types=arg_types or [],
                is_const=is_const,
                source="tclass"
            )
        except Exception:
            # If any TClass API call fails, return None to fall back to schema
            return None
    
    def _resolve_method_from_schema(self, class_name: str,
                                     method_name: str) -> Optional[MethodInfo]:
        """Try to resolve method from schema."""
        if not self.schema:
            return None
        
        methods = self.schema.get("methods", {})
        
        # Try fully qualified name first
        method_key = f"{class_name}::{method_name}"
        if method_key in methods:
            return MethodInfo.from_schema(class_name, method_name,
                                          methods[method_key])
        
        return None
    
    def _raise_method_not_found(self, tclass, class_name: str, 
                                 method_name: str) -> None:
        """Raise appropriate error for method not found."""
        if tclass is None:
            raise IRError(
                IRErrorKind.MISSING_DICT,
                f"Class '{class_name}' not found (no dictionary). "
                "Load the appropriate dictionary or add required headers.",
                suggestions=[
                    f"ROOT.gSystem.Load('lib{class_name}')",
                    "Or provide method signature in schema"
                ]
            )
        
        # Class exists but method not found
        suggestions = self._fuzzy_match_methods(tclass, method_name)
        raise IRError(
            IRErrorKind.TYPE_ERROR,  # Using TYPE_ERROR since METHOD_NOT_FOUND may not exist
            f"Method '{method_name}' not found in class '{class_name}'",
            suggestions=suggestions
        )
    
    def resolve_property(self, class_name: str, 
                         property_name: str) -> PropertyInfo:
        """
        Resolve property/data member info. TClass first, schema fallback.
        
        Args:
            class_name: C++ class name
            property_name: Property/member name to resolve
            
        Returns:
            PropertyInfo with property details
            
        Raises:
            IRError: If property cannot be resolved
        """
        cache_key = (class_name, property_name)
        if cache_key in self._property_cache:
            return self._property_cache[cache_key]
        
        # Step 1: Try TClass reflection first
        tclass = self._get_tclass(class_name)
        
        if tclass:
            info = self._resolve_property_from_tclass(tclass, class_name,
                                                       property_name)
            if info:
                self._property_cache[cache_key] = info
                return info
        
        # Step 2: If reflection fails, try schema
        info = self._resolve_property_from_schema(class_name, property_name)
        if info:
            self._property_cache[cache_key] = info
            return info
        
        # Step 3: Neither worked - raise appropriate error
        self._raise_property_not_found(tclass, class_name, property_name)
    
    def _resolve_property_from_tclass(self, tclass, class_name: str,
                                       property_name: str) -> Optional[PropertyInfo]:
        """Try to resolve property from TClass."""
        try:
            member = tclass.GetDataMember(property_name)
            
            if member is None:
                return None
            
            return PropertyInfo(
                class_name=class_name,
                property_name=property_name,
                property_type=member.GetTypeName(),
                source="tclass"
            )
        except Exception:
            # If any TClass API call fails, return None to fall back to schema
            return None
    
    def _resolve_property_from_schema(self, class_name: str,
                                       property_name: str) -> Optional[PropertyInfo]:
        """Try to resolve property from schema."""
        if not self.schema:
            return None
        
        properties = self.schema.get("properties", {})
        
        # Try fully qualified name
        property_key = f"{class_name}::{property_name}"
        if property_key in properties:
            return PropertyInfo.from_schema(class_name, property_name,
                                            properties[property_key])
        
        return None
    
    def _raise_property_not_found(self, tclass, class_name: str,
                                   property_name: str) -> None:
        """Raise appropriate error for property not found."""
        if tclass is None:
            raise IRError(
                IRErrorKind.MISSING_DICT,
                f"Class '{class_name}' not found (no dictionary).",
                suggestions=["Load the appropriate dictionary"]
            )
        
        # Class exists but property not found
        suggestions = self._fuzzy_match_properties(tclass, property_name)
        raise IRError(
            IRErrorKind.TYPE_ERROR,  # Using TYPE_ERROR since PROPERTY_NOT_FOUND may not exist
            f"Property '{property_name}' not found in class '{class_name}'",
            suggestions=suggestions
        )
    
    def _fuzzy_match_methods(self, tclass, target: str) -> List[str]:
        """Find similar method names for suggestions."""
        if tclass is None:
            return []
        
        suggestions = []
        target_lower = target.lower()
        
        try:
            methods = tclass.GetListOfMethods()
            if methods:
                seen = set()
                for i in range(methods.GetSize()):
                    method = methods.At(i)
                    if method is None:
                        continue
                    name = method.GetName()
                    
                    # Skip duplicates, operators, and destructors
                    if name in seen or name.startswith("operator") or name.startswith("~"):
                        continue
                    seen.add(name)
                    
                    name_lower = name.lower()
                    # Case-insensitive match or substring match
                    if (target_lower == name_lower or
                        target_lower in name_lower or
                        name_lower in target_lower):
                        suggestions.append(f"Did you mean '{name}'?")
        except:
            pass
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _fuzzy_match_properties(self, tclass, target: str) -> List[str]:
        """Find similar property names for suggestions."""
        if tclass is None:
            return []
        
        suggestions = []
        target_lower = target.lower()
        
        try:
            members = tclass.GetListOfDataMembers()
            if members:
                for i in range(members.GetSize()):
                    member = members.At(i)
                    if member is None:
                        continue
                    name = member.GetName()
                    name_lower = name.lower()
                    
                    if (target_lower == name_lower or
                        target_lower in name_lower or
                        name_lower in target_lower):
                        suggestions.append(f"Did you mean '{name}'?")
        except:
            pass
        
        return suggestions[:3]
    
    def has_class(self, class_name: str) -> bool:
        """Check if class is available via TClass or schema."""
        tclass = self._get_tclass(class_name)
        if tclass:
            return True
        
        # Check if any methods/properties for this class in schema
        if self.schema:
            prefix = f"{class_name}::"
            methods = self.schema.get("methods", {})
            properties = self.schema.get("properties", {})
            
            for key in methods:
                if key.startswith(prefix):
                    return True
            for key in properties:
                if key.startswith(prefix):
                    return True
        
        return False
    
    def list_methods(self, class_name: str) -> List[str]:
        """List available methods for a class."""
        methods = []
        
        tclass = self._get_tclass(class_name)
        if tclass:
            try:
                method_list = tclass.GetListOfMethods()
                if method_list:
                    seen = set()
                    for i in range(method_list.GetSize()):
                        method = method_list.At(i)
                        if method:
                            name = method.GetName()
                            if name not in seen and not name.startswith("~"):
                                methods.append(name)
                                seen.add(name)
            except:
                pass
        
        # Add schema methods
        if self.schema:
            prefix = f"{class_name}::"
            for key in self.schema.get("methods", {}):
                if key.startswith(prefix):
                    method_name = key[len(prefix):]
                    if method_name not in methods:
                        methods.append(method_name)
        
        return sorted(methods)
    
    def list_properties(self, class_name: str) -> List[str]:
        """List available properties for a class."""
        properties = []
        
        tclass = self._get_tclass(class_name)
        if tclass:
            try:
                member_list = tclass.GetListOfDataMembers()
                if member_list:
                    for i in range(member_list.GetSize()):
                        member = member_list.At(i)
                        if member:
                            properties.append(member.GetName())
            except:
                pass
        
        # Add schema properties
        if self.schema:
            prefix = f"{class_name}::"
            for key in self.schema.get("properties", {}):
                if key.startswith(prefix):
                    prop_name = key[len(prefix):]
                    if prop_name not in properties:
                        properties.append(prop_name)
        
        return sorted(properties)
    
    def clear_cache(self) -> None:
        """Clear all cached reflection results."""
        self._method_cache.clear()
        self._property_cache.clear()
        self._class_cache.clear()
    
    def describe(self, class_name: str) -> str:
        """Return human-readable description of class."""
        lines = [f"Class: {class_name}"]
        
        tclass = self._get_tclass(class_name)
        if tclass:
            lines.append("  Source: TClass (dictionary available)")
        elif self.has_class(class_name):
            lines.append("  Source: Schema only")
        else:
            lines.append("  Source: NOT FOUND")
            return "\n".join(lines)
        
        methods = self.list_methods(class_name)
        if methods:
            lines.append(f"  Methods ({len(methods)}):")
            for m in methods[:10]:  # Show first 10
                lines.append(f"    - {m}")
            if len(methods) > 10:
                lines.append(f"    ... and {len(methods) - 10} more")
        
        properties = self.list_properties(class_name)
        if properties:
            lines.append(f"  Properties ({len(properties)}):")
            for p in properties[:10]:
                lines.append(f"    - {p}")
            if len(properties) > 10:
                lines.append(f"    ... and {len(properties) - 10} more")
        
        return "\n".join(lines)

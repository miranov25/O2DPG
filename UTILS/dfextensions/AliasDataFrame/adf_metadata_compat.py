"""
adf_metadata_compat.py — backward-compatible ADF metadata read-precedence resolver.

Phase 13.59.ADF. Single metadata-read entry point implementing AD-3/13.59.ADF
reading precedence (first source that succeeds wins):

  1. ROOT UserInfo        (if ROOT is importable)            -> _source='root_userinfo'
  2. uproot UserInfo      (minimal_ttree_metadata=False)     -> _source='uproot_userinfo'
  3. standalone key       <tree>__adfmeta__ (TObjString JSON)-> _source='key'
  4. names reconstruction from <tree>__subframe__<name>      -> _source='names'
                                                                schema_source='names_only'

Note the ORDER: UserInfo (1,2) is preferred over the standalone key (3), per the
architect decision — old, trusted files carry UserInfo; the standalone key is the
additive escape hatch. (The earlier 13.58 prototype tried the key first; that order
is superseded by AD-3.)

Reading metadata only touches the TTree header, not branch data, so it is cheap even
for multi-GB files. Levels 2-4 are ROOT-free (uproot only).
"""
import json
import warnings
from typing import Optional

import uproot

ADFMETA_SUFFIX = "__adfmeta__"
SUBFRAME_DELIM = "__subframe__"
SCHEMA_KEY = "__alias_dataframe_schema__"


def _key_names(f) -> list:
    """Top-level key names without the ROOT ';cycle' suffix."""
    return [k.rsplit(";", 1)[0] for k in f.keys(recursive=False)]


def _subframes_from_names(f, tree_name: str) -> list:
    prefix = f"{tree_name}{SUBFRAME_DELIM}"
    return [n[len(prefix):] for n in _key_names(f) if n.startswith(prefix)]


def _normalize(meta: dict, source: str) -> dict:
    """Flatten a parsed metadata JSON into a stable shape.

    Prefers the nested unified schema (__alias_dataframe_schema__) when present,
    mirroring the eager ROOT path; otherwise uses top-level legacy fields.
    """
    subframes = list(meta.get("subframes", []) or [])
    subframe_indices = dict(meta.get("subframe_indices", {}) or {})
    aliases = dict(meta.get("aliases", {}) or {})
    column_dtypes = dict(meta.get("column_dtypes", {}) or {})

    sch = meta.get(SCHEMA_KEY)
    if isinstance(sch, dict):
        sf = sch.get("subframes", {}) or {}
        if sf:
            subframes = list(sf.keys())
            subframe_indices = {
                k: (v.get("index_columns") or v.get("index"))
                for k, v in sf.items()
            }
        if sch.get("column_dtypes"):
            column_dtypes = dict(sch["column_dtypes"])

    return {
        "_source": source,
        "subframes": subframes,
        "subframe_indices": subframe_indices,
        "aliases": aliases,
        "column_dtypes": column_dtypes,
        "raw": meta,
    }


def _try_root_userinfo(file_path: str, tree_name: str) -> Optional[dict]:
    """Level 1: read UserInfo via ROOT if ROOT is importable. None if unavailable/empty."""
    try:
        import ROOT  # noqa: import inside function — ROOT is optional
    except Exception:
        return None
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return None
    try:
        tree = f.Get(tree_name)
        if not tree:
            return None
        ui = tree.GetUserInfo()
        if not ui:
            return None
        for i in range(ui.GetEntries()):
            obj = ui.At(i)
            if isinstance(obj, ROOT.TObjString):
                try:
                    meta = json.loads(obj.GetString().Data())
                except Exception:
                    continue
                if isinstance(meta, dict) and (SCHEMA_KEY in meta or "subframes" in meta):
                    return meta
        return None
    finally:
        f.Close()


def read_adf_metadata(file_path: str, tree_name: str,
                      f: Optional["uproot.ReadOnlyDirectory"] = None,
                      prefer_root: bool = True) -> dict:
    """Recover ADF metadata following AD-3/13.59.ADF read precedence.

    Returns a dict with keys: _source ('root_userinfo'|'uproot_userinfo'|'key'|'names'),
    subframes, subframe_indices, aliases, column_dtypes, raw. For the names fallback the
    result also carries schema_source='names_only'.

    Parameters
    ----------
    prefer_root : bool, default True
        If True, try ROOT UserInfo (level 1) before the uproot path. When ROOT is not
        importable this is skipped automatically and level 2 (uproot UserInfo) is used.
    """
    # Level 1 — ROOT UserInfo (if available)
    if prefer_root:
        meta = _try_root_userinfo(file_path, tree_name)
        if meta is not None:
            return _normalize(meta, "root_userinfo")

    own = f is None
    if own:
        f = uproot.open(file_path, minimal_ttree_metadata=False)
    try:
        # Level 2 — uproot UserInfo. Loop ALL entries (mirrors the ROOT path); the ADF
        # JSON is not guaranteed to be the first UserInfo object.
        t = f[tree_name]
        ui = t.all_members.get("fUserInfo")
        saw_userinfo_parse_error = False
        if ui is not None:
            for item in list(ui):
                try:
                    meta = json.loads(str(item))
                except Exception:
                    saw_userinfo_parse_error = True
                    continue
                if isinstance(meta, dict) and (SCHEMA_KEY in meta or "subframes" in meta):
                    return _normalize(meta, "uproot_userinfo")
        if saw_userinfo_parse_error:
            # A UserInfo entry was present but unparseable (possible corrupt metadata) and
            # no usable ADF entry was found. Warn rather than silently fall through.
            warnings.warn(
                f"UserInfo for '{tree_name}' contained an entry that failed JSON parsing; "
                f"no usable ADF metadata in UserInfo — falling through to lower-precedence "
                f"sources."
            )

        # Level 3 — standalone key
        if f"{tree_name}{ADFMETA_SUFFIX}" in _key_names(f):
            meta = json.loads(str(f[f"{tree_name}{ADFMETA_SUFFIX}"]))
            return _normalize(meta, "key")

        # Level 4 — names reconstruction (structure only, never overrides 1-3)
        names = _subframes_from_names(f, tree_name)
        out = _normalize({"subframes": names}, "names")
        out["schema_source"] = "names_only"
        if names:
            warnings.warn(
                f"ADF metadata for '{tree_name}' reconstructed from subframe key names; "
                f"aliases/dtypes unavailable (schema_source='names_only')."
            )
        return out
    finally:
        if own:
            f.close()


def write_adf_metadata_key(file_obj, tree_name: str, meta: dict) -> str:
    """Write metadata as the standalone key <tree>__adfmeta__ (TObjString JSON).

    file_obj must be a writable uproot file (uproot.recreate/update). uproot cannot
    write TTree UserInfo, so this is the uproot-only write path (used only when ROOT
    is absent, per AD-3 write precedence). Returns the key name written.
    """
    key = f"{tree_name}{ADFMETA_SUFFIX}"
    file_obj[key] = json.dumps(meta)
    return key

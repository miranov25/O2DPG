"""PHASE_13_73_FIX_ADF — guard: every public method of the API must have a docstring.

WHY THIS EXISTS
`adf.add_alias?` printed "<no docstring>" for THREE PHASES (13.70 -> 13.73), on the most-used
method in the whole API. Phase 13.70 turned add_alias into a thin dispatcher and the docstring
went with the implementation into the PRIVATE _add_scalar_alias. Every review passed, because
nobody ever typed `adf.add_alias?`.

This test is that missing check. A public method with no docstring is an undocumented feature:
it fails the 15-minute cold-read standard (MTTU) no matter how good the code is.

If this test fails: write a docstring. Do not add the method to an exemption list.
"""
import ast
import inspect
import os

import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402
from LazyTreeReader import LazyTreeReader    # noqa: E402
from LazyChainReader import LazyChainReader  # noqa: E402


def _public_methods(cls):
    """Public callables + properties defined ON this class (not inherited).

    Properties are unwrapped to their getter: a @property's docstring lives on fget,
    and reading the attribute off the class would otherwise execute/return the descriptor.
    """
    out = []
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            if member.fget is not None:
                out.append((name, member.fget, "property"))
        elif isinstance(member, staticmethod):
            out.append((name, member.__func__, "staticmethod"))
        elif isinstance(member, classmethod):
            out.append((name, member.__func__, "classmethod"))
        elif inspect.isfunction(member):
            out.append((name, member, "method"))
    return sorted(out)


PUBLIC = _public_methods(AliasDataFrame)

# D-4 (as originally specified): the lazy readers are part of the public surface too.
# Both are currently clean, so this is free coverage that keeps them that way.
READERS = ([("LazyTreeReader." + n, f, k) for n, f, k in _public_methods(LazyTreeReader)] +
           [("LazyChainReader." + n, f, k) for n, f, k in _public_methods(LazyChainReader)])


def test_public_api_is_not_empty():
    """Sanity: the reflection above actually found the API (guards against a silent no-op)."""
    assert len(PUBLIC) > 100, f"only found {len(PUBLIC)} public members — reflection is broken"


@pytest.mark.parametrize("name,func,kind", PUBLIC, ids=[p[0] for p in PUBLIC])
def test_public_method_has_docstring(name, func, kind):
    """Every public method/property of AliasDataFrame must have a non-empty docstring."""
    doc = inspect.getdoc(func)
    assert doc and doc.strip(), (
        f"AliasDataFrame.{name} ({kind}) has NO docstring — it is an undocumented public "
        f"feature. `adf.{name}?` shows nothing to a user. Write one; do not exempt it."
    )


def test_add_alias_documents_its_real_surface():
    """add_alias's docstring must cover the surface it actually has.

    Pinned explicitly because this is the method that regressed: the 13.70 dispatcher split
    stranded the docstring on the private _add_scalar_alias, and 13.73 then added `source=`
    to a signature that had no docs at all.
    """
    doc = inspect.getdoc(AliasDataFrame.add_alias)
    assert doc, "add_alias has no docstring"
    low = doc.lower()
    for token, why in [
        ("source", "the source= subframe-binding kwarg (13.73)"),
        ("dtype", "the dtype argument"),
        ("list", "the vector/group alias form, where name is a LIST (13.70)"),
    ]:
        assert token in low, f"add_alias docstring does not mention {token!r} — {why}"


@pytest.mark.parametrize("name,func,kind", READERS, ids=[p[0] for p in READERS])
def test_reader_public_method_has_docstring(name, func, kind):
    """LazyTreeReader / LazyChainReader are public API too (users read available_branches,
    loaded_branches, entries...). D-4 specified them; keeping them covered keeps them clean."""
    doc = inspect.getdoc(func)
    assert doc and doc.strip(), f"{name} ({kind}) has NO docstring — undocumented public API."


def test_add_alias_docstring_return_contract_matches_reality():
    """PHASE_13_73_FIX_ADF (panel GPT20/GPT21): the docstring's CLAIMS must match what the
    code actually DOES — verified by running it, not by reading it.

    The first cut of this docstring said "Returns: None". That is true for a scalar alias but
    FALSE for a vector alias, which returns list(names) — a contract that had existed since
    13.70 and that the one document whose job was to describe it got wrong. A keyword-presence
    test would have passed. This one executes the call and compares.
    """
    import numpy as np, pandas as pd
    adf = AliasDataFrame(pd.DataFrame({"a": np.arange(4) * 1.0, "b": np.arange(4) + 1.0}))
    adf.register_function("pair2", lambda a, b: np.column_stack([a + b, a - b]))

    assert adf.add_alias("s", "a*2") is None                      # scalar -> None
    assert adf.add_alias(["one"], "a*3") is None                  # 1-elem list -> scalar -> None
    got = adf.add_alias(["u", "w"], "pair2(a,b)", dtype=["float32", "float32"])
    assert got == ["u", "w"], f"vector alias returned {got!r}, not the member-name list"

    # Scope the check to the RETURNS section: "list of str" also appears under Parameters
    # (`name : str or list of str`), so a whole-docstring substring test is a false negative.
    doc = inspect.getdoc(AliasDataFrame.add_alias) or ""
    assert "Returns" in doc, "add_alias docstring has no Returns section"
    returns_block = doc.split("Returns", 1)[1].split("Raises", 1)[0]
    assert "list" in returns_block, (
        "add_alias returns list(names) for a vector alias, but its RETURNS section does not "
        f"say so. Returns section reads: {returns_block.strip()[:120]!r}"
    )


def test_add_alias_docstring_is_honest_about_probe_evaluation():
    """The docstring must not claim the expression is never evaluated during add_alias.

    A vector alias with available inputs runs a ONE-ROW probe at definition (13.70 fail-fast
    arity check), so the user's function IS called here. The first cut of the docstring said
    "The alias is NOT evaluated here" — false for the most common interactive case.
    """
    import numpy as np, pandas as pd
    calls = {"n": 0}

    def spy(a, b):
        calls["n"] += 1
        return np.column_stack([a + b, a - b])

    adf = AliasDataFrame(pd.DataFrame({"a": np.arange(4) * 1.0, "b": np.arange(4) + 1.0}))
    adf.register_function("spy", spy)
    adf.add_alias(["p", "q"], "spy(a,b)", dtype=["float32", "float32"])
    assert calls["n"] >= 1, "expected a probe evaluation at definition (13.70 fail-fast)"

    doc = (inspect.getdoc(AliasDataFrame.add_alias) or "").lower()
    assert "probe" in doc, "docstring hides the definition-time probe evaluation"
    assert "not evaluated here" not in doc, (
        "docstring still claims the alias is NOT evaluated here — false for vector aliases "
        "with available inputs"
    )


def test_is_constant_and_fill_value_reach_vector_members():
    """PHASE_13_73_FIX_ADF (architect token): forward, do not silently drop.

    add_alias advertises is_constant and fill_value, but the vector dispatch used to call
    _add_group_alias(names, expression, dtype) — dropping BOTH on the floor with no warning.
    They must now reach every member.
    """
    import numpy as np, pandas as pd
    adf = AliasDataFrame(pd.DataFrame({"a": [1., 2., 0., 4.], "b": [1., 0., 0., 2.]}))
    adf.register_function("div2", lambda a, b: np.column_stack([a / b, b / a]))
    adf.add_alias(["u", "w"], "div2(a,b)", dtype=["float32", "float32"], fill_value=-1.0)
    adf.materialize_aliases(names=["u", "w"])
    assert np.isfinite(adf.df["u"]).all(), "fill_value not applied to vector member 'u'"
    assert np.isfinite(adf.df["w"]).all(), "fill_value not applied to vector member 'w'"
    assert (adf.df["u"] == -1.0).any(), "fill_value did not replace the non-finite entries"


def test_no_governance_jargon_in_add_alias_docstring():
    """Docstrings are for users, not for the review process.

    SCOPE: this checks add_alias SPECIFICALLY (the method this phase repaired), not all
    user-facing help across the codebase — a broader prose-quality audit is a separate job.

    A user at an IPython prompt has no idea what 'PHASE_13_73_ADF' or '[X-7]' mean. Phase IDs
    and bracket codes belong in commit messages and CRRs, not in `?` output. (This is the
    defect that made the removed add_aliases undocumented-in-practice despite having text.)
    """
    doc = inspect.getdoc(AliasDataFrame.add_alias) or ""
    assert "PHASE_" not in doc, "add_alias docstring leaks a phase ID into user-facing help"
    for code in ("[X-", "[D-", "[R1", "[R2", "[F-"):
        assert code not in doc, f"add_alias docstring leaks governance code {code!r}"


def test_add_aliases_is_gone():
    """PHASE_13_73_FIX_ADF (D-1): add_aliases was removed.

    It was never requested (only `add_alias(source=)` was), it could not express per-alias
    dtypes (one dtype was applied to the whole mapping), and it created a second public entry
    point for one concept. A loop over add_alias(..., source=...) is strictly more expressive.
    """
    assert not hasattr(AliasDataFrame, "add_aliases"), (
        "add_aliases still exists — it was removed in PHASE_13_73_FIX_ADF"
    )

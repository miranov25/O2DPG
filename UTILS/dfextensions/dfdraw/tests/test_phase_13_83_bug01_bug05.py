"""PHASE_13_83_DF red reproducer tests for BUG-01 and BUG-05.

These tests are intentionally ordinary assertions, not xfails.  On the
PHASE_13_83 BEGIN baseline they must fail and thereby demonstrate the product
defects before any production fix is installed.

After the bounded fixes they become permanent passing invariance tests.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dfdraw import DFDraw


def test_B01_selection_singleton_matches_scalar_selection_with_facet():
    """BUG-01: one-element selection_vector must keep the scalar selection effect."""
    rng = np.random.default_rng(138301)
    n = 120
    left_x = rng.normal(-1.0, 0.10, n)
    right_x = rng.normal(+1.0, 0.10, n)
    frame = pd.DataFrame(
        {
            "x": np.concatenate([left_x, right_x]),
            "y": np.arange(2 * n, dtype=float),
            # Deliberately correlated with the selection so losing the
            # selection creates a second visible facet.
            "facet": np.concatenate(
                [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
            ),
        }
    )
    d = DFDraw(frame)

    try:
        _, _, scalar = d.profile(
            "y:x",
            bins=6,
            return_data=True,
            selection="x < 0",
            facet_by="facet",
        )
        _, _, singleton = d.profile(
            "y:x",
            bins=6,
            return_data=True,
            selection_vector=["x < 0"],
            facet_by="facet",
        )
    finally:
        plt.close("all")

    # Independent fixture truth.
    assert int((frame["x"] < 0).sum()) == n

    # Scalar selection is the positive control.
    assert scalar["n_total"] == n
    assert scalar["groups"] == [0]

    # AD-67 requires the singleton-vector spelling to be semantically identical.
    assert singleton["n_total"] == scalar["n_total"]
    assert singleton["groups"] == scalar["groups"]


def test_B01_weights_singleton_matches_scalar_weights():
    """BUG-01: one-element weights_vector must keep the scalar weight effect."""
    n = 200
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, n),
            "y": np.concatenate(
                [np.zeros(n // 2), np.full(n // 2, 10.0)]
            ),
            "w": np.concatenate(
                [np.ones(n // 2), np.full(n // 2, 10.0)]
            ),
        }
    )
    expected_mean = float(np.average(frame["y"], weights=frame["w"]))
    expected_sum_weights = float(frame["w"].sum())

    d = DFDraw(frame)
    try:
        _, _, scalar = d.profile(
            "y:x",
            bins=1,
            range=(0.0, 1.0),
            return_data=True,
            weights="w",
        )
        _, _, singleton = d.profile(
            "y:x",
            bins=1,
            range=(0.0, 1.0),
            return_data=True,
            weights_vector=["w"],
        )
    finally:
        plt.close("all")

    scalar_row = scalar["profile_data"].iloc[0]
    singleton_row = singleton["profile_data"].iloc[0]

    np.testing.assert_allclose(
        scalar_row["y_mean"], expected_mean, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        scalar_row["sum_weights"], expected_sum_weights, rtol=0.0, atol=1e-12
    )

    # AD-67 requires the singleton-vector spelling to be the same weighted plot.
    np.testing.assert_allclose(
        singleton_row["y_mean"], scalar_row["y_mean"], rtol=0.0, atol=1e-12
    )
    assert "sum_weights" in singleton["profile_data"].columns
    np.testing.assert_allclose(
        singleton_row["sum_weights"],
        scalar_row["sum_weights"],
        rtol=0.0,
        atol=1e-12,
    )



def test_B01_selection_singleton_composes_with_global_selection():
    """AD-66: global selection AND singleton selection_vector[0]."""
    frame = pd.DataFrame({
        "x": np.linspace(-2.0, 2.0, 401),
        "y": np.linspace(0.0, 1.0, 401),
    })
    d = DFDraw(frame)
    scalar_selection = "(x > -1.5) & (x < -0.5)"
    try:
        _, _, scalar = d.profile(
            "y:x", bins=5, return_data=True,
            selection=scalar_selection,
        )
        _, _, singleton = d.profile(
            "y:x", bins=5, return_data=True,
            selection="x > -1.5",
            selection_vector=["x < -0.5"],
        )
    finally:
        plt.close("all")

    expected = int(((frame["x"] > -1.5) & (frame["x"] < -0.5)).sum())
    assert scalar["n"] == expected
    assert singleton["n"] == expected


def test_B01_weights_singleton_multiplies_global_weights():
    """AD-66: global weights * singleton weights_vector[0]."""
    n = 200
    frame = pd.DataFrame({
        "x": np.linspace(0.0, 1.0, n),
        "y": np.concatenate([np.zeros(n // 2), np.full(n // 2, 10.0)]),
        "w0": np.linspace(1.0, 2.0, n),
        "w1": np.linspace(2.0, 5.0, n),
    })
    d = DFDraw(frame)
    try:
        _, _, scalar = d.profile(
            "y:x", bins=1, range=(0.0, 1.0), return_data=True,
            weights="w0 * w1",
        )
        _, _, singleton = d.profile(
            "y:x", bins=1, range=(0.0, 1.0), return_data=True,
            weights="w0", weights_vector=["w1"],
        )
    finally:
        plt.close("all")

    s = scalar["profile_data"].iloc[0]
    v = singleton["profile_data"].iloc[0]
    expected_w = frame["w0"] * frame["w1"]
    expected_mean = float(np.average(frame["y"], weights=expected_w))
    expected_sum = float(expected_w.sum())
    np.testing.assert_allclose(s["y_mean"], expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v["y_mean"], expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s["sum_weights"], expected_sum, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v["sum_weights"], expected_sum, rtol=0, atol=1e-12)


def test_B01_singleton_selection_hist_matches_scalar_selection():
    """Singleton lowering is shared by histogram, not patched only in profile."""
    frame = pd.DataFrame({"x": np.arange(20.0)})
    d = DFDraw(frame)
    try:
        _, _, scalar = d.hist("x", bins=4, selection="x < 7")
        _, _, singleton = d.hist("x", bins=4, selection_vector=["x < 7"])
    finally:
        plt.close("all")
    assert scalar["n"] == 7
    assert singleton["n"] == scalar["n"]


def test_B01_empty_selection_vector_still_refuses():
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})
    d = DFDraw(frame)
    import pytest
    with pytest.raises(ValueError, match="selection_vector must be non-empty"):
        d.profile("y:x", selection_vector=[])


def test_B01_singleton_selection_label_refuses_explicitly():
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})
    d = DFDraw(frame)
    import pytest
    with pytest.raises(ValueError, match="selection_labels is not applicable"):
        d.profile(
            "y:x",
            selection_vector=["x >= 0"],
            selection_labels=["kept"],
        )

def test_B05_draw_bracket_vector_delta_matches_typed_profile():
    """BUG-05: draw(type='profile') must reach the existing normalize owner."""
    n = 240
    x = np.linspace(-3.0, 3.0, n)
    y_ref = 0.5 * x - 2.0
    expected_delta = 1.25
    y_sig = y_ref + expected_delta

    frame = pd.DataFrame({"x": x, "y_sig": y_sig, "y_ref": y_ref})
    d = DFDraw(frame)

    try:
        _, _, typed = d.profile(
            "[y_sig,y_ref]:x",
            bins=12,
            range=(-3.0, 3.0),
            normalize="delta",
        )
        _, _, generic = d.draw(
            "[y_sig,y_ref]:x",
            type="profile",
            bins=12,
            range=(-3.0, 3.0),
            normalize="delta",
        )
    finally:
        plt.close("all")

    # Independent numerical truth first.
    assert isinstance(typed, dict)
    assert typed["normalize_mode"] == "delta"
    typed_nd = typed["normalize_data"]
    valid = ~typed_nd["mask_undefined"].to_numpy(dtype=bool)
    assert valid.any()
    np.testing.assert_allclose(
        typed_nd.loc[valid, "value"].to_numpy(),
        expected_delta,
        rtol=0.0,
        atol=1e-12,
    )

    # Door-equivalence contract: draw() must expose the same normalized result.
    assert isinstance(generic, dict)
    assert generic["normalize_mode"] == typed["normalize_mode"]
    pd.testing.assert_frame_equal(
        generic["normalize_data"].reset_index(drop=True),
        typed["normalize_data"].reset_index(drop=True),
    )


def test_B05_draw_bracket_vector_ratio_matches_typed_profile():
    """BUG-05 neighbor: ratio uses the same typed normalize owner."""
    n = 240
    x = np.linspace(-3.0, 3.0, n)
    y_ref = 2.0 + 0.1 * x
    expected_ratio = 1.5
    y_sig = expected_ratio * y_ref
    frame = pd.DataFrame({"x": x, "y_sig": y_sig, "y_ref": y_ref})
    d = DFDraw(frame)

    try:
        _, _, typed = d.profile(
            "[y_sig,y_ref]:x", bins=12, range=(-3.0, 3.0), normalize="ratio"
        )
        _, _, generic = d.draw(
            "[y_sig,y_ref]:x", type="profile", bins=12,
            range=(-3.0, 3.0), normalize="ratio"
        )
    finally:
        plt.close("all")

    assert typed["normalize_mode"] == "ratio"
    valid = ~typed["normalize_data"]["mask_undefined"].to_numpy(dtype=bool)
    np.testing.assert_allclose(
        typed["normalize_data"].loc[valid, "value"].to_numpy(),
        expected_ratio, rtol=0.0, atol=1e-12,
    )
    assert generic["normalize_mode"] == "ratio"
    pd.testing.assert_frame_equal(
        generic["normalize_data"].reset_index(drop=True),
        typed["normalize_data"].reset_index(drop=True),
    )


def test_B05_draw_bracket_vector_callable_matches_typed_profile():
    """BUG-05 neighbor: callable normalize is routed to the typed owner too."""
    n = 240
    x = np.linspace(-3.0, 3.0, n)
    y_ref = 1.0 + 0.2 * x
    y_sig = y_ref + 0.75
    frame = pd.DataFrame({"x": x, "y_sig": y_sig, "y_ref": y_ref})
    d = DFDraw(frame)

    def custom(S):
        return S[0] - S[1]

    try:
        _, _, typed = d.profile(
            "[y_sig,y_ref]:x", bins=12, range=(-3.0, 3.0), normalize=custom
        )
        _, _, generic = d.draw(
            "[y_sig,y_ref]:x", type="profile", bins=12,
            range=(-3.0, 3.0), normalize=custom
        )
    finally:
        plt.close("all")

    assert typed["normalize_mode"] == "callable"
    assert generic["normalize_mode"] == "callable"
    pd.testing.assert_frame_equal(
        generic["normalize_data"].reset_index(drop=True),
        typed["normalize_data"].reset_index(drop=True),
    )


def test_B05_no_normalize_bracket_vector_draw_keeps_existing_vector_shape():
    """BUG-05 negative neighbor: no-normalize draw stays on generic vector dispatch."""
    n = 120
    x = np.linspace(-1.0, 1.0, n)
    frame = pd.DataFrame({"x": x, "a": x + 1.0, "b": 2.0 * x - 1.0})
    d = DFDraw(frame)

    try:
        _, _, generic = d.draw(
            "[a,b]:x", type="profile", bins=8, range=(-1.0, 1.0)
        )
    finally:
        plt.close("all")

    assert isinstance(generic, list)
    assert len(generic) == 2

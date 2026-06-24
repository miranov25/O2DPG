"""
Phase 13.61.ADF — A1 peak-RSS regression gate (R4-C1, §7.3).

Measures the peak resident-memory bump (resource ru_maxrss high-water mark) from a
single draw() that triggers the group_by-expression copy. With D-ADF-1, df_subset
is projected before that copy, so the bump is a few columns instead of the whole
(10-100 GB) frame.

Run in a SUBPROCESS so the high-water mark is clean: ru_maxrss is monotonic, so an
in-process measurement under `pytest -n` can false-pass when a prior test on the
same worker already raised the peak. The child inherits the parent's sys.path so it
resolves dfextensions/AliasDataFrame exactly as pytest did (no PYTHONPATH guess).

Regression property (FM#12, via public draw()): pre-fix df_subset = self.df, so the
copy is the FULL frame and the bump ~= frame size -> FAILS. With the projection the
bump is a few columns -> well under the scale-aware threshold.

Marked 'memory': opt-in. On alma2: pytest tests/test_phase1361_a1_memory.py -m memory
(register `markers = memory` in pytest.ini to silence the unknown-mark warning).
"""
import os, sys, subprocess, textwrap
import pytest

N_ROWS = 1_000_000
N_DECOY = 64                 # frame ~= (N_DECOY+3) cols * N_ROWS * 8 bytes ~= 540 MB
THRESHOLD_FRAC = 0.4         # bump must be < 40% of frame size; full-frame copy ~= 100%

_CHILD = textwrap.dedent(r"""
    import sys, resource, gc, warnings
    import numpy as np, pandas as pd
    warnings.simplefilter("ignore")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from AliasDataFrame import AliasDataFrame
    N_ROWS, N_DECOY = {n_rows}, {n_decoy}
    def peak_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
    np.random.seed(0)
    base = {{"pt": np.random.rand(N_ROWS)+0.5, "tgl": np.random.randn(N_ROWS),
             "sector": np.random.randint(0,18,N_ROWS)}}
    for i in range(N_DECOY): base["decoy%d"%i] = np.random.randn(N_ROWS)
    adf = AliasDataFrame(pd.DataFrame(base)); adf.add_alias("qpt","1/pt")
    adf.materialize_aliases(names=["qpt"]); gc.collect()
    n_cols = adf.df.shape[1]; frame_mb = N_ROWS*n_cols*8/1e6
    base_peak = peak_mb()
    adf.draw(expr="qpt", group_by="pt>1", type="hist")
    plt.close("all"); gc.collect()
    print("FRAME_MB=%.1f" % frame_mb)
    print("DELTA_MB=%.1f" % (peak_mb()-base_peak))
""")


@pytest.mark.memory
@pytest.mark.invariance
def test_draw_path_peak_rss_bounded():
    script = _CHILD.format(n_rows=N_ROWS, n_decoy=N_DECOY)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    vals = dict(l.split("=", 1) for l in out.stdout.splitlines() if "=" in l)
    frame_mb = float(vals["FRAME_MB"]); delta = float(vals["DELTA_MB"])
    limit = frame_mb * THRESHOLD_FRAC
    assert delta < limit, (
        f"draw-path peak-RSS bump {delta:.0f}MB exceeds {limit:.0f}MB "
        f"({THRESHOLD_FRAC:.0%} of {frame_mb:.0f}MB frame) — projection likely "
        f"reverted (full-frame copy)."
    )

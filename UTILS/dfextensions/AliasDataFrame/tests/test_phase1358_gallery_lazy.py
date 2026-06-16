"""
Phase 13.58.ADF — D4 / AC-1 / AC-2 gallery lazy-vs-eager double-run, as a pytest test.

This drives examples/time_series/time_series_draw.validate_lazy_vs_eager() against the
TIME-SERIES production file (the large file with the timeMS/sector schema the gallery's
build_adf expects) — NOT calibITS. calibITS has a different schema and its own real-data
test (test_phase1358_lazy_calibITS.py). Needs the time-series ROOT file + dfdraw; SKIPS
cleanly when absent. Point it at the time-series file:

    ADF_TS_ROOT=/path/to/timeseries.root ADF_TS_TREE=<tree> \
        python -m pytest tests/test_phase1358_gallery_lazy.py -v 2>&1 | tee gallery.log
"""
import os
import sys
import importlib

import pytest

GALLERY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "time_series",
)
ROOT_FILE = os.environ.get("ADF_TS_ROOT")
TREE = os.environ.get("ADF_TS_TREE", "tree")


@pytest.mark.skipif(
    not ROOT_FILE,
    reason="set ADF_TS_ROOT=<gallery .root file> (and optional ADF_TS_TREE) to run the "
           "D4 gallery double-run",
)
def test_AC1_AC2_gallery_lazy_vs_eager():
    """AC-1 (clean + genuinely lazy) and AC-2 (lazy==eager stats) on the real gallery."""
    if not os.path.exists(ROOT_FILE):
        pytest.skip(f"ADF_TS_ROOT not found: {ROOT_FILE}")
    sys.path.insert(0, GALLERY_DIR)
    try:
        tsd = importlib.import_module("time_series_draw")
    except Exception as e:  # dfdraw / perfmonitor / env not present
        pytest.skip(f"gallery import failed (dfdraw env): {e}")
    tsd.validate_lazy_vs_eager(ROOT_FILE, tree_name=TREE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

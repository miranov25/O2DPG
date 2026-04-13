"""pytest conftest at the cpp/ level.

Placed here (NOT in cpp/tests/) so pytest loads it while descending
from the repo-level rootdir (where pytest.ini lives) into cpp/tests/.
This ensures sys.path contains cpp/ BEFORE any test module is imported,
making `from dfGB_to_root import ...` resolve correctly.

Required because the repo has a top-level pytest.ini that owns rootdir,
and the cpp/ subproject is not a Python package (no __init__.py).
"""
import sys
from pathlib import Path

CPP_ROOT = Path(__file__).resolve().parent
if str(CPP_ROOT) not in sys.path:
    sys.path.insert(0, str(CPP_ROOT))

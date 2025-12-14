"""
Pytest configuration for dfdraw tests.

Adds the parent directory to sys.path so that 'from dfdraw import ...'
works when running tests from the dfdraw directory.
"""

import sys
from pathlib import Path

# Add the parent directory (dfdraw/) to sys.path
# This allows 'from dfdraw import ...' to work
dfdraw_root = Path(__file__).parent.parent
if str(dfdraw_root) not in sys.path:
    sys.path.insert(0, str(dfdraw_root))

# Also add the grandparent (dfextensions/) for AliasDataFrame imports
dfextensions_root = dfdraw_root.parent
if str(dfextensions_root) not in sys.path:
    sys.path.insert(0, str(dfextensions_root))

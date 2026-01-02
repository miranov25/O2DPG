"""Pytest config - adds UTILS to path for development."""
import sys
from pathlib import Path
UTILS_DIR = Path(__file__).parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

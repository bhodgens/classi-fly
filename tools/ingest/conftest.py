"""Make sibling ingest modules importable when tests run from any cwd."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

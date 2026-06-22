"""
Main entry point for the FFP optimization methods project.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ffp.cli import main

if __name__ == "__main__":
    main()

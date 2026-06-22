"""Uproszczony punkt wejscia: `python run.py` (lub `poetry run python run.py`).

Dziala takze bez `poetry install` - dodaje katalog src/ do sciezki importow.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ffp.cli import main

if __name__ == "__main__":
    main()

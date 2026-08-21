"""Canonical project paths.

Import these instead of hard-coding relative paths, so notebooks and scripts
resolve data the same way regardless of the working directory they run from.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

NOTEBOOKS = ROOT / "notebooks"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist yet."""
    for path in (INTERIM, PROCESSED, OUTPUTS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)

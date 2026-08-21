"""Canonical project paths.

Import these instead of hard-coding relative paths, so notebooks and scripts
resolve data the same way regardless of the working directory they run from.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

MODELS = ROOT / "models"  # HF weight cache - large, gitignored
RESULTS = ROOT / "results"  # every output, in a per-model subfolder

NOTEBOOKS = ROOT / "notebooks"


def model_slug(model_id: str) -> str:
    """Filesystem-safe folder name for a model id or Ollama tag.

    ``google/gemma-3-1b-it`` -> ``google_gemma-3-1b-it``
    ``gemma3:1b``            -> ``gemma3_1b``
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_id).strip()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a folder name from {model_id!r}")
    return slug


def results_dir(model_id: str, *, create: bool = True) -> Path:
    """Where results for one model belong: ``results/<model-slug>/``.

    Every output - figure, table, JSON, checkpoint - goes here, so results are
    never mixed between models and a stale chart cannot be mistaken for a new
    one. Pair it with :func:`research.models.load` on the same ``MODEL_ID``.
    """
    path = RESULTS / model_slug(model_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    """Create the writable directories if they do not exist yet."""
    for path in (INTERIM, PROCESSED, MODELS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)

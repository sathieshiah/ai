"""Research project package.

Every model used by this project lives in ``<project>/models`` on D:. Importing
this package enforces that by pointing the Hugging Face cache at
``research.paths.MODELS``, so nothing is downloaded to — or read from — the
default profile cache on C:.

The pin is **forced, not a default**: a stray ``HF_HUB_CACHE`` in the ambient
environment must not be able to redirect weights back outside the project.

It must run before ``huggingface_hub`` is imported, because that module reads
the cache location once at import time. In practice: import ``research`` (or
anything from it) before ``transformers`` or ``huggingface_hub``. If that order
is violated, :func:`research.models.load` raises rather than silently reading
from the wrong cache.
"""

import os

from .paths import MODELS

__version__ = "0.1.0"

MODELS.mkdir(parents=True, exist_ok=True)

# Forced assignment, not setdefault - see module docstring.
os.environ["HF_HUB_CACHE"] = str(MODELS)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS)  # legacy name, still read by older code

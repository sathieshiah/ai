"""Research project package.

Importing this package pins the Hugging Face cache to the project's own
``models/`` directory on D:, so weights land on the large drive rather than in
the default location under the user profile on C:.

This must happen before ``huggingface_hub`` is imported, because that module
reads the cache location once at import time. In practice: import ``research``
(or anything from it) before ``transformers`` or ``huggingface_hub``.

An existing ``HF_HUB_CACHE`` in the environment always wins, so this can be
overridden without editing code.
"""

import os

from .paths import MODELS

__version__ = "0.1.0"

MODELS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", str(MODELS))

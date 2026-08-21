"""Access model weights, layers, and activations.

TransformerLens and nnsight have no Windows-ARM64 wheels (both pull in
`datasets` -> `pyarrow`, which ships none), so this module provides the same
core capabilities directly on top of PyTorch hooks and safetensors.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import MODELS


class ExternalModelError(RuntimeError):
    """Raised when a model would be read from outside the project's models/."""


def assert_cache_is_local() -> None:
    """Fail if huggingface_hub is not pointed at this project's models/.

    This happens when ``transformers`` or ``huggingface_hub`` was imported
    before ``research``: the cache location is read once at import time, so the
    pin set in ``research/__init__.py`` arrives too late and downloads would
    silently land in the profile cache on C:.
    """
    import huggingface_hub.constants as hub_constants

    resolved = Path(hub_constants.HF_HUB_CACHE)
    if resolved != MODELS:
        raise ExternalModelError(
            f"Hugging Face cache resolves to {resolved}, not {MODELS}. "
            "Import `research` before `transformers` / `huggingface_hub` - "
            "the cache location is read once, at import time."
        )


def assert_is_local(model_id: str | Path) -> None:
    """Fail if a filesystem path points outside the project's models/.

    A bare repo id (``'gpt2'``, ``'google/gemma-3-1b-it'``) is fine: it resolves
    through the pinned cache. An absolute path to weights elsewhere on the
    machine is not - this project keeps every model under ``models/``.
    """
    path = Path(model_id)
    if not path.exists():
        return  # a hub repo id, not a local path

    resolved = path.resolve()
    if resolved != MODELS and MODELS not in resolved.parents:
        raise ExternalModelError(
            f"{resolved} is outside {MODELS}. Models used by this project must "
            "live under models/. Re-download the weights by repo id instead of "
            "pointing at an external copy."
        )


def load(model_id: str, dtype: torch.dtype = torch.float32, **kwargs):
    """Load a model and tokenizer onto the CPU, from this project's models/.

    Defaults to float32: this machine has no GPU, and CPU float16 matmul is
    slower than float32, not faster. Use bfloat16 only to halve memory when a
    model would otherwise not fit.

    Raises :class:`ExternalModelError` rather than reading weights from outside
    ``models/`` - including the pre-existing profile cache on C:.
    """
    assert_cache_is_local()
    assert_is_local(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODELS)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, cache_dir=MODELS, **kwargs
    )
    model.eval()
    return model, tokenizer


def layers(model, kind: type | None = None) -> list[tuple[str, torch.nn.Module]]:
    """Every named submodule, optionally filtered to one module class."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and (kind is None or isinstance(module, kind))
    ]


def layer_table(model) -> list[dict]:
    """One row per parameterised submodule: name, class, parameter count, shapes.

    Tied weights are counted once, against the module that owns them first.
    GPT-2, for instance, ties ``lm_head.weight`` to ``transformer.wte.weight``;
    counting both would report 163 M parameters for a 124 M model. Modules whose
    parameters are all tied elsewhere get ``params=0`` and ``tied=True``, so
    ``sum(row["params"])`` reconciles with ``sum(p.numel() for p in
    model.parameters())``.

    Returns plain dicts so the caller can hand it to pandas or print it raw.
    """
    rows = []
    seen: dict[int, str] = {}

    for name, module in model.named_modules():
        if not name:
            continue

        shapes = {}
        own = 0
        tied_to = None

        for param_name, param in module.named_parameters(recurse=False):
            shapes[param_name] = tuple(param.shape)
            key = param.data_ptr()
            if key in seen:
                tied_to = seen[key]
            else:
                seen[key] = name
                own += param.numel()

        if not shapes:
            continue

        rows.append(
            {
                "name": name,
                "class": type(module).__name__,
                "params": own,
                "tied": tied_to is not None,
                "tied_to": tied_to,
                "shapes": shapes,
            }
        )
    return rows


def weight(model, name: str) -> torch.Tensor:
    """Fetch one parameter tensor by its dotted name."""
    return dict(model.named_parameters())[name].detach()


@contextlib.contextmanager
def capture(model, names: Sequence[str]) -> Iterator[dict[str, torch.Tensor]]:
    """Capture the outputs of named submodules during a forward pass.

    >>> with capture(model, ["transformer.h.0"]) as acts:
    ...     model(**inputs)
    >>> acts["transformer.h.0"].shape
    """
    store: dict[str, torch.Tensor] = {}
    handles = []
    lookup = dict(model.named_modules())

    for name in names:
        if name not in lookup:
            raise KeyError(f"no submodule named {name!r}")

        def hook(_module, _inp, out, _name=name):
            store[_name] = (out[0] if isinstance(out, tuple) else out).detach()

        handles.append(lookup[name].register_forward_hook(hook))

    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def raw_tensors(model_id_or_path: str | Path) -> dict[str, tuple]:
    """Read tensor names, shapes and dtypes from safetensors without loading.

    Works on a local .safetensors file or on a directory of shards. Lets you
    inspect a model far larger than RAM, since nothing is materialised.
    """
    path = Path(model_id_or_path)
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {path}")

    out = {}
    for file in files:
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                slice_ = handle.get_slice(key)
                out[key] = (tuple(slice_.get_shape()), slice_.get_dtype(), file.name)
    return out


def read_tensor(file: str | Path, name: str) -> torch.Tensor:
    """Materialise exactly one tensor from a safetensors file."""
    with safe_open(Path(file), framework="pt") as handle:
        return handle.get_tensor(name)


def gguf_tensors(path: str | Path) -> list[dict]:
    """List tensors in a GGUF file — e.g. an Ollama blob under ~/.ollama.

    Lets you compare quantised Ollama weights against the original
    safetensors weights of the same model.
    """
    from gguf import GGUFReader

    reader = GGUFReader(Path(path))
    return [
        {
            "name": t.name,
            "shape": tuple(int(d) for d in t.shape),
            "quant": getattr(t.tensor_type, "name", str(t.tensor_type)),
            "n_elements": int(t.n_elements),
        }
        for t in reader.tensors
    ]


def ollama_blob(model_tag: str) -> Path:
    """Resolve an Ollama model tag to the GGUF blob holding its weights.

    >>> ollama_blob("gemma3:1b")
    """
    root = Path.home() / ".ollama" / "models"
    name, _, tag = model_tag.partition(":")
    tag = tag or "latest"
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest.is_file():
        raise FileNotFoundError(f"no Ollama manifest for {model_tag!r} at {manifest}")

    data = json.loads(manifest.read_text())
    for layer in data["layers"]:
        if layer["mediaType"].endswith("model"):
            digest = layer["digest"].replace(":", "-")
            return root / "blobs" / digest
    raise LookupError(f"no model layer in manifest for {model_tag!r}")


def blocks(model) -> list[tuple[str, torch.nn.Module]]:
    """The repeated transformer blocks, whatever this architecture calls them.

    Naming is not portable: GPT-2 uses ``transformer.h``, Llama/Gemma/Qwen use
    ``model.layers``, GPT-NeoX uses ``gpt_neox.layers``, OPT uses
    ``model.decoder.layers``. Rather than hard-coding any of those, this finds
    the longest ``ModuleList`` whose entries all share one class - which is what
    a stack of identical decoder blocks looks like structurally.

    Use this instead of writing a literal module path, so switching models
    stays a one-line change of model id.
    """
    best: list[tuple[str, torch.nn.Module]] = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        if len({type(child) for child in module}) != 1:
            continue
        if len(module) > len(best):
            best = [(f"{name}.{i}", child) for i, child in enumerate(module)]

    if not best:
        raise LookupError(
            "could not find a stack of transformer blocks; inspect "
            "layer_table(model) and address the modules directly"
        )
    return best


def block_names(model) -> list[str]:
    """Dotted names of the transformer blocks, in depth order."""
    return [name for name, _ in blocks(model)]


def find(model, pattern: str) -> list[tuple[str, torch.nn.Module]]:
    """Submodules whose dotted name matches a regex.

    Portable way to reach a role that every architecture gives a different
    name, e.g. ``find(model, "q_proj|c_attn")`` for the query projection.
    """
    regex = re.compile(pattern)
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and regex.search(name)
    ]

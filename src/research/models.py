"""Access model weights, layers, and activations.

TransformerLens and nnsight have no Windows-ARM64 wheels (both pull in
`datasets` -> `pyarrow`, which ships none), so this module provides the same
core capabilities directly on top of PyTorch hooks and safetensors.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import MODELS


def load(model_id: str, dtype: torch.dtype = torch.float32, **kwargs):
    """Load a model and tokenizer onto the CPU.

    Defaults to float32: this machine has no GPU, and CPU float16 matmul is
    slower than float32, not faster. Use bfloat16 only to halve memory when a
    model would otherwise not fit.
    """
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

# research

Research on model weights, layers, and activations — running locally on a
Snapdragon X (Windows ARM64) laptop.

## Layout

```
src/research/     importable package
  paths.py        canonical project paths
  models.py       weights / layers / activations access
notebooks/        exploratory notebooks
scripts/          runnable entry points
models/           HF weight cache (gitignored, large)
data/raw|interim|processed/    gitignored contents, tracked shape
tests/            pytest suite
docs/             findings and methodology
outputs/          generated figures (gitignored)
```

## Setup

This project runs on **native ARM64 Python 3.13** — not the emulated x64 build.
`torch` must come from the PyTorch index; PyPI ships no `win_arm64` wheel.

```bash
py -3.13-arm64 -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

## Working in Jupyter

Coding happens in notebooks. Launch JupyterLab from the project root:

```bash
.venv/Scripts/python -m jupyterlab
```

- **Select the `Python (research ARM64)` kernel.** The `python3` kernel on this
  machine is an emulated x64 interpreter with no PyTorch — it will fail on
  `import torch`.
- `notebooks/00-getting-started.ipynb` walks the whole workflow end to end:
  loading a model, listing layers, reading weights, capturing activations, and
  inspecting models too large for RAM. Outputs are kept in that one, so it reads
  as documentation without being run.
- `notebooks/_template.ipynb` is the starting point for new work.
- Every notebook opens with `import research` — before `transformers` — so the
  model cache is pinned to D:.

If the kernel is missing from the launcher, register it:

```bash
.venv/Scripts/python -m ipykernel install --user --name research --display-name "Python (research ARM64)"
```

## Checks

```bash
.venv/Scripts/python -m pytest -m "not slow"
```

```bash
.venv/Scripts/python -m ruff check .
```

## Where models are downloaded

All weights land in `models/` at the project root (on D:), never in the default
cache under your user profile on C:. `research/__init__.py` sets `HF_HUB_CACHE`
to do this, which means **import `research` before `transformers`**:

```python
import research                      # pins the cache - must come first
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")   # -> D:/research/models
```

Or skip the ordering concern entirely by using the helper:

```python
from research import models
model, tok = models.load("gpt2")
```

This is **enforced**, not merely defaulted. Loading a model from anywhere
else - including the pre-existing HF cache in your user profile on C: -
raises `ExternalModelError`:

```python
models.load(profile_cache / 'models--Qwen--Qwen3-0.6B')
# ExternalModelError: C:/Users/.../models--Qwen--Qwen3-0.6B is outside
#                     D:/research/models
```

Re-download by repo id instead. `models/` is gitignored; put `HF_TOKEN` in
`.env` for gated repos such as Gemma.

## Accessing weights and layers

```python
from research import models

model, tok = models.load("gpt2")           # cached under models/

# Every parameterised submodule, with shapes
for row in models.layer_table(model)[:5]:
    print(row["name"], row["class"], row["params"], row["shapes"])

# One weight tensor by dotted name
w = models.weight(model, "transformer.h.0.attn.c_attn.weight")   # (768, 2304)

# Activations, via forward hooks
import torch
with models.capture(model, ["transformer.h.0", "transformer.ln_f"]) as acts:
    with torch.no_grad():
        model(**tok("The capital of France is", return_tensors="pt"))
acts["transformer.h.0"].shape        # (1, 5, 768)
```

Inspect weights **without loading the model** — works on files larger than RAM,
since nothing is materialised:

```python
t = models.raw_tensors("models/models--gpt2/snapshots/<hash>")
t["h.0.attn.c_attn.weight"]          # ((768, 2304), 'F32', 'model.safetensors')
one = models.read_tensor(path, "h.0.attn.c_attn.weight")
```

Read the quantised GGUF weights Ollama already has on disk, to compare against
the original safetensors:

```python
blob = models.ollama_blob("gemma3:1b")
models.gguf_tensors(blob)[:3]
```

## Known limits on this machine

- **16 GB unified RAM, CPU only.** A model at float32 needs ~4 bytes/param, so
  the practical ceiling for a full torch load is roughly **1–3 B parameters**.
  Use `dtype=torch.bfloat16` to halve that when a model would not otherwise fit.
- **`datasets`, `transformer_lens`, and `nnsight` cannot be installed.** All
  three require `pyarrow`, which ships no `win_arm64` wheel. `models.capture()`
  covers the hook-based workflow those libraries wrap.
- See [docs/arm64-python-stack.md](docs/arm64-python-stack.md) for the full
  wheel-availability survey, and
  [docs/local-llm-on-snapdragon-x.md](docs/local-llm-on-snapdragon-x.md) for
  measured inference throughput.

## Conventions

- Anything reused by more than one notebook moves into `src/research/`.
- Resolve paths via `research.paths`, never with relative strings.
- Secrets live in `.env` (gitignored); document new keys in `.env.example`.

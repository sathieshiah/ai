# Python ML stack on Windows ARM64 — what installs and what does not

Surveyed 2026-08-21 against PyPI and the PyTorch index, on Snapdragon X /
Windows 11 ARM64. Every claim below was checked by querying the package index
for `win_arm64` wheels, not from memory.

## The interpreter trap

`python` on this machine's PATH was **Python 3.13.7 (AMD64)** — an x64 build
running under emulation, despite `platform.machine()` reporting `ARM64`. The
only reliable check is `sys.version`, which names the build architecture:

```python
import sys; print(sys.version)
# ...[MSC v.1944 64 bit (ARM64)]   <- native
# ...[MSC v.1944 64 bit (AMD64)]   <- emulated
```

Version choice is forced, and the window is narrow:

| Interpreter | torch available? |
|---|---|
| 3.14 ARM64 (was the machine default) | **No** — no cp314 `win_arm64` wheel exists |
| 3.13 x64 | Yes, torch 2.13.0 — but every op runs emulated |
| **3.13 ARM64** | **Yes, torch 2.9.1+cpu — native. This is the one to use.** |

So the stack is pinned to **Python 3.13 ARM64**, and `requires-python` is
capped at `<3.14` in `pyproject.toml` to stop an accidental upgrade.

## torch is not on PyPI for this platform

PyPI's `torch` 2.13.0 publishes `win_amd64` wheels only. Native ARM64 builds
live on the PyTorch index and stop at **2.9.1+cpu** (cp311/cp312/cp313):

```bash
pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Measured after install: **51.8 GFLOP/s** fp32 on a 1024³ matmul across 8 Oryon
cores, 8 threads.

## Wheel availability

| Package | `win_arm64` | Notes |
|---|---|---|
| numpy 2.5.2 | ✅ cp312/313/314 | |
| pandas 3.0.5 | ✅ cp311/312/313 | |
| safetensors 0.8.0 | ✅ abi3 | one wheel covers 3.10+ |
| tokenizers 0.23.1 | ✅ abi3 | |
| transformers 5.15.1 | n/a | pure Python (`py3-none-any`) |
| accelerate, einops, gguf | n/a | pure Python |
| **pyarrow** | ❌ **none, any version** | checked 22.0.0 → 25.0.1 |
| **fastparquet** | ❌ none | the usual pyarrow fallback is also unavailable |
| **datasets** | ❌ blocked | hard dependency on pyarrow |
| **transformer_lens** | ❌ blocked | pure Python, but requires `datasets` |
| **nnsight** | ❌ blocked | 25 wheels published, none `win_arm64`; sdist only |

## Consequence: no TransformerLens, no nnsight

Both mechanistic-interpretability wrappers are unreachable, and installing
`transformer_lens --no-deps` does not help — it imports `datasets` at module
load and fails immediately.

This costs less than it appears. Both libraries wrap PyTorch's own
`register_forward_hook`, which works natively. `research.models.capture()`
provides the equivalent context-manager API:

```python
with models.capture(model, ["transformer.h.0", "transformer.h.11.mlp"]) as acts:
    model(**inputs)
```

What is genuinely lost: TransformerLens's weight-folding and its normalised
`HookedTransformer` naming scheme, so cross-architecture code must address
modules by their native HF names.

## Working around the missing `datasets`

Parquet is unreadable without pyarrow *or* fastparquet. For evaluation corpora,
prefer formats that need neither:

- `huggingface_hub.hf_hub_download()` for raw JSON / JSONL / text files
- `pandas.read_json(..., lines=True)` and `pandas.read_csv()`
- Plain `json` for small local benchmark sets

## Verified working end to end

Loading `gpt2` (124.4 M params, float32, 56.5 s cold): 76 parameterised
modules enumerated, `transformer.h.0.attn.c_attn.weight` read as (768, 2304)
with mean +0.00005 / std 0.19962, activations captured at four depths, and a
correct next-token prediction. Lazy safetensors reads listed all 160 tensors
without materialising any.

GGUF reading against Ollama's own blobs also works, and immediately shows that
`gemma3:1b`'s nominal "Q4" is a mixed-precision layout:

| Type | Tensors |
|---|---|
| F32 | 157 (norms) |
| Q5_0 | 117 |
| Q4_K | 39 |
| Q8_0 | 14 (token embeddings) |
| Q6_K | 13 |

Note that reading a 815 MB GGUF took over 5 minutes — budget for it, or cache
the tensor listing.

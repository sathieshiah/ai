# research

Beam search over recent open-weight language models, run on a Colab GPU.

The deliverable is a finding, not an app. The whole experiment lives in one
self-sufficient notebook; this repo holds that notebook, the tests that execute
its code, and the write-ups.

## Layout

```
notebooks/cloud/naive-beam-search-colab.ipynb   the experiment
tests/test_beam.py                              runs the notebook's own cells
results/<model>/                                every output, split per model
docs/                                           findings and methodology
data/raw|interim|processed/                     gitignored contents, tracked shape
models/                                         HF weight cache (gitignored, large)
```

There is no `src/` package. The notebook inlines everything it needs, and the
tests read the notebook — see **Self-sufficiency** below.

## Setup

The local environment exists to run the **tests and the linter**. The experiment
itself does not run here; it runs on Colab.

This project needs **native ARM64 Python 3.13** — not the emulated x64 build also
on PATH, and not 3.14 (no `torch` wheel for cp314). PyPI ships no `win_arm64`
torch wheel either, so `pyproject.toml` routes `torch` to the PyTorch CPU index.
That mapping is uv-specific: a plain `pip install` would resolve torch from PyPI
and fail.

```bash
uv sync --extra dev
```

That creates `.venv` and installs the locked versions. The project is **not** an
installable package (`[tool.uv] package = false`), so nothing is built — there is
no `research` module to import.

`uv sync` takes about **7 seconds** on NTFS, because uv hardlinks from its cache
instead of copying. On the exFAT volume this project used to live on, which
supports neither symlinks nor hardlinks, the same command took **19 minutes**. If
it is ever slow again, check the filesystem before anything else.

## Checks

```bash
.venv/Scripts/python -m pytest        # 46 tests
.venv/Scripts/python -m ruff check .
```

The suite is pure CPU and needs no weights: it executes the notebook's beam
search against a four-token toy model whose every step can be worked out by
hand, and checks the result against brute force.

## The experiment

`notebooks/cloud/naive-beam-search-colab.ipynb` runs a beam search on one
open-ended comic prompt across **seven architectures**, and compares each result
against its own greedy baseline. Width 10, 100 tokens, top-10 candidates per
beam, seed 0. Every selection decision is traced, which is the point: the trace
is what `beam_trace.csv` and `token-level-probabilities.csv` hold.

It walks a registry of models in order — download, load to GPU, search, save
under `results/<model-slug>/`, free VRAM, **delete the weights**, next model.

**The search runs without a KV cache**, and that is deliberate. Four of the seven
models are hybrid SSM/attention architectures whose recurrent state cannot be
permuted when beam search re-parents its hypotheses, so a cache would apply to
three models out of seven — a confound rather than an optimisation. The cost is
O(n²) in generated length: at 100 tokens, roughly 40× the work of 10 rather than
10×. See [docs/beam-search-without-a-kv-cache.md](docs/beam-search-without-a-kv-cache.md).

### Running it

Upload the notebook to Colab, set *Runtime > Change runtime type > A100*, and run
top to bottom.

| Runtime | VRAM | Models that run | Download |
|---|---|---|---|
| T4 | 15 GB | **none** — and no bfloat16, so it falls back to float16 | — |
| L4 | 22.5 GB | the first five, up to Zaya1-8B | ~75 GiB |
| A100 | 40 GB | **all seven** | ~127 GiB |

- **A T4 runs nothing.** The budget is VRAM minus 2.5 GiB of headroom for
  activations and the logits tensor, and the smallest model needs 12.9 GiB
  against a T4's 12.5. A fit check prints the verdict per model *before* anything
  downloads.
- **An L4 drops the control.** `Mistral-Nemo-Base-2407` is the plain dense
  transformer the comparison is measured against; without it the remaining six
  are much harder to interpret.
- **~127 GiB of downloads.** Peak disk stays near 31 GiB because each model's
  weights are deleted after use — that deletion is load-bearing, not
  housekeeping, and happens even when a download fails part-way.
- **Results go to Drive**, not to Colab's ephemeral disk. The sweep is resumable:
  a model whose `summary-row.json` is already on Drive is skipped, so a
  disconnect costs only the model in flight.
- **No repo is needed on Colab and none is gated.** All seven repos load with
  `trust_remote_code` off; the notebook depends on nothing in this repository.
- **Budget for the runtime.** With no cache the search is quadratic in generated
  length, so 100 tokens per model across seven models is the dominant cost after
  the downloads. The sweep is resumable, so a disconnect is survivable.

## Self-sufficiency

The notebook is standalone on purpose. Read it top to bottom and it works: the
algorithm lives in the notebook, not behind an import. There is no `research`
package to install on Colab, and `pyproject.toml` pins a **CPU-only** torch index
that would silently install the wrong build on a GPU box.

Self-sufficiency normally costs test coverage. It does not here: the cells
carrying the algorithm are tagged,

```python
# cell metadata: {"tags": ["beam-search-implementation"]}
```

and `tests/test_beam.py` reads those cells out of the `.ipynb`, `exec`s them, and
tests what the notebook actually runs — not a copy that could drift.

## Results

Every output lands in `results/`, in a subfolder named for the model, so results
are never mixed between models. The notebook derives that folder from the model
id rather than typing it, which is what stops a model switch from overwriting
another model's data.

Results are tracked in git — they are the deliverable. Tensors and checkpoints
under `results/` are gitignored.

> **The beam-search results currently in `results/` are superseded.** They come
> from a first run made before the chat template was applied and before the
> vision-language models were dropped, so instruction-tuned models completed the
> prompt instead of answering it. They are kept as the evidence for why the
> notebook is built the way it is. The `tensor-manifest.csv` files are unrelated
> and still current — they belong to the weight-layout comparison in `docs/`.

## Known limits on this machine

Snapdragon X, Windows ARM64, 16 GB unified RAM, **CPU only**.

- **No local inference at this scale.** Measured on Falcon-H1R-7B at bfloat16, a
  single batched forward pass took **376 s**, and a warm pass was no faster than
  a cold one: 14.1 GiB of weights cannot stay cached in ~8 GiB, so every pass
  re-streams the model off D: at ~38 MB/s. This is why the experiment is on
  Colab.
- **`datasets`, `transformer_lens` and `nnsight` cannot be installed.** All three
  need `pyarrow`, which ships no `win_arm64` wheel at any version.
- **`models/` is a directory junction onto `D:/research/models`** and still holds
  weights from earlier work. Nothing in this repo reads it any more — the cache
  guard was removed with `src/research/`, so a local `from_pretrained` now falls
  back to the profile cache under `~/.cache/huggingface` unless you set
  `HF_HUB_CACHE` yourself. Recreate the junction with
  `mklink /J C:\research\models D:\research\models`; do not delete
  `D:/research/models`, it is the real folder rather than a copy.

## Docs

- [docs/beam-search-without-a-kv-cache.md](docs/beam-search-without-a-kv-cache.md)
  — why the search has no KV cache, and which architectures cannot support one
- [docs/candidate-models-7b-2026.md](docs/candidate-models-7b-2026.md) — how the
  model shortlist was chosen
- [docs/weight-layout-comparison.md](docs/weight-layout-comparison.md) — weight
  layouts across the shortlisted architectures
- [docs/arm64-python-stack.md](docs/arm64-python-stack.md) — what installs on
  ARM64 and what does not
- [docs/local-llm-on-snapdragon-x.md](docs/local-llm-on-snapdragon-x.md) —
  measured local inference throughput

Secrets live in `.env` (gitignored); document new keys in `.env.example`.

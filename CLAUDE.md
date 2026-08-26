# Project conventions (research)

## What this project is
A Python research project. The output is a **finding**, not a shipped app.
Correctness of the analysis matters more than code elegance — but reproducibility
is non-negotiable: any result must be re-derivable by running code that lives in
this repo.

The whole experiment is one notebook,
`notebooks/cloud/naive-beam-search-colab.ipynb`, which runs on a Colab GPU. This
machine runs the tests and the linter; it does not run the experiment.

## Definition of done
- Tests green: `.venv/Scripts/python -m pytest`.
- `.venv/Scripts/python -m ruff check .` clean.
- **`README.md`** updated when layout, setup, or how-to-run changes.
- **`docs/`** updated when a finding, method, or data source changes — a result
  that exists only in a notebook cell is not finished.
- New dependency → added to `pyproject.toml`, not just pip-installed.
- New secret / config key → added to `.env.example` with a comment.

## The notebook is the working surface
Exploration and analysis belong in the notebook, with markdown cells explaining
what is being tested and why.

- **The notebook is self-sufficient, and this is load-bearing.** Run it top to
  bottom and it works on its own: the logic it demonstrates lives in the
  notebook, not behind an import. There is no `src/` package in this repo, and
  there is nothing to install on Colab — the notebook inlines the few helpers it
  needs. Do not "fix" that by extracting them.
- **Tag the cells that carry the algorithm** so the tests run the notebook's own
  code rather than a copy of it:

  ```python
  # cell metadata: {"tags": ["beam-search-implementation"]}
  ```

  `tests/test_beam.py` reads the tagged cells out of the `.ipynb`, `exec`s them,
  and tests what the notebook actually runs. That is what keeps self-sufficiency
  from costing test coverage. A new algorithm cell means a new tag and a new
  test.
- **State the brief at the top.** When the notebook implements a written
  specification, paste it verbatim into a markdown cell after the title (a
  collapsed `<details>` block keeps it out of the way), so the notebook records
  what it was asked to do, not just what it does.
- **Clear outputs before committing.** Long-running cells must say so in the
  markdown above them.
- **A finding that lives only in a cell is not finished** — write it up in
  `docs/` with the model id, revision, dtype and seed needed to reproduce it.

## Where the code runs — and it is two places
This is the single easiest thing to get wrong here.

**The notebook targets a Colab GPU.** It uses CUDA, bfloat16 and a KV cache
deliberately. Do not strip those out.

**The tests run locally on CPU, ARM64, with no GPU at all.** Never add `.cuda()`
or `device_map="auto"` to a test; there is no device to move to.

Other rules that still hold in both places:

- **Seed anything stochastic** (`torch.manual_seed`) and record the seed in the
  result, or the finding is not reproducible.
- **Wrap inference in `torch.no_grad()` / `@torch.inference_mode()`.** Nothing
  here trains; an autograd graph wastes memory that is already the constraint.
- **Check the parameter count against the memory budget before loading.** On
  Colab the notebook's fit check does this before anything downloads. Locally,
  assume nothing above ~1–3 B parameters will load at all.
- **CPU `float16` is slower than `float32`, not faster.** That is a CPU rule and
  does not apply on the GPU, where float16 is fine.

## Code must be model-agnostic
**Switching models is a text change, never a code change.** The model registry is
declared once, at the top, and nothing downstream hard-codes anything about it.

Module naming is *not* portable between architectures — GPT-2 uses
`transformer.h.N`, Llama/Gemma/Qwen use `model.layers.N`, GPT-NeoX uses
`gpt_neox.layers.N`. So:

- **Never write a literal module path.** Discover the block stack structurally —
  the notebook's `blocks()` takes the longest `ModuleList` whose children share a
  class — or reach a role by regex across naming conventions.
- **Use the portable accessors.** `model.get_input_embeddings()`, not
  `transformer.wte`. `model.config`, not a memorised hidden size.
- **Never hard-code a layer count, hidden size, vocab size, or head count.**
- **Never assume a chat template, BOS/EOS convention, or that a tokenizer has a
  pad token. Ask the tokenizer.** This has already cost one full run: without the
  template, instruction-tuned models *completed* the prompt instead of answering
  it, producing `The user says: "Write a funny story..."` on repeat. Base models
  have no template and must fall back to the raw string, so which happened is
  recorded per model — it changes what the numbers mean.
- **Never assume a model fits.** Check against the VRAM budget first.

The test for this: changing a model id to a different family should either work,
or fail with a clear error. It must never silently analyse the wrong tensors.

## Results go to `results/<model>/`
**Every output is written under `results/`, in a per-model subfolder.** Never to
the project root, never to `notebooks/`, never to a bare `outputs/`.

- **Always derive the folder from the model id**, never type the folder name.
  Hard-coding it means changing models silently overwrites another model's
  results with the wrong data — the exact failure the per-model split prevents.
- **Name files for what they contain**, not the notebook that made them:
  `token-level-probabilities.csv`, not `output1.csv`.
- **Results are tracked in git** — they are the deliverable. Keep them small
  (figures, tables, JSON); tensors and checkpoints are gitignored.
- A figure in `results/` still needs its finding written up in `docs/`.

The beam-search results currently in `results/` are **superseded** — they predate
the chat-template fix and the removal of the vision-language models. They are
kept as evidence for why the notebook is built as it is. Do not read them as
current findings. The `tensor-manifest.csv` files are a separate, still-valid
experiment.

## Data rules
- `data/raw/` is **immutable**. Never edit or overwrite files there; derive into
  `data/interim/` or `data/processed/`.
- `data/` contents are gitignored. If a dataset must be reproducible, commit the
  script that fetches it, not the file.
- Record the provenance of every raw dataset in `docs/` — where it came from,
  when it was pulled, and any licence or usage restriction.

## Platform constraints — read before adding a dependency
This machine is **Snapdragon X / Windows ARM64, 16 GB unified RAM, CPU only**.

1. **Python is pinned to 3.13 ARM64.** Not 3.14 (no `torch` cp314 wheel), not the
   x64 3.13 on PATH (runs emulated). Verify with `sys.version` — it names the
   build arch; `platform.machine()` reports the host and will lie to you.
2. **Check for a `win_arm64` wheel before adding any compiled dependency.**
   `pyarrow` has none at any version, which transitively blocks `datasets`,
   `transformer_lens` and `nnsight`.
3. **Symlink creation fails here.** NTFS raises `WinError 1314` because Windows
   gates symlinks behind elevation or Developer Mode. Nothing sets
   `HF_HUB_DISABLE_SYMLINKS` for you any more — if you download weights locally,
   set it yourself, or a large download fails at the *final* file, after every
   byte has already transferred.

`torch` is a normal entry in `pyproject.toml` (`torch==2.9.1`), but PyPI ships no
`win_arm64` wheel, so `[tool.uv.sources]` routes it to the `pytorch-cpu` index.
That mapping is uv-specific: plain `pip install` would resolve torch from PyPI
and fail. **Use `uv sync`.**

Note that index is **CPU-only**, which is one reason the notebook cannot depend
on this project's environment: installing it on a GPU box would silently give you
a CPU build.

Full survey: `docs/arm64-python-stack.md`.

## How things run (dev)
- Single venv at `.venv`, uv-managed and pinned by `uv.lock`: `uv sync --extra dev`.
  It is the only venv — do not create a second one alongside it.
- The project is **not** an installable package (`[tool.uv] package = false`).
  There is no `research` module; nothing imports from this repo.
- Tests: `.venv/Scripts/python -m pytest`
- Lint: `.venv/Scripts/python -m ruff check .`
- Editing the notebook locally: `.venv/Scripts/python -m jupyterlab`. It is a
  Colab notebook, so its GPU cells have nothing to run on here — edit and lint
  locally, execute on Colab.

## Docs index
`README.md` (layout, setup, how to run the sweep) ·
`notebooks/cloud/naive-beam-search-colab.ipynb` (the experiment) ·
`tests/test_beam.py` (executes the notebook's own tagged cells) ·
`docs/candidate-models-7b-2026.md` (how the shortlist was chosen) ·
`docs/weight-layout-comparison.md` (weight layouts across architectures) ·
`docs/arm64-python-stack.md` (what installs on ARM64) ·
`docs/local-llm-on-snapdragon-x.md` (measured local inference throughput)

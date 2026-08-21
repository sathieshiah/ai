# Project conventions (research)

## What this project is
A Python research / data-analysis project. Output is findings, not a shipped app.
Correctness of the analysis matters more than code elegance — but reproducibility
is non-negotiable: any result must be re-derivable from `data/raw` by running code
that lives in this repo.

## Definition of done
- Code + **tests** green: `.venv/Scripts/python -m pytest`.
- `.venv/Scripts/python -m ruff check .` clean.
- **`README.md`** updated when layout, setup, or how-to-run changes.
- **`docs/`** updated when a finding, method, or data source changes — a result
  that exists only in a notebook cell is not finished.
- New dependency → added to `pyproject.toml`, not just pip-installed.
- New secret / config key → added to `.env.example` with a comment.

## Data rules
- `data/raw/` is **immutable**. Never edit or overwrite files there; derive into
  `data/interim/` or `data/processed/` instead.
- `data/` contents are gitignored. If a dataset must be reproducible, commit the
  script that fetches it, not the file.
- Record the provenance of every raw dataset in `docs/` — where it came from,
  when it was pulled, and any licence or usage restriction.

## Code rules
- Resolve paths through `research.paths` (`RAW`, `INTERIM`, `PROCESSED`, ...).
  Never hard-code relative paths — notebooks and scripts run from different cwds.

## Notebooks are the working surface
**Do the coding in Jupyter.** Exploration, model probing, and analysis belong in
`notebooks/`, with markdown cells explaining what is being tested and why. Start
from `notebooks/_template.ipynb`; `notebooks/00-getting-started.ipynb` is the
worked example of the whole workflow.

```bash
.venv/Scripts/python -m jupyterlab
```

- **Kernel must be `Python (research ARM64)`.** The machine also exposes a
  `python3` kernel pointing at an emulated x64 interpreter with no PyTorch;
  selecting it yields confusing `ModuleNotFoundError`s. If the kernel is
  missing, re-register it:
  `.venv/Scripts/python -m ipykernel install --user --name research --display-name "Python (research ARM64)"`
- **First cell is always `import research`**, before `transformers` or
  `huggingface_hub` — it pins the model cache to D:. See "Getting models".
- **Name notebooks `NN-short-slug.ipynb`** so they sort in reading order.
- **Logic a second notebook needs graduates to `src/research/`** and gets a
  test. Copy-pasting a cell between notebooks is the signal to move it.
- **Clear outputs before committing**, unless the output *is* the documentation
  (`00-getting-started.ipynb` is the standing exception). Long-running cells
  must say so in the markdown above them.
- **A finding that lives only in a cell is not finished** — write it up in
  `docs/` with the model id, revision, dtype and seed needed to reproduce it.

## Getting models — always with `transformers`, always onto D:
Weights are fetched with `transformers` / `huggingface_hub` and land in
`models/` at the project root (`D:/research/models`). Never in the default
cache under the user profile on C:.

This is enforced by `research/__init__.py`, which sets `HF_HUB_CACHE` to
`research.paths.MODELS`. That works only if it runs before `huggingface_hub`
is imported, which gives one hard rule:

> **Import `research` before `transformers` or `huggingface_hub`.**

```python
import research                      # pins the cache — must come first
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")   # -> D:/research/models
```

Prefer the helper, which is correct regardless of import order:

```python
from research import models
model, tok = models.load("gpt2")     # passes cache_dir explicitly as well
```

- `models/` is gitignored. **Never commit weights.** Commit the model id and
  revision instead, so a result can be re-derived.
- Record every model used in `docs/` — id, revision/commit, dtype, and date
  pulled. "gemma3:1b" is not a provenance record; `google/gemma-3-1b-it` at a
  pinned revision is.
- Set `HF_TOKEN` in `.env` for gated repos and higher rate limits.
- An existing `HF_HUB_CACHE` in the environment overrides the pin, by design.

## Writing PyTorch code
- **There is no GPU.** Never write `.cuda()`, `.to("cuda")`, or
  `device_map="auto"`. There is no device to move to and these will fail or
  silently no-op. Everything is CPU; do not add device-juggling abstractions.
- **Default to `float32`.** CPU `float16` matmul is *slower* than `float32`,
  not faster. Reach for `bfloat16` only to halve memory when a model would
  otherwise not fit — and say so in a comment.
- **Check the parameter count against the memory budget before loading.**
  ~4 bytes/param at float32 caps a full load near 1–3 B parameters. For
  anything larger use `models.raw_tensors()` / `models.read_tensor()`, which
  read shapes and individual tensors without materialising the model.
- **Wrap inference in `torch.no_grad()`.** Nothing here trains; building an
  autograd graph wastes memory that is already the binding constraint.
- **Capture activations with `models.capture()`**, not hand-rolled hooks — it
  removes them on exit, including when the body raises. A leaked forward hook
  silently corrupts every later forward pass in the session.
- **Seed anything stochastic** (`torch.manual_seed`) and record the seed in the
  result, or the finding is not reproducible.

## Platform constraints — read before adding a dependency
This machine is **Snapdragon X / Windows ARM64, 16 GB unified RAM, CPU only**.
Two rules follow, and both have already bitten:

1. **Python is pinned to 3.13 ARM64.** Not 3.14 (no `torch` cp314 wheel), not
   the x64 3.13 on PATH (runs emulated). Verify with `sys.version` — it names
   the build arch; `platform.machine()` reports the host and will lie to you.
2. **Check for a `win_arm64` wheel before adding any compiled dependency.**
   `pyarrow` has none at any version, which transitively blocks `datasets`,
   `transformer_lens`, and `nnsight`. Use `models.capture()` for hook-based
   interpretability instead.

`torch` is deliberately absent from `pyproject.toml` dependencies — it is not
installable from PyPI here and must come from the PyTorch index.

Full survey: `docs/arm64-python-stack.md`.

## Memory budget
A float32 load costs ~4 bytes/param, so full `torch` loads are capped around
**1–3 B parameters**. Prefer `models.raw_tensors()` for inspecting anything
larger — it reads shapes and dtypes without materialising the weights.

## How things run (dev)
- Single venv at `.venv` built from `py -3.13-arm64`; the package is installed
  editable (`pip install -e ".[dev]"`), so `import research` works everywhere.
- Tests: `.venv/Scripts/python -m pytest -m "not slow"`
  (drop the marker filter to include tests that read multi-hundred-MB files)
- Lint: `.venv/Scripts/python -m ruff check .`
- Notebooks: `.venv/Scripts/python -m jupyterlab`

## Docs index
`README.md` (layout + setup) · `notebooks/00-getting-started.ipynb` (worked
example of the whole workflow) · `notebooks/_template.ipynb` (start here for new
work) · `docs/arm64-python-stack.md` (what installs on ARM64) ·
`docs/local-llm-on-snapdragon-x.md` (measured inference throughput).

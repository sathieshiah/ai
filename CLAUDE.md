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
- Logic that more than one notebook needs moves into `src/research/` and gets a
  test. Notebooks are for narrative and charts, not for the implementation.
- Notebooks are committed with outputs cleared unless a chart is the point.

## How things run (dev)
- Single venv at `.venv` in the project root; the package is installed editable
  (`pip install -e ".[dev]"`), so `import research` works everywhere.
- Tests: `.venv/Scripts/python -m pytest`
- Lint: `.venv/Scripts/python -m ruff check .`
- Notebooks: `.venv/Scripts/python -m jupyterlab`

## Docs index
`README.md` (layout + setup) · `docs/` (findings and methodology notes).

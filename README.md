# research

Research and data-analysis project.

## Layout

```
src/research/     importable package code (paths, loaders, analysis helpers)
notebooks/        exploratory notebooks — prose + charts, thin on logic
scripts/          runnable entry points (ETL, report generation)
data/raw/         immutable source data — never edited in place
data/interim/     intermediate artefacts
data/processed/   analysis-ready datasets
tests/            pytest suite
docs/             findings, methodology notes
outputs/          generated figures and reports (gitignored)
```

`data/` contents are gitignored; only the directory shape is tracked.

## Setup

```bash
cd /d/research && python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
```

## Run the tests

```bash
cd /d/research && .venv/Scripts/python -m pytest
```

## Start JupyterLab

```bash
cd /d/research && .venv/Scripts/python -m jupyterlab
```

## Conventions

- Anything reused by more than one notebook moves into `src/research/`.
- Resolve paths via `research.paths`, never with relative strings.
- Secrets live in `.env` (gitignored); document new keys in `.env.example`.

"""Guardrails for the conventions in CLAUDE.md, "Notebooks are the working surface".

These catch the two mistakes that cost the most time: a notebook saved against
the wrong kernel (the emulated x64 `python3`, which has no torch), and a
notebook that imports transformers before `research` has pinned the HF cache.
"""

import nbformat
import pytest

from research.paths import NOTEBOOKS

KERNEL = "research"
NOTEBOOK_FILES = sorted(NOTEBOOKS.glob("*.ipynb"))


def _read(path):
    return nbformat.read(path, as_version=4)


def _code_cells(nb):
    return [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]


def test_there_is_at_least_one_notebook():
    assert NOTEBOOK_FILES, f"no notebooks found under {NOTEBOOKS}"


@pytest.mark.parametrize("path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_notebook_uses_the_project_kernel(path):
    spec = _read(path).metadata.get("kernelspec", {})
    assert spec.get("name") == KERNEL, (
        f"{path.name} is saved against kernel {spec.get('name')!r}; "
        f"expected {KERNEL!r} (the native ARM64 venv)"
    )


@pytest.mark.parametrize("path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_notebook_imports_research_first(path):
    """`research` pins HF_HUB_CACHE, and huggingface_hub reads it at import time."""
    cells = _code_cells(_read(path))
    if not cells:
        pytest.skip("no code cells")

    first = cells[0].source
    assert "import research" in first, (
        f"{path.name}: first code cell must import research before anything else"
    )

    research_at = first.index("import research")
    for late in ("import transformers", "from transformers", "import huggingface_hub"):
        if late in first:
            assert first.index(late) > research_at, (
                f"{path.name}: {late!r} appears before `import research`"
            )


@pytest.mark.parametrize("path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_notebook_starts_with_a_markdown_heading(path):
    """Every notebook states what question it answers, before any code."""
    cells = _read(path).cells
    assert cells, f"{path.name} is empty"
    assert cells[0].cell_type == "markdown", f"{path.name} must open with markdown"
    assert cells[0].source.lstrip().startswith("#"), (
        f"{path.name} must open with a heading"
    )


@pytest.mark.parametrize("path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_every_code_cell_compiles(path):
    """Catch syntax errors without paying for a full notebook execution."""
    failures = []
    for i, cell in enumerate(_read(path).cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        try:
            compile(cell.source, f"{path.name}:cell{i}", "exec")
        except SyntaxError as exc:
            failures.append(f"cell {i}: {exc}")
    assert not failures, f"{path.name} has cells that do not parse:\n" + "\n".join(failures)


@pytest.mark.parametrize("path", NOTEBOOK_FILES, ids=lambda p: p.name)
def test_no_committed_error_outputs(path):
    """A notebook kept for its outputs must not ship a traceback.

    `nbconvert --execute` exits 0 even when a cell raises, so the exit code is
    not proof the notebook ran. This checks the outputs themselves.
    """
    errors = [
        (i, out.get("ename"), (out.get("evalue") or "")[:120])
        for i, cell in enumerate(_read(path).cells)
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors, f"{path.name} has error outputs: {errors}"

from research import paths


def test_root_contains_pyproject():
    assert (paths.ROOT / "pyproject.toml").is_file()


def test_data_dirs_are_under_root():
    for path in (paths.RAW, paths.INTERIM, paths.PROCESSED):
        assert paths.ROOT in path.parents


def test_ensure_dirs_is_idempotent(tmp_path, monkeypatch):
    for name in ("INTERIM", "PROCESSED", "MODELS", "RESULTS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    paths.ensure_dirs()
    paths.ensure_dirs()
    for name in ("interim", "processed", "models", "results"):
        assert (tmp_path / name).is_dir()


def test_results_is_a_sibling_of_data_under_root():
    assert paths.RESULTS.parent == paths.ROOT
    assert paths.RESULTS.name == "results"

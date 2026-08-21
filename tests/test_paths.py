from research import paths


def test_root_contains_pyproject():
    assert (paths.ROOT / "pyproject.toml").is_file()


def test_data_dirs_are_under_root():
    for path in (paths.RAW, paths.INTERIM, paths.PROCESSED):
        assert paths.ROOT in path.parents


def test_ensure_dirs_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "INTERIM", tmp_path / "interim")
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(paths, "FIGURES", tmp_path / "outputs" / "figures")
    paths.ensure_dirs()
    paths.ensure_dirs()
    assert (tmp_path / "outputs" / "figures").is_dir()

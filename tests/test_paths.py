"""Repository-layout and path-resolution checks."""

from pathlib import Path

from reviewerbrain import paths


def test_root_layout():
    assert (paths.ROOT / "README.md").exists()
    assert (paths.ROOT / "configs" / "rag" / "default.yaml").exists()
    assert (paths.ROOT / "pyproject.toml").exists()


def test_data_dirs():
    assert paths.RAW_DIR.is_dir()
    assert paths.PROCESSED_DATA_DIR.is_dir()
    for p in paths.RAW_FILES.values():
        assert p.exists(), f"missing raw dataset: {p}"
    for p in paths.CLEAN_FILES.values():
        assert p.exists(), f"missing cleaned dataset: {p}"
    assert paths.COMBINED_CLEAN_FILE.exists()


def test_index_dir():
    assert paths.CHROMA_V2_DIR.is_dir()
    assert (paths.CHROMA_V2_DIR / "chroma.sqlite3").exists()


def test_no_absolute_paths_in_source():
    """Guard: source files must not hard-code machine-specific paths."""
    banned = ("D:/Projects", "D:\\Projects", "C:\\Users", "/home/")
    search_dirs = [
        paths.ROOT / "src",
        paths.ROOT / "scripts",
        paths.ROOT / "tests",
        paths.ROOT / "configs",
    ]
    self_path = Path(__file__).resolve()
    for d in search_dirs:
        for f in d.rglob("*.py"):
            if f.resolve() == self_path:
                continue  # this file contains the patterns by definition
            text = f.read_text(encoding="utf-8")
            for b in banned:
                assert b not in text, f"{f} contains hard-coded path {b!r}"

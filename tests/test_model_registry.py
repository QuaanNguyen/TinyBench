"""Tests for model_registry — discovery of all available GGUF models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.services.model_registry import BENCH_MODEL_GGUF_STEMS, list_models


def _stub_models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "gguf"
    root.mkdir()
    for stem in BENCH_MODEL_GGUF_STEMS:
        (root / f"{stem}.gguf").write_bytes(b"")
    return root


class TestListModels:
    def test_discovers_all_models(self, tmp_path: Path):
        """Bench manifest must match every .gguf stem under MODELS_DIR."""
        stub = _stub_models_dir(tmp_path)
        with patch("app.services.model_registry.MODELS_DIR", stub):
            models = list_models()
        stems = {p.stem for p in models}
        assert stems == BENCH_MODEL_GGUF_STEMS

    def test_returns_path_objects(self, tmp_path: Path):
        stub = _stub_models_dir(tmp_path)
        with patch("app.services.model_registry.MODELS_DIR", stub):
            models = list_models()
        for m in models:
            assert isinstance(m, Path)
            assert m.suffix == ".gguf"

    def test_empty_when_no_dir(self):
        with patch("app.services.model_registry.MODELS_DIR", Path("/nonexistent")):
            models = list_models()
        assert models == []

"""Tests for model_registry — discovery of all available GGUF models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.services.model_registry import list_models

EXPECTED_MODELS = {
    "Phi-4-mini-instruct-Q4_K_M",
    "Qwen3-4B-Q4_K_M",
    "gemma-3-1b-it-q4_0_s",
    "gemma-3-4b-it-q4_0_s",
}


class TestListModels:
    def test_discovers_all_models(self):
        """list_models() must find every .gguf file in MODELS_DIR."""
        models = list_models()
        stems = {p.stem for p in models}
        assert stems == EXPECTED_MODELS

    def test_returns_path_objects(self):
        models = list_models()
        for m in models:
            assert isinstance(m, Path)
            assert m.suffix == ".gguf"

    def test_empty_when_no_dir(self):
        with patch("app.services.model_registry.MODELS_DIR", Path("/nonexistent")):
            models = list_models()
        assert models == []

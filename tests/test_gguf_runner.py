"""Tests for GGUFRunner — load, generate, error handling."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from app.services.inference.gguf_runner import GGUFRunner
from app.services.model_registry import BENCH_MODEL_GGUF_STEMS

ALL_MODELS = sorted(BENCH_MODEL_GGUF_STEMS)

FAKE_MODEL = Path("/tmp/fake-model.gguf")


def _chat_chunk(content: str) -> dict:
    """Build a streamed create_chat_completion chunk."""
    return {"choices": [{"delta": {"content": content}}]}


class TestLoad:
    """Runner.load() behaviour."""

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_succeeds(self, mock_llama_cls):
        runner = GGUFRunner(n_threads=2)
        runner.load(FAKE_MODEL)

        mock_llama_cls.assert_called_once()
        assert runner._loaded_path == FAKE_MODEL

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_uses_auto_context(self, mock_llama_cls):
        """n_ctx=0 lets llama.cpp auto-detect from model metadata."""
        runner = GGUFRunner()
        runner.load(FAKE_MODEL)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_ctx"] == 0

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_caches_model(self, mock_llama_cls):
        """Second load with same path should skip re-creation."""
        runner = GGUFRunner()
        runner.load(FAKE_MODEL)
        runner.load(FAKE_MODEL)

        assert mock_llama_cls.call_count == 1

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_reloads_on_different_path(self, mock_llama_cls):
        runner = GGUFRunner()
        runner.load(FAKE_MODEL)
        runner.load(Path("/tmp/other.gguf"))

        assert mock_llama_cls.call_count == 2

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_propagates_error(self, mock_llama_cls):
        """When Llama() raises, load propagates directly."""
        mock_llama_cls.side_effect = ValueError("context init failed")
        runner = GGUFRunner()

        with pytest.raises(ValueError, match="context init failed"):
            runner.load(FAKE_MODEL)

        assert runner._model is None
        assert runner._loaded_path is None

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_succeeds_per_model(self, mock_llama_cls, model_name):
        """Each known model should load successfully (mocked)."""
        mock_llama_cls.return_value = MagicMock()
        runner = GGUFRunner()
        model_path = Path(f"/tmp/{model_name}.gguf")
        runner.load(model_path)

        assert runner._loaded_path == model_path
        assert runner._model is not None

    @patch("app.services.inference.gguf_runner.Llama")
    def test_n_gpu_layers_passed_to_llama(self, mock_llama_cls):
        """n_gpu_layers should be forwarded to Llama()."""
        runner = GGUFRunner(n_gpu_layers=0)
        runner.load(FAKE_MODEL)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0

    @patch("app.services.inference.gguf_runner.Llama")
    def test_default_n_gpu_layers_is_zero(self, mock_llama_cls):
        """Default should be CPU-only (n_gpu_layers=0)."""
        runner = GGUFRunner()
        runner.load(FAKE_MODEL)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0


class TestGenerate:
    """Runner.generate() behaviour."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_yields_tokens_per_model(self, mock_llama_cls, model_name):
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([
            _chat_chunk("Hello"),
            _chat_chunk(" world"),
        ])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner()
        runner.load(Path(f"/tmp/{model_name}.gguf"))
        tokens = list(runner.generate("test prompt", max_tokens=10))

        assert tokens == ["Hello", " world"]

    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_sends_chat_messages(self, mock_llama_cls):
        """generate() should wrap the prompt as a user message."""
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner()
        runner.load(FAKE_MODEL)
        list(runner.generate("hello there"))

        call_kwargs = mock_model.create_chat_completion.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello there"}]

    def test_generate_raises_without_model(self):
        runner = GGUFRunner()
        with pytest.raises(RuntimeError, match="No model loaded"):
            list(runner.generate("test"))

    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_respects_cancel(self, mock_llama_cls):
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([
            _chat_chunk("tok1"),
            _chat_chunk("tok2"),
            _chat_chunk("tok3"),
        ])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner()
        runner.load(FAKE_MODEL)

        cancel = Event()
        cancel.set()
        tokens = list(runner.generate("test", cancel=cancel))

        assert tokens == []

    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_skips_empty_tokens(self, mock_llama_cls):
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter([
            _chat_chunk("a"),
            _chat_chunk(""),
            _chat_chunk("b"),
        ])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner()
        runner.load(FAKE_MODEL)
        tokens = list(runner.generate("test"))

        assert tokens == ["a", "b"]

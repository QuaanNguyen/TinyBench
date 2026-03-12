"""Tests for GGUFRunner — load, generate, error handling, and context fallback."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from app.services.inference.gguf_runner import GGUFRunner

ALL_MODELS = [
    "Phi-4-mini-instruct-Q4_K_M",
    "Qwen3-4B-Q4_K_M",
    "gemma-3-1b-it-q4_0_s",
    "gemma-3-4b-it-q4_0_s",
]

FAKE_MODEL = Path("/tmp/fake-model.gguf")


class TestLoad:
    """Runner.load() behaviour."""

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_succeeds(self, mock_llama_cls):
        runner = GGUFRunner(n_ctx=512, n_threads=2)
        runner.load(FAKE_MODEL)

        mock_llama_cls.assert_called_once()
        assert runner._loaded_path == FAKE_MODEL

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_caches_model(self, mock_llama_cls):
        """Second load with same path should skip re-creation."""
        runner = GGUFRunner(n_ctx=512)
        runner.load(FAKE_MODEL)
        runner.load(FAKE_MODEL)

        assert mock_llama_cls.call_count == 1

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_reloads_on_different_path(self, mock_llama_cls):
        runner = GGUFRunner(n_ctx=512)
        runner.load(FAKE_MODEL)
        runner.load(Path("/tmp/other.gguf"))

        assert mock_llama_cls.call_count == 2

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_raises_on_context_failure(self, mock_llama_cls):
        """When Llama() raises ValueError (context alloc), load must propagate
        after exhausting retries."""
        mock_llama_cls.side_effect = ValueError("Failed to create llama_context")
        runner = GGUFRunner(n_ctx=2048)

        with pytest.raises(ValueError, match="Failed to create llama_context"):
            runner.load(FAKE_MODEL)

        assert runner._model is None
        assert runner._loaded_path is None

    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_retries_with_smaller_ctx(self, mock_llama_cls):
        """If initial n_ctx fails, runner should retry with halved context."""

        def side_effect(*args, **kwargs):
            if kwargs.get("n_ctx", 0) > 512:
                raise ValueError("Failed to create llama_context")
            return MagicMock()

        mock_llama_cls.side_effect = side_effect
        runner = GGUFRunner(n_ctx=2048)
        runner.load(FAKE_MODEL)

        assert runner._model is not None
        assert runner._loaded_path == FAKE_MODEL
        last_call_kwargs = mock_llama_cls.call_args[1]
        assert last_call_kwargs["n_ctx"] <= 512

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_succeeds_per_model(self, mock_llama_cls, model_name):
        """Each known model should load successfully (mocked)."""
        mock_llama_cls.return_value = MagicMock()
        runner = GGUFRunner(n_ctx=2048)
        model_path = Path(f"/tmp/{model_name}.gguf")
        runner.load(model_path)

        assert runner._loaded_path == model_path
        assert runner._model is not None

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @patch("app.services.inference.gguf_runner.Llama")
    def test_load_retries_per_model(self, mock_llama_cls, model_name):
        """Context fallback should work identically for every model."""

        def side_effect(*args, **kwargs):
            if kwargs.get("n_ctx", 0) > 512:
                raise ValueError("Failed to create llama_context")
            return MagicMock()

        mock_llama_cls.side_effect = side_effect
        runner = GGUFRunner(n_ctx=2048)
        runner.load(Path(f"/tmp/{model_name}.gguf"))

        assert runner._model is not None
        assert mock_llama_cls.call_args[1]["n_ctx"] <= 512

    @patch("app.services.inference.gguf_runner.Llama")
    def test_retry_sequence_halves_correctly(self, mock_llama_cls):
        """Verify the exact n_ctx values tried: 2048 -> 1024 -> 512 -> 256."""
        tried = []

        def side_effect(*args, **kwargs):
            ctx = kwargs.get("n_ctx", 0)
            tried.append(ctx)
            if ctx > 256:
                raise ValueError("Failed to create llama_context")
            return MagicMock()

        mock_llama_cls.side_effect = side_effect
        runner = GGUFRunner(n_ctx=2048)
        runner.load(FAKE_MODEL)

        assert tried == [2048, 1024, 512, 256]

    @patch("app.services.inference.gguf_runner.Llama")
    def test_all_retries_exhausted_raises(self, mock_llama_cls):
        """If even n_ctx=256 fails, a clear error is raised."""
        mock_llama_cls.side_effect = ValueError("Failed to create llama_context")
        runner = GGUFRunner(n_ctx=2048)

        with pytest.raises(ValueError, match="tried n_ctx down to"):
            runner.load(FAKE_MODEL)

    @patch("app.services.inference.gguf_runner.Llama")
    def test_n_gpu_layers_passed_to_llama(self, mock_llama_cls):
        """n_gpu_layers should be forwarded to Llama()."""
        runner = GGUFRunner(n_ctx=512, n_gpu_layers=0)
        runner.load(FAKE_MODEL)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0

    @patch("app.services.inference.gguf_runner.Llama")
    def test_default_n_gpu_layers_is_zero(self, mock_llama_cls):
        """Default should be CPU-only (n_gpu_layers=0)."""
        runner = GGUFRunner(n_ctx=512)
        runner.load(FAKE_MODEL)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0


class TestGenerate:
    """Runner.generate() behaviour."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_yields_tokens_per_model(self, mock_llama_cls, model_name):
        mock_model = MagicMock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
        ])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner(n_ctx=512)
        runner.load(Path(f"/tmp/{model_name}.gguf"))
        tokens = list(runner.generate("test prompt", max_tokens=10))

        assert tokens == ["Hello", " world"]

    def test_generate_raises_without_model(self):
        runner = GGUFRunner()
        with pytest.raises(RuntimeError, match="No model loaded"):
            list(runner.generate("test"))

    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_respects_cancel(self, mock_llama_cls):
        mock_model = MagicMock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "tok1"}]},
            {"choices": [{"text": "tok2"}]},
            {"choices": [{"text": "tok3"}]},
        ])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner(n_ctx=512)
        runner.load(FAKE_MODEL)

        cancel = Event()
        cancel.set()
        tokens = list(runner.generate("test", cancel=cancel))

        assert tokens == []

    @patch("app.services.inference.gguf_runner.Llama")
    def test_generate_skips_empty_tokens(self, mock_llama_cls):
        mock_model = MagicMock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "a"}]},
            {"choices": [{"text": ""}]},
            {"choices": [{"text": "b"}]},
        ])
        mock_llama_cls.return_value = mock_model

        runner = GGUFRunner(n_ctx=512)
        runner.load(FAKE_MODEL)
        tokens = list(runner.generate("test"))

        assert tokens == ["a", "b"]

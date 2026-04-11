"""Tests for job_manager — job lifecycle, error propagation, cancellation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from threading import Event
from unittest.mock import MagicMock, patch

from app.services import job_manager
from app.services.model_registry import BENCH_MODEL_GGUF_STEMS

ALL_MODELS = sorted(BENCH_MODEL_GGUF_STEMS)


@pytest.fixture(autouse=True)
def _clean_jobs():
    """Reset global state between tests."""
    job_manager._jobs.clear()
    yield
    job_manager._jobs.clear()


class TestCreateJob:
    @pytest.mark.asyncio
    @patch.object(job_manager, "runner")
    async def test_returns_job_id(self, mock_runner):
        mock_runner.load = MagicMock()
        mock_runner.generate = MagicMock(return_value=iter([]))

        job_id = await job_manager.create_job("fake-model", "hello")
        assert isinstance(job_id, str)
        assert len(job_id) == 12


class TestTokenStream:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_id", ALL_MODELS)
    @patch.object(job_manager, "_resolve_model_path")
    @patch.object(job_manager, "runner")
    async def test_streams_tokens_per_model(self, mock_runner, mock_resolve, model_id):
        mock_resolve.return_value = Path(f"/tmp/{model_id}.gguf")
        mock_runner.load = MagicMock()
        mock_runner.generate = MagicMock(return_value=iter(["Hello", " world"]))

        job_id = await job_manager.create_job(model_id, "prompt")
        events = []
        async for evt in job_manager.token_stream(job_id):
            events.append(evt)

        types = [e["event"] for e in events]
        assert "token" in types
        assert types[-1] == "done"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_id", ALL_MODELS)
    @patch.object(job_manager, "_resolve_model_path")
    @patch.object(job_manager, "runner")
    async def test_streams_error_on_load_failure(
        self, mock_runner, mock_resolve, model_id
    ):
        mock_resolve.return_value = Path(f"/tmp/{model_id}.gguf")
        mock_runner.load.side_effect = ValueError("Failed to create llama_context")

        job_id = await job_manager.create_job(model_id, "prompt")
        events = []
        async for evt in job_manager.token_stream(job_id):
            events.append(evt)

        assert any(e["event"] == "error" for e in events)
        err_data = next(e["data"] for e in events if e["event"] == "error")
        assert "llama_context" in err_data

    @pytest.mark.asyncio
    async def test_stream_unknown_job(self):
        events = []
        async for evt in job_manager.token_stream("nonexistent"):
            events.append(evt)

        assert events == [{"event": "error", "data": "Job not found"}]

    @pytest.mark.asyncio
    @patch.object(job_manager, "_resolve_model_path")
    async def test_stream_model_not_found(self, mock_resolve):
        mock_resolve.side_effect = FileNotFoundError("Model not found: bogus")

        job_id = await job_manager.create_job("bogus", "prompt")
        events = []
        async for evt in job_manager.token_stream(job_id):
            events.append(evt)

        assert any(e["event"] == "error" for e in events)
        err_data = next(e["data"] for e in events if e["event"] == "error")
        assert "not found" in err_data.lower()


class TestCancelJob:
    @pytest.mark.asyncio
    @patch.object(job_manager, "_resolve_model_path")
    @patch.object(job_manager, "runner")
    async def test_cancel_sets_event(self, mock_runner, mock_resolve):
        mock_resolve.return_value = Path("/tmp/fake.gguf")

        def slow_generate(prompt, *, max_tokens=512, cancel=None):
            for t in ["a", "b", "c"]:
                if cancel and cancel.is_set():
                    break
                yield t

        mock_runner.load = MagicMock()
        mock_runner.generate = slow_generate

        job_id = await job_manager.create_job("m", "p")
        assert job_manager.cancel_job(job_id) is True

    def test_cancel_nonexistent(self):
        assert job_manager.cancel_job("nope") is False

    @pytest.mark.asyncio
    @patch.object(job_manager, "_resolve_model_path")
    @patch.object(job_manager, "runner")
    async def test_cancel_during_streaming_truncates_and_ends_with_done(
        self, mock_runner, mock_resolve
    ):
        """Cancel mid-stream: output is truncated, stream ends with 'done'."""
        mock_resolve.return_value = Path("/tmp/fake.gguf")
        mock_runner.load = MagicMock()

        gate = Event()

        def blocking_generate(prompt, *, max_tokens=512, cancel=None):
            for t in ["tok1", "tok2", "tok3", "tok4", "tok5"]:
                if cancel and cancel.is_set():
                    break
                yield t
                gate.wait(timeout=0.05)

        mock_runner.generate = blocking_generate

        job_id = await job_manager.create_job("m", "p")
        await asyncio.sleep(0.08)

        job_manager.cancel_job(job_id)

        events = []
        async for evt in job_manager.token_stream(job_id):
            events.append(evt)

        types = [e["event"] for e in events]
        assert types[-1] == "done"
        token_count = sum(1 for t in types if t == "token")
        assert token_count < 5


@pytest.fixture
def patched_models_dir(tmp_path: Path):
    root = tmp_path / "gguf"
    root.mkdir()
    for stem in BENCH_MODEL_GGUF_STEMS:
        (root / f"{stem}.gguf").write_bytes(b"")
    with patch.object(job_manager, "MODELS_DIR", root):
        yield root


class TestResolveModelPath:
    @pytest.mark.parametrize("model_id", ALL_MODELS)
    def test_resolves_each_model(self, patched_models_dir, model_id):
        path = job_manager._resolve_model_path(model_id)
        assert path.stem == model_id
        assert path.suffix == ".gguf"

    def test_raises_on_unknown_model(self):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            job_manager._resolve_model_path("nonexistent-model")

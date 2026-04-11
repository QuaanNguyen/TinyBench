from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Event, Lock
from typing import Any, Iterator

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - depends on optional native package
    Llama = None  # type: ignore[assignment]

from app.services.inference.base import InferenceRunner

logger = logging.getLogger(__name__)


class GGUFRunner(InferenceRunner):
    """llama-cpp-python wrapper with single-model caching."""

    def __init__(
        self,
        n_ctx: int = 0,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ) -> None:
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._model: Any | None = None
        self._loaded_path: Path | None = None
        self._lock = Lock()
        self._last_metrics: dict[str, object] | None = None
        self._context_length: int | None = None
        self._known_context_lengths: dict[str, int] = {}

    def load(self, model_path: Path) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the package and its "
                "native build prerequisites to enable GGUF inference."
            )

        with self._lock:
            if self._loaded_path == model_path and self._model is not None:
                return

            logger.info("Loading GGUF model: %s (n_ctx=%d)", model_path, self._n_ctx)
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            self._loaded_path = model_path

            try:
                self._context_length = self._model.n_ctx()
                self._known_context_lengths[model_path.stem] = self._context_length
            except Exception:
                self._context_length = None

            logger.info(
                "Model loaded: %s (n_ctx=%s)", model_path.name, self._context_length
            )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        cancel: Event | None = None,
    ) -> Iterator[str]:
        if self._model is None:
            raise RuntimeError("No model loaded")

        self._last_metrics = None

        messages = [{"role": "user", "content": prompt}]
        stream = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )

        n_tokens = 0
        t_start = time.perf_counter()

        for chunk in stream:
            if cancel and cancel.is_set():
                logger.info("Generation cancelled")
                break
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                n_tokens += 1
                yield token

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        tps = n_tokens / (t_elapsed_ms / 1000) if t_elapsed_ms > 0 else 0.0
        self._last_metrics = {
            "n_tokens": n_tokens,
            "generation_ms": round(t_elapsed_ms, 2),
            "throughput_tps": round(tps, 2),
        }

    def get_last_metrics(self) -> dict[str, object] | None:
        """Return metrics from the most recent generate() call."""
        return self._last_metrics

    def get_model_context_length(self) -> int | None:
        """Return context length of the currently loaded model."""
        return self._context_length

    def get_known_context_lengths(self) -> dict[str, int]:
        """Return cached context lengths for all models loaded so far."""
        return dict(self._known_context_lengths)

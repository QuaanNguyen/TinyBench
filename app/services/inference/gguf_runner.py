from __future__ import annotations

import gc
import logging
from pathlib import Path
from threading import Event, Lock
from typing import Any, Iterator

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - depends on optional native package
    Llama = None  # type: ignore[assignment]

from app.services.inference.base import InferenceRunner

logger = logging.getLogger(__name__)

_MIN_CTX = 256


class GGUFRunner(InferenceRunner):
    """llama-cpp-python wrapper with single-model caching."""

    def __init__(
        self,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ) -> None:
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._model: Any | None = None
        self._loaded_path: Path | None = None
        self._lock = Lock()

    def load(self, model_path: Path) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install the package and its "
                "native build prerequisites to enable GGUF inference."
            )

        with self._lock:
            if self._loaded_path == model_path and self._model is not None:
                return

            ctx = self._n_ctx
            while ctx >= _MIN_CTX:
                try:
                    logger.info("Loading GGUF model: %s (n_ctx=%d)", model_path, ctx)
                    self._model = Llama(
                        model_path=str(model_path),
                        n_ctx=ctx,
                        n_threads=self._n_threads,
                        n_gpu_layers=self._n_gpu_layers,
                        verbose=False,
                    )
                    self._loaded_path = model_path
                    logger.info("Model loaded: %s (n_ctx=%d)", model_path.name, ctx)
                    return
                except ValueError:
                    logger.warning(
                        "Context allocation failed (n_ctx=%d), retrying smaller", ctx
                    )
                    self._model = None
                    gc.collect()
                    ctx //= 2

            raise ValueError(
                f"Failed to create llama_context for {model_path.name} "
                f"(tried n_ctx down to {_MIN_CTX})"
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

        stream = self._model.create_completion(
            prompt,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if cancel and cancel.is_set():
                logger.info("Generation cancelled")
                break
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token

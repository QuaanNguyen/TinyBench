from abc import ABC, abstractmethod
from pathlib import Path
from threading import Event
from typing import Iterator


class InferenceRunner(ABC):
    """Contract for all inference backends (GGUF, BitNet, etc.)."""

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load (or re-load) the model from *model_path*."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        cancel: Event | None = None,
    ) -> Iterator[str]:
        """Yield token strings. Stop early if *cancel* is set."""

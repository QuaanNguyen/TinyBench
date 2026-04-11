from pathlib import Path

from app.config import MODELS_DIR

# GGUF filename stems (without .gguf) that belong to the TinyBench suite.
BENCH_MODEL_GGUF_STEMS: frozenset[str] = frozenset(
    {
        "Phi-4-mini-instruct-Q4_K_M",
        "Qwen3-4B-Q4_K_M",
        "gemma-3-1b-it-q4_0_s",
        "gemma-3-4b-it-q4_0_s",
        "gemma-4-E2B-it-BF16",
        "LFM2-8B-A1B-F16",
    }
)


def list_models() -> list[Path]:
    return list(MODELS_DIR.glob("**/*.gguf"))
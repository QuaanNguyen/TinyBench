from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from threading import Event
from typing import AsyncIterator

from app.config import MODELS_DIR
from app.services.inference.gguf_runner import GGUFRunner

logger = logging.getLogger(__name__)

_SENTINEL = object()

runner = GGUFRunner()


@dataclass
class Job:
    job_id: str
    model_id: str
    prompt: str
    queue: asyncio.Queue[object] = field(default_factory=asyncio.Queue)
    cancel_event: Event = field(default_factory=Event)
    done: bool = False


_jobs: dict[str, Job] = {}


def _resolve_model_path(model_id: str):
    for p in MODELS_DIR.glob("**/*.gguf"):
        if p.stem == model_id:
            return p
    raise FileNotFoundError(f"Model not found: {model_id}")


def _run_inference(job: Job, loop: asyncio.AbstractEventLoop) -> None:
    """Blocking inference executed in a worker thread."""
    try:
        model_path = _resolve_model_path(job.model_id)
        runner.load(model_path)
        for token in runner.generate(
            job.prompt, max_tokens=512, cancel=job.cancel_event
        ):
            loop.call_soon_threadsafe(job.queue.put_nowait, token)
    except Exception as exc:
        logger.exception("Inference error for job %s", job.job_id)
        loop.call_soon_threadsafe(
            job.queue.put_nowait, RuntimeError(str(exc))
        )
    finally:
        loop.call_soon_threadsafe(job.queue.put_nowait, _SENTINEL)
        job.done = True


async def create_job(model_id: str, prompt: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, model_id=model_id, prompt=prompt)
    _jobs[job_id] = job

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_inference, job, loop)
    return job_id


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job is None:
        return False
    job.cancel_event.set()
    return True


async def token_stream(job_id: str) -> AsyncIterator[dict]:
    """Yield SSE-shaped dicts: {event, data}."""
    job = _jobs.get(job_id)
    if job is None:
        yield {"event": "error", "data": "Job not found"}
        return

    while True:
        item = await job.queue.get()
        if item is _SENTINEL:
            yield {"event": "done", "data": ""}
            break
        if isinstance(item, Exception):
            yield {"event": "error", "data": str(item)}
            break
        yield {"event": "token", "data": item}

    _jobs.pop(job_id, None)

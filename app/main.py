import random
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.services.model_registry import list_models
from app.services import job_manager
from app.services import fight_store
from app.services import ranking_store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    fight_store.init_db()
    ranking_store.init_db()
    yield


app = FastAPI(title="Tiny Bench", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the landing page."""
    return templates.TemplateResponse(
        request=request, name="chat.html"
    )


@app.get("/ranking", response_class=HTMLResponse)
async def ranking(request: Request):
    """Render the ranking page."""
    return templates.TemplateResponse(
        request=request, name="ranking.html"
    )


@app.get("/power", response_class=HTMLResponse)
async def power(request: Request):
    """Render the power monitoring page."""
    return templates.TemplateResponse(
        request=request, name="power.html"
    )


@app.get("/fight", response_class=HTMLResponse)
async def fight(request: Request):
    """Render the fight page."""
    return templates.TemplateResponse(
        request=request, name="fight.html"
    )


@app.get("/api/models")
async def get_models():
    models = list_models()
    return [
        {"id": p.stem, "name": p.stem}
        for p in sorted(models, key=lambda p: p.name)
    ]


class ChatRequest(BaseModel):
    model_id: str
    prompt: str


@app.post("/api/chat")
async def start_chat(req: ChatRequest):
    job_id = await job_manager.create_job(req.model_id, req.prompt)
    return {"job_id": job_id}


@app.get("/sse/chat/{job_id}")
async def stream_chat(job_id: str):
    async def event_generator():
        async for evt in job_manager.token_stream(job_id):
            yield f"event: {evt['event']}\ndata: {evt['data']}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/{job_id}/cancel")
async def cancel_chat(job_id: str):
    ok = job_manager.cancel_job(job_id)
    if not ok:
        return {"status": "not_found"}
    return {"status": "cancelled"}


# ── Ranking endpoints ──


@app.get("/api/ranking")
async def get_ranking():
    ctx = job_manager.get_context_lengths()
    return ranking_store.compute_rankings(context_lengths=ctx)


# ── Fight endpoints ──


class FightRequest(BaseModel):
    prompt: str


class VoteRequest(BaseModel):
    winner: str  # "A" or "B"


@app.post("/api/fight")
async def start_fight(req: FightRequest):
    models = list_models()
    if len(models) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models for a fight")

    chosen = random.sample(models, 2)
    model_a, model_b = chosen[0].stem, chosen[1].stem

    fight_id = uuid.uuid4().hex[:12]
    job_a = await job_manager.create_job(model_a, req.prompt)
    job_b = await job_manager.create_job(model_b, req.prompt)

    fight_store.save_fight(fight_id, model_a, model_b, req.prompt)

    return {
        "fight_id": fight_id,
        "jobs": [
            {"job_id": job_a, "label": "A"},
            {"job_id": job_b, "label": "B"},
        ],
    }


@app.post("/api/fight/{fight_id}/vote")
async def vote_fight(fight_id: str, req: VoteRequest):
    if req.winner not in ("A", "B"):
        raise HTTPException(status_code=400, detail="winner must be 'A' or 'B'")

    result = fight_store.record_vote(fight_id, req.winner)
    if result is None:
        raise HTTPException(status_code=404, detail="Fight not found")

    return result

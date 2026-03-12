from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.services.model_registry import list_models
from app.services import job_manager

app = FastAPI(title="Tiny Bench")

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

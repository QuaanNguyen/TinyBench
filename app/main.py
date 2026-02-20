from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

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

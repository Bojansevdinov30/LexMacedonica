"""LexMacedonica — FastAPI entry point.

Serves the four interfaces (Jinja2 templates + static files) and includes the
JSON API routers. Run with:  uvicorn main:app --reload
"""
import app.config  # noqa: F401  — loads .env + the TLS fix FIRST (order matters)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi.errors import RateLimitExceeded

from app.limits import limiter
from app.routers import admin, chat, lawyer, simulation

app = FastAPI(title="LexMacedonica")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={
        "detail": "Премногу барања за кратко време — почекајте минута и обидете се повторно."})

# all the HTML template responses are here
@app.get("/", response_class=HTMLResponse, include_in_schema=False, name="home")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active": "home"})


@app.get("/simulacija", response_class=HTMLResponse, include_in_schema=False, name="simulation")
async def simulation_page(request: Request):
    return templates.TemplateResponse(request, "simulation.html", {"active": "simulation"})


@app.get("/advokati", response_class=HTMLResponse, include_in_schema=False, name="lawyer")
async def lawyer_page(request: Request):
    return templates.TemplateResponse(request, "lawyer.html", {"active": "lawyer"})


@app.get("/administracija", response_class=HTMLResponse, include_in_schema=False, name="admin")
async def admin_page(request: Request):
    return templates.TemplateResponse(request, "admin.html", {"active": "admin"})


# everything with JSON goes with the /api
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(lawyer.router, prefix="/api", tags=["lawyer"])
app.include_router(simulation.router, prefix="/api", tags=["simulation"])
app.include_router(admin.router, prefix="/api", tags=["admin"])

"""LexMacedonica — FastAPI entry point.

Serves the four interfaces (Jinja2 templates + static files) and the JSON API
the frontend talks to. Run with:  uvicorn main:app --reload
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="LexMacedonica")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Pages ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"active": "home"})


@app.get("/simulacija", response_class=HTMLResponse)
async def simulation(request: Request):
    return templates.TemplateResponse(request, "simulation.html", {"active": "simulation"})


@app.get("/advokati", response_class=HTMLResponse)
async def lawyer(request: Request):
    return templates.TemplateResponse(request, "lawyer.html", {"active": "lawyer"})


@app.get("/administracija", response_class=HTMLResponse)
async def admin(request: Request):
    return templates.TemplateResponse(request, "admin.html", {"active": "admin"})


# ---------- API ----------

class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main assistant endpoint.

    Phase 0 stub: returns a hardcoded response in the exact shape the real RAG
    chain will produce in Phase 2 (answer + probability + cited cases), so the
    frontend can be built and tested against the final contract from day one.
    """
    return {
        "answer": (
            "Ова е тест-одговор (системот сè уште не е поврзан со базата на случаи). "
            "Вашето прашање беше: „" + req.question.strip() + "“"
        ),
        "probability": 70,
        "cases": [
            {
                "case_number": "П1-94/24",
                "court": "Основен суд Битола",
                "date": "25.11.2024",
                "outcome": "СЕ УСВОЈУВА",
                "summary": "Пример-случај за демонстрација на изгледот на картичките.",
            }
        ],
    }

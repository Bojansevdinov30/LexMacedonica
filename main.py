"""LexMacedonica — FastAPI entry point.

Serves the four interfaces (Jinja2 templates + static files) and the JSON API
the frontend talks to. Run with:  uvicorn main:app --reload
"""
import app.config  # noqa: F401  (loads .env + the machine's TLS fix before anything else)
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


class LawyerRequest(BaseModel):
    question: str
    mode: str = "laws"   # "laws" = закони + пракса, "cases" = само пракса


class SimTurnRequest(BaseModel):
    scenario: str
    history: list[dict] = []


class AnonymizeRequest(BaseModel):
    text: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Main assistant endpoint (sync `def` on purpose: FastAPI runs it in a
    worker thread, so the LLM calls don't block other requests)."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "answer": ("Системот сè уште не е поврзан со OpenAI. Додадете "
                       "OPENAI_API_KEY во .env датотеката (видете .env.example)."),
            "probability": None, "cases": [],
        }
    try:
        from app.rag.chains import answer_question
        return answer_question(req.question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Грешка при обработката: {type(e).__name__}. "
                      "Проверете дали е изграден индексот (build_index).",
            "probability": None, "cases": [],
        }


@app.post("/api/lawyer")
def lawyer_api(req: LawyerRequest):
    try:
        from app.lawyer.rag import lawyer_answer
        return lawyer_answer(req.question, req.mode)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"reasoning": "", "sources": [], "mode": req.mode,
                "answer": f"Грешка при обработката: {type(e).__name__}."}


@app.post("/api/simulate/turn")
def simulate_turn(req: SimTurnRequest):
    try:
        from app.agents.simulation import next_turn
        turn = next_turn(req.scenario, req.history)
        return turn if turn else {"done": True}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"role": "narrator", "name": "Систем", "icon": "⚠️",
                "text": f"Грешка: {type(e).__name__}", "done": True}


@app.post("/api/anonymize")
def anonymize_api(req: AnonymizeRequest):
    try:
        from app.admin.anonymize import anonymize
        return anonymize(req.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"anonymized": "", "replacements": [],
                "reasoning": f"Грешка при обработката: {type(e).__name__}."}

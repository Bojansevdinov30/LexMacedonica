"""Правен асистент — the main chat endpoint"""
import re
import os
import traceback
from fastapi import APIRouter, HTTPException, Request, status

from app.config import TXT_DIR
from app.limits import limiter
from app.schemas import ChatRequest
from app.rag.chains import answer_question

router = APIRouter()


@router.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, req: ChatRequest):
    """Main assistant endpoint (sync `def` on purpose: FastAPI runs it in a
    worker thread, so the LLM calls don't block other requests)."""

    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "answer": "Системот сè уште не е поврзан со OpenAI.",
            "probability": None, "cases": [],
        }
    try:
        return answer_question(req.question, req.history)
    except Exception as e:
        traceback.print_exc()
        return {
            "answer": f"Грешка при обработката: {type(e).__name__}. "
                      "Проверете дали е изграден индексот (build_index).",
            "probability": None, "cases": [],
        }


# full decision text for a case card's «Повеќе» button.
@router.get("/case/{case_id}/text")
def case_text(case_id: str):
    if not re.fullmatch(r"[\w-]+", case_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Непознат предмет.")
    path = TXT_DIR / f"{case_id}.txt"
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Непознат предмет.")
    return {"case_id": case_id, "text": path.read_text(encoding="utf-8")}

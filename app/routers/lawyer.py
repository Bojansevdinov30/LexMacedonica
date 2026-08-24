"""Адвокати — specialized law-first assistant endpoint."""
from fastapi import APIRouter, Request
import traceback
from app.limits import limiter
from app.schemas import LawyerRequest
from app.lawyer.rag import lawyer_answer
router = APIRouter()

@router.post("/lawyer")
@limiter.limit("10/minute")
def lawyer_api(request: Request, req: LawyerRequest):
    try:
        return lawyer_answer(req.question, req.mode)
    except Exception as e:
        traceback.print_exc()
        return {"reasoning": "", "sources": [], "mode": req.mode,
                "answer": f"Грешка при обработката: {type(e).__name__}."}

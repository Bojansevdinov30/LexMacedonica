"""Адвокати — specialized law-first assistant endpoint."""
from fastapi import APIRouter, Request

from app.limits import limiter
from app.schemas import LawyerRequest

router = APIRouter()

@router.post("/lawyer")
@limiter.limit("10/minute")
def lawyer_api(request: Request, req: LawyerRequest):
    try:
        from app.lawyer.rag import lawyer_answer
        return lawyer_answer(req.question, req.mode)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"reasoning": "", "sources": [], "mode": req.mode,
                "answer": f"Грешка при обработката: {type(e).__name__}."}

"""Администрација — anonymization endpoint."""
from fastapi import APIRouter, Request
import traceback
from app.limits import limiter
from app.schemas import AnonymizeRequest
from app.admin.anonymize import anonymize

router = APIRouter()


@router.post("/anonymize")
@limiter.limit("10/minute")
def anonymize_api(request: Request, req: AnonymizeRequest):
    try:
        return anonymize(req.text)
    except Exception as e:
        traceback.print_exc()
        return {"anonymized": "", "replacements": [],
                "reasoning": f"Грешка при обработката: {type(e).__name__}."}

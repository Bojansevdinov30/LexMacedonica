"""Симулација — one endpoint per courtroom turn (the browser drives the loop)."""
import json
import traceback

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.limits import limiter
from app.schemas import SimTurnRequest
from app.agents.simulation import stream_turn
router = APIRouter()


@router.post("/simulate/turn")
@limiter.limit("30/minute")  # one full simulation = 6 sequential turn calls
def simulate_turn(request: Request, req: SimTurnRequest):
    def gen():
        try:
            for event in stream_turn(req.scenario, req.history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error",
                              "detail": f"Грешка: {type(e).__name__}"},
                             ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

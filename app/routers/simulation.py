"""Симулација — one endpoint per courtroom turn (the browser drives the loop).

Од 04.08.2026 одговорот е NDJSON стрим (еден JSON објект по линија) наместо
еден JSON: серверот испраќа meta → token* → final, па учесникот „куца" во живо.
Зошто NDJSON преку POST, а не SSE: EventSource не може да прати JSON тело, па
SSE овде би значело истата fetch-reader јамка + уште и "data:"-рамки. Sync
генератор е доволен — Starlette сам го тера низ threadpool.
"""
import json
import traceback

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.limits import limiter
from app.schemas import SimTurnRequest

router = APIRouter()

@router.post("/simulate/turn")
@limiter.limit("30/minute")   # one full simulation = 6 sequential turn calls
def simulate_turn(request: Request, req: SimTurnRequest):
    def gen():
        # try/except ВНАТРЕ во генераторот е задолжително: штом почне
        # стримувањето HTTP статусот е веќе 200, па грешка може да се
        # пренесе само како настан во самиот стрим
        try:
            from app.agents.simulation import stream_turn
            for event in stream_turn(req.scenario, req.history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error",
                              "detail": f"Грешка: {type(e).__name__}"},
                             ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

"""Pydantic request models for the JSON API.

One flat file (as in classic FastAPI projects): the frontend sends small JSON
bodies, so every request shape is defined here and imported by the routers.
"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    # [{"who": "user"|"bot", "text": "..."}]
    history: list[dict] = []


class LawyerRequest(BaseModel):
    question: str
    mode: str = "laws"   # "laws" = закони + пракса, "cases" = само пракса


class SimTurnRequest(BaseModel):
    scenario: str
    history: list[dict] = []


class AnonymizeRequest(BaseModel):
    text: str

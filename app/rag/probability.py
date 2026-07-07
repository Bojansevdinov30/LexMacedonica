"""Outcome probability from similar past cases.

The number the user sees is OUR statistic, not something the LLM invents:
take the N most similar cases, look up their real outcomes in SQLite metadata,
and compute a similarity-weighted share for the dominant outcome. More similar
cases influence the estimate more.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import TOP_CASES_FOR_PROBABILITY
from app.rag.retriever import RetrievalResult

# procedural endings say nothing about who was right — exclude from the stats
NON_MERITS = {"НЕПОЗНАТО", "ЗАПРЕНА ПОСТАПКА", "СПОГОДБА"}

READABLE = {
    "УСВОЕНО": "тужбеното барање е усвоено (тужителот добил)",
    "ДЕЛУМНО УСВОЕНО": "тужбеното барање е делумно усвоено",
    "ОДБИЕНО": "тужбеното барање е одбиено (тужениот добил)",
    "ОТФРЛЕНО": "тужбата е отфрлена од процедурални причини",
    "СПОГОДБА": "странките склучиле спогодба/порамнување",
    "ЗАПРЕНА ПОСТАПКА": "постапката е запрена (нпр. повлечена тужба)",
    "ПОТВРДЕНО": "првостепената одлука е потврдена",
    "УКИНАТО": "првостепената одлука е укината",
}


@dataclass
class ProbabilityEstimate:
    outcome: str            # dominant outcome label
    outcome_readable: str
    percent: int            # weighted share of that outcome, 0-100
    counts: dict[str, int]  # raw outcome counts among the sample
    sample_size: int


def estimate(result: RetrievalResult) -> ProbabilityEstimate | None:
    cases = result.top_cases(TOP_CASES_FOR_PROBABILITY)
    weighted: dict[str, float] = {}
    counts: dict[str, int] = {}

    for chunk in cases:
        outcome = chunk.metadata.get("outcome", "НЕПОЗНАТО")
        if outcome in NON_MERITS:
            continue
        # rrf_score reflects how similar the case is to the user's situation
        weight = chunk.rrf_score or 0.001
        weighted[outcome] = weighted.get(outcome, 0.0) + weight
        counts[outcome] = counts.get(outcome, 0) + 1

    if not weighted or sum(counts.values()) < 3:
        return None  # too few comparable cases for an honest percentage

    total = sum(weighted.values())
    dominant = max(weighted, key=weighted.get)
    percent = round(100 * weighted[dominant] / total)

    return ProbabilityEstimate(
        outcome=dominant,
        outcome_readable=READABLE.get(dominant, dominant),
        percent=percent,
        counts=counts,
        sample_size=sum(counts.values()),
    )

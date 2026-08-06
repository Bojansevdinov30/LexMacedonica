"""Outcome probability from similar past cases."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import TOP_CASES_FOR_PROBABILITY
from app.rag.retriever import RetrievalResult

NON_MERITS = {"НЕПОЗНАТО", "ЗАПРЕНА ПОСТАПКА", "СПОГОДБА", "ПОТВРДЕНО", "УКИНАТО"}

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

    for case in cases:
        outcome = case.metadata.get("outcome", "НЕПОЗНАТО")
        if outcome in NON_MERITS:
            continue
        # similarity of the case SUMMARY to the user's situation — cases that
        # match the situation better influence the estimate more
        weight = max(case.similarity, 0.001)
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

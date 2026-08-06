"""Structured metadata out of court decision text + the SQLite schema.

The probability feature needs a machine-readable OUTCOME per case. Macedonian
decisions are formulaic: right after the "ПРЕСУДА"/"РЕШЕНИЕ" header comes the
dispositive (изрека) with standard phrases. Regex catches most of them for
free; the few misses can later go through a cheap LLM fallback.
"""
from __future__ import annotations

import re

from sqlalchemy import create_engine, String, Text
from sqlalchemy import text as sql_text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from app.config import SQLITE_PATH

# --- SQLite schema ---------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_number: Mapped[str] = mapped_column(String(50), default="")
    court: Mapped[str] = mapped_column(String(120), default="")
    date: Mapped[str] = mapped_column(String(10), default="")   # dd.mm.yyyy
    outcome: Mapped[str] = mapped_column(String(30), default="НЕПОЗНАТО")
    outcome_sentence: Mapped[str] = mapped_column(Text, default="")
    preview: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")      # LLM summary
    # секцијата «Повеќе податоци» од sud.mk:
    judge: Mapped[str] = mapped_column(String(120), default="")
    legal_area: Mapped[str] = mapped_column(String(80), default="")
    case_type: Mapped[str] = mapped_column(String(80), default="")
    case_subtype: Mapped[str] = mapped_column(String(120), default="")
    foundation_type: Mapped[str] = mapped_column(String(120), default="")
    foundation: Mapped[str] = mapped_column(String(200), default="")


def ensure_case_columns(engine) -> None:
    """SQLite migration: create_all never ADDS columns to an existing table,
    so compare the live schema with the model and ALTER TABLE the gap.
    (Generalization of the old ensure_summary_column from summarize.py.)"""
    with engine.connect() as conn:
        live = {r[1] for r in conn.execute(sql_text("PRAGMA table_info(cases)"))}
        for col in Case.__table__.columns:
            if col.name not in live:
                conn.execute(sql_text(
                    f"ALTER TABLE cases ADD COLUMN {col.name} TEXT DEFAULT ''"))
                print(f"added cases.{col.name} column")
        conn.commit()


def get_engine():
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    Base.metadata.create_all(engine)
    ensure_case_columns(engine)   # existing DBs get any newly added columns
    return engine


# --- Outcome extraction ----------------------------------------------------

# Order matters: more specific patterns first.
OUTCOME_PATTERNS: list[tuple[str, str]] = [
    ("ДЕЛУМНО УСВОЕНО", r"ДЕЛУМНО\s+СЕ\s+УСВОЈУВА|СЕ\s+УСВОЈУВА\s+ДЕЛУМНО"
                        r"|ДЕЛУМНО\s+СЕ\s+УВАЖУВА|СЕ\s+УВАЖУВА\s+ДЕЛУМНО"),
    # УВАЖУВА is the older synonym of УСВОЈУВА, still used by some courts
    ("УСВОЕНО",         r"СЕ\s+УСВОЈУВА|СЕ\s+УВАЖУВА"),
    ("ОДБИЕНО",         r"СЕ\s+ОДБИВА"),
    # ОТ[РФ]{2}ЛА tolerates the "ОТРФЛА" typo that real decisions contain
    ("ОТФРЛЕНО",        r"СЕ\s+ОТ[РФ]{2}ЛА|НЕНАДЛЕЖЕН"),
    ("ЗАПРЕНА ПОСТАПКА", r"СЕ\s+ЗАПИРА|СЕ\s+ПОВЛЕКУВА|ПОВЛЕЧЕНА|ПОВЛЕКУВАЊЕ"),
    ("СПОГОДБА",        r"ПОРАМНУВАЊЕ|СПОГОДБА"),   # settlement
    ("ПОТВРДЕНО",       r"СЕ\s+ПОТВРДУВА"),   # appeal decisions
    ("УКИНАТО",         r"СЕ\s+УКИНУВА"),      # appeal decisions
]

# The header may be typeset with spaces between letters: "П Р Е С У Д А"
_HEADER_RE = re.compile(
    r"П\s*Р\s*Е\s*С\s*У\s*Д\s*А|Р\s*Е\s*Ш\s*Е\s*Н\s*И\s*Е", re.IGNORECASE
)


def extract_outcome(text: str) -> tuple[str, str]:
    """Returns (outcome_label, matched_sentence). Label НЕПОЗНАТО when unsure."""
    upper = re.sub(r"\s+", " ", text.upper())

    # search window: dispositive = ~2500 chars after the first header
    m = _HEADER_RE.search(upper)
    window = upper[m.end():m.end() + 2500] if m else upper[:3000]

    for label, pattern in OUTCOME_PATTERNS:
        hit = re.search(pattern, window)
        if hit:
            start = max(0, hit.start() - 60)
            return label, window[start:hit.end() + 160].strip()
    return "НЕПОЗНАТО", ""

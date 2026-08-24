"""Anonymization tool (Interface 4): personal data out of legal documents.

Two passes, cheapest first:
1. REGEX for machine-recognizable data: ЕМБГ (13 digits), phone numbers,
   emails, bank accounts. Deterministic, free, never misses the pattern.
2. LLM for what regex can't see: names of people, addresses, company names —
   returns a JSON list of replacements that we apply with plain str.replace.

"""
from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL


REGEX_RULES = [
    ("ЕМБГ", re.compile(r"\b\d{13}\b"), "«ЕМБГ»"),
    ("трансакциска сметка", re.compile(r"\b\d{3}[- ]?\d{10,12}[- ]?\d{2}\b"), "«СМЕТКА»"),
    ("телефон", re.compile(r"\b(?:\+389|0)7\d[\s/-]?\d{3}[\s/-]?\d{3}\b"), "«ТЕЛЕФОН»"),
    ("е-пошта", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "«Е-ПОШТА»"),
]

LLM_PROMPT = (
    "Ти си алатка за анонимизација на македонски правни документи. Најди ги "
    "СИТЕ лични податоци во текстот: имиња и презимиња на физички лица, адреси, "
    "градови, држави, називи на фирми/правни лица.\n"
    "НЕ ги анонимизирај: судови, институции, закони, судии и адвокати во "
    "службена улога.\n"
    "Врати JSON со точно две полиња:\n"
    '  "reasoning": кратко објаснување што си нашол и зошто одлучи да го замениш '
    "(или да не го замениш)\n"
    '  "replacements": листа од објекти {"original": "...", "replacement": "..."} '
    "каде replacement е иницијали за лица (пр. „Марко Марковски“ -> „М.М.“), "
    "„«АДРЕСА»“ за адреси, „«ФИРМА»“ за правни лица.\n"
    "Ако нема ништо за замена, врати празна листа."
)


def anonymize(text: str) -> dict:
    replacements = []

    # pass 1: regex
    result = text
    for label, pattern, token in REGEX_RULES:
        for match in set(pattern.findall(result)):
            replacements.append({"original": match, "replacement": token,
                                 "method": f"правило ({label})"})
            result = result.replace(match, token)

    # pass 2: LLM for names/addresses/companies
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})
    response = llm.invoke([("system", LLM_PROMPT), ("user", result)])

    reasoning = ""
    try:
        parsed = json.loads(response.content)
        reasoning = parsed.get("reasoning", "")
        for r in parsed.get("replacements", []):
            original = str(r.get("original", ""))
            if original and original in result:
                replacement = str(r.get("replacement", "«XXX»"))
                replacements.append({"original": original,
                                     "replacement": replacement,
                                     "method": "АИ (име/адреса/фирма)"})
                result = result.replace(original, replacement)
    except json.JSONDecodeError:
        reasoning = "Грешка при читање на АИ-одговорот; применети се само правилата."

    return {"anonymized": result, "replacements": replacements,
            "reasoning": reasoning}

"""Courtroom simulation (Interface 2): four agents play out the user's case.

Simplest possible multi-agent design: a FIXED turn script (narrator, plaintiff
lawyer, defense lawyer, one rebuttal each, judge). Every turn is one LLM call
that sees the scenario + the transcript so far, with a role-specific system
prompt. The frontend asks for one turn at a time, so the courtroom "types"
turn by turn and the server stays stateless.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL
from app.costs import log_llm_response

ROLES = {
    "narrator": ("Наратор", "🎙️"),
    "plaintiff": ("Адвокат на тужителот", "⚖️"),
    "defendant": ("Адвокат на тужениот", "🛡️"),
    "judge": ("Судија", "👨‍⚖️"),
}

# the fixed courtroom script
SCRIPT = ["narrator", "plaintiff", "defendant", "plaintiff", "defendant", "judge"]

PROMPTS = {
    "narrator": (
        "Ти си наратор во симулација на македонско судење. Во 3-4 реченици "
        "претстави го случајот и странките пред публиката, неутрално и живописно. "
        "Зборувај само македонски."),
    "plaintiff": (
        "Ти си адвокат на ТУЖИТЕЛОТ во македонско судење. Застапувај ја страната "
        "на тужителот убедливо: изнеси аргументи, повикај се на фактите од "
        "сценариото, одговори на аргументите на спротивната страна ако ги има. "
        "Биди краток — максимум 4-5 реченици. Зборувај само македонски."),
    "defendant": (
        "Ти си адвокат на ТУЖЕНИОТ во македонско судење. Брани ја страната на "
        "тужениот убедливо: оспорувај ги аргументите на тужителот, изнеси "
        "противаргументи. Биди краток — максимум 4-5 реченици. Зборувај само "
        "македонски."),
    "judge": (
        "Ти си СУДИЈА во македонско судење. Ислушај ги двете страни од "
        "транскриптот и донеси кратка пресуда: кој добива, зошто, и што "
        "следува. Биди достоинствен и јасен, максимум 5-6 реченици. Зборувај "
        "само македонски."),
}


def next_turn(scenario: str, history: list[dict]) -> dict | None:
    """history = [{"role": "plaintiff", "text": "..."}]. None = trial over."""
    step = len(history)
    if step >= len(SCRIPT):
        return None
    role = SCRIPT[step]

    transcript = "\n\n".join(
        f"{ROLES[h['role']][0]}: {h['text']}" for h in history
    ) or "(судењето штотуку почнува)"

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.7, max_tokens=400)
    response = llm.invoke([
        ("system", PROMPTS[role]),
        ("user", f"Сценарио на случајот:\n{scenario}\n\n"
                 f"Досегашен тек на судењето:\n{transcript}\n\n"
                 f"Твојот настап сега:"),
    ])
    log_llm_response(response, label=f"sim_{role}")

    name, icon = ROLES[role]
    return {"role": role, "name": name, "icon": icon,
            "text": response.content.strip(),
            "done": step + 1 >= len(SCRIPT)}

"""Courtroom simulation (Interface 2): four agents play out the user's case."""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL


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


@lru_cache(maxsize=1)
def _sim_llm() -> ChatOpenAI:
    # one client reused for every turn
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.7, max_tokens=400)


def next_turn(scenario: str, history: list[dict]) -> dict | None:
    """history = [{"role": "plaintiff", "text": "..."}]. None = trial over."""
    step = len(history)
    if step >= len(SCRIPT):
        return None
    role = SCRIPT[step]

    transcript = "\n\n".join(
        f"{ROLES[h['role']][0]}: {h['text']}" for h in history
    ) or "(судењето штотуку почнува)"

    response = _sim_llm().invoke([
        ("system", PROMPTS[role]),
        ("user", f"Сценарио на случајот:\n{scenario}\n\n"
                 f"Досегашен тек на судењето:\n{transcript}\n\n"
                 f"Твојот настап сега:"),
    ])

    name, icon = ROLES[role]
    return {"role": role, "name": name, "icon": icon,
            "text": response.content.strip(),
            "done": step + 1 >= len(SCRIPT)}


def stream_turn(scenario: str, history: list[dict]):
    """Streaming variant of next_turn: yields NDJSON-ready dicts."""
    step = len(history)
    if step >= len(SCRIPT):
        yield {"type": "final", "text": "", "done": True}
        return
    role = SCRIPT[step]
    name, icon = ROLES[role]
    yield {"type": "meta", "role": role, "name": name, "icon": icon}

    transcript = "\n\n".join(
        f"{ROLES[h['role']][0]}: {h['text']}" for h in history
    ) or "(судењето штотуку почнува)"

    pieces: list[str] = []
    for chunk in _sim_llm().stream([
        ("system", PROMPTS[role]),
        ("user", f"Сценарио на случајот:\n{scenario}\n\n"
                 f"Досегашен тек на судењето:\n{transcript}\n\n"
                 f"Твојот настап сега:"),
    ]):
        if chunk.content:
            pieces.append(chunk.content)
            yield {"type": "token", "text": chunk.content}

    yield {"type": "final", "role": role, "name": name, "icon": icon,
           "text": "".join(pieces).strip(),
           "done": step + 1 >= len(SCRIPT)}

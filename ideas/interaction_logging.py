"""ИДЕЈА (не е вклучена во апликацијата): логирање на секоја chat-интеракција.

Она што LangSmith го прави хостирано (следење на секој LLM повик: влез, излез,
латенција, токени), ова го прави локално — еден JSONL ред по прашање, без нови
зависности. Целата промена би живеела во app/rag/chains.py; НИШТО друго не се
менува (endpoints, frontend, retriever остануваат исти).

Напомена: .env веќе содржи LANGSMITH_* клучеви — LangChain автоматски праќа
трага за СЕКОЈ повик на LangSmith ако се постави LANGSMITH_TRACING=true во
.env. Тоа е нула-код алтернатива на овој файл; ова овде е локалната верзија
(податоците остануваат кај нас, работи и без интернет-сметка).

Формат на data/interactions.jsonl (еден ред = една интеракција):
    {"ts": "2026-08-05T14:31:07", "question": "...", "standalone": "...",
     "cached": false, "gated": false, "probability": 89,
     "cases": ["РО-333/10", ...], "answer": "...",
     "input_tokens": 1841, "output_tokens": 236, "seconds": 5.2}
"""

# ============================================================================
# 1. Новиот помошник — би стоел во app/rag/chains.py (или мал app/rag/log.py)
# ============================================================================

# import json, time
# from datetime import datetime
# from app.config import DATA_DIR
#
# INTERACTIONS_PATH = DATA_DIR / "interactions.jsonl"
#
# def log_interaction(**fields) -> None:
#     """Append-only JSONL — ист шаблон како cases_meta.jsonl од скрејперот."""
#     fields["ts"] = datetime.now().isoformat(timespec="seconds")
#     with INTERACTIONS_PATH.open("a", encoding="utf-8") as f:
#         f.write(json.dumps(fields, ensure_ascii=False) + "\n")


# ============================================================================
# 2. Што се менува во answer_question() — три излезни точки, три повици.
#    (Функцијата има точно три `return`-а; секој добива по еден лог-ред.)
# ============================================================================

# def answer_question(question, history=None):
#     t0 = time.perf_counter()                                   # НОВО
#     ...кондензација, embed, кеш...
#
#     if cached is not None:
#         cached["cached"] = True
#         log_interaction(question=question, standalone=question,  # НОВО
#                         cached=True, gated=False,
#                         probability=cached.get("probability"),
#                         answer=cached["answer"],
#                         seconds=round(time.perf_counter() - t0, 2))
#         return cached
#
#     ...retrieve...
#
#     if result.max_similarity < MIN_SIMILARITY_FOR_ANSWER or not result.chunks:
#         log_interaction(question=question, standalone=question,  # НОВО
#                         cached=False, gated=True,
#                         max_similarity=round(result.max_similarity, 3),
#                         seconds=round(time.perf_counter() - t0, 2))
#         return dict(NO_ANSWER)
#
#     ...probability, LLM повик...
#
#     payload = {...}
#     cache.store(qvec, payload)
#     usage = response.usage_metadata or {}                       # НОВО —
#     log_interaction(                                            # LangChain го
#         question=question, standalone=question,                 # враќа бројот
#         cached=False, gated=False,                              # токени на
#         probability=payload["probability"],                     # секој одговор
#         cases=[c["case_number"] for c in payload["cases"]],
#         answer=payload["answer"],
#         input_tokens=usage.get("input_tokens"),
#         output_tokens=usage.get("output_tokens"),
#         seconds=round(time.perf_counter() - t0, 2))
#     return payload


# ============================================================================
# 3. Зошто вака, а не декоратор/middleware
# ============================================================================
# - Middleware на /api/chat би го видел само влезот и излезот — не знае дали
#   одговорот бил кеширан, гејтиран, ниту колку токени чинел. Тие факти ги
#   знае само answer_question, па логот припаѓа таму.
# - JSONL наместо SQLite-табела: append е атомски, нема миграции, а анализата
#   е еден pandas.read_json(lines=True) кога ќе затреба.
# - Приватност: прашањата на корисниците се лични податоци — датотеката оди
#   во data/ (гитигнорирана) и НЕ се качува никаде. Тоа е и аргументот за
#   локално логирање наместо LangSmith во продукциска верзија.
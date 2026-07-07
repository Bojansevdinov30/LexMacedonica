"""Cost tracking for every OpenAI call (budget rule: < $20 total, see CLAUDE.md).

Every LLM/embedding call reports its token usage here; we convert to dollars
and keep a running total in data/costs.json so the number survives restarts.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime

from app.config import DATA_DIR, BUDGET_WARN_USD

COSTS_PATH = DATA_DIR / "costs.json"
_lock = threading.Lock()

# USD per 1M tokens (input, output)
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def log_cost(model: str, input_tokens: int, output_tokens: int, label: str = "") -> float:
    """Record one call. Returns the cumulative total in USD."""
    p_in, p_out = PRICES.get(model, (0.15, 0.60))  # unknown model -> assume mini pricing
    usd = (input_tokens * p_in + output_tokens * p_out) / 1_000_000

    with _lock:
        data = {"total_usd": 0.0, "calls": []}
        if COSTS_PATH.exists():
            data = json.loads(COSTS_PATH.read_text(encoding="utf-8"))
        data["total_usd"] = round(data["total_usd"] + usd, 6)
        data["calls"].append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": model, "label": label,
            "in": input_tokens, "out": output_tokens, "usd": round(usd, 6),
        })
        COSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        COSTS_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
        total = data["total_usd"]

    if total >= BUDGET_WARN_USD:
        print(f"!!! BUDGET WARNING: cumulative OpenAI spend is ${total:.2f} "
              f"(limit ${BUDGET_WARN_USD}) !!!")
    return total


def log_llm_response(response, label: str = "") -> None:
    """Convenience for LangChain AIMessage objects (usage_metadata)."""
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", {}) or {}
    model = meta.get("model_name", "gpt-4o-mini")
    log_cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), label)


def total_spent() -> float:
    if COSTS_PATH.exists():
        return json.loads(COSTS_PATH.read_text(encoding="utf-8"))["total_usd"]
    return 0.0

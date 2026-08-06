"""Shared slowapi limiter.

The limiter lives in its own tiny module so that both main.py and the routers
can import it without a circular import (main -> routers -> main would break).
main.py attaches it to the app (app.state.limiter) and registers the 429 handler.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Per-IP rate limiting: /api/chat alone can trigger several sequential OpenAI
# calls, so without a cap a single misbehaving client could burn the whole
# budget in minutes. Pages and free endpoints stay unlimited.
limiter = Limiter(key_func=get_remote_address)

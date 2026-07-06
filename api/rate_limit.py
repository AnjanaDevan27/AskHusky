"""
Rate Limiting — AskHusky
Protects Claude API and Pinecone from abuse and cost blowout.
Uses slowapi for per-IP rate limiting on FastAPI endpoints.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

# ── Limiter Setup ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ── Rate Limit Error Handler ──────────────────────────────────────────────────

async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded
) -> JSONResponse:
    """
    Return a friendly error when rate limit is exceeded.
    Tells the student to wait rather than showing a generic 429.
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too many requests",
            "message": "You've sent too many messages in a short time. "
                       "Please wait a moment before trying again.",
            "retry_after": "60 seconds"
        }
    )


# ── Rate Limit Decorators ─────────────────────────────────────────────────────

# Use these as decorators on FastAPI route handlers:
#
# @app.post("/chat")
# @limiter.limit("10/minute")
# async def chat(request: Request, ...):
#     ...
#
# @app.post("/voice")
# @limiter.limit("3/minute")
# async def voice(request: Request, ...):
#     ...

CHAT_LIMIT  = "10/minute"   # text chat endpoint
VOICE_LIMIT = "3/minute"    # voice endpoint — stricter due to Whisper + ElevenLabs cost
HEALTH_LIMIT = "60/minute"  # health check endpoint
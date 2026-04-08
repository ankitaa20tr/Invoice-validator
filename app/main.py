"""
FastAPI server that wraps InvoiceValidationEnv and exposes it over HTTP.

Endpoints
---------
POST /reset              — start a new episode, returns InvoiceObservation
POST /openenv/reset      — alias required by OpenEnv checker
POST /step               — take an action, returns StepResult
POST /openenv/step       — alias required by OpenEnv checker
GET  /state              — dump raw InvoiceState (debug / grader use)
GET  /openenv/state      — alias required by OpenEnv checker
GET  /health             — liveness probe for Docker / HF Spaces
GET  /openenv/health     — alias required by OpenEnv checker

The server holds one environment instance per process. This is fine for
single-agent evaluation but you'd swap to a session map for concurrent use.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.env import InvoiceValidationEnv
from app.models import InvoiceAction, InvoiceObservation, InvoiceState, StepResult


app = FastAPI(
    title="Invoice Validation Environment",
    description=(
        "An OpenEnv-compatible RL environment that simulates a finance team "
        "reviewing invoices for fraud and errors."
    ),
    version="1.0.0",
)

# Single global env instance — one episode at a time
_env = InvoiceValidationEnv()


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: str = "easy"  # one of: easy, medium, hard


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Shared handler functions (avoid code duplication across route aliases)
# ---------------------------------------------------------------------------

def _handle_reset(request: Optional[ResetRequest]) -> InvoiceObservation:
    """Core reset logic shared by /reset and /openenv/reset."""
    difficulty = (request.difficulty if request else None) or "easy"
    try:
        return _env.reset(difficulty=difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _handle_step(action: InvoiceAction) -> StepResult:
    """Core step logic shared by /step and /openenv/step."""
    try:
        return _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _handle_state() -> InvoiceState:
    """Core state logic shared by /state and /openenv/state."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["utility"])
def root():
    """Landing page — confirms the server is up and shows available endpoints."""
    return {
        "name": "Invoice Validation Environment",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/reset", "/step", "/state", "/openenv/reset", "/openenv/step", "/openenv/state"],
        "usage": "POST /reset with {\"difficulty\": \"easy\"} to start an episode",
    }


@app.get("/health", response_model=HealthResponse, tags=["utility"])
@app.get("/openenv/health", response_model=HealthResponse, tags=["utility"])
def health_check():
    """Liveness probe used by Docker and HF Spaces."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=InvoiceObservation, tags=["environment"])
def reset_environment(request: Optional[ResetRequest] = None):
    """
    Reset the environment to a fresh episode.

    Pass ``difficulty`` as one of ``easy``, ``medium``, or ``hard``.
    Body is **optional** — if omitted (or sent as null), defaults to ``easy``.
    Returns the initial InvoiceObservation the agent should start reasoning from.
    """
    return _handle_reset(request)


@app.post("/openenv/reset", response_model=InvoiceObservation, tags=["environment"])
def openenv_reset(request: Optional[ResetRequest] = None):
    """OpenEnv checker alias for POST /reset. Body is optional."""
    return _handle_reset(request)


@app.post("/step", response_model=StepResult, tags=["environment"])
def take_step(action: InvoiceAction):
    """
    Apply one action to the environment.

    The body must be a valid InvoiceAction JSON, e.g.::

        {"action_type": "add_issue", "value": "missing_gst_number"}

    Returns a StepResult containing the new observation, reward, done flag,
    and a free-form info dict.
    """
    return _handle_step(action)


@app.post("/openenv/step", response_model=StepResult, tags=["environment"])
def openenv_step(action: InvoiceAction):
    """OpenEnv checker alias for POST /step."""
    return _handle_step(action)


@app.get("/state", response_model=InvoiceState, tags=["debug"])
def get_state():
    """
    Return the raw InvoiceState including ground-truth fields.
    Intended for graders and developers, not for the agent.
    """
    return _handle_state()


@app.get("/openenv/state", response_model=InvoiceState, tags=["debug"])
def openenv_state():
    """OpenEnv checker alias for GET /state."""
    return _handle_state()

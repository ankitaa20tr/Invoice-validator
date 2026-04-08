"""
FastAPI server that wraps InvoiceValidationEnv and exposes it over HTTP.

Endpoints
---------
POST /reset          — start a new episode, returns InvoiceObservation
POST /step           — take an action, returns StepResult
GET  /state          — dump raw InvoiceState (debug / grader use)
GET  /health         — liveness probe for Docker / HF Spaces

The server holds one environment instance per process. This is fine for
single-agent evaluation but you'd swap to a session map for concurrent use.
"""

from __future__ import annotations

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
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["utility"])
def health_check():
    """Liveness probe used by Docker health checks and HF Spaces."""
    return HealthResponse(status="ok")


@app.post("/reset", response_model=InvoiceObservation, tags=["environment"])
def reset_environment(request: ResetRequest):
    """
    Reset the environment to a fresh episode.

    Pass `difficulty` as one of `easy`, `medium`, or `hard`.
    Returns the initial InvoiceObservation the agent should start reasoning from.
    """
    try:
        obs = _env.reset(difficulty=request.difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResult, tags=["environment"])
def take_step(action: InvoiceAction):
    """
    Apply one action to the environment.

    The body must be a valid InvoiceAction JSON, e.g.:
      {"action_type": "add_issue", "value": "missing_gst_number"}

    Returns a StepResult containing the new observation, reward, done flag,
    and a free-form info dict.
    """
    try:
        result = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state", response_model=InvoiceState, tags=["debug"])
def get_state():
    """
    Return the raw InvoiceState including ground-truth fields.
    Intended for graders and developers, not for the agent.
    """
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

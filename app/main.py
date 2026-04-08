"""
FastAPI server that wraps InvoiceValidationEnv and exposes it over HTTP.

Endpoints
---------
POST /reset                      — start a new episode, returns InvoiceObservation
POST /openenv/reset              — alias required by OpenEnv checker
POST /step                       — take an action, returns StepResult
POST /openenv/step               — alias required by OpenEnv checker
GET  /state                      — dump raw InvoiceState (debug / grader use)
GET  /openenv/state              — alias required by OpenEnv checker
GET  /health                     — liveness probe for Docker / HF Spaces
GET  /openenv/health             — alias required by OpenEnv checker
GET  /tasks                      — list all available tasks with graders
GET  /openenv/tasks              — alias required by OpenEnv checker
POST /tasks/{task_id}/grade      — grade a completed episode for a task
POST /openenv/tasks/{task_id}/grade — alias required by OpenEnv checker

The server holds one environment instance per process. This is fine for
single-agent evaluation but you'd swap to a session map for concurrent use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.env import InvoiceValidationEnv
from app.grader import grade_submission
from app.models import InvoiceAction, InvoiceObservation, InvoiceState, StepResult
from app.tasks import TASKS, load_task


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


class TaskInfo(BaseModel):
    id: str
    description: str
    max_steps: int
    target_score: float
    has_grader: bool = True


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]


class GradeRequest(BaseModel):
    """Submission payload for grading a completed episode."""
    detected_issues: List[str] = Field(default_factory=list)
    final_action:    str       = ""
    steps_taken:     int       = 0


class GradeResponse(BaseModel):
    task_id:  str
    score:    float
    passed:   bool
    feedback: str


# Task catalogue — mirrors openenv.yaml
_TASK_CATALOGUE: List[TaskInfo] = [
    TaskInfo(
        id="easy",
        description="Invoice with a single missing GST number",
        max_steps=20,
        target_score=0.8,
    ),
    TaskInfo(
        id="medium",
        description="Invoice with an incorrect total calculation",
        max_steps=20,
        target_score=0.75,
    ),
    TaskInfo(
        id="hard",
        description="Invoice with five overlapping issues including duplication",
        max_steps=20,
        target_score=0.7,
    ),
]


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


def _handle_tasks() -> TaskListResponse:
    """Return all tasks with grader metadata."""
    return TaskListResponse(tasks=_TASK_CATALOGUE)


def _handle_grade(task_id: str, req: GradeRequest) -> GradeResponse:
    """Grade a completed episode against the ground-truth task definition."""
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task '{task_id}'. Available: {list(TASKS)}",
        )
    task_state = load_task(task_id)

    score = grade_submission(
        detected_issues=req.detected_issues,
        expected_issues=task_state.expected_issues,
        final_action=req.final_action,
        expected_action=task_state.expected_action or "",
        steps_taken=req.steps_taken,
    )

    # Find target score for this task
    target = next((t.target_score for t in _TASK_CATALOGUE if t.id == task_id), 0.5)
    passed = score >= target

    feedback_parts = []
    for issue in task_state.expected_issues:
        if issue in req.detected_issues:
            feedback_parts.append(f"✓ {issue}")
        else:
            feedback_parts.append(f"✗ {issue} (missed)")

    if req.final_action == task_state.expected_action:
        feedback_parts.append(f"✓ final_action={req.final_action}")
    else:
        feedback_parts.append(
            f"✗ final_action={req.final_action} (expected {task_state.expected_action})"
        )

    return GradeResponse(
        task_id=task_id,
        score=score,
        passed=passed,
        feedback="; ".join(feedback_parts),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["utility"])
def root() -> Dict[str, Any]:
    """Landing page — confirms the server is up and shows available endpoints."""
    return {
        "name": "Invoice Validation Environment",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/reset", "/step", "/state", "/tasks",
            "/openenv/reset", "/openenv/step", "/openenv/state", "/openenv/tasks",
            "/tasks/{task_id}/grade", "/openenv/tasks/{task_id}/grade",
        ],
        "usage": "POST /reset with {\"difficulty\": \"easy\"} to start an episode",
    }


@app.get("/health", response_model=HealthResponse, tags=["utility"])
@app.get("/openenv/health", response_model=HealthResponse, tags=["utility"])
def health_check() -> HealthResponse:
    """Liveness probe used by Docker and HF Spaces."""
    return HealthResponse(status="ok")


# --- Tasks list ---

@app.get("/tasks", response_model=TaskListResponse, tags=["tasks"])
def list_tasks() -> TaskListResponse:
    """Return all available tasks with grader metadata."""
    return _handle_tasks()


@app.get("/openenv/tasks", response_model=TaskListResponse, tags=["tasks"])
def openenv_list_tasks() -> TaskListResponse:
    """OpenEnv checker alias for GET /tasks."""
    return _handle_tasks()


# --- Grade endpoints ---

@app.post("/tasks/{task_id}/grade", response_model=GradeResponse, tags=["tasks"])
def grade_task(task_id: str, req: GradeRequest) -> GradeResponse:
    """
    Grade a completed episode for the given task.

    Body example::

        {
          "detected_issues": ["missing_gst_number"],
          "final_action": "request_missing_info",
          "steps_taken": 3
        }
    """
    return _handle_grade(task_id, req)


@app.post("/openenv/tasks/{task_id}/grade", response_model=GradeResponse, tags=["tasks"])
def openenv_grade_task(task_id: str, req: GradeRequest) -> GradeResponse:
    """OpenEnv checker alias for POST /tasks/{task_id}/grade."""
    return _handle_grade(task_id, req)


# --- Reset ---

@app.post("/reset", response_model=InvoiceObservation, tags=["environment"])
def reset_environment(request: Optional[ResetRequest] = None) -> InvoiceObservation:
    """
    Reset the environment to a fresh episode.

    Pass ``difficulty`` as one of ``easy``, ``medium``, or ``hard``.
    Body is **optional** — defaults to ``easy`` if omitted.
    """
    return _handle_reset(request)


@app.post("/openenv/reset", response_model=InvoiceObservation, tags=["environment"])
def openenv_reset(request: Optional[ResetRequest] = None) -> InvoiceObservation:
    """OpenEnv checker alias for POST /reset. Body is optional."""
    return _handle_reset(request)


# --- Step ---

@app.post("/step", response_model=StepResult, tags=["environment"])
def take_step(action: InvoiceAction) -> StepResult:
    """
    Apply one action to the environment.

    Body example::

        {"action_type": "add_issue", "value": "missing_gst_number"}
    """
    return _handle_step(action)


@app.post("/openenv/step", response_model=StepResult, tags=["environment"])
def openenv_step(action: InvoiceAction) -> StepResult:
    """OpenEnv checker alias for POST /step."""
    return _handle_step(action)


# --- State ---

@app.get("/state", response_model=InvoiceState, tags=["debug"])
def get_state() -> InvoiceState:
    """Return raw InvoiceState including ground-truth fields (for graders/debug)."""
    return _handle_state()


@app.get("/openenv/state", response_model=InvoiceState, tags=["debug"])
def openenv_state() -> InvoiceState:
    """OpenEnv checker alias for GET /state."""
    return _handle_state()


# ---------------------------------------------------------------------------
# Server entry point (used by [project.scripts] server = "app.main:main")
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the uvicorn server. Called by the `server` console script."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()

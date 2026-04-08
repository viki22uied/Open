"""
FastAPI server exposing the Invoice Review environment as HTTP endpoints.

Endpoints:
    GET  /                -> Health check
    GET  /tasks           -> List available tasks
    POST /reset           -> Reset environment with a task_id
    POST /step            -> Execute an action
    GET  /state           -> Get current internal state
    GET  /health          -> Health check (for Docker/HF Space probes)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.environment import InvoiceReviewEnv
from src.models import Action, ActionType, ErrorCategory, Severity
from src.tasks.registry import list_tasks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & environment instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Invoice Review OpenEnv",
    description=(
        "An OpenEnv benchmark environment for AI agent evaluation. "
        "Simulates real-world invoice/procurement review workflows."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = InvoiceReviewEnv()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    action_type: str
    invoice_id: Optional[str] = None
    error_category: Optional[str] = None
    error_description: Optional[str] = None
    severity: Optional[str] = None
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check / welcome."""
    return {
        "environment": "Invoice Review OpenEnv",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    """Health check for container orchestration."""
    return {"status": "healthy"}


@app.get("/tasks")
def get_tasks():
    """List all available tasks with metadata."""
    tasks = list_tasks()
    return {"tasks": [t.model_dump() for t in tasks]}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    """Reset the environment with the specified task.

    Returns the initial observation.
    """
    if request is None:
        request = ResetRequest()
    try:
        obs = env.reset(task_id=request.task_id)
        logger.info(f"Environment reset with task_id={request.task_id!r}")
        return {"observation": obs.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """Execute an action in the environment.

    Returns observation, reward, done, and info.
    """
    try:
        # Parse action_type
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid action_type: {request.action_type!r}. "
                    f"Valid types: {[a.value for a in ActionType]}"
                ),
            )

        # Parse optional enums
        error_category = None
        if request.error_category:
            try:
                error_category = ErrorCategory(request.error_category)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid error_category: {request.error_category!r}.",
                )

        severity = None
        if request.severity:
            try:
                severity = Severity(request.severity)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity: {request.severity!r}.",
                )

        action = Action(
            action_type=action_type,
            invoice_id=request.invoice_id,
            error_category=error_category,
            error_description=request.error_description,
            severity=severity,
            reason=request.reason,
        )

        result = env.step(action)
        logger.info(
            f"Step {result.observation.step_number}: "
            f"{request.action_type} -> reward={result.reward.value:.3f}, "
            f"done={result.done}"
        )
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "info": result.info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Step error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    """Return the full internal environment state."""
    try:
        s = env.state()
        return {"state": s.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

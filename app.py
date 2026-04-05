"""
app.py - FastAPI HTTP server for City Pothole Repair Scheduler
Serves the OpenEnv environment via REST API on port 7860 (HuggingFace default).
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import PotholeRepairEnv
from models import Action
from tasks import list_tasks


# ─────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────

class AppState:
    env: Optional[PotholeRepairEnv] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create default env
    app_state.env = PotholeRepairEnv(task_name="critical_repair")
    yield
    # Shutdown
    if app_state.env:
        app_state.env.close()


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="City Pothole Repair Scheduler",
    description="OpenEnv environment for AI agent training — schedules road repair crews.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "critical_repair"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def health():
    """Health check — judges ping this first."""
    return {
        "status": "ok",
        "env": "PotholeRepairEnv",
        "version": "1.0.0",
    }

@app.get("/reset")
def reset():
    """
    Reset environment to a fresh episode.
    Returns initial Observation.
    """
    try:
        app_state.env = PotholeRepairEnv()
        obs = app_state.env.reset()
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """
    Reset environment to a fresh episode.
    Returns initial Observation.
    """
    try:
        app_state.env = PotholeRepairEnv(task_name=request.task_name)
        obs = app_state.env.reset(task_name=request.task_name)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
def step(action: Action):
    """
    Execute one action.
    Returns StepResult with observation, reward, done, info.
    """
    if app_state.env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        result = app_state.env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
def state():
    """Return current environment state without advancing the episode."""
    if app_state.env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        obs = app_state.env.state()
        return obs.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/score")
def score():
    """Return current final score from grader (useful for debugging)."""
    if app_state.env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {
        "task": app_state.env.task_name,
        "score": app_state.env.get_final_score(),
    }


@app.get("/tasks")
def tasks():
    """Return all available tasks with descriptions."""
    return {"tasks": list_tasks()}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
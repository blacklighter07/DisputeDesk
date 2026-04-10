from __future__ import annotations

import os

from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from openenv.core.env_server import create_app

from dispute_desk.baseline import run_baseline
from dispute_desk.config import load_environment
from dispute_desk.models import (
    BaselineResponse,
    EnvironmentStateModel,
    GraderResponse,
    TaskCatalogResponse,
    CaseAction,
    CaseObservation,
)
from dispute_desk.server.dispute_desk_environment import DisputeDeskEnvironment

load_environment()

app = create_app(
    DisputeDeskEnvironment,
    CaseAction,
    CaseObservation,
    env_name="dispute_desk_env",
    max_concurrent_envs=4,
)
app.title = "DisputeDesk Environment API"
app.version = "0.1.0"
app.description = "OpenEnv-style HTTP API for marketplace dispute resolution."

environment = DisputeDeskEnvironment()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    destination = "/web" if _web_interface_enabled() else "/docs"
    return RedirectResponse(url=destination, status_code=307)


@app.get("/tasks", response_model=TaskCatalogResponse, tags=["Tasks"])
def tasks() -> TaskCatalogResponse:
    return TaskCatalogResponse(
        tasks=environment.tasks(),
        action_schema=CaseAction.model_json_schema(),
    )


@app.get("/grader", response_model=GraderResponse, tags=["Tasks"])
def grader() -> GraderResponse:
    return environment.grader_report()


@app.get("/baseline", response_model=BaselineResponse, tags=["Tasks"])
def baseline() -> BaselineResponse:
    try:
        return run_baseline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


def _web_interface_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"}

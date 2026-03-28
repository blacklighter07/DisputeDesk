from __future__ import annotations

from fastapi import FastAPI, HTTPException
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.types import ServerMode

from dispute_desk.baseline import run_baseline
from dispute_desk.config import load_environment
from dispute_desk.models import (
    BaselineResponse,
    CaseAction,
    CaseObservation,
    EnvironmentStateModel,
    GraderResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    TaskCatalogResponse,
)
from dispute_desk.server.dispute_desk_environment import DisputeDeskEnvironment

load_environment()

app = FastAPI(
    title="DisputeDesk Environment API",
    version="0.1.0",
    description="OpenEnv-style HTTP API for marketplace dispute resolution.",
)
environment = DisputeDeskEnvironment()
openenv_server = HTTPEnvServer(
    DisputeDeskEnvironment,
    CaseAction,
    CaseObservation,
    max_concurrent_envs=4,
)
openenv_server.register_routes(app, mode=ServerMode.PRODUCTION)


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
def reset(request: ResetRequest | None = None) -> ResetResponse:
    request = request or ResetRequest()
    observation = environment.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    return ResetResponse(observation=observation, reward=observation.reward, done=observation.done)


@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step(request: StepRequest) -> StepResponse:
    observation = environment.step(request.action)
    return StepResponse(observation=observation, reward=observation.reward, done=observation.done)


@app.get("/state", response_model=EnvironmentStateModel, tags=["State"])
def state() -> EnvironmentStateModel:
    return environment.state


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

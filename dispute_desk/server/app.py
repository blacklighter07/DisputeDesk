from __future__ import annotations

import time
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_COOKIE_NAME = "dispute_desk_demo_session"
DEMO_COOKIE_TTL_SECONDS = 60 * 60 * 12
MAX_DEMO_SESSIONS = 24

_demo_lock = Lock()
_demo_sessions: dict[str, tuple[DisputeDeskEnvironment, float]] = {}

app.mount("/demo/assets", StaticFiles(directory=STATIC_DIR), name="demo-assets")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo", status_code=307)


@app.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html")


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


@app.post("/demo/api/reset", response_model=ResetResponse, tags=["Demo"])
def demo_reset(
    http_request: Request,
    response: Response,
    request: ResetRequest | None = None,
) -> ResetResponse:
    request = request or ResetRequest()
    demo_environment = _get_demo_environment(http_request, response)
    observation = demo_environment.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    return ResetResponse(observation=observation, reward=observation.reward, done=observation.done)


@app.post("/demo/api/step", response_model=StepResponse, tags=["Demo"])
def demo_step(http_request: Request, request: StepRequest) -> StepResponse:
    demo_environment = _require_demo_environment(http_request)
    observation = demo_environment.step(request.action)
    return StepResponse(observation=observation, reward=observation.reward, done=observation.done)


@app.get("/demo/api/state", response_model=EnvironmentStateModel, tags=["Demo"])
def demo_state(http_request: Request) -> EnvironmentStateModel:
    demo_environment = _require_demo_environment(http_request)
    return demo_environment.state


@app.get("/demo/api/grader", response_model=GraderResponse, tags=["Demo"])
def demo_grader(http_request: Request) -> GraderResponse:
    demo_environment = _require_demo_environment(http_request)
    return demo_environment.grader_report()


@app.get("/demo/api/tasks", response_model=TaskCatalogResponse, tags=["Demo"])
def demo_tasks() -> TaskCatalogResponse:
    return tasks()


def _get_demo_environment(http_request: Request, response: Response) -> DisputeDeskEnvironment:
    session_id = http_request.cookies.get(DEMO_COOKIE_NAME)
    now = time.time()

    with _demo_lock:
        _prune_demo_sessions(now)

        if session_id and session_id in _demo_sessions:
            demo_environment, _ = _demo_sessions[session_id]
            _demo_sessions[session_id] = (demo_environment, now)
            return demo_environment

        session_id = str(uuid.uuid4())
        demo_environment = DisputeDeskEnvironment()
        _demo_sessions[session_id] = (demo_environment, now)
        response.set_cookie(
            key=DEMO_COOKIE_NAME,
            value=session_id,
            max_age=DEMO_COOKIE_TTL_SECONDS,
            httponly=True,
            samesite="lax",
        )
        return demo_environment


def _require_demo_environment(http_request: Request) -> DisputeDeskEnvironment:
    session_id = http_request.cookies.get(DEMO_COOKIE_NAME)
    now = time.time()
    with _demo_lock:
        _prune_demo_sessions(now)
        if not session_id or session_id not in _demo_sessions:
            raise HTTPException(
                status_code=409,
                detail="Demo session not initialized. Start a case from the UI first.",
            )
        demo_environment, _ = _demo_sessions[session_id]
        _demo_sessions[session_id] = (demo_environment, now)
        return demo_environment


def _prune_demo_sessions(now: float) -> None:
    stale_ids = [
        session_id
        for session_id, (_, updated_at) in _demo_sessions.items()
        if now - updated_at > DEMO_COOKIE_TTL_SECONDS
    ]
    for session_id in stale_ids:
        demo_environment, _ = _demo_sessions.pop(session_id)
        demo_environment.close()

    if len(_demo_sessions) <= MAX_DEMO_SESSIONS:
        return

    overflow = len(_demo_sessions) - MAX_DEMO_SESSIONS
    oldest_sessions = sorted(_demo_sessions.items(), key=lambda item: item[1][1])[:overflow]
    for session_id, (demo_environment, _) in oldest_sessions:
        _demo_sessions.pop(session_id, None)
        demo_environment.close()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

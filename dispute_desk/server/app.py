from __future__ import annotations

import time
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
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
DEMO_HTML_PATH = STATIC_DIR / "demo.html"
DEMO_CSS_PATH = STATIC_DIR / "styles.css"
DEMO_JS_PATH = STATIC_DIR / "app.js"
DEMO_STATIC_READY = DEMO_HTML_PATH.is_file() and DEMO_CSS_PATH.is_file() and DEMO_JS_PATH.is_file()
DEMO_COOKIE_NAME = "dispute_desk_demo_session"
DEMO_COOKIE_TTL_SECONDS = 60 * 60 * 12
MAX_DEMO_SESSIONS = 24

_demo_lock = Lock()
_demo_sessions: dict[str, tuple[DisputeDeskEnvironment, float]] = {}

if DEMO_STATIC_READY:
    app.mount("/demo/assets", StaticFiles(directory=STATIC_DIR), name="demo-assets")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo", status_code=307)


@app.get("/demo", include_in_schema=False, response_model=None)
def demo() -> Response:
    if DEMO_STATIC_READY:
        return FileResponse(DEMO_HTML_PATH)
    return HTMLResponse(_fallback_demo_html())


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


def _fallback_demo_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DisputeDesk Demo</title>
    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #16324a;
        background: linear-gradient(180deg, #f8f2e8 0%, #efe3d3 100%);
      }
      .shell {
        max-width: 960px;
        margin: 32px auto;
        padding: 24px;
      }
      .panel {
        background: rgba(255, 252, 247, 0.92);
        border: 1px solid rgba(22, 50, 74, 0.08);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(32, 47, 64, 0.12);
        padding: 24px;
      }
      h1, h2 {
        margin: 0 0 12px;
      }
      p {
        line-height: 1.6;
        color: #53667a;
      }
      .row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 16px 0;
      }
      select, button {
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid rgba(22, 50, 74, 0.12);
        font: inherit;
      }
      button {
        background: #0d6a73;
        color: white;
        border: 0;
        cursor: pointer;
      }
      .muted {
        font-size: 14px;
      }
      .stack {
        display: grid;
        gap: 12px;
        margin-top: 16px;
      }
      .card {
        border: 1px solid rgba(22, 50, 74, 0.1);
        border-radius: 16px;
        padding: 16px;
        background: white;
      }
      code {
        background: rgba(22, 50, 74, 0.06);
        padding: 2px 6px;
        border-radius: 8px;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="panel">
        <h1>DisputeDesk Demo</h1>
        <p>
          This deployment is running in fallback mode because the static demo bundle
          was not present in the image. The environment API is still live and usable.
        </p>
        <div class="row">
          <select id="taskSelect"></select>
          <button id="startBtn">Start Case</button>
          <button onclick="window.location.href='/docs'">Open API Docs</button>
        </div>
        <p class="muted">
          Expected asset directory: <code>dispute_desk/server/static/</code>
        </p>
        <div class="stack">
          <div class="card">
            <h2 id="objective">No task started</h2>
            <p id="summary">Start a task to preview the case summary and evidence list.</p>
          </div>
          <div class="card">
            <h2>Artifacts</h2>
            <div id="artifacts">No artifacts loaded.</div>
          </div>
        </div>
      </div>
    </div>
    <script>
      async function fetchJson(url, options = {}) {
        const response = await fetch(url, {
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          ...options,
        });
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json();
      }
      async function loadTasks() {
        const payload = await fetchJson("/demo/api/tasks");
        const select = document.getElementById("taskSelect");
        select.innerHTML = payload.tasks
          .map((task) => `<option value="${task.task_id}">${task.task_id}</option>`)
          .join("");
      }
      async function startCase() {
        const taskId = document.getElementById("taskSelect").value;
        const payload = await fetchJson("/demo/api/reset", {
          method: "POST",
          body: JSON.stringify({ task_id: taskId }),
        });
        const observation = payload.observation;
        document.getElementById("objective").textContent = observation.objective;
        document.getElementById("summary").textContent = observation.case_summary;
        document.getElementById("artifacts").innerHTML = observation.available_artifacts
          .map((artifact) => `<div><strong>${artifact.title}</strong>: ${artifact.summary}</div>`)
          .join("");
      }
      document.getElementById("startBtn").addEventListener("click", startCase);
      loadTasks().catch((error) => {
        document.getElementById("summary").textContent = error.message;
      });
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    main()

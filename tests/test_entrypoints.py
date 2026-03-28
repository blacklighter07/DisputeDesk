import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from dispute_desk.server.app import app


PROJECT_PARENT = Path(__file__).resolve().parents[2]


def test_wrapper_imports_cleanly_from_parent_directory():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "module = importlib.import_module('dispute_desk_env.server.app'); "
                "print(module.app.title)"
            ),
        ],
        cwd=PROJECT_PARENT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DisputeDesk Environment API" in result.stdout


def test_reset_accepts_missing_body():
    client = TestClient(app)
    response = client.post("/reset")

    assert response.status_code == 200
    payload = response.json()
    assert payload["done"] is False
    assert payload["observation"]["task_id"]


def test_root_inference_wrapper_imports_cleanly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                "runpy.run_path('dispute_desk_env/inference.py', run_name='not_main'); "
                "print('ok')"
            ),
        ],
        cwd=PROJECT_PARENT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "ok"

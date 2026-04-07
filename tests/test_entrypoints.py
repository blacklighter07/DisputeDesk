import importlib.util
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from dispute_desk.baseline import _fallback_decision
from dispute_desk.server.app import app


PROJECT_PARENT = Path(__file__).resolve().parents[2]


def _load_root_inference_module():
    target = PROJECT_PARENT / "dispute_desk_env" / "inference.py"
    spec = importlib.util.spec_from_file_location("dispute_desk_env_root_inference", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_root_inference_emits_structured_stdout_blocks(monkeypatch):
    module = _load_root_inference_module()
    emitted_lines: list[str] = []

    monkeypatch.setattr(module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(module, "get_api_base_url", lambda: "https://api.openai.com/v1")
    monkeypatch.setattr(module, "get_model_name", lambda default: "test-model")
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(module, "_write_baseline_output", lambda result: None)

    result = module.run_inference(
        model="test-model",
        emit=emitted_lines.append,
        decision_resolver=lambda client, model, observation_payload: _fallback_decision(observation_payload),
    )

    assert any(line.startswith("[START] task=late_delivery_refund") for line in emitted_lines)
    assert any(line.startswith("[STEP] task=late_delivery_refund") for line in emitted_lines)
    assert any(line.startswith("[END] task=suspicious_refund_abuse") for line in emitted_lines)
    assert result.average_score >= 0.98

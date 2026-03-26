from __future__ import annotations

from dataclasses import dataclass

from dispute_desk import DisputeDeskEnv
from dispute_desk.models import CaseAction
from dispute_desk.server.dispute_desk_environment import DisputeDeskEnvironment


@dataclass
class _FakeResponse:
    payload: dict

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_client_parses_step_results_and_state_models():
    environment = DisputeDeskEnvironment(default_task_id="late_delivery_refund")
    client = DisputeDeskEnv(base_url="http://127.0.0.1:8000")

    observation = environment.reset(task_id="late_delivery_refund")
    result = client._parse_result(
        {
            "observation": observation.model_dump(mode="json"),
            "reward": observation.reward,
            "done": observation.done,
        }
    )

    assert result.observation.task_id == "late_delivery_refund"
    assert result.done is False

    observation = environment.step(
        CaseAction(action_type="review_artifact", artifact_id="order_summary")
    )
    state = client._parse_state(environment.state.model_dump(mode="json"))

    assert any(
        artifact.artifact_id == "order_summary"
        for artifact in observation.revealed_artifacts
    )
    assert state.task_id == "late_delivery_refund"
    assert state.reviewed_artifact_ids == ["order_summary"]


def test_client_http_helpers_validate_typed_responses(monkeypatch):
    environment = DisputeDeskEnvironment(default_task_id="late_delivery_refund")
    observation = environment.reset(task_id="late_delivery_refund")
    grader = environment.grader_report().model_dump(mode="json")
    client = DisputeDeskEnv(base_url="http://127.0.0.1:8000")

    payloads = {
        "http://127.0.0.1:8000/health": {"status": "healthy"},
        "http://127.0.0.1:8000/metadata": environment.metadata(),
        "http://127.0.0.1:8000/schema": {
            "action": CaseAction.model_json_schema(),
            "observation": observation.model_json_schema(),
            "state": environment.state.model_json_schema(),
        },
        "http://127.0.0.1:8000/tasks": {
            "tasks": [task.model_dump(mode="json") for task in environment.tasks()],
            "action_schema": CaseAction.model_json_schema(),
        },
        "http://127.0.0.1:8000/grader": grader,
    }

    def fake_get(url: str, timeout: float):
        assert timeout == 10.0
        return _FakeResponse(payload=payloads[url])

    monkeypatch.setattr("dispute_desk.client.requests.get", fake_get)

    assert client.health().status == "healthy"
    assert client.metadata().name == "DisputeDesk"
    assert "properties" in client.schema().action
    assert any(task.task_id == "late_delivery_refund" for task in client.tasks().tasks)
    assert client.grader().task_id == "late_delivery_refund"


def test_client_normalizes_websocket_urls_for_http_helpers():
    client = DisputeDeskEnv(base_url="ws://127.0.0.1:8000/ws")
    assert client.http_base_url == "http://127.0.0.1:8000"

from __future__ import annotations

from typing import Any

import requests
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from dispute_desk.models import (
    CaseAction,
    CaseObservation,
    EnvironmentStateModel,
    GraderResponse,
    HealthResponse,
    MetadataResponse,
    SchemaResponse,
    TaskCatalogResponse,
)


class DisputeDeskEnv(EnvClient[CaseAction, CaseObservation, EnvironmentStateModel]):
    """Typed OpenEnv client for the DisputeDesk environment."""

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        provider: Any | None = None,
        mode: str | None = None,
        request_timeout_s: float = 10.0,
    ):
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            provider=provider,
            mode=mode,
        )
        self._http_base_url = _to_http_base_url(base_url)
        self._request_timeout_s = request_timeout_s

    @property
    def http_base_url(self) -> str:
        return self._http_base_url

    def _step_payload(self, action: CaseAction) -> dict[str, Any]:
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CaseObservation]:
        observation_payload = dict(payload.get("observation", {}))
        observation_payload.setdefault("done", payload.get("done", False))
        if "reward" in payload:
            observation_payload.setdefault("reward", payload["reward"])

        observation = CaseObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> EnvironmentStateModel:
        return EnvironmentStateModel.model_validate(payload)

    def health(self) -> HealthResponse:
        return self._get_model("/health", HealthResponse)

    def metadata(self) -> MetadataResponse:
        return self._get_model("/metadata", MetadataResponse)

    def schema(self) -> SchemaResponse:
        return self._get_model("/schema", SchemaResponse)

    def tasks(self) -> TaskCatalogResponse:
        return self._get_model("/tasks", TaskCatalogResponse)

    def grader(self) -> GraderResponse:
        return self._get_model("/grader", GraderResponse)

    def _get_model(self, path: str, model_type: Any):
        response = requests.get(
            f"{self._http_base_url}{path}",
            timeout=self._request_timeout_s,
        )
        response.raise_for_status()
        return model_type.model_validate(response.json())


def _to_http_base_url(base_url: str) -> str:
    normalized = base_url.removesuffix("/").removesuffix("/ws")
    if normalized.startswith("ws://"):
        return "http://" + normalized[len("ws://") :]
    if normalized.startswith("wss://"):
        return "https://" + normalized[len("wss://") :]
    return normalized

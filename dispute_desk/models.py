from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


Classification = Literal[
    "item_not_received",
    "partial_damage",
    "wrong_item",
    "suspected_abuse",
    "other",
]
Severity = Literal["low", "medium", "high"]
ResolutionType = Literal[
    "refund_full",
    "refund_partial",
    "replace_item",
    "deny",
    "escalate",
]
EscalationTarget = Literal["none", "seller_support", "trust_safety", "billing_ops"]
MessageTemplate = Literal[
    "refund_approved",
    "partial_refund_approved",
    "replacement_offered",
    "escalated_for_review",
    "claim_denied",
]
Difficulty = Literal["easy", "medium", "hard"]
ArtifactRelevance = Literal["required", "helpful", "distractor"]


class ArtifactPreview(BaseModel):
    artifact_id: str
    title: str
    summary: str
    reviewed: bool = False


class ArtifactDetail(ArtifactPreview):
    content: str
    relevance: ArtifactRelevance


class ResolutionDraft(BaseModel):
    resolution: ResolutionType | None = None
    refund_amount: float | None = Field(default=None, ge=0)
    require_return: bool | None = None
    escalation_target: EscalationTarget | None = None
    reason_code: str | None = None
    message_template: MessageTemplate | None = None


class CaseAction(Action):
    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "review_artifact",
        "classify_case",
        "request_more_context",
        "resolve_case",
    ]
    artifact_id: str | None = None
    classification: Classification | None = None
    severity: Severity | None = None
    context_key: str | None = None
    resolution: ResolutionType | None = None
    refund_amount: float | None = Field(default=None, ge=0)
    require_return: bool | None = None
    escalation_target: EscalationTarget | None = None
    reason_code: str | None = None
    message_template: MessageTemplate | None = None


class CaseObservation(Observation):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    case_id: str
    difficulty: Difficulty
    objective: str
    case_summary: str
    available_artifacts: list[ArtifactPreview]
    revealed_artifacts: list[ArtifactDetail]
    revealed_context: dict[str, str] = Field(default_factory=dict)
    last_event: str
    allowed_actions: list[str]
    reviewed_artifact_ids: list[str]
    steps_remaining: int
    current_classification: Classification | None = None
    current_severity: Severity | None = None
    provisional_score: float = 0.0


class EnvironmentStateModel(State):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    case_id: str | None = None
    reviewed_artifact_ids: list[str] = Field(default_factory=list)
    requested_context_keys: list[str] = Field(default_factory=list)
    revealed_context: dict[str, str] = Field(default_factory=dict)
    classification: Classification | None = None
    severity: Severity | None = None
    current_resolution: ResolutionDraft | None = None
    cumulative_reward: float = 0.0
    provisional_score: float = 0.0
    final_score: float | None = None
    done: bool = False


class TaskSummary(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    objective: str
    max_steps: int


class TaskCatalogResponse(BaseModel):
    tasks: list[TaskSummary]
    action_schema: dict[str, Any]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = Field(default=None, ge=0)
    episode_id: str | None = None
    task_id: str | None = None


class ResetResponse(BaseModel):
    observation: CaseObservation
    reward: float | None = None
    done: bool = False


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: CaseAction


class StepResponse(BaseModel):
    observation: CaseObservation
    reward: float | None = None
    done: bool = False


class SchemaResponse(BaseModel):
    action: dict[str, Any]
    observation: dict[str, Any]
    state: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"


class MetadataResponse(BaseModel):
    name: str
    description: str
    version: str
    author: str


class GraderResponse(BaseModel):
    task_id: str
    score: float
    passed: bool
    components: dict[str, float]
    notes: list[str]


class BaselineTaskResult(BaseModel):
    task_id: str
    score: float
    steps: int
    passed: bool


class BaselineResponse(BaseModel):
    model: str
    average_score: float
    tasks: list[BaselineTaskResult]

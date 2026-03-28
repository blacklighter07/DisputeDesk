from __future__ import annotations

import random
import uuid

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from dispute_desk.grading import estimate_progress_score, grade_episode
from dispute_desk.models import (
    ArtifactDetail,
    ArtifactPreview,
    CaseAction,
    CaseObservation,
    EnvironmentStateModel,
    GraderResponse,
    ResolutionDraft,
)
from dispute_desk.scenarios import (
    TASK_VARIANTS,
    DisputeScenario,
    SCENARIOS,
    get_scenario,
    task_catalog,
)


class DisputeDeskEnvironment(Environment[CaseAction, CaseObservation, EnvironmentStateModel]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task_id: str | None = None):
        super().__init__()
        self._rng = random.Random(0)
        self._default_task_id = default_task_id
        self._scenario: DisputeScenario | None = None
        self._state = EnvironmentStateModel()
        self._revealed_artifacts: dict[str, ArtifactDetail] = {}
        self._last_report: GraderResponse | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
    ) -> CaseObservation:
        if seed is not None:
            self._rng.seed(seed)
        scenario = self._pick_scenario(task_id=task_id, seed=seed)
        self._scenario = scenario
        self._revealed_artifacts = {}
        self._last_report = None
        self._state = EnvironmentStateModel(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=scenario.task_id,
            case_id=scenario.case_id,
            reviewed_artifact_ids=[],
            requested_context_keys=[],
            revealed_context={},
            classification=None,
            severity=None,
            current_resolution=None,
            cumulative_reward=0.0,
            provisional_score=0.0,
            final_score=None,
            done=False,
        )
        return self._build_observation(
            last_event="Episode reset. Review the case evidence before taking a final action.",
            reward=None,
        )

    def step(self, action: CaseAction) -> CaseObservation:
        scenario = self._require_scenario()
        if self._state.done:
            return self._build_observation(
                last_event="Episode already finished. Reset before taking more actions.",
                reward=-0.05,
                done=True,
            )

        self._state.step_count += 1
        reward = -0.01
        last_event = "Action processed."

        if action.action_type == "review_artifact":
            reward, last_event = self._handle_review_artifact(action, scenario)
        elif action.action_type == "classify_case":
            reward, last_event = self._handle_classification(action, scenario)
        elif action.action_type == "request_more_context":
            reward, last_event = self._handle_more_context(action, scenario)
        elif action.action_type == "resolve_case":
            reward, last_event = self._handle_resolution(action, scenario)

        self._state.provisional_score = estimate_progress_score(scenario, self._state)
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)

        if not self._state.done and self._state.step_count >= scenario.max_steps:
            self._last_report = grade_episode(scenario, self._state)
            self._state.done = True
            self._state.final_score = self._last_report.score
            reward = round(reward - 0.10, 4)
            last_event = "Step limit reached before a clean resolution was completed."

        return self._build_observation(
            last_event=last_event,
            reward=reward,
            done=self._state.done,
        )

    @property
    def state(self) -> EnvironmentStateModel:
        return self._state

    def grader_report(self) -> GraderResponse:
        scenario = self._require_scenario()
        if self._last_report is None:
            self._last_report = grade_episode(scenario, self._state)
        return self._last_report

    def tasks(self):
        return task_catalog()

    def metadata(self) -> dict[str, str]:
        return {
            "name": "DisputeDesk",
            "description": "Marketplace dispute resolution environment with deterministic graders.",
            "version": "0.1.0",
            "author": "Kapil Verma + Codex",
        }

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(**self.metadata())

    def _pick_scenario(self, task_id: str | None, seed: int | None) -> DisputeScenario:
        effective_task_id = task_id or self._default_task_id
        if effective_task_id:
            return get_scenario(effective_task_id, seed=seed)

        scenario = self._rng.choice(SCENARIOS)
        variant_count = len(TASK_VARIANTS[scenario.task_id])
        if variant_count == 1:
            return scenario
        return get_scenario(scenario.task_id, variant_index=self._rng.randrange(variant_count))

    def _handle_review_artifact(
        self, action: CaseAction, scenario: DisputeScenario
    ) -> tuple[float, str]:
        if not action.artifact_id:
            return -0.08, "review_artifact requires an artifact_id."

        artifact = next(
            (item for item in scenario.artifacts if item.artifact_id == action.artifact_id),
            None,
        )
        if artifact is None:
            return -0.08, f"Artifact '{action.artifact_id}' does not exist."
        if artifact.artifact_id in self._revealed_artifacts:
            return -0.03, f"Artifact '{artifact.title}' was already reviewed."

        detail = ArtifactDetail(
            artifact_id=artifact.artifact_id,
            title=artifact.title,
            summary=artifact.summary,
            reviewed=True,
            content=artifact.content,
            relevance=artifact.relevance,
        )
        self._revealed_artifacts[artifact.artifact_id] = detail
        self._state.reviewed_artifact_ids.append(artifact.artifact_id)

        if artifact.relevance == "required":
            reward = 0.05
        elif artifact.relevance == "helpful":
            reward = 0.015
        else:
            reward = -0.01
        return reward, f"Reviewed artifact '{artifact.title}'."

    def _handle_classification(
        self, action: CaseAction, scenario: DisputeScenario
    ) -> tuple[float, str]:
        if not action.classification or not action.severity:
            return -0.08, "classify_case requires both classification and severity."

        self._state.classification = action.classification
        self._state.severity = action.severity

        reviewed_required = len(
            set(self._state.reviewed_artifact_ids) & set(scenario.expected.required_artifact_ids)
        )
        reward = -0.015
        if reviewed_required < max(1, len(scenario.expected.required_artifact_ids) // 2):
            reward -= 0.015
        if action.classification == scenario.expected.classification:
            reward += 0.05
        if action.severity == scenario.expected.severity:
            reward += 0.03
        return reward, "Updated case classification."

    def _handle_more_context(
        self, action: CaseAction, scenario: DisputeScenario
    ) -> tuple[float, str]:
        if not action.context_key:
            return -0.08, "request_more_context requires a context_key."
        if action.context_key not in scenario.extra_context:
            return -0.06, f"Context key '{action.context_key}' is unavailable."
        if action.context_key in self._state.requested_context_keys:
            return -0.03, f"Context '{action.context_key}' was already requested."

        self._state.requested_context_keys.append(action.context_key)
        self._state.revealed_context[action.context_key] = scenario.extra_context[action.context_key]
        reward = 0.05 if action.context_key in scenario.expected.required_context_keys else 0.01
        return reward, f"Retrieved context '{action.context_key}'."

    def _handle_resolution(
        self, action: CaseAction, scenario: DisputeScenario
    ) -> tuple[float, str]:
        if not action.resolution:
            return -0.08, "resolve_case requires a resolution."

        self._state.current_resolution = ResolutionDraft(
            resolution=action.resolution,
            refund_amount=action.refund_amount,
            require_return=action.require_return,
            escalation_target=action.escalation_target,
            reason_code=action.reason_code,
            message_template=action.message_template,
        )
        self._last_report = grade_episode(scenario, self._state)
        self._state.done = True
        self._state.final_score = self._last_report.score

        missing_artifacts = set(scenario.expected.required_artifact_ids) - set(
            self._state.reviewed_artifact_ids
        )
        missing_context = set(scenario.expected.required_context_keys) - set(
            self._state.requested_context_keys
        )
        process_penalty = 0.0
        if missing_artifacts:
            process_penalty -= 0.06
        if missing_context:
            process_penalty -= 0.05
        if self._state.classification is None or self._state.severity is None:
            process_penalty -= 0.04

        terminal_reward = round((self._last_report.score * 0.85) - 0.12 + process_penalty, 4)
        return terminal_reward, "Submitted final resolution and graded the episode."

    def _build_observation(
        self,
        last_event: str,
        reward: float | None,
        done: bool | None = None,
    ) -> CaseObservation:
        scenario = self._require_scenario()
        effective_done = self._state.done if done is None else done
        return CaseObservation(
            done=effective_done,
            reward=reward,
            metadata={
                "score_if_resolved_now": self.grader_report().score if self._state.done else None,
                "available_context_keys": list(scenario.extra_context.keys()),
            },
            task_id=scenario.task_id,
            case_id=scenario.case_id,
            difficulty=scenario.difficulty,
            objective=scenario.objective,
            case_summary=scenario.case_summary,
            available_artifacts=[
                ArtifactPreview(
                    artifact_id=artifact.artifact_id,
                    title=artifact.title,
                    summary=artifact.summary,
                    reviewed=artifact.artifact_id in self._revealed_artifacts,
                )
                for artifact in scenario.artifacts
            ],
            revealed_artifacts=list(self._revealed_artifacts.values()),
            revealed_context=dict(self._state.revealed_context),
            last_event=last_event,
            allowed_actions=[
                "review_artifact",
                "classify_case",
                "request_more_context",
                "resolve_case",
            ],
            reviewed_artifact_ids=list(self._state.reviewed_artifact_ids),
            steps_remaining=max(scenario.max_steps - self._state.step_count, 0),
            current_classification=self._state.classification,
            current_severity=self._state.severity,
            provisional_score=self._state.final_score or self._state.provisional_score,
        )

    def _require_scenario(self) -> DisputeScenario:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset before use.")
        return self._scenario

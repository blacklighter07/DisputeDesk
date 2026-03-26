from __future__ import annotations

from math import isclose

from dispute_desk.models import EnvironmentStateModel, GraderResponse, ResolutionDraft
from dispute_desk.scenarios import DisputeScenario


WEIGHTS = {
    "evidence": 0.15,
    "context": 0.12,
    "classification": 0.15,
    "resolution": 0.18,
    "refund_amount": 0.14,
    "escalation": 0.08,
    "return_requirements": 0.06,
    "reason_code": 0.06,
    "message_template": 0.04,
    "efficiency": 0.02,
}


def estimate_progress_score(scenario: DisputeScenario, state: EnvironmentStateModel) -> float:
    evidence_score = _evidence_score(
        reviewed_ids=state.reviewed_artifact_ids,
        required_ids=scenario.expected.required_artifact_ids,
    )
    context_score = _context_score(
        requested_keys=state.requested_context_keys,
        required_keys=scenario.expected.required_context_keys,
    )
    classification_score = 1.0 if state.classification == scenario.expected.classification else 0.0
    severity_score = 1.0 if state.severity == scenario.expected.severity else 0.0

    weighted_score = (
        evidence_score * WEIGHTS["evidence"]
        + context_score * WEIGHTS["context"]
        + ((classification_score * 0.7) + (severity_score * 0.3)) * WEIGHTS["classification"]
    )
    max_progress_weight = WEIGHTS["evidence"] + WEIGHTS["context"] + WEIGHTS["classification"]
    return round(weighted_score / max_progress_weight, 4)


def grade_episode(scenario: DisputeScenario, state: EnvironmentStateModel) -> GraderResponse:
    resolution = state.current_resolution or ResolutionDraft()
    components = {
        "evidence": round(
            _evidence_score(
                reviewed_ids=state.reviewed_artifact_ids,
                required_ids=scenario.expected.required_artifact_ids,
            ),
            4,
        ),
        "context": round(
            _context_score(
                requested_keys=state.requested_context_keys,
                required_keys=scenario.expected.required_context_keys,
            ),
            4,
        ),
        "classification": round(
            _classification_score(
                actual_classification=state.classification,
                expected_classification=scenario.expected.classification,
                actual_severity=state.severity,
                expected_severity=scenario.expected.severity,
            ),
            4,
        ),
        "resolution": round(
            1.0 if resolution.resolution == scenario.expected.resolution else 0.0,
            4,
        ),
        "refund_amount": round(
            _refund_score(
                actual_amount=resolution.refund_amount,
                expected_amount=scenario.expected.refund_amount,
            ),
            4,
        ),
        "escalation": round(
            _escalation_score(
                actual_target=resolution.escalation_target,
                expected_target=scenario.expected.escalation_target,
            ),
            4,
        ),
        "return_requirements": round(
            _return_requirement_score(
                actual_require_return=resolution.require_return,
                expected_require_return=scenario.expected.require_return,
            ),
            4,
        ),
        "reason_code": round(
            1.0 if resolution.reason_code == scenario.expected.reason_code else 0.0,
            4,
        ),
        "message_template": round(
            1.0 if resolution.message_template == scenario.expected.message_template else 0.0,
            4,
        ),
        "efficiency": round(
            _efficiency_score(
                step_count=state.step_count,
                max_steps=scenario.max_steps,
                required_artifact_count=len(scenario.expected.required_artifact_ids),
                required_context_count=len(scenario.expected.required_context_keys),
            ),
            4,
        ),
    }
    total_score = round(
        sum(components[name] * weight for name, weight in WEIGHTS.items()),
        4,
    )
    notes: list[str] = []
    if components["evidence"] < 1.0:
        notes.append("Required evidence was not fully reviewed.")
    if components["context"] < 1.0:
        notes.append("Required policy or operations context was not fully requested.")
    if components["classification"] < 1.0:
        notes.append("Classification or severity did not match expected policy handling.")
    if components["resolution"] < 1.0:
        notes.append("Final resolution type was incorrect.")
    if components["refund_amount"] < 1.0:
        notes.append("Refund amount did not match the policy-compliant outcome.")
    if components["escalation"] < 1.0:
        notes.append("Escalation target was incorrect.")
    if components["return_requirements"] < 1.0:
        notes.append("Return requirement did not match the expected policy flow.")
    if components["reason_code"] < 1.0:
        notes.append("Reason code did not match the expected policy rationale.")
    if components["message_template"] < 1.0:
        notes.append("Customer message template did not match the expected outcome.")
    return GraderResponse(
        task_id=scenario.task_id,
        score=total_score,
        passed=total_score >= 0.8,
        components=components,
        notes=notes,
    )


def _evidence_score(reviewed_ids: list[str], required_ids: list[str]) -> float:
    if not required_ids:
        return 1.0
    reviewed = len(set(reviewed_ids) & set(required_ids))
    return min(reviewed / len(required_ids), 1.0)


def _context_score(requested_keys: list[str], required_keys: list[str]) -> float:
    if not required_keys:
        return 1.0
    requested = len(set(requested_keys) & set(required_keys))
    return min(requested / len(required_keys), 1.0)


def _classification_score(
    actual_classification: str | None,
    expected_classification: str,
    actual_severity: str | None,
    expected_severity: str,
) -> float:
    classification_component = 1.0 if actual_classification == expected_classification else 0.0
    severity_component = 1.0 if actual_severity == expected_severity else 0.0
    return (classification_component * 0.7) + (severity_component * 0.3)


def _refund_score(actual_amount: float | None, expected_amount: float) -> float:
    if expected_amount == 0:
        return 1.0 if actual_amount in (None, 0, 0.0) else 0.0
    if actual_amount is None:
        return 0.0
    if isclose(actual_amount, expected_amount, abs_tol=0.01):
        return 1.0
    relative_error = abs(actual_amount - expected_amount) / expected_amount
    return max(0.0, 1.0 - (relative_error * 2.0))


def _escalation_score(actual_target: str | None, expected_target: str) -> float:
    if expected_target == "none":
        return 1.0 if actual_target in (None, "none") else 0.0
    return 1.0 if actual_target == expected_target else 0.0


def _return_requirement_score(
    actual_require_return: bool | None,
    expected_require_return: bool,
) -> float:
    return 1.0 if actual_require_return is expected_require_return else 0.0


def _efficiency_score(
    step_count: int,
    max_steps: int,
    required_artifact_count: int,
    required_context_count: int,
) -> float:
    target_steps = required_artifact_count + required_context_count + 2
    target_steps = max(3, min(max_steps, target_steps))
    if step_count <= target_steps:
        return 1.0
    remaining_band = max(max_steps - target_steps, 1)
    overflow = step_count - target_steps
    return max(0.0, 1.0 - (overflow / remaining_band))

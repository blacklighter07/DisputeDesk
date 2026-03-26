import pytest

from dispute_desk.baseline import _fallback_decision
from dispute_desk.models import CaseAction
from dispute_desk.scenarios import SCENARIOS, get_scenario
from dispute_desk.server.dispute_desk_environment import DisputeDeskEnvironment


@pytest.mark.parametrize(
    "task_id",
    [scenario.task_id for scenario in SCENARIOS],
)
def test_fallback_decision_matches_expected_policy(task_id: str):
    scenario = get_scenario(task_id)
    environment = DisputeDeskEnvironment(default_task_id=task_id)
    observation = environment.reset(task_id=task_id)

    for artifact in observation.available_artifacts:
        observation = environment.step(
            CaseAction(action_type="review_artifact", artifact_id=artifact.artifact_id)
        )
    for context_key in observation.metadata["available_context_keys"]:
        observation = environment.step(
            CaseAction(action_type="request_more_context", context_key=context_key)
        )

    decision = _fallback_decision(observation.model_dump(mode="json"))

    assert decision.classification == scenario.expected.classification
    assert decision.severity == scenario.expected.severity
    assert decision.resolution == scenario.expected.resolution
    assert decision.refund_amount == scenario.expected.refund_amount
    assert decision.require_return == scenario.expected.require_return
    assert decision.escalation_target == scenario.expected.escalation_target
    assert decision.reason_code == scenario.expected.reason_code
    assert decision.message_template == scenario.expected.message_template

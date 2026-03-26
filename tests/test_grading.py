from dispute_desk.grading import grade_episode
from dispute_desk.models import EnvironmentStateModel, ResolutionDraft
from dispute_desk.scenarios import get_scenario


def test_hard_task_perfect_grade_is_high():
    scenario = get_scenario("suspicious_refund_abuse")
    state = EnvironmentStateModel(
        step_count=4,
        task_id=scenario.task_id,
        case_id=scenario.case_id,
        reviewed_artifact_ids=list(scenario.expected.required_artifact_ids),
        requested_context_keys=list(scenario.expected.required_context_keys),
        classification=scenario.expected.classification,
        severity=scenario.expected.severity,
        current_resolution=ResolutionDraft(
            resolution=scenario.expected.resolution,
            refund_amount=scenario.expected.refund_amount,
            require_return=scenario.expected.require_return,
            escalation_target=scenario.expected.escalation_target,
            reason_code=scenario.expected.reason_code,
            message_template=scenario.expected.message_template,
        ),
    )
    report = grade_episode(scenario, state)
    assert report.score >= 0.95
    assert report.passed is True

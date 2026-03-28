from dispute_desk.models import CaseAction
from dispute_desk.server.dispute_desk_environment import DisputeDeskEnvironment


def test_easy_task_can_be_completed():
    environment = DisputeDeskEnvironment(default_task_id="late_delivery_refund")
    observation = environment.reset(task_id="late_delivery_refund")
    assert observation.task_id == "late_delivery_refund"

    observation = environment.step(CaseAction(action_type="review_artifact", artifact_id="order_summary"))
    assert "order_summary" in observation.reviewed_artifact_ids

    observation = environment.step(
        CaseAction(
            action_type="classify_case",
            classification="item_not_received",
            severity="medium",
        )
    )
    assert observation.current_classification == "item_not_received"

    observation = environment.step(
        CaseAction(
            action_type="resolve_case",
            resolution="refund_full",
            refund_amount=84.50,
            require_return=False,
            escalation_target="none",
            reason_code="carrier_no_delivery_scan",
            message_template="refund_approved",
        )
    )
    assert observation.done is True


def test_request_more_context_reveals_policy_text():
    environment = DisputeDeskEnvironment(default_task_id="late_delivery_refund")
    observation = environment.reset(task_id="late_delivery_refund")

    observation = environment.step(
        CaseAction(action_type="request_more_context", context_key="carrier_policy")
    )

    assert "carrier_policy" in observation.revealed_context
    assert "full refund" in observation.revealed_context["carrier_policy"].lower()


def test_wrong_item_replacement_flow_can_be_completed():
    environment = DisputeDeskEnvironment(default_task_id="wrong_item_premium_exchange")
    observation = environment.reset(task_id="wrong_item_premium_exchange")

    for artifact_id in [
        "order_summary",
        "warehouse_pick_ticket",
        "buyer_photo_report",
    ]:
        observation = environment.step(
            CaseAction(action_type="review_artifact", artifact_id=artifact_id)
        )

    for context_key in ["inventory_status", "return_policy"]:
        observation = environment.step(
            CaseAction(action_type="request_more_context", context_key=context_key)
        )

    observation = environment.step(
        CaseAction(
            action_type="classify_case",
            classification="wrong_item",
            severity="medium",
        )
    )
    observation = environment.step(
        CaseAction(
            action_type="resolve_case",
            resolution="replace_item",
            refund_amount=0.0,
            require_return=True,
            escalation_target="none",
            reason_code="seller_misfulfillment_wrong_item",
            message_template="replacement_offered",
        )
    )

    assert observation.done is True


def test_seeded_variant_changes_safety_return_flow():
    environment = DisputeDeskEnvironment(default_task_id="safety_risk_damage_replacement")
    observation = environment.reset(task_id="safety_risk_damage_replacement", seed=1)

    assert observation.case_id == "CASE-004B"

    for artifact_id in [
        "order_summary",
        "photo_report",
        "diagnostics_note",
        "seller_policy",
    ]:
        observation = environment.step(
            CaseAction(action_type="review_artifact", artifact_id=artifact_id)
        )

    for context_key in ["safety_policy", "replacement_eta", "return_safety_policy"]:
        observation = environment.step(
            CaseAction(action_type="request_more_context", context_key=context_key)
        )

    observation = environment.step(
        CaseAction(
            action_type="classify_case",
            classification="partial_damage",
            severity="high",
        )
    )
    observation = environment.step(
        CaseAction(
            action_type="resolve_case",
            resolution="replace_item",
            refund_amount=0.0,
            require_return=False,
            escalation_target="none",
            reason_code="unsafe_return_waived_replacement",
            message_template="replacement_offered",
        )
    )

    assert observation.done is True


def test_seeded_variant_changes_partial_refund_amount():
    environment = DisputeDeskEnvironment(default_task_id="partial_damage_partial_refund")
    observation = environment.reset(task_id="partial_damage_partial_refund", seed=1)

    assert observation.case_id == "CASE-002B"

    for artifact_id in [
        "order_summary",
        "photo_report",
        "seller_policy",
    ]:
        observation = environment.step(
            CaseAction(action_type="review_artifact", artifact_id=artifact_id)
        )

    observation = environment.step(
        CaseAction(action_type="request_more_context", context_key="replacement_eta")
    )

    observation = environment.step(
        CaseAction(
            action_type="classify_case",
            classification="partial_damage",
            severity="medium",
        )
    )
    observation = environment.step(
        CaseAction(
            action_type="resolve_case",
            resolution="refund_partial",
            refund_amount=32.0,
            require_return=False,
            escalation_target="none",
            reason_code="multiple_items_damaged",
            message_template="partial_refund_approved",
        )
    )

    assert observation.done is True


def test_seeded_variant_can_require_refund_when_replacement_is_too_slow():
    environment = DisputeDeskEnvironment(default_task_id="safety_risk_damage_replacement")
    observation = environment.reset(task_id="safety_risk_damage_replacement", seed=2)

    assert observation.case_id == "CASE-004C"

    for artifact_id in [
        "order_summary",
        "photo_report",
        "diagnostics_note",
        "seller_policy",
    ]:
        observation = environment.step(
            CaseAction(action_type="review_artifact", artifact_id=artifact_id)
        )

    for context_key in ["safety_policy", "replacement_eta", "return_safety_policy"]:
        observation = environment.step(
            CaseAction(action_type="request_more_context", context_key=context_key)
        )

    observation = environment.step(
        CaseAction(
            action_type="classify_case",
            classification="partial_damage",
            severity="high",
        )
    )
    observation = environment.step(
        CaseAction(
            action_type="resolve_case",
            resolution="refund_full",
            refund_amount=182.0,
            require_return=False,
            escalation_target="none",
            reason_code="safety_risk_no_timely_replacement",
            message_template="refund_approved",
        )
    )

    assert observation.done is True


def test_seeded_variant_can_escalate_linked_account_abuse():
    environment = DisputeDeskEnvironment(default_task_id="suspicious_refund_abuse")
    observation = environment.reset(task_id="suspicious_refund_abuse", seed=3)

    assert observation.case_id == "CASE-005D"

    for artifact_id in [
        "order_summary",
        "shipment_timeline",
        "concession_history",
        "authentication_note",
    ]:
        observation = environment.step(
            CaseAction(action_type="review_artifact", artifact_id=artifact_id)
        )

    for context_key in ["trust_policy", "counterfeit_review_policy", "abuse_pattern_policy"]:
        observation = environment.step(
            CaseAction(action_type="request_more_context", context_key=context_key)
        )

    observation = environment.step(
        CaseAction(
            action_type="classify_case",
            classification="suspected_abuse",
            severity="high",
        )
    )
    observation = environment.step(
        CaseAction(
            action_type="resolve_case",
            resolution="escalate",
            refund_amount=0.0,
            require_return=False,
            escalation_target="trust_safety",
            reason_code="linked_account_abuse_pattern",
            message_template="escalated_for_review",
        )
    )

    assert observation.done is True

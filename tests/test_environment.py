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

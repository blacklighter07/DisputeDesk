from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from dispute_desk.models import (
    ArtifactRelevance,
    Classification,
    Difficulty,
    EscalationTarget,
    MessageTemplate,
    ResolutionType,
    Severity,
    TaskSummary,
)


class ArtifactSpec(BaseModel):
    artifact_id: str
    title: str
    summary: str
    content: str
    relevance: ArtifactRelevance


class ScenarioExpectation(BaseModel):
    required_artifact_ids: list[str]
    required_context_keys: list[str] = Field(default_factory=list)
    classification: Classification
    severity: Severity
    resolution: ResolutionType
    refund_amount: float
    require_return: bool
    escalation_target: EscalationTarget
    reason_code: str
    message_template: MessageTemplate


class DisputeScenario(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    case_id: str
    objective: str
    case_summary: str
    max_steps: int = Field(default=8, ge=4)
    order_total: float
    currency: str = "USD"
    artifacts: list[ArtifactSpec]
    extra_context: Dict[str, str]
    expected: ScenarioExpectation


SCENARIOS: list[DisputeScenario] = [
    DisputeScenario(
        task_id="late_delivery_refund",
        title="Late Delivery Refund",
        difficulty="easy",
        case_id="CASE-001",
        objective="Review the shipment evidence and resolve the buyer's non-delivery claim.",
        case_summary=(
            "Buyer says a handmade lamp never arrived. The seller claims the package is "
            "still in transit, but the buyer says the promised delivery window has passed."
        ),
        max_steps=8,
        order_total=84.50,
        artifacts=[
            ArtifactSpec(
                artifact_id="order_summary",
                title="Order Summary",
                summary="Single-item order for one handmade lamp worth $84.50.",
                content=(
                    "Order total: $84.50. Item: handmade lamp. Order date: 2026-03-08. "
                    "Promised delivery date: 2026-03-15."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="shipment_timeline",
                title="Shipment Timeline",
                summary="Carrier scan history with no delivered scan.",
                content=(
                    "Carrier scans: label created 2026-03-09, accepted 2026-03-10, "
                    "distribution center delay 2026-03-14, no final delivery scan as of 2026-03-20."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="seller_response",
                title="Seller Response",
                summary="Seller asks for patience but provides no proof of delivery.",
                content=(
                    "Seller note: We believe the package is delayed and should arrive soon. "
                    "We do not have a delivered scan to share."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="buyer_message",
                title="Buyer Message",
                summary="Buyer requests a refund due to missed delivery promise.",
                content=(
                    "Buyer note: The package never arrived and the gift date already passed. "
                    "Please issue a refund."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="loyalty_status",
                title="Buyer Loyalty Status",
                summary="Gold loyalty tier with no special shipping guarantees.",
                content=(
                    "Account note: buyer is Gold tier. Loyalty status affects coupon eligibility "
                    "but does not override shipment-claim policy."
                ),
                relevance="distractor",
            ),
        ],
        extra_context={
            "carrier_policy": (
                "Marketplace policy: if no delivered scan exists and the promised delivery "
                "window is exceeded by 3 or more days, the buyer is eligible for a full refund."
            ),
            "gift_service_policy": (
                "Gift-service credits apply only to missed premium gift-wrapping or note errors, "
                "not to shipping disputes."
            ),
        },
        expected=ScenarioExpectation(
            required_artifact_ids=["order_summary", "shipment_timeline"],
            required_context_keys=["carrier_policy"],
            classification="item_not_received",
            severity="medium",
            resolution="refund_full",
            refund_amount=84.50,
            require_return=False,
            escalation_target="none",
            reason_code="carrier_no_delivery_scan",
            message_template="refund_approved",
        ),
    ),
    DisputeScenario(
        task_id="partial_damage_partial_refund",
        title="Partial Damage Partial Refund",
        difficulty="medium",
        case_id="CASE-002",
        objective="Review evidence and refund only the damaged portion of the order.",
        case_summary=(
            "Buyer ordered two ceramic mugs and one arrived shattered. The buyer wants the "
            "entire order refunded, but the undamaged mug is still usable."
        ),
        max_steps=8,
        order_total=46.00,
        artifacts=[
            ArtifactSpec(
                artifact_id="order_summary",
                title="Order Summary",
                summary="Two mugs at $18 each plus $10 shipping.",
                content=(
                    "Order total: $46.00. Items: 2 ceramic mugs at $18.00 each. "
                    "Shipping: $10.00 flat."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="photo_report",
                title="Photo Assessment",
                summary="Photo review confirms exactly one broken mug.",
                content=(
                    "Photo review: one mug is visibly shattered on unpacking. "
                    "Second mug appears intact and consistent with listing."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="seller_policy",
                title="Seller Policy",
                summary="Damage policy allows replacement or refund for affected items only.",
                content=(
                    "Policy: when only part of a shipment is damaged, refund or replacement "
                    "should be limited to the affected item unless the remaining items are unusable."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="buyer_message",
                title="Buyer Message",
                summary="Buyer requests a full order refund.",
                content=(
                    "Buyer note: One mug arrived broken. I want a full refund for the whole order."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="coupon_history",
                title="Coupon History",
                summary="Promo discount used at checkout.",
                content=(
                    "Checkout applied SPRING10 promo code. Coupon usage does not change the "
                    "damage-resolution policy."
                ),
                relevance="distractor",
            ),
        ],
        extra_context={
            "replacement_eta": (
                "Replacement stock is backordered for 10 days. Partial refund is the fastest policy-compliant option."
            ),
            "shipping_refund_policy": (
                "Shipping is only refunded when the full order is unusable or returned in full."
            ),
        },
        expected=ScenarioExpectation(
            required_artifact_ids=["order_summary", "photo_report", "seller_policy"],
            required_context_keys=["replacement_eta"],
            classification="partial_damage",
            severity="medium",
            resolution="refund_partial",
            refund_amount=18.00,
            require_return=False,
            escalation_target="none",
            reason_code="single_item_damaged",
            message_template="partial_refund_approved",
        ),
    ),
    DisputeScenario(
        task_id="wrong_item_premium_exchange",
        title="Wrong Item Premium Exchange",
        difficulty="medium",
        case_id="CASE-003",
        objective="Confirm the wrong-item fulfillment error and offer a replacement with the required return flow.",
        case_summary=(
            "Buyer ordered a navy wool coat but received an olive bomber jacket instead. "
            "The buyer wants a refund and says they do not want to send the wrong item back."
        ),
        max_steps=10,
        order_total=148.00,
        artifacts=[
            ArtifactSpec(
                artifact_id="order_summary",
                title="Order Summary",
                summary="Single premium coat order for $148.00.",
                content=(
                    "Order total: $148.00. Item ordered: navy wool coat, size M. "
                    "Delivery completed on 2026-03-18."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="warehouse_pick_ticket",
                title="Warehouse Pick Ticket",
                summary="Fulfillment scan shows the wrong SKU was packed.",
                content=(
                    "Picker scan: BOMBER-OLV-M scanned at pack station instead of COAT-NVY-M. "
                    "Pack station exception was not corrected before carrier handoff."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="buyer_photo_report",
                title="Buyer Photo Report",
                summary="Buyer photo shows an olive bomber with mismatched SKU tag.",
                content=(
                    "Photo review: delivered item tag reads BOMBER-OLV-M. "
                    "Listing confirmation in app shows ordered SKU COAT-NVY-M."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="seller_response",
                title="Seller Response",
                summary="Seller apologizes and confirms the coat is still stocked.",
                content=(
                    "Seller note: We packed the wrong item by mistake. The ordered coat is in stock "
                    "and can be sent immediately."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="buyer_message",
                title="Buyer Message",
                summary="Buyer asks for a refund and says they do not want the hassle of a return.",
                content=(
                    "Buyer note: This is the wrong jacket. I want a refund, and I do not want to "
                    "ship anything back."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="gift_wrap_note",
                title="Gift Wrap Note",
                summary="Gift wrap preference from checkout.",
                content=(
                    "Checkout note: add gift wrapping and leave the package at the front desk if possible."
                ),
                relevance="distractor",
            ),
        ],
        extra_context={
            "inventory_status": (
                "The ordered navy wool coat is in stock and can ship within 24 hours."
            ),
            "return_policy": (
                "Wrong-item orders above $100 require a prepaid return of the incorrect item before "
                "the replacement is finalized."
            ),
        },
        expected=ScenarioExpectation(
            required_artifact_ids=[
                "order_summary",
                "warehouse_pick_ticket",
                "buyer_photo_report",
            ],
            required_context_keys=["inventory_status", "return_policy"],
            classification="wrong_item",
            severity="medium",
            resolution="replace_item",
            refund_amount=0.00,
            require_return=True,
            escalation_target="none",
            reason_code="seller_misfulfillment_wrong_item",
            message_template="replacement_offered",
        ),
    ),
    DisputeScenario(
        task_id="safety_risk_damage_replacement",
        title="Safety Risk Damage Replacement",
        difficulty="hard",
        case_id="CASE-004",
        objective="Detect the safety-critical damage pattern and replace the item instead of offering a keep-item credit.",
        case_summary=(
            "Buyer says a countertop blender arrived with a cracked pitcher and asks for a partial refund "
            "so they can keep using the base unit. The damage may create a product-safety issue."
        ),
        max_steps=10,
        order_total=182.00,
        artifacts=[
            ArtifactSpec(
                artifact_id="order_summary",
                title="Order Summary",
                summary="Blender order worth $182.00.",
                content=(
                    "Order total: $182.00. Item: high-speed countertop blender. "
                    "Delivered on 2026-03-17."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="photo_report",
                title="Damage Photo Review",
                summary="Photo shows a visible crack along the pitcher wall.",
                content=(
                    "Photo review: crack runs from the pitcher rim down 4 cm along the blending jar. "
                    "Unit powers on, but the cracked component is a food-contact part."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="diagnostics_note",
                title="Product Specialist Note",
                summary="Specialist flags a shatter and leak risk during operation.",
                content=(
                    "Diagnostics: cracked pitcher can leak or shatter at high RPM. "
                    "This is a safety risk and should not be resolved with a keep-item credit."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="seller_policy",
                title="Seller Policy",
                summary="Seller supports replacement for damaged appliances when replacement inventory exists.",
                content=(
                    "Policy: for damaged appliances, replacement is preferred when the main unit can be restored "
                    "safely with a replacement component or full replacement unit."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="buyer_message",
                title="Buyer Message",
                summary="Buyer requests a 40% refund and wants to keep the blender.",
                content=(
                    "Buyer note: The blender jar is cracked. Give me a 40% refund and I will keep the rest."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="warranty_banner",
                title="Warranty Banner",
                summary="Standard warranty upsell copy.",
                content=(
                    "Optional extended warranty available at checkout for accidental-damage coverage."
                ),
                relevance="distractor",
            ),
        ],
        extra_context={
            "safety_policy": (
                "Cracked food-contact components on powered appliances are safety-critical. "
                "Partial keep-item refunds are not allowed when continued use may be unsafe."
            ),
            "replacement_eta": (
                "A replacement blender can ship within 2 business days from the regional warehouse."
            ),
        },
        expected=ScenarioExpectation(
            required_artifact_ids=[
                "order_summary",
                "photo_report",
                "diagnostics_note",
                "seller_policy",
            ],
            required_context_keys=["safety_policy", "replacement_eta"],
            classification="partial_damage",
            severity="high",
            resolution="replace_item",
            refund_amount=0.00,
            require_return=True,
            escalation_target="none",
            reason_code="safety_risk_component_damage",
            message_template="replacement_offered",
        ),
    ),
    DisputeScenario(
        task_id="suspicious_refund_abuse",
        title="Suspicious Refund Abuse",
        difficulty="hard",
        case_id="CASE-005",
        objective="Detect abuse signals and escalate instead of issuing an immediate refund.",
        case_summary=(
            "Buyer claims the received sneakers are fake and requests an instant refund. "
            "There are multiple prior concessions on the same account."
        ),
        max_steps=9,
        order_total=129.00,
        artifacts=[
            ArtifactSpec(
                artifact_id="order_summary",
                title="Order Summary",
                summary="Premium sneaker order worth $129.00.",
                content=(
                    "Order total: $129.00. Item: limited-edition sneakers. "
                    "Delivered on 2026-03-16."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="shipment_timeline",
                title="Shipment Timeline",
                summary="Delivery completed with signed handoff and matching package weight.",
                content=(
                    "Carrier scans: delivered 2026-03-16. Signature captured. "
                    "Delivered package weight matches seller outbound weight within 0.02 lb."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="concession_history",
                title="Concession History",
                summary="Three high-value concessions in the last 45 days.",
                content=(
                    "Buyer account concessions: $88 missing-item refund, $64 damaged-item credit, "
                    "$112 counterfeit-claim refund in past 45 days."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="authentication_note",
                title="Authentication Note",
                summary="Seller provided proof of authenticity.",
                content=(
                    "Marketplace authentication passed before shipment. SKU, box tag, and serial "
                    "numbers match seller inventory records."
                ),
                relevance="required",
            ),
            ArtifactSpec(
                artifact_id="buyer_message",
                title="Buyer Message",
                summary="Buyer demands immediate refund without return.",
                content=(
                    "Buyer note: These shoes are fake. Refund me now. I will not return the item."
                ),
                relevance="helpful",
            ),
            ArtifactSpec(
                artifact_id="vip_status_note",
                title="VIP Service Note",
                summary="VIP label from a past promotion.",
                content=(
                    "Account was tagged VIP during a holiday campaign. VIP tags do not bypass trust-and-safety review."
                ),
                relevance="distractor",
            ),
        ],
        extra_context={
            "trust_policy": (
                "Policy: repeat high-value counterfeit claims with contradictory delivery evidence "
                "must be escalated to trust and safety before any refund is issued."
            ),
            "appeasement_policy": (
                "Small appeasement credits are not allowed on high-value counterfeit disputes while abuse review is pending."
            ),
        },
        expected=ScenarioExpectation(
            required_artifact_ids=[
                "order_summary",
                "shipment_timeline",
                "concession_history",
                "authentication_note",
            ],
            required_context_keys=["trust_policy"],
            classification="suspected_abuse",
            severity="high",
            resolution="escalate",
            refund_amount=0.00,
            require_return=False,
            escalation_target="trust_safety",
            reason_code="repeat_high_value_claim_pattern",
            message_template="escalated_for_review",
        ),
    ),
]

SCENARIO_MAP = {scenario.task_id: scenario for scenario in SCENARIOS}


def get_scenario(task_id: str) -> DisputeScenario:
    return SCENARIO_MAP[task_id]


def task_catalog() -> list[TaskSummary]:
    return [
        TaskSummary(
            task_id=scenario.task_id,
            title=scenario.title,
            difficulty=scenario.difficulty,
            objective=scenario.objective,
            max_steps=scenario.max_steps,
        )
        for scenario in SCENARIOS
    ]

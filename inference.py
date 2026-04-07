from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from dispute_desk.baseline import (
    DEFAULT_MODEL,
    BaselineDecision,
    _artifact_priority,
    _choose_decision,
    _context_priority,
    _write_baseline_output,
)
from dispute_desk.config import (
    get_api_base_url,
    get_api_key,
    get_model_name,
    load_environment,
)
from dispute_desk.models import BaselineResponse, BaselineTaskResult, CaseAction
from dispute_desk.scenarios import SCENARIOS
from dispute_desk.server.dispute_desk_environment import DisputeDeskEnvironment


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    api_base_url: str
    model_name: str
    has_api_key: bool


StructuredEmitter = Callable[[str], None]
DecisionResolver = Callable[[OpenAI, str, dict[str, Any]], BaselineDecision]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DisputeDesk inference baseline with the OpenAI client. "
            "Reads OPENAI_API_KEY and OPENAI_MODEL by default, with support for "
            "API_BASE_URL, HF_TOKEN, and MODEL_NAME as compatibility aliases."
        )
    )
    parser.add_argument("--model", default=None, help="Override the OpenAI model id.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the final baseline response as JSON after the structured stdout blocks.",
    )
    return parser


def resolve_runtime_config(model_override: str | None) -> InferenceRuntimeConfig:
    return InferenceRuntimeConfig(
        api_base_url=get_api_base_url(),
        model_name=model_override or get_model_name(DEFAULT_MODEL),
        has_api_key=bool(get_api_key()),
    )


def format_structured_line(kind: str, **fields: Any) -> str:
    tokens = [f"[{kind}]"]
    for key, value in fields.items():
        if value is None:
            continue
        tokens.append(f"{key}={_structured_value(value)}")
    return " ".join(tokens)


def run_inference(
    model: str | None = None,
    *,
    emit: StructuredEmitter | None = None,
    decision_resolver: DecisionResolver | None = None,
) -> BaselineResponse:
    load_environment()
    runtime = resolve_runtime_config(model)
    api_key = get_api_key()
    if not runtime.has_api_key or not api_key:
        raise RuntimeError(
            "Missing credentials. Set OPENAI_API_KEY, or provide HF_TOKEN as a compatibility alias."
        )

    structured_emit = emit or _stdout_emit
    resolve_decision = decision_resolver or _choose_decision
    client = OpenAI(
        base_url=runtime.api_base_url,
        api_key=api_key,
        timeout=30.0,
        max_retries=1,
    )

    results: list[BaselineTaskResult] = []
    for scenario in SCENARIOS:
        environment = DisputeDeskEnvironment(default_task_id=scenario.task_id)
        observation = environment.reset(task_id=scenario.task_id)
        _emit(
            structured_emit,
            "START",
            task=scenario.task_id,
            case=observation.case_id,
            difficulty=observation.difficulty,
            max_steps=scenario.max_steps,
        )
        observation = _collect_case_signal_with_trace(
            environment=environment,
            observation=observation,
            task_id=scenario.task_id,
            case_id=observation.case_id,
            emit=structured_emit,
        )
        decision = resolve_decision(
            client=client,
            model=runtime.model_name,
            observation_payload=observation.model_dump(mode="json"),
        )

        if not observation.done:
            observation = _step_with_trace(
                environment=environment,
                task_id=scenario.task_id,
                case_id=observation.case_id,
                action=CaseAction(
                    action_type="classify_case",
                    classification=decision.classification,
                    severity=decision.severity,
                ),
                emit=structured_emit,
            )

        if not observation.done:
            observation = _step_with_trace(
                environment=environment,
                task_id=scenario.task_id,
                case_id=observation.case_id,
                action=CaseAction(
                    action_type="resolve_case",
                    resolution=decision.resolution,
                    refund_amount=decision.refund_amount,
                    require_return=decision.require_return,
                    escalation_target=decision.escalation_target,
                    reason_code=decision.reason_code,
                    message_template=decision.message_template,
                ),
                emit=structured_emit,
            )

        report = environment.grader_report()
        result = BaselineTaskResult(
            task_id=scenario.task_id,
            score=report.score,
            steps=environment.state.step_count,
            passed=report.passed,
        )
        results.append(result)
        _emit(
            structured_emit,
            "END",
            task=scenario.task_id,
            case=observation.case_id,
            score=report.score,
            steps=environment.state.step_count,
            passed=report.passed,
        )

    average_score = round(sum(item.score for item in results) / len(results), 4)
    response = BaselineResponse(model=runtime.model_name, average_score=average_score, tasks=results)
    _write_baseline_output(response)
    return response


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference(model=args.model)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2), flush=True)


def _collect_case_signal_with_trace(
    environment: DisputeDeskEnvironment,
    observation: Any,
    *,
    task_id: str,
    case_id: str,
    emit: StructuredEmitter,
):
    candidate_artifacts = sorted(
        observation.available_artifacts,
        key=_artifact_priority,
        reverse=True,
    )
    for artifact in candidate_artifacts:
        if artifact.reviewed:
            continue
        remaining_positive_contexts = sum(
            1
            for context_key in observation.metadata.get("available_context_keys", [])
            if context_key not in observation.revealed_context and _context_priority(context_key) > 0
        )
        if observation.steps_remaining <= remaining_positive_contexts + 2:
            break
        if _artifact_priority(artifact) <= 0:
            continue
        observation = _step_with_trace(
            environment=environment,
            task_id=task_id,
            case_id=case_id,
            action=CaseAction(action_type="review_artifact", artifact_id=artifact.artifact_id),
            emit=emit,
        )

    candidate_context_keys = sorted(
        observation.metadata.get("available_context_keys", []),
        key=_context_priority,
        reverse=True,
    )
    for context_key in candidate_context_keys:
        if context_key in observation.revealed_context:
            continue
        if observation.steps_remaining <= 2:
            break
        if _context_priority(context_key) <= 0:
            continue
        observation = _step_with_trace(
            environment=environment,
            task_id=task_id,
            case_id=case_id,
            action=CaseAction(action_type="request_more_context", context_key=context_key),
            emit=emit,
        )
    return observation


def _step_with_trace(
    environment: DisputeDeskEnvironment,
    *,
    task_id: str,
    case_id: str,
    action: CaseAction,
    emit: StructuredEmitter,
):
    observation = environment.step(action)
    _emit(
        emit,
        "STEP",
        task=task_id,
        case=case_id,
        step=environment.state.step_count,
        action=action.action_type,
        reward=getattr(observation, "reward", None),
        done=observation.done,
        artifact=action.artifact_id,
        context=action.context_key,
        classification=action.classification,
        severity=action.severity,
        resolution=action.resolution,
        refund_amount=action.refund_amount,
        require_return=action.require_return,
        escalation_target=action.escalation_target,
        reason_code=action.reason_code,
        message_template=action.message_template,
    )
    return observation


def _emit(emit: StructuredEmitter, kind: str, **fields: Any) -> None:
    emit(format_structured_line(kind, **fields))


def _stdout_emit(line: str) -> None:
    print(line, flush=True)


def _structured_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        formatted = f"{value:.4f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return str(value).replace(" ", "_")


if __name__ == "__main__":
    main()

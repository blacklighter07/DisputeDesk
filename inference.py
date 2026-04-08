from __future__ import annotations

import argparse
import contextlib
import io
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


BENCHMARK_NAME = "dispute_desk"
NULL_ERROR = "null"


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    api_base_url: str
    model_name: str
    api_key: str


StructuredEmitter = Callable[[str], None]
DecisionResolver = Callable[[OpenAI, str, dict[str, Any]], BaselineDecision]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DisputeDesk inference baseline with the OpenAI client. "
            "Uses API_BASE_URL, MODEL_NAME, and HF_TOKEN by default, while still "
            "accepting OPENAI_API_KEY and OPENAI_MODEL compatibility aliases."
        )
    )
    parser.add_argument("--model", default=None, help="Override the model id for inference.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the final baseline response JSON after the structured stdout logs.",
    )
    return parser


def resolve_runtime_config(model_override: str | None) -> InferenceRuntimeConfig:
    load_environment()
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing credentials. Set HF_TOKEN, or provide OPENAI_API_KEY as a compatibility alias."
        )
    return InferenceRuntimeConfig(
        api_base_url=get_api_base_url(),
        model_name=model_override or get_model_name(DEFAULT_MODEL),
        api_key=api_key,
    )


def log_start(emit: StructuredEmitter, task: str, env: str, model: str) -> None:
    emit(f"[START] task={task} env={env} model={model}")


def log_step(
    emit: StructuredEmitter,
    *,
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None,
) -> None:
    emit(
        "[STEP] "
        f"step={step} "
        f"action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={_format_error(error)}"
    )


def log_end(
    emit: StructuredEmitter,
    *,
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    emit(
        "[END] "
        f"success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}"
    )


def run_inference(
    model: str | None = None,
    *,
    emit: StructuredEmitter | None = None,
    decision_resolver: DecisionResolver | None = None,
) -> BaselineResponse:
    runtime = resolve_runtime_config(model)
    structured_emit = emit or _stdout_emit
    resolve_decision = decision_resolver or _choose_decision
    client = OpenAI(
        base_url=runtime.api_base_url,
        api_key=runtime.api_key,
        timeout=30.0,
        max_retries=1,
    )

    results: list[BaselineTaskResult] = []
    for scenario in SCENARIOS:
        result = _run_task_episode(
            client=client,
            model_name=runtime.model_name,
            task_id=scenario.task_id,
            emit=structured_emit,
            decision_resolver=resolve_decision,
        )
        results.append(result)

    average_score = round(sum(item.score for item in results) / len(results), 4)
    response = BaselineResponse(model=runtime.model_name, average_score=average_score, tasks=results)
    _write_baseline_output(response)
    return response


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference(model=args.model)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2), flush=True)


def _run_task_episode(
    *,
    client: OpenAI,
    model_name: str,
    task_id: str,
    emit: StructuredEmitter,
    decision_resolver: DecisionResolver,
) -> BaselineTaskResult:
    environment = DisputeDeskEnvironment(default_task_id=task_id)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    observation = None

    log_start(emit=emit, task=task_id, env=BENCHMARK_NAME, model=model_name)
    try:
        observation = environment.reset(task_id=task_id)
        observation = _collect_case_signal_with_trace(
            environment=environment,
            observation=observation,
            rewards=rewards,
            emit=emit,
        )
        with contextlib.redirect_stderr(io.StringIO()):
            decision = decision_resolver(
                client=client,
                model=model_name,
                observation_payload=observation.model_dump(mode="json"),
            )

        if not observation.done:
            observation = _step_with_trace(
                environment=environment,
                action=CaseAction(
                    action_type="classify_case",
                    classification=decision.classification,
                    severity=decision.severity,
                ),
                rewards=rewards,
                emit=emit,
            )

        if not observation.done:
            observation = _step_with_trace(
                environment=environment,
                action=CaseAction(
                    action_type="resolve_case",
                    resolution=decision.resolution,
                    refund_amount=decision.refund_amount,
                    require_return=decision.require_return,
                    escalation_target=decision.escalation_target,
                    reason_code=decision.reason_code,
                    message_template=decision.message_template,
                ),
                rewards=rewards,
                emit=emit,
            )

        report = environment.grader_report()
        score = float(report.score)
        success = bool(report.passed)
        steps_taken = environment.state.step_count
    except Exception:
        steps_taken = environment.state.step_count
    finally:
        _close_environment(environment)
        log_end(
            emit=emit,
            success=success,
            steps=steps_taken,
            score=_clamp_score(score),
            rewards=rewards,
        )

    return BaselineTaskResult(
        task_id=task_id,
        score=_clamp_score(score),
        steps=steps_taken,
        passed=success,
    )


def _collect_case_signal_with_trace(
    environment: DisputeDeskEnvironment,
    observation: Any,
    *,
    rewards: list[float],
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
            action=CaseAction(action_type="review_artifact", artifact_id=artifact.artifact_id),
            rewards=rewards,
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
            action=CaseAction(action_type="request_more_context", context_key=context_key),
            rewards=rewards,
            emit=emit,
        )
    return observation


def _step_with_trace(
    environment: DisputeDeskEnvironment,
    *,
    action: CaseAction,
    rewards: list[float],
    emit: StructuredEmitter,
):
    observation = environment.step(action)
    reward = float(getattr(observation, "reward", 0.0) or 0.0)
    rewards.append(reward)
    log_step(
        emit=emit,
        step=environment.state.step_count,
        action=_action_to_string(action),
        reward=reward,
        done=bool(observation.done),
        error=NULL_ERROR,
    )
    return observation


def _action_to_string(action: CaseAction) -> str:
    if action.action_type == "review_artifact":
        return f"review_artifact({action.artifact_id})"
    if action.action_type == "request_more_context":
        return f"request_more_context({action.context_key})"
    if action.action_type == "classify_case":
        return f"classify_case({action.classification},{action.severity})"
    if action.action_type == "resolve_case":
        refund_amount = 0.0 if action.refund_amount is None else float(action.refund_amount)
        require_return = str(bool(action.require_return)).lower()
        escalation_target = action.escalation_target or "none"
        reason_code = action.reason_code or "null"
        message_template = action.message_template or "null"
        return (
            "resolve_case("
            f"{action.resolution},{refund_amount:.2f},{require_return},{escalation_target},"
            f"{reason_code},{message_template})"
        )
    return action.action_type


def _format_error(error: str | None) -> str:
    if not error:
        return NULL_ERROR
    return error.replace("\n", "\\n")


def _clamp_score(score: float) -> float:
    return min(max(float(score), 0.0), 1.0)


def _close_environment(environment: DisputeDeskEnvironment) -> None:
    close_method = getattr(environment, "close", None)
    if callable(close_method):
        close_method()


def _stdout_emit(line: str) -> None:
    print(line, flush=True)


if __name__ == "__main__":
    main()

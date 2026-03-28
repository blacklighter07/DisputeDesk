from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from dispute_desk.baseline import run_baseline
from dispute_desk.config import get_api_base_url, get_api_key, get_model_name


@dataclass(frozen=True)
class InferenceRuntimeConfig:
    api_base_url: str
    model_name: str
    has_api_key: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DisputeDesk inference baseline with the OpenAI client. "
            "Reads OPENAI_API_KEY and OPENAI_MODEL by default, with support for "
            "API_BASE_URL, HF_TOKEN, and MODEL_NAME as compatibility aliases."
        )
    )
    parser.add_argument("--model", default=None, help="Override the OpenAI model id.")
    return parser


def resolve_runtime_config(model_override: str | None) -> InferenceRuntimeConfig:
    return InferenceRuntimeConfig(
        api_base_url=get_api_base_url(),
        model_name=model_override or get_model_name("gpt-5-mini-2025-08-07"),
        has_api_key=bool(get_api_key()),
    )


def main() -> None:
    args = build_parser().parse_args()
    runtime = resolve_runtime_config(args.model)
    if not runtime.has_api_key:
        raise RuntimeError(
            "Missing credentials. Set OPENAI_API_KEY, or provide HF_TOKEN as a compatibility alias."
        )
    print(
        (
            "[inference] starting baseline with "
            f"API_BASE_URL={runtime.api_base_url} MODEL_NAME={runtime.model_name}"
        ),
        file=sys.stderr,
    )
    result = run_baseline(model=args.model)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()

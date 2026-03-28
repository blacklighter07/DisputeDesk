from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_DOTENV_LOADED = False


def load_environment() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    _DOTENV_LOADED = True


def get_api_base_url(default: str = "https://router.huggingface.co/v1") -> str:
    load_environment()
    return os.getenv("API_BASE_URL") or default


def get_api_key() -> str | None:
    load_environment()
    return os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")


def get_model_name(default_model: str) -> str:
    load_environment()
    return os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", default_model)


def get_openai_model(default_model: str) -> str:
    return get_model_name(default_model)

from dispute_desk.config import get_api_base_url, get_api_key, get_model_name


def test_competition_environment_variables_take_priority(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("MODEL_NAME", "hf/model")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_test_token")
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")

    assert get_api_base_url() == "https://router.huggingface.co/v1"
    assert get_api_key() == "hf_test_token"
    assert get_model_name("default-model") == "hf/model"


def test_legacy_environment_variables_remain_supported(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai_test_token")
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")

    assert get_api_base_url() == "https://router.huggingface.co/v1"
    assert get_api_key() == "openai_test_token"
    assert get_model_name("default-model") == "legacy-model"


def test_blank_api_base_url_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "")

    assert get_api_base_url() == "https://router.huggingface.co/v1"

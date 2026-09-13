"""Тесты валидации/нормализации параметров генерации."""

from server.services.generation import GenerationSettings, sanitize_settings


def test_top_p_zero_normalized():
    s = sanitize_settings({"top_p": 0})
    assert s.top_p == 0.01


def test_temperature_range_enforced():
    assert sanitize_settings({"temperature": 5}).temperature is None
    assert sanitize_settings({"temperature": -1}).temperature is None
    assert sanitize_settings({"temperature": 0}).temperature == 0
    assert sanitize_settings({"temperature": 2}).temperature == 2


def test_top_p_range():
    assert sanitize_settings({"top_p": 2}).top_p is None
    assert sanitize_settings({"top_p": 0.5}).top_p == 0.5


def test_top_k_kept_in_settings_but_not_upstream():
    s = sanitize_settings({"top_k": 5})
    assert s.top_k == 5
    assert "top_k" not in s.to_upstream()


def test_max_tokens():
    assert sanitize_settings({"max_tokens": 100}).max_tokens == 100
    assert sanitize_settings({"max_tokens": 0}).max_tokens is None
    assert sanitize_settings({"max_tokens": -5}).max_tokens is None


def test_stop_filtered_and_limited():
    words = [str(i) for i in range(30)]
    s = sanitize_settings({"stop": words + [""]})
    assert len(s.stop) == 16
    assert "" not in s.stop


def test_response_format():
    s = sanitize_settings({"response_format": {"type": "json_object"}})
    assert s.to_upstream()["response_format"] == {"type": "json_object"}
    assert sanitize_settings({"response_format": {"type": ""}}).response_format is None


def test_non_dict_returns_defaults():
    s = sanitize_settings(None)
    assert isinstance(s, GenerationSettings)
    assert s.to_upstream() == {}


def test_context_summary_valid():
    s = sanitize_settings({"context_summary": {"enabled": True, "requests_per_summary": 7}})
    assert s.context_summary is not None
    assert s.context_summary.enabled is True
    assert s.context_summary.requests_per_summary == 7
    assert s.to_dict()["context_summary"] == {"enabled": True, "requests_per_summary": 7}


def test_context_summary_clamped_and_defaulted():
    assert sanitize_settings({"context_summary": {"enabled": True, "requests_per_summary": 0}}).context_summary.requests_per_summary == 1
    assert sanitize_settings({"context_summary": {"enabled": True, "requests_per_summary": 99}}).context_summary.requests_per_summary == 20
    # невалидный requests_per_summary -> дефолт 5
    assert sanitize_settings({"context_summary": {"enabled": True, "requests_per_summary": "x"}}).context_summary.requests_per_summary == 5


def test_context_summary_enabled_must_be_bool():
    s = sanitize_settings({"context_summary": {"enabled": "yes", "requests_per_summary": 3}})
    assert s.context_summary.enabled is False
    assert sanitize_settings({"context_summary": None}).context_summary is None
    assert sanitize_settings({"context_summary": {}}).context_summary.enabled is False


def test_context_summary_not_upstream():
    s = sanitize_settings({"context_summary": {"enabled": True, "requests_per_summary": 3}})
    assert "context_summary" not in s.to_upstream()

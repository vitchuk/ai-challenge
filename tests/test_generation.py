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

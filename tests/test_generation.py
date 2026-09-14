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


def test_context_strategy_valid():
    s = sanitize_settings(
        {"context_strategy": {"strategy": "sliding", "n": 7, "k": 3}}
    )
    assert s.context_strategy is not None
    assert s.context_strategy.strategy == "sliding"
    assert s.context_strategy.n == 7
    assert s.context_strategy.k == 3
    assert s.to_dict()["context_strategy"] == {
        "strategy": "sliding", "n": 7, "k": 3,
    }


def test_context_strategy_invalid_falls_back_to_none():
    s = sanitize_settings({"context_strategy": {"strategy": "bogus", "n": 5}})
    assert s.context_strategy.strategy == "none"
    assert sanitize_settings({}).context_strategy is None


def test_context_strategy_params_clamped_and_defaulted():
    s = sanitize_settings({"context_strategy": {"strategy": "sliding", "n": 99, "k": 0}})
    assert s.context_strategy.n == 20
    assert s.context_strategy.k == 1
    # невалидные параметры -> дефолты (n=5 для summarize, n=10 для sliding, k=10)
    assert sanitize_settings({"context_strategy": {"strategy": "summarize", "n": "x"}}).context_strategy.n == 5
    assert sanitize_settings({"context_strategy": {"strategy": "sliding", "n": "x"}}).context_strategy.n == 10
    assert sanitize_settings({"context_strategy": {"strategy": "facts", "k": "x"}}).context_strategy.k == 10


def test_legacy_context_summary_migrates():
    # включённая старая саммаризация -> strategy=summarize с n
    s = sanitize_settings(
        {"context_summary": {"enabled": True, "requests_per_summary": 3}}
    )
    assert s.context_strategy.strategy == "summarize"
    assert s.context_strategy.n == 3
    # выключенная -> none
    s = sanitize_settings(
        {"context_summary": {"enabled": False, "requests_per_summary": 3}}
    )
    assert s.context_strategy.strategy == "none"
    # самый старый ключ keep_recent тоже поддерживается при миграции
    s = sanitize_settings({"context_summary": {"enabled": True, "keep_recent": 4}})
    assert s.context_strategy.n == 4


def test_context_strategy_wins_over_legacy():
    s = sanitize_settings({
        "context_summary": {"enabled": True, "requests_per_summary": 3},
        "context_strategy": {"strategy": "facts", "n": 5, "k": 2},
    })
    assert s.context_strategy.strategy == "facts"
    assert s.context_strategy.k == 2


def test_context_strategy_not_upstream():
    s = sanitize_settings({"context_strategy": {"strategy": "sliding", "n": 3}})
    assert "context_strategy" not in s.to_upstream()

"""Тесты расчёта стоимости ответов."""

from server.pricing import MODEL_CONTEXT, MODEL_PRICES, message_cost, model_context, model_price


def test_message_cost_known_model():
    cost = message_cost("deepseek-chat", 1000, 2000)
    assert cost == (1000 * 0.22 + 2000 * 0.66) / 1e6


def test_message_cost_free_model():
    assert message_cost("opencode/big-pickle", 100, 100) == 0


def test_message_cost_unknown_model():
    assert message_cost("unknown-model", 100, 100) is None


def test_model_price():
    assert model_price("opencode/glm-5.3") == 4.4
    assert model_price("opencode/big-pickle") == 0
    assert model_price("nope") is None


def test_model_context():
    assert model_context("deepseek-v4-flash") == 1048576
    assert model_context("opencode/kimi-k2.6") == 262144
    assert model_context("opencode/nemotron-3-ultra-free") == 1000000
    assert model_context("unknown-model") is None
    # лимит GLM апстримом не раскрывается — значения нет (в UI «—»)
    assert model_context("opencode/glm-5.3") is None


def test_context_values_are_positive():
    assert all(isinstance(v, int) and v > 0 for v in MODEL_CONTEXT.values())

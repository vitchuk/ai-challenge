"""Расчёт стоимости ответов LLM.

Карта цен на 1M токенов (вход/выход) используется и для метаданных
ответов (``MessageMeta.cost_usd``), и для маркировки моделей в
``GET /api/models``.
"""

from __future__ import annotations

from typing import Optional

# Цены в долларах за 1M токенов: {"in": ..., "out": ...}
MODEL_PRICES: dict[str, dict[str, float]] = {
    "deepseek-flash": {"in": 0.22, "out": 0.66},
    "deepseek-v4-flash": {"in": 0.22, "out": 0.66},
    "deepseek-v4-flash-vision-exp": {"in": 0.22, "out": 0.66},
    "deepseek-v4-pro": {"in": 0.66, "out": 1.98},
    "deepseek-chat": {"in": 0.22, "out": 0.66},
    "deepseek-reasoner": {"in": 0.22, "out": 0.66},
    "opencode/glm-5.3-flash": {"in": 0.15, "out": 0.5},
    "opencode/glm-5.3": {"in": 1.4, "out": 4.4},
    "opencode/glm-5.2": {"in": 1.4, "out": 4.4},
    "opencode/glm-5.1": {"in": 1.4, "out": 4.4},
    "opencode/kimi-k3": {"in": 3, "out": 15},
    "opencode/kimi-k2.7-code": {"in": 0.95, "out": 4},
    "opencode/kimi-k2.6": {"in": 0.95, "out": 4},
    "opencode/longcat-2.0": {"in": 0.3, "out": 1.2},
    "opencode/deepseek-v4-pro": {"in": 0.66, "out": 1.98},
    "opencode/deepseek-v4-flash": {"in": 0.22, "out": 0.66},
    "opencode/deepseek-v4-flash-vision-exp": {"in": 0.22, "out": 0.66},
    "opencode/mimo-v2.5": {"in": 0.14, "out": 0.28},
    "opencode/mimo-v2.5-pro": {"in": 0.435, "out": 0.87},
    "opencode/hy4-preview": {"in": 0.834, "out": 2.501},
    "opencode/hy3": {"in": 0.14, "out": 0.58},
    "opencode/omen-alpha": {"in": 0.2, "out": 0.66},
    "opencode/big-pickle": {"in": 0, "out": 0},
    "opencode/deepseek-v4-flash-free": {"in": 0, "out": 0},
    "opencode/mimo-v2.5-free": {"in": 0, "out": 0},
    "opencode/ling-3.0-flash-fin-free": {"in": 0, "out": 0},
    "opencode/nemotron-3-ultra-free": {"in": 0, "out": 0},
    "opencode/nemotron-3.5-lightning-free": {"in": 0, "out": 0},
}


def model_price(model_id: str) -> Optional[float]:
    """Цена за 1M *выходных* токенов модели (для маркировки в UI).

    Returns:
        Цена за 1M выходных токенов в USD или ``None``, если модель
        отсутствует в карте.
    """
    p = MODEL_PRICES.get(model_id)
    if p is None:
        return None
    return p["out"]


def message_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Стоимость одного ответа (в USD).

    Args:
        model_id: идентификатор модели (например ``opencode/glm-5.3``).
        prompt_tokens: число входных токенов.
        completion_tokens: число выходных токенов.

    Returns:
        Стоимость в USD или ``None``, если для модели нет карты цен.
    """
    p = MODEL_PRICES.get(model_id)
    if p is None:
        return None
    return (prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1e6


# Максимальный размер контекста моделей (в токенах). Значения получены
# probe-запросами к апстримам (валидация prompt+max_tokens до генерации).
# Для моделей, чей лимит апстрим не раскрывает, значение отсутствует —
# в UI показывается «—».
MODEL_CONTEXT: dict[str, int] = {
    "deepseek-flash": 1048576,
    "deepseek-v4-flash": 1048576,
    "deepseek-v4-flash-vision-exp": 1048576,
    "deepseek-v4-pro": 1048576,
    "deepseek-chat": 1048576,
    "deepseek-reasoner": 1048576,
    "opencode/deepseek-v4-flash": 1048576,
    "opencode/deepseek-v4-flash-vision-exp": 1048576,
    "opencode/deepseek-v4-pro": 1048576,
    "opencode/kimi-k2.6": 262144,
    "opencode/longcat-2.0": 1048580,
    "opencode/hy4-preview": 1048576,
    "opencode/hy3": 262144,
    "opencode/ling-3.0-flash-fin-free": 262144,
    "opencode/nemotron-3-ultra-free": 1000000,
    "opencode/nemotron-3.5-lightning-free": 1000000,
}


def model_context(model_id: str) -> Optional[int]:
    """Максимальный размер контекста модели в токенах.

    Args:
        model_id: идентификатор модели.

    Returns:
        Размер контекста в токенах или ``None``, если модель неизвестна.
    """
    return MODEL_CONTEXT.get(model_id)

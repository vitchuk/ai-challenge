"""Маршрутизация моделей по провайдерам.

Перенесено из прежней Node-версии: префикс ``opencode/`` маршрутизирует
в OpenCode (Go-подписка или бесплатные Zen-модели), иначе — в DeepSeek.
"""

from __future__ import annotations

from .base import ProviderSpec

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODELS_URL = "https://api.deepseek.com/models"
OPENCODE_CHAT_URL = "https://opencode.ai/zen/go/v1/chat/completions"
OPENCODE_MODELS_URL = "https://opencode.ai/zen/go/v1/models"
OPENCODE_ZEN_CHAT_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENCODE_ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"

OPENCODE_PREFIX = "opencode/"
DEFAULT_MODEL = "deepseek-chat"
MODEL_NAME_RE_PATTERN = r"[a-z0-9][a-z0-9._-]*"

GO_CHAT_MODELS = {
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "glm-5.1",
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "longcat-2.0",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "hy4-preview",
    "hy3",
    "omen-alpha",
}

ZEN_FREE_MODELS = {
    "big-pickle",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
}

USER_AGENT = "pomogator2k/2.0 (https://github.com/vitchuk/ai-challenge)"


class UnsupportedModelError(Exception):
    """Модель с префиксом ``opencode/`` не входит в белый список."""


def resolve_provider(
    raw_model: str | None,
    deepseek_api_key: str | None,
    opencode_api_key: str | None,
    session_id: str,
) -> ProviderSpec:
    """Определяет апстрим для запрошенной модели.

    Args:
        raw_model: строка модели из запроса (может быть ``None``).
        deepseek_api_key: ключ DeepSeek (может быть ``None``).
        opencode_api_key: ключ OpenCode (может быть ``None``).
        session_id: идентификатор сессии сервера (для заголовка OpenCode).

    Returns:
        Описание вызова апстрима (:class:`ProviderSpec`).

    Raises:
        ValueError: если нужный ключ не задан.
        UnsupportedModelError: если ``opencode/…``-модель не поддержана.
    """
    if not isinstance(raw_model, str) or not raw_model.strip():
        model = DEFAULT_MODEL
    else:
        model = raw_model.strip()

    if model.startswith(OPENCODE_PREFIX):
        inner = model[len(OPENCODE_PREFIX) :]
        if inner in GO_CHAT_MODELS:
            endpoint = OPENCODE_CHAT_URL
        elif inner in ZEN_FREE_MODELS:
            endpoint = OPENCODE_ZEN_CHAT_URL
        else:
            raise UnsupportedModelError(inner)
        if not opencode_api_key:
            raise ValueError("OPENCODE_API_KEY is not set on the server")
        return ProviderSpec(
            endpoint=endpoint,
            api_key=opencode_api_key,
            model=inner,
            provider_name="OpenCode",
            model_label=model,
            headers={
                "x-opencode-session": session_id,
                "User-Agent": USER_AGENT,
            },
        )

    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set on the server")
    return ProviderSpec(
        endpoint=DEEPSEEK_URL,
        api_key=deepseek_api_key,
        model=model,
        provider_name="DeepSeek",
        model_label=model,
        headers={"User-Agent": USER_AGENT},
    )


def model_name(raw_model: str | None) -> str:
    """Возвращает чистый (без префикса) идентификатор модели для отображения."""
    if not isinstance(raw_model, str) or not raw_model.strip():
        return DEFAULT_MODEL
    m = raw_model.strip()
    if m.startswith(OPENCODE_PREFIX):
        return m[len(OPENCODE_PREFIX) :]
    return m

"""Маршрут ``GET /api/models`` — агрегированный список моделей с ценами."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request

from ..pricing import model_context, model_price
from ..providers.routing import (
    DEEPSEEK_MODELS_URL,
    GO_CHAT_MODELS,
    OPENCODE_MODELS_URL,
    OPENCODE_PREFIX,
    OPENCODE_ZEN_MODELS_URL,
    USER_AGENT,
    ZEN_FREE_MODELS,
)

router = APIRouter(tags=["models"])


async def _fetch_ids(client: httpx.AsyncClient, url: str, headers: dict) -> list[str]:
    """Загружает список идентификаторов моделей из каталога апстрима."""
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [m["id"] for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)]


@router.get("/api/models")
async def list_models(request: Request) -> dict:
    """Агрегирует модели всех настроенных провайдеров.

    Отказ одного апстрима не роняет весь список — возвращаются модели
    оставшихся (с предупреждением в лог).

    Args:
        request: HTTP-запрос.

    Returns:
        ``{"object": "list", "data": [{"id", "owned_by", "price", "context"}]}``
        (``context`` — максимальный размер контекста в токенах или ``None``).

    Raises:
        HTTPException: если не задан ни один ключ (500) или список пуст (502).
    """
    from ..config import get_settings

    settings = get_settings()
    if not settings.deepseek_api_key and not settings.opencode_api_key:
        raise HTTPException(500, "No API keys configured (DEEPSEEK_API_KEY / OPENCODE_API_KEY)")

    client: httpx.AsyncClient = request.app.state.http_client
    models: list[dict] = []
    errors: list[str] = []

    def item(model_id: str, owned_by: str) -> dict:
        return {
            "id": model_id,
            "owned_by": owned_by,
            "price": model_price(model_id),
            "context": model_context(model_id),
        }

    if settings.deepseek_api_key:
        try:
            ids = await _fetch_ids(
                client,
                DEEPSEEK_MODELS_URL,
                {"Authorization": f"Bearer {settings.deepseek_api_key}"},
            )
            for mid in ids:
                models.append(item(mid, "deepseek"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"DeepSeek: {exc}")

    if settings.opencode_api_key:
        base_headers = {
            "Authorization": f"Bearer {settings.opencode_api_key}",
            "x-opencode-session": request.app.state.opencode_session_id,
            "User-Agent": USER_AGENT,
        }
        try:
            ids = await _fetch_ids(client, OPENCODE_MODELS_URL, base_headers)
            for mid in ids:
                if mid in GO_CHAT_MODELS:
                    models.append(item(f"{OPENCODE_PREFIX}{mid}", "opencode"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"OpenCode Go: {exc}")
        try:
            ids = await _fetch_ids(client, OPENCODE_ZEN_MODELS_URL, base_headers)
            for mid in ids:
                if mid in ZEN_FREE_MODELS:
                    models.append(item(f"{OPENCODE_PREFIX}{mid}", "opencode"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"OpenCode Zen (free): {exc}")

    if not models:
        raise HTTPException(502, f"Failed to load models: {'; '.join(errors)}")

    return {"object": "list", "data": models}

"""Маршруты журнала промптов (вкладка «Логи»).

Журнал хранится в памяти реестра (не персистится): итоговые сообщения,
отправленные в LLM, и ответы модели по всем запросам — основным и скрытым
(саммаризация, извлечение фактов).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("")
async def list_logs(request: Request) -> dict:
    """Возвращает журнал промптов (сначала новые).

    Args:
        request: HTTP-запрос (реестр из состояния приложения).

    Returns:
        ``{"logs": [{time, source, title, kind, messages, response, model,
        prompt_tokens, completion_tokens, reasoning_tokens}]}``.
    """
    return {"logs": request.app.state.registry.list_prompt_logs()}


@router.delete("", status_code=204)
async def clear_logs(request: Request) -> Response:
    """Очищает журнал промптов.

    Args:
        request: HTTP-запрос.

    Returns:
        Пустой ответ ``204``.
    """
    request.app.state.registry.clear_prompt_logs()
    return Response(status_code=204)

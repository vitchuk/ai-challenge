"""Маршруты управления сессиями и стриминга ответов."""

from __future__ import annotations

import json
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..providers import UnsupportedModelError, resolve_provider
from ..providers.client import StreamedCompletion
from ..schemas import MessageCreateRequest, SessionCreateRequest, SessionCreateResponse
from ..services import ChatService, SessionKind, sanitize_settings
from ..services.registry import SessionRegistry

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _sse(event: dict) -> str:
    """Форматирует событие как строку Server-Sent Events."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _resolve_spec(
    registry: SessionRegistry,
    session: ChatService,
    model_override: Optional[str],
    opencode_session_id: str,
):
    """Определяет апстрим для запроса с учётом переопределения модели.

    Raises:
        HTTPException: при ошибках маршрутизации.
    """
    from ..config import get_settings

    settings = get_settings()
    raw_model = model_override or session.model
    try:
        return resolve_provider(
            raw_model,
            settings.deepseek_api_key,
            settings.opencode_api_key,
            opencode_session_id,
        )
    except UnsupportedModelError as exc:
        raise HTTPException(400, f"Unsupported model: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("", status_code=201, response_model=SessionCreateResponse)
async def create_session(
    body: SessionCreateRequest,
    request: Request,
) -> SessionCreateResponse:
    """Создаёт новую сессию (чат/итоги/изолированную).

    Args:
        body: параметры сессии.
        request: HTTP-запрос (для реестра из состояния приложения).

    Returns:
        Идентификатор и модель созданной сессии.
    """
    registry: SessionRegistry = request.app.state.registry
    kind = body.kind if body.kind in {k.value for k in SessionKind} else SessionKind.CHAT.value
    gen_settings = sanitize_settings(body.settings)
    session = registry.create(
        kind=SessionKind(kind),
        model=body.model,
        settings=gen_settings,
        system_prompt=body.system_prompt,
    )
    return SessionCreateResponse(id=session.id, model=session.model)


@router.get("")
async def list_sessions(request: Request) -> dict:
    """Возвращает все сессии с историей (для восстановления клиента).

    Изолированные (ephemeral) сессии — временные и в список не включаются.

    Args:
        request: HTTP-запрос.

    Returns:
        ``{"data": [{id, kind, model, system_prompt, settings, last_active,
        history}], "active_id": "…"|null}`` — активная сессия (открытая вкладка).
    """
    registry: SessionRegistry = request.app.state.registry
    data = []
    for session in registry.list_sessions():
        if session.kind == SessionKind.EPHEMERAL:
            continue
        data.append(
            {
                "id": session.id,
                "kind": session.kind.value,
                "model": session.model,
                "system_prompt": session.system_prompt,
                "settings": session.settings.to_dict(),
                "last_active": session.last_active,
                "history": [m.to_dict() for m in session.history],
            }
        )
    return {"data": data, "active_id": registry.get_active()}


@router.post("/{session_id}/activate", status_code=204)
async def activate_session(session_id: str, request: Request) -> None:
    """Отмечает сессию как активную (открытую вкладку пользователя).

    Используется клиентом при переключении вкладок, чтобы восстановить
    именно открытый чат после перезапуска сервера/перезагрузки страницы.

    Args:
        session_id: идентификатор сессии.
        request: HTTP-запрос.
    """
    registry: SessionRegistry = request.app.state.registry
    if not registry.set_active(session_id):
        raise HTTPException(404, "Session not found")


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    """Возвращает полное состояние сессии (история с метаданными).

    Args:
        session_id: идентификатор сессии.
        request: HTTP-запрос.

    Returns:
        Состояние сессии (см. :class:`SessionMeta`).
    """
    registry: SessionRegistry = request.app.state.registry
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {
        "id": session.id,
        "kind": session.kind.value,
        "model": session.model,
        "system_prompt": session.system_prompt,
        "settings": session.settings.to_dict(),
        "last_active": session.last_active,
        "history": [m.to_dict() for m in session.history],
    }


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> None:
    """Удаляет сессию из памяти сервера.

    Args:
        session_id: идентификатор сессии.
        request: HTTP-запрос.
    """
    registry: SessionRegistry = request.app.state.registry
    if not registry.delete(session_id):
        raise HTTPException(404, "Session not found")


@router.post("/{session_id}/messages")
async def send_message(session_id: str, body: MessageCreateRequest, request: Request):
    """Отправляет сообщение пользователя и стримит ответ модели (SSE).

    Args:
        session_id: идентификатор сессии.
        body: содержимое сообщения и переопределения настроек/модели.
        request: HTTP-запрос.

    Returns:
        Поток Server-Sent Events с событиями ``session/reasoning_start/
        reasoning_end/done/error``.
    """
    registry: SessionRegistry = request.app.state.registry
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.busy:
        raise HTTPException(409, "Session is busy")

    opencode_session_id = request.app.state.opencode_session_id
    spec = _resolve_spec(registry, session, body.model, opencode_session_id)
    # Запоминаем модель, с которой было обращение к этому чату.
    if body.model:
        session.model = body.model
    # Настройки запроса инкапсулируются за чатом (если присланы).
    if body.settings is not None:
        session.settings = sanitize_settings(body.settings)
    gen_settings = session.settings

    # Первое сообщение обычного чата становится системным промптом.
    if (
        session.kind == SessionKind.CHAT
        and not session.history
        and session.system_prompt is None
    ):
        session.seed_system_message(body.content)
    else:
        session.add_user_message(body.content)

    # Для чата «Итоги» собираем актуальный TOON-контекст на каждый запрос.
    if session.kind == SessionKind.SUMMARY:
        session.system_prompt = registry.summary_system_prompt(session.id)

    runner = StreamedCompletion(client=request.app.state.http_client)

    async def event_stream():
        yield _sse({"type": "session", "id": session_id})
        try:
            async for event in session.stream_completion(
                runner, spec, gen_settings, extra_system=body.system_prompt
            ):
                yield _sse(event)
            # Успешное завершение: сохраняем пару user+assistant в БД.
            registry.remember_pair(session)
        except Exception as exc:  # noqa: BLE001 - отдаём ошибку клиенту как событие
            session.rollback_user_message()
            yield _sse({"type": "error", "error": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

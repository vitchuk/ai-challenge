"""Маршруты управления сессиями и стриминга ответов."""

from __future__ import annotations

import json
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..providers import UnsupportedModelError, resolve_provider
from ..providers.base import ProviderError
from ..providers.client import StreamedCompletion
from ..schemas import (
    BranchRequest,
    MemorySyncRequest,
    MessageCreateRequest,
    SessionCreateRequest,
    SessionCreateResponse,
)
from ..services import ChatService, MemoryStore, SessionKind, sanitize_settings
from ..services.registry import SessionRegistry

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

#: Лимиты памяти чата (невалидное молча отбрасывается).
MAX_MEMORY_STORES = 50
MAX_MEMORY_ITEMS = 500
MAX_MEMORY_NAME = 100
MAX_MEMORY_KEY = 200
MAX_MEMORY_VALUE = 2000


def _sanitize_memory(raw_stores) -> list:
    """Валидирует и нормализует список вкладок памяти чата.

    Args:
        raw_stores: «сырой» список вкладок из запроса.

    Returns:
        Список :class:`MemoryStore` (невалидные элементы отброшены).
    """
    if not isinstance(raw_stores, list):
        return []
    stores = []
    for raw in raw_stores[:MAX_MEMORY_STORES]:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        name = name.strip()[:MAX_MEMORY_NAME] if isinstance(name, str) else ""
        if not name:
            continue
        store_id = raw.get("id")
        store_id = store_id.strip()[:64] if isinstance(store_id, str) and store_id.strip() else None
        items = []
        raw_items = raw.get("items")
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            key, value = item[0], item[1]
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            key, value = key.strip(), value.strip()
            if key and value:
                items.append([key[:MAX_MEMORY_KEY], value[:MAX_MEMORY_VALUE]])
            if len(items) >= MAX_MEMORY_ITEMS:
                break
        stores.append(
            MemoryStore(
                id=store_id or f"mem-{len(stores) + 1}",
                name=name,
                persistent=bool(raw.get("persistent", False)),
                items=items,
            )
        )
    return stores


def _thousands(n: int) -> str:
    """Форматирует число с пробелами-разделителями тысяч."""
    return f"{n:,}".replace(",", " ")


def _provider_error_event(exc: ProviderError, session: ChatService) -> dict:
    """Формирует SSE-событие ошибки провайдера.

    Для ошибки лимита контекста возвращает локализованное сообщение и
    машинный ``code`` (клиент показывает спец-блок с действиями).
    """
    if exc.code == "context_length_exceeded":
        max_ctx = exc.details.get("max_context")
        requested = exc.details.get("requested")
        parts = []
        if max_ctx is not None:
            parts.append(f"макс. {_thousands(max_ctx)} токенов")
        if requested is not None:
            parts.append(f"запрос {_thousands(requested)} токенов")
        detail = f" ({', '.join(parts)})" if parts else ""
        msg = (
            f"Лимит контекста модели достигнут{detail}. История чата слишком "
            "большая — начните новый чат или сожмите контекст («Подвести итоги»)."
        )
        return {"type": "error", "error": msg, "code": exc.code}
    return {"type": "error", "error": str(exc)}


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
        history, requests}], "active_id": "…"|null}`` — активная сессия
        (открытая вкладка).
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
                "requests": [r.to_dict() for r in session.requests],
                "facts": list(session.facts),
                "memory": [s.to_dict() for s in session.memory_stores],
                "title": session.title,
                "parent_id": session.parent_id,
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
        "requests": [r.to_dict() for r in session.requests],
        "facts": list(session.facts),
        "memory": [s.to_dict() for s in session.memory_stores],
        "title": session.title,
        "parent_id": session.parent_id,
    }


@router.put("/{session_id}/memory")
async def sync_memory(
    session_id: str, body: MemorySyncRequest, request: Request
) -> dict:
    """Синхронизирует память чата (полное состояние вкладок «Память»).

    Клиент присылает все вкладки со своими флагами персистентности; сервер
    заменяет состояние и сохраняет в БД только персистентные вкладки.

    Args:
        session_id: идентификатор сессии.
        body: список вкладок памяти.
        request: HTTP-запрос.

    Returns:
        Санированное состояние памяти ``{"memory": [...]}``.

    Raises:
        HTTPException: 404 — сессия не найдена; 400 — не обычный чат.
    """
    registry: SessionRegistry = request.app.state.registry
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.kind != SessionKind.CHAT:
        raise HTTPException(400, "Memory is available for chat sessions only")
    session.memory_stores = _sanitize_memory(body.stores)
    registry.persist_memory(session)
    return {"memory": [s.to_dict() for s in session.memory_stores]}


@router.post("/{session_id}/branch", status_code=201, response_model=SessionCreateResponse)
async def branch_session(
    session_id: str, body: BranchRequest, request: Request
) -> SessionCreateResponse:
    """Создаёт чат-снапшот (ветку) от существующего обычного чата.

    Args:
        session_id: идентификатор исходной сессии.
        body: название новой вкладки (необязательно).
        request: HTTP-запрос.

    Returns:
        Идентификатор и модель нового чата.

    Raises:
        HTTPException: 404 — сессия не найдена; 400 — не обычный чат.
    """
    registry: SessionRegistry = request.app.state.registry
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.kind != SessionKind.CHAT:
        raise HTTPException(400, "Only chat sessions can be branched")
    branch = registry.branch(session, body.title)
    return SessionCreateResponse(id=branch.id, model=branch.model)


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
        reasoning_end/request_log/done/facts/error``.
    """
    registry: SessionRegistry = request.app.state.registry
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.busy:
        raise HTTPException(409, "Session is busy")

    opencode_session_id = request.app.state.opencode_session_id
    spec = _resolve_spec(registry, session, body.model, opencode_session_id)
    first_message = not session.history
    # Модель и «привязываемые» параметры (temperature/top_p/top_k) фиксируются
    # первым сообщением чата; далее изменения игнорируются.
    if body.model and first_message:
        session.model = body.model
    if body.settings is not None:
        incoming = sanitize_settings(body.settings)
        if first_message:
            session.settings = incoming
        else:
            # «Гибкие» параметры можно менять на лету, привязываемые — нет.
            session.settings.max_tokens = incoming.max_tokens
            session.settings.stop = incoming.stop
            session.settings.response_format = incoming.response_format
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
        # Помечаем чат занятым до первого await: саммаризация (если включена)
        # идёт до основного стрима, и параллельный запрос должен получить 409.
        session.busy = True
        # Записи, уже отданные клиенту в прошлых ответах, повторно не шлём.
        emitted = len(session.requests)

        def drain_request_logs() -> list[str]:
            nonlocal emitted
            frames = []
            while emitted < len(session.requests):
                record = session.requests[emitted]
                frames.append(_sse({"type": "request_log", "record": record.to_dict()}))
                emitted += 1
            return frames

        yield _sse({"type": "session", "id": session_id})
        try:
            # Скрытая чанковая саммаризация истории (если включена).
            summary_items = await session.ensure_summaries(runner, spec, gen_settings)
            # Токены саммаризации уже сожжены — сохраняем состояние и отдаём их сразу.
            registry.persist_summary_state(session)
            registry.persist_new_requests(session)
            for frame in drain_request_logs():
                yield frame

            async for event in session.stream_completion(
                runner,
                spec,
                gen_settings,
                extra_system=body.system_prompt,
                summary_items=summary_items,
            ):
                if event.get("type") == "done":
                    for frame in drain_request_logs():
                        yield frame
                yield _sse(event)
            # Успешное завершение: сохраняем пару user+assistant и записи.
            registry.remember_pair(session)
            # Стратегия facts: после ответа обновляем канонические факты.
            updated_facts = await session.extract_facts(
                runner, spec, gen_settings, body.content
            )
            if updated_facts is not None:
                registry.persist_facts(session)
            registry.persist_new_requests(session)
            for frame in drain_request_logs():
                yield frame
            if updated_facts is not None:
                yield _sse({"type": "facts", "items": updated_facts})
        except ProviderError as exc:
            session.rollback_user_message()
            registry.persist_new_requests(session)
            yield _sse(_provider_error_event(exc, session))
        except Exception as exc:  # noqa: BLE001 - отдаём ошибку клиенту как событие
            session.rollback_user_message()
            registry.persist_new_requests(session)
            yield _sse({"type": "error", "error": str(exc)})
        finally:
            session.busy = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

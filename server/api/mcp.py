"""Маршруты управления MCP-серверами (вкладка «MCP»).

Конфигурации хранятся в БД (формат OpenCode: ``local`` — stdio,
``remote`` — streamable-http) и живут в менеджере MCP. Синхронизация
полного состояния подключает/отключает серверы; отдельные операции
позволяют переподключить или отключить сервер вручную.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request

from ..schemas import McpSyncRequest
from ..services.mcp import MAX_MCP_SERVERS, McpError, McpServerConfig

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


def _sanitize_servers(raw_servers) -> list[McpServerConfig]:
    """Валидирует и нормализует список конфигураций MCP-серверов.

    Невалидные конфигурации (без имени/типа, с пустой командой или плохим
    URL) отбрасываются; отсутствующие/дублирующиеся идентификаторы
    перегенерируются.

    Args:
        raw_servers: «сырой» список конфигураций из запроса.

    Returns:
        Список :class:`McpServerConfig` (невалидное отброшено).
    """
    if not isinstance(raw_servers, list):
        return []
    servers: list[McpServerConfig] = []
    seen: set[str] = set()
    for raw in raw_servers[:MAX_MCP_SERVERS]:
        config = McpServerConfig.from_dict(raw)
        if config is None:
            continue
        if not config.id or config.id in seen:
            config.id = f"mcp-{secrets.token_hex(6)}"
        seen.add(config.id)
        servers.append(config)
    return servers


def _manager(request: Request):
    """Менеджер MCP из состояния приложения."""
    return request.app.state.mcp


def _state(manager) -> dict:
    """Текущее состояние серверов: ``{"servers": [...]}``."""
    return {"servers": manager.list_state()}


@router.get("")
async def list_servers(request: Request) -> dict:
    """Возвращает все MCP-серверы с конфигурацией, статусом и инструментами.

    Args:
        request: HTTP-запрос (менеджер из состояния приложения).

    Returns:
        ``{"servers": [{id, name, type, enabled, …, status, error, tools}]}``.
    """
    return _state(_manager(request))


@router.put("")
async def sync_servers(body: McpSyncRequest, request: Request) -> dict:
    """Синхронизирует MCP-серверы (полное состояние).

    Клиент присылает все конфигурации; сервер сохраняет их, подключает
    новые/включённые и отключает удалённые/выключенные/изменённые.

    Args:
        body: список конфигураций MCP-серверов.
        request: HTTP-запрос.

    Returns:
        Санированное состояние ``{"servers": [...]}``.
    """
    manager = _manager(request)
    servers = _sanitize_servers(body.servers)
    await manager.apply_config(servers)
    return _state(manager)


@router.post("/{server_id}/connect")
async def connect_server(server_id: str, request: Request) -> dict:
    """Принудительно (пере)подключает MCP-сервер.

    Args:
        server_id: идентификатор сервера.
        request: HTTP-запрос.

    Returns:
        ``{"server": {…}}`` с актуальным статусом и инструментами.

    Raises:
        HTTPException: 404 — сервер не найден.
    """
    manager = _manager(request)
    try:
        state = await manager.connect(server_id)
    except McpError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"server": state}


@router.post("/{server_id}/disconnect")
async def disconnect_server(server_id: str, request: Request) -> dict:
    """Отключает MCP-сервер.

    Args:
        server_id: идентификатор сервера.
        request: HTTP-запрос.

    Returns:
        ``{"server": {…}}`` со статусом ``disabled``.

    Raises:
        HTTPException: 404 — сервер не найден.
    """
    manager = _manager(request)
    try:
        state = await manager.disconnect(server_id)
    except McpError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"server": state}

"""Клиент MCP: конфигурация серверов и менеджер подключений.

MCP (Model Context Protocol) позволяет подключать к чатам внешние
инструменты. Менеджер хранит конфигурации серверов в стиле OpenCode
(``local`` — stdio-процесс, ``remote`` — streamable-http), поддерживает
живые подключения и отдаёт инструменты в OpenAI-формате для tool calling.

Подключения живут в фоновых задачах: на каждый включённый сервер одна
задача открывает транспорт, инициализирует сессию, кэширует список
инструментов и держит соединение до отключения.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

#: Тип транспорта: локальный stdio-процесс / удалённый streamable-http.
LOCAL = "local"
REMOTE = "remote"
TYPES = (LOCAL, REMOTE)

#: Статусы подключения сервера.
STATUS_DISABLED = "disabled"
STATUS_CONNECTING = "connecting"
STATUS_CONNECTED = "connected"
STATUS_ERROR = "error"

#: Лимиты конфигурации (невалидное молча отбрасывается).
MAX_MCP_SERVERS = 20
MAX_MCP_NAME = 100
MAX_MCP_COMMAND_ITEMS = 64
MAX_MCP_ARG_LEN = 500
MAX_MCP_MAP_ITEMS = 32
MAX_MCP_KEY_LEN = 128
MAX_MCP_VALUE_LEN = 2000
MAX_MCP_URL_LEN = 1000
MIN_TIMEOUT_MS = 1000
MAX_TIMEOUT_MS = 600_000
DEFAULT_TIMEOUT_MS = 5000

#: Таймаут одного вызова инструмента (секунды).
TOOL_CALL_TIMEOUT_S = 120

#: Таймаут ожидания подключения/отключения при ручных операциях (секунды).
CONNECT_GRACE_S = 5.0

#: Максимум инструментов, отдаваемых модели в одном запросе.
MAX_TOOLS = 64

#: Максимум символов результата инструмента, уходящего в контекст/на клиент.
MAX_TOOL_RESULT_CHARS = 20_000

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_MAP_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]")


class McpError(Exception):
    """Ошибка MCP-менеджера (недоступный инструмент/сервер)."""


def _sanitize_token(value: str, limit: int = 40) -> str:
    """Превращает имя в безопасный токен (для имени функции OpenAI)."""
    token = _NAME_RE.sub("_", value).strip("_")
    return token[:limit] or "server"


def _sanitize_map(raw: Any, key_limit: int = MAX_MCP_MAP_ITEMS) -> dict[str, str]:
    """Валидирует словарь строк (переменные окружения/заголовки)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in list(raw.items())[:key_limit]:
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        key, value = key.strip(), value.strip()
        if key and value:
            out[key[:MAX_MCP_KEY_LEN]] = value[:MAX_MCP_VALUE_LEN]
    return out


def _sanitize_command(raw: Any) -> list[str]:
    """Валидирует команду запуска stdio-сервера (argv)."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw[:MAX_MCP_COMMAND_ITEMS]:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item:
            out.append(item[:MAX_MCP_ARG_LEN])
    return out


def _sanitize_timeout(raw: Any) -> int:
    """Нормализует таймаут получения инструментов (мс)."""
    if not isinstance(raw, int) or isinstance(raw, bool):
        return DEFAULT_TIMEOUT_MS
    return max(MIN_TIMEOUT_MS, min(MAX_TIMEOUT_MS, raw))


@dataclass
class McpServerConfig:
    """Конфигурация одного MCP-сервера (формат OpenCode).

    Attributes:
        id: идентификатор сервера.
        name: имя сервера (префикс имён инструментов).
        type: ``local`` (stdio) или ``remote`` (streamable-http).
        enabled: подключать ли сервер при старте/синхронизации.
        command: команда запуска stdio-процесса (argv).
        environment: переменные окружения stdio-процесса.
        cwd: рабочий каталог stdio-процесса (или ``None``).
        url: адрес remote-сервера.
        headers: HTTP-заголовки remote-сервера.
        timeout_ms: таймаут получения списка инструментов (мс).
    """

    id: str
    name: str
    type: str = LOCAL
    enabled: bool = True
    command: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    @property
    def is_local(self) -> bool:
        """Транспорт — локальный stdio-процесс."""
        return self.type == LOCAL

    def to_dict(self) -> dict[str, Any]:
        """Представляет конфигурацию как словарь (для API/БД)."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "command": list(self.command),
            "environment": dict(self.environment),
            "cwd": self.cwd,
            "url": self.url,
            "headers": dict(self.headers),
            "timeout": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["McpServerConfig"]:
        """Валидирует и восстанавливает конфигурацию из словаря.

        Args:
            data: «сырой» словарь конфигурации.

        Returns:
            :class:`McpServerConfig` или ``None``, если конфигурация
            невалидна (нет имени/типа, пустая команда или плохой URL).
        """
        if not isinstance(data, dict):
            return None
        name = data.get("name")
        name = name.strip()[:MAX_MCP_NAME] if isinstance(name, str) else ""
        if not name:
            return None
        server_type = data.get("type")
        if server_type not in TYPES:
            return None
        server_id = data.get("id")
        server_id = (
            server_id.strip()[:64]
            if isinstance(server_id, str) and server_id.strip()
            else ""
        )
        config = cls(
            id=server_id,
            name=name,
            type=server_type,
            enabled=bool(data.get("enabled", True)),
            timeout_ms=_sanitize_timeout(data.get("timeout")),
        )
        if config.is_local:
            config.command = _sanitize_command(data.get("command"))
            if not config.command:
                return None
            config.environment = _sanitize_map(data.get("environment"))
            cwd = data.get("cwd")
            config.cwd = (
                cwd.strip()[:MAX_MCP_ARG_LEN]
                if isinstance(cwd, str) and cwd.strip()
                else None
            )
        else:
            url = data.get("url")
            url = url.strip() if isinstance(url, str) else ""
            if not url.startswith(("http://", "https://")):
                return None
            config.url = url[:MAX_MCP_URL_LEN]
            config.headers = _sanitize_map(data.get("headers"), key_limit=16)
        return config


def _resolve_command(command: list[str], cwd: Optional[str]) -> list[str]:
    """Приводит исполняемый файл команды к абсолютному пути.

    Относительный путь резолвится от ``cwd`` (или текущего каталога
    процесса) — иначе на Windows CreateProcess ищет его не там.
    """
    if not command:
        return command
    first = command[0]
    if os.path.isabs(first):
        return command
    base = cwd or os.getcwd()
    return [os.path.abspath(os.path.join(base, first)), *command[1:]]


def _default_local_transport(config: McpServerConfig):
    """Транспорт stdio: запускает процесс сервера."""
    command = _resolve_command(config.command, config.cwd)
    params = StdioServerParameters(
        command=command[0],
        args=command[1:],
        env=dict(config.environment) if config.environment else None,
        cwd=config.cwd,
    )
    return stdio_client(params)


def _default_remote_transport(config: McpServerConfig):
    """Транспорт streamable-http: подключается к URL сервера."""
    return streamablehttp_client(
        config.url,
        headers=config.headers or None,
        timeout=config.timeout_ms / 1000,
    )


def _tools_from_result(result: Any) -> list[dict[str, Any]]:
    """Превращает ответ ``list_tools`` в список словарей инструментов."""
    tools = []
    for tool in getattr(result, "tools", []) or []:
        schema = getattr(tool, "inputSchema", None)
        tools.append(
            {
                "name": str(getattr(tool, "name", "")),
                "description": getattr(tool, "description", None) or "",
                "input_schema": schema if isinstance(schema, dict) else {},
            }
        )
    return tools


def _result_text(result: Any) -> str:
    """Собирает текст результата ``call_tool`` из блоков контента."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif getattr(block, "data", None) is not None:
            parts.append("[бинарные данные]")
        else:
            parts.append(str(block))
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False))
    text = "\n".join(parts).strip()
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "… (обрезано)"
    return text


@dataclass
class McpConnection:
    """Живое подключение к MCP-серверу.

    Attributes:
        config: конфигурация сервера.
        status: текущий статус (см. ``STATUS_*``).
        error: текст ошибки подключения (при ``status=error``).
        tools: кэш инструментов сервера.
        session: активная клиентская сессия (или ``None``).
        lock: сериализация вызовов инструментов одного сервера.
        ready: событие готовности подключения (успех или ошибка).
        stop: событие остановки фоновой задачи.
        task: фоновая задача обслуживания подключения.
    """

    config: McpServerConfig
    status: str = STATUS_DISABLED
    error: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    session: Optional[ClientSession] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready: Optional[asyncio.Event] = None
    stop: Optional[asyncio.Event] = None
    task: Optional[asyncio.Task] = None


class MCPManager:
    """Менеджер MCP-серверов: конфигурации, подключения, инструменты.

    Args:
        store: хранилище SQLite для персистентности конфигураций.
        local_transport: фабрика транспорта для ``local`` (для тестов).
        remote_transport: фабрика транспорта для ``remote`` (для тестов).
        session_factory: фабрика клиентской сессии (для тестов).
    """

    def __init__(
        self,
        store=None,
        *,
        local_transport: Optional[Callable[[McpServerConfig], Any]] = None,
        remote_transport: Optional[Callable[[McpServerConfig], Any]] = None,
        session_factory: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self._store = store
        self._local_transport = local_transport or _default_local_transport
        self._remote_transport = remote_transport or _default_remote_transport
        self._session_factory = session_factory or ClientSession
        self._configs: list[McpServerConfig] = []
        self._connections: dict[str, McpConnection] = {}
        self._tool_index: dict[str, tuple[str, str]] = {}

    # --- конфигурация -----------------------------------------------------

    def load(self) -> None:
        """Загружает конфигурации из хранилища (если оно задано)."""
        if self._store is None:
            return
        raw = self._store.load_mcp_servers()
        self._configs = [
            config
            for config in (McpServerConfig.from_dict(item) for item in raw)
            if config is not None
        ]

    def list_configs(self) -> list[McpServerConfig]:
        """Возвращает текущий список конфигураций."""
        return list(self._configs)

    def replace_configs(self, configs: list[McpServerConfig]) -> None:
        """Полностью заменяет конфигурации и сохраняет их в БД."""
        self._configs = list(configs)
        if self._store is not None:
            self._store.save_mcp_servers([c.to_dict() for c in self._configs])

    def _get_config(self, server_id: str) -> Optional[McpServerConfig]:
        for config in self._configs:
            if config.id == server_id:
                return config
        return None

    # --- жизненный цикл ---------------------------------------------------

    async def startup(self) -> None:
        """Загружает конфигурации и подключает включённые серверы."""
        self.load()
        for config in self._configs:
            if config.enabled:
                self._start_connect(config)

    async def shutdown(self) -> None:
        """Отключает все серверы (при остановке приложения)."""
        for server_id in list(self._connections):
            await self._disconnect(server_id)

    async def apply_config(self, configs: list[McpServerConfig]) -> None:
        """Синхронизирует подключения с новым списком конфигураций.

        Удалённые, выключенные и изменённые серверы отключаются; новые и
        включённые — подключаются в фоне.

        Args:
            configs: новый полный список конфигураций.
        """
        incoming = {config.id: config for config in configs}
        for server_id, conn in list(self._connections.items()):
            target = incoming.get(server_id)
            if (
                target is None
                or not target.enabled
                or target.to_dict() != conn.config.to_dict()
            ):
                await self._disconnect(server_id)
        self.replace_configs(configs)
        started: list[McpServerConfig] = []
        for config in configs:
            if not config.enabled:
                continue
            conn = self._connections.get(config.id)
            if conn is not None and conn.status in (STATUS_CONNECTING, STATUS_CONNECTED):
                continue
            self._start_connect(config)
            started.append(config)
        for config in started:
            conn = self._connections.get(config.id)
            if conn is not None:
                await self._await_ready(conn, config.timeout_ms / 1000 + CONNECT_GRACE_S)

    async def _await_ready(self, conn: McpConnection, timeout_s: float) -> None:
        """Дожидается готовности подключения (успех или ошибка)."""
        if conn.ready is None:
            return
        try:
            await asyncio.wait_for(conn.ready.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            conn.status = STATUS_ERROR
            conn.error = "Таймаут подключения"

    def _start_connect(self, config: McpServerConfig) -> None:
        """Запускает фоновую задачу подключения к серверу."""
        conn = McpConnection(config=config, status=STATUS_CONNECTING)
        conn.ready = asyncio.Event()
        conn.stop = asyncio.Event()
        self._connections[config.id] = conn
        conn.task = asyncio.create_task(self._serve(conn))

    async def _serve(self, conn: McpConnection) -> None:
        """Держит подключение сервера: транспорт, сессия, инструменты."""
        config = conn.config
        timeout_s = max(config.timeout_ms / 1000, 1.0)
        try:
            async with AsyncExitStack() as stack:
                transport = (
                    self._local_transport(config)
                    if config.is_local
                    else self._remote_transport(config)
                )
                streams = await stack.enter_async_context(transport)
                session = await stack.enter_async_context(
                    self._session_factory(streams[0], streams[1])
                )
                await asyncio.wait_for(session.initialize(), timeout=timeout_s)
                result = await asyncio.wait_for(
                    session.list_tools(), timeout=timeout_s
                )
                conn.tools = _tools_from_result(result)
                conn.session = session
                conn.status = STATUS_CONNECTED
                conn.error = ""
                conn.ready.set()
                await conn.stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - статус вместо падения приложения
            conn.status = STATUS_ERROR
            conn.error = f"{type(exc).__name__}: {exc}"
            conn.tools = []
            logger.warning(
                "MCP-сервер %s не подключился: %s", config.name, conn.error
            )
        finally:
            conn.session = None
            if conn.status == STATUS_CONNECTED:
                conn.status = STATUS_DISABLED
            if conn.ready is not None and not conn.ready.is_set():
                conn.ready.set()

    async def _disconnect(self, server_id: str) -> None:
        """Останавливает подключение и дожидается завершения задачи."""
        conn = self._connections.pop(server_id, None)
        if conn is None:
            return
        if conn.stop is not None:
            conn.stop.set()
        task = conn.task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=CONNECT_GRACE_S)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            except Exception:  # noqa: BLE001 - задача сама пишет статус
                pass
        conn.session = None

    # --- операции API -----------------------------------------------------

    async def connect(self, server_id: str) -> dict[str, Any]:
        """Принудительно (пере)подключает сервер.

        Args:
            server_id: идентификатор сервера.

        Returns:
            Состояние сервера (конфигурация + статус + инструменты).

        Raises:
            McpError: если сервер не найден.
        """
        config = self._get_config(server_id)
        if config is None:
            raise McpError(f"MCP-сервер {server_id} не найден")
        await self._disconnect(server_id)
        self._start_connect(config)
        conn = self._connections[server_id]
        await self._await_ready(conn, config.timeout_ms / 1000 + CONNECT_GRACE_S)
        return self._state(conn)

    async def disconnect(self, server_id: str) -> dict[str, Any]:
        """Отключает сервер.

        Args:
            server_id: идентификатор сервера.

        Returns:
            Состояние сервера (``disabled``).

        Raises:
            McpError: если сервер не найден.
        """
        config = self._get_config(server_id)
        if config is None:
            raise McpError(f"MCP-сервер {server_id} не найден")
        await self._disconnect(server_id)
        return self._state(McpConnection(config=config, status=STATUS_DISABLED))

    def list_state(self) -> list[dict[str, Any]]:
        """Возвращает состояние всех серверов (конфигурация + статус)."""
        states = []
        for config in self._configs:
            conn = self._connections.get(config.id)
            if conn is None or conn.config.to_dict() != config.to_dict():
                conn = McpConnection(config=config, status=STATUS_DISABLED)
            states.append(self._state(conn))
        return states

    @staticmethod
    def _state(conn: McpConnection) -> dict[str, Any]:
        """Собирает состояние сервера для API."""
        data = conn.config.to_dict()
        data["status"] = conn.status
        data["error"] = conn.error
        data["tools"] = [dict(tool) for tool in conn.tools]
        return data

    # --- инструменты ------------------------------------------------------

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Возвращает OpenAI-описания инструментов подключённых серверов.

        Имена инструментов префиксуются именем сервера
        (``mcp_{сервер}_{инструмент}``); карта соответствий сохраняется
        для :meth:`call_tool`.
        """
        tools: list[dict[str, Any]] = []
        index: dict[str, tuple[str, str]] = {}
        used: set[str] = set()
        for config in self._configs:
            conn = self._connections.get(config.id)
            if conn is None or conn.status != STATUS_CONNECTED:
                continue
            for tool in conn.tools:
                name = _unique_tool_name(config.name, tool["name"], used)
                used.add(name)
                index[name] = (config.id, tool["name"])
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.get("description")
                            or f"MCP-инструмент {tool['name']} ({config.name})",
                            "parameters": tool.get("input_schema")
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
                if len(tools) >= MAX_TOOLS:
                    break
            if len(tools) >= MAX_TOOLS:
                break
        self._tool_index = index
        return tools

    async def call_tool(
        self, name: str, arguments: Optional[dict[str, Any]] = None
    ) -> tuple[str, bool]:
        """Вызывает инструмент по имени OpenAI-описания.

        Args:
            name: префиксованное имя инструмента.
            arguments: аргументы вызова.

        Returns:
            Кортеж ``(текст результата, is_error)``.

        Raises:
            McpError: если инструмент неизвестен или сервер не подключён.
        """
        target = self._tool_index.get(name)
        if target is None:
            self.tool_definitions()
            target = self._tool_index.get(name)
        if target is None:
            raise McpError(f"Инструмент {name} не найден")
        server_id, tool_name = target
        conn = self._connections.get(server_id)
        if conn is None or conn.session is None or conn.status != STATUS_CONNECTED:
            server_label = conn.config.name if conn else server_id
            raise McpError(f"MCP-сервер {server_label} не подключён")
        async with conn.lock:
            result = await conn.session.call_tool(
                tool_name,
                arguments or {},
                read_timeout_seconds=timedelta(seconds=TOOL_CALL_TIMEOUT_S),
            )
        return _result_text(result), bool(getattr(result, "isError", False))


def _unique_tool_name(server_name: str, tool_name: str, used: set[str]) -> str:
    """Строит уникальное имя функции OpenAI для инструмента сервера."""
    base = f"mcp_{_sanitize_token(server_name)}_{_sanitize_token(tool_name)}"[:64]
    name = base
    counter = 2
    while name in used:
        suffix = f"_{counter}"
        name = base[: 64 - len(suffix)] + suffix
        counter += 1
    return name

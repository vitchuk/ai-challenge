"""Общие фейки MCP для тестов (транспорт, сессия, инструменты).

Файл не собирается pytest (нет префикса ``test_``), но импортируется
тестовыми модулями.
"""

from __future__ import annotations

import asyncio

from server.services.mcp import MCPManager, McpServerConfig


class FakeTool:
    """Инструмент MCP-сервера (аналог ``mcp.types.Tool``)."""

    def __init__(self, name, description="", schema=None):
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {}}


class FakeListTools:
    """Ответ ``list_tools``."""

    def __init__(self, tools):
        self.tools = tools


class FakeBlock:
    """Блок контента результата вызова."""

    def __init__(self, text):
        self.text = text


class FakeCallResult:
    """Ответ ``call_tool``."""

    def __init__(self, texts, is_error=False):
        self.content = [FakeBlock(text) for text in texts]
        self.isError = is_error
        self.structuredContent = None


class FakeSession:
    """Клиентская сессия MCP: инициализация, список и вызов инструментов."""

    def __init__(self, tools, *, fail_init=False, fail_call=False, is_error=False):
        self._tools = tools
        self._fail_init = fail_init
        self._fail_call = fail_call
        self._is_error = is_error
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        if self._fail_init:
            raise RuntimeError("init boom")

    async def list_tools(self):
        return FakeListTools(self._tools)

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
        if self._fail_call:
            raise RuntimeError("call boom")
        self.calls.append((name, arguments))
        return FakeCallResult([f"{name}:ok"], is_error=self._is_error)


class FakeStreams:
    """Контекст транспорта: отдаёт пару потоков ``(read, write)``."""

    async def __aenter__(self):
        return (None, None)

    async def __aexit__(self, *exc):
        return False


def make_manager(session, *, store=None):
    """Менеджер с фейковым транспортом и одной сессией на подключение."""
    return MCPManager(
        store=store,
        local_transport=lambda config: FakeStreams(),
        remote_transport=lambda config: FakeStreams(),
        session_factory=lambda read, write: session,
    )


def local_config(server_id="s1", name="demo", command=("python", "srv.py"), **kwargs):
    """Валидная local-конфигурация (или ``AssertionError``)."""
    data = {
        "id": server_id,
        "name": name,
        "type": "local",
        "command": list(command),
        **kwargs,
    }
    config = McpServerConfig.from_dict(data)
    assert config is not None
    return config


async def wait_ready(manager, server_id):
    """Дожидается готовности подключения сервера."""
    conn = manager._connections[server_id]
    await asyncio.wait_for(conn.ready.wait(), timeout=2)
    return conn

"""Тесты менеджера MCP: конфигурации, подключения, инструменты."""

import asyncio

import pytest

from server.services.mcp import (
    McpError,
    McpServerConfig,
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_ERROR,
    _unique_tool_name,
)
from tests.mcp_fakes import (
    FakeSession,
    FakeTool,
    local_config,
    make_manager,
    wait_ready,
)


# --- конфигурация ---------------------------------------------------------


def test_config_local_valid():
    config = local_config(environment={"A": "1"}, timeout=7000)
    assert config.is_local
    assert config.command == ["python", "srv.py"]
    assert config.environment == {"A": "1"}
    assert config.timeout_ms == 7000


def test_config_local_requires_command():
    assert McpServerConfig.from_dict({"name": "x", "type": "local"}) is None
    assert McpServerConfig.from_dict(
        {"name": "x", "type": "local", "command": ["  ", ""]}
    ) is None


def test_config_remote_requires_http_url():
    assert McpServerConfig.from_dict({"name": "x", "type": "remote"}) is None
    assert (
        McpServerConfig.from_dict({"name": "x", "type": "remote", "url": "ftp://x"})
        is None
    )
    config = McpServerConfig.from_dict(
        {"name": "x", "type": "remote", "url": "http://127.0.0.1:8001/mcp"}
    )
    assert config is not None and config.url.endswith("/mcp")


def test_config_requires_name_and_known_type():
    assert McpServerConfig.from_dict({"type": "local", "command": ["x"]}) is None
    assert McpServerConfig.from_dict({"name": "x", "type": "other"}) is None
    assert McpServerConfig.from_dict("nope") is None


def test_config_timeout_clamped():
    assert local_config(timeout=1).timeout_ms == 1000
    assert local_config(timeout=10**9).timeout_ms == 600_000
    assert local_config(timeout="bad").timeout_ms == 5000


# --- подключения ----------------------------------------------------------


async def test_connect_sets_connected_and_tools():
    session = FakeSession([FakeTool("list_items", "список")])
    manager = make_manager(session)
    manager.replace_configs([local_config()])

    state = await manager.connect("s1")
    assert state["status"] == STATUS_CONNECTED
    assert [tool["name"] for tool in state["tools"]] == ["list_items"]
    await manager.shutdown()


async def test_connect_failure_sets_error():
    session = FakeSession([], fail_init=True)
    manager = make_manager(session)
    manager.replace_configs([local_config()])

    state = await manager.connect("s1")
    assert state["status"] == STATUS_ERROR
    assert "init boom" in state["error"]
    assert state["tools"] == []
    await manager.shutdown()


async def test_connect_unknown_server_raises():
    manager = make_manager(FakeSession([]))
    with pytest.raises(McpError):
        await manager.connect("nope")


async def test_disconnect_sets_disabled():
    manager = make_manager(FakeSession([FakeTool("t")]))
    manager.replace_configs([local_config()])
    await manager.connect("s1")

    state = await manager.disconnect("s1")
    assert state["status"] == STATUS_DISABLED
    assert "s1" not in manager._connections


async def test_list_state_disabled_without_connection():
    manager = make_manager(FakeSession([FakeTool("t")]))
    manager.replace_configs([local_config(enabled=False)])
    states = manager.list_state()
    assert len(states) == 1
    assert states[0]["status"] == STATUS_DISABLED


async def test_apply_config_disconnects_removed_and_skips_disabled():
    manager = make_manager(FakeSession([FakeTool("t")]))
    manager.replace_configs([local_config("s1"), local_config("s2", name="two")])
    await manager.connect("s1")
    await manager.connect("s2")

    await manager.apply_config([local_config("s2", name="two", enabled=False)])
    assert "s1" not in manager._connections
    assert "s2" not in manager._connections
    assert [config.id for config in manager.list_configs()] == ["s2"]


async def test_apply_config_reconnects_changed():
    session = FakeSession([FakeTool("t")])
    manager = make_manager(session)
    manager.replace_configs([local_config()])
    await manager.connect("s1")
    assert manager._connections["s1"].config.command == ["python", "srv.py"]

    await manager.apply_config([local_config(command=("python", "other.py"))])
    conn = await wait_ready(manager, "s1")
    assert conn.status == STATUS_CONNECTED
    assert conn.config.command == ["python", "other.py"]
    await manager.shutdown()


# --- инструменты ----------------------------------------------------------


async def test_tool_definitions_prefixed_and_indexed():
    session = FakeSession([FakeTool("get_item", "получить", {"type": "object"})])
    manager = make_manager(session)
    manager.replace_configs([local_config(name="jsonplaceholder")])
    await manager.connect("s1")

    tools = manager.tool_definitions()
    assert len(tools) == 1
    function = tools[0]["function"]
    assert function["name"] == "mcp_jsonplaceholder_get_item"
    assert function["description"] == "получить"
    assert manager._tool_index[function["name"]] == ("s1", "get_item")
    await manager.shutdown()


async def test_tool_definitions_empty_without_connections():
    manager = make_manager(FakeSession([FakeTool("t")]))
    manager.replace_configs([local_config(enabled=False)])
    assert manager.tool_definitions() == []


async def test_call_tool_success():
    session = FakeSession([FakeTool("get_item")])
    manager = make_manager(session)
    manager.replace_configs([local_config()])
    await manager.connect("s1")
    manager.tool_definitions()

    text, is_error = await manager.call_tool(
        "mcp_demo_get_item", {"resource": "posts", "id": 1}
    )
    assert text == "get_item:ok"
    assert is_error is False
    assert session.calls == [("get_item", {"resource": "posts", "id": 1})]
    await manager.shutdown()


async def test_call_tool_error_flag_and_unknown():
    session = FakeSession([FakeTool("t")], is_error=True)
    manager = make_manager(session)
    manager.replace_configs([local_config()])
    await manager.connect("s1")
    manager.tool_definitions()

    _, is_error = await manager.call_tool("mcp_demo_t")
    assert is_error is True
    with pytest.raises(McpError):
        await manager.call_tool("mcp_demo_missing")
    await manager.shutdown()


async def test_call_tool_disconnected_raises():
    session = FakeSession([FakeTool("t")])
    manager = make_manager(session)
    manager.replace_configs([local_config()])
    await manager.connect("s1")
    manager.tool_definitions()
    await manager.disconnect("s1")
    with pytest.raises(McpError):
        await manager.call_tool("mcp_demo_t")


def test_unique_tool_name_collision():
    used = set()
    first = _unique_tool_name("demo", "run", used)
    used.add(first)
    second = _unique_tool_name("demo", "run", used)
    assert first == "mcp_demo_run"
    assert second == "mcp_demo_run_2"


def test_unique_tool_name_sanitizes():
    name = _unique_tool_name("my server.v2", "do/it", set())
    assert name == "mcp_my_server_v2_do_it"

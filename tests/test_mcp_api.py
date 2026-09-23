"""Тесты HTTP API MCP-серверов и подключения инструментов к чатам."""

import json

import pytest_asyncio

from tests.mcp_fakes import FakeSession, FakeTool, make_manager


@pytest_asyncio.fixture
async def mcp_app(app):
    """Приложение с фейковым менеджером MCP (транспорт не ходит в сеть)."""
    session = FakeSession([FakeTool("get_item", "получить объект")])
    app.state.mcp = make_manager(session)
    app.state.mcp_session = session
    yield app
    await app.state.mcp.shutdown()


async def test_list_empty(mcp_app, client):
    res = await client.get("/api/mcp")
    assert res.status_code == 200
    assert res.json() == {"servers": []}


async def test_sync_connects_and_returns_tools(mcp_app, client):
    payload = {
        "servers": [
            {
                "id": "mcp-json",
                "name": "jsonplaceholder",
                "type": "local",
                "command": ["python", "srv.py"],
                "enabled": True,
                "timeout": 5000,
            }
        ]
    }
    res = await client.put("/api/mcp", json=payload)
    assert res.status_code == 200
    servers = res.json()["servers"]
    assert len(servers) == 1
    assert servers[0]["status"] == "connected"
    assert servers[0]["tools"][0]["name"] == "get_item"
    assert servers[0]["tools"][0]["description"] == "получить объект"

    again = await client.get("/api/mcp")
    assert again.json()["servers"][0]["status"] == "connected"


async def test_sync_drops_invalid_and_regenerates_ids(mcp_app, client):
    payload = {
        "servers": [
            {"name": "", "type": "local", "command": ["x"]},  # без имени
            {"name": "bad-type", "type": "nope"},  # неизвестный тип
            {"name": "no-command", "type": "local"},  # пустая команда
            {"name": "no-url", "type": "remote"},  # нет url
            {
                "name": "dup",
                "type": "local",
                "command": ["python", "a.py"],
                "id": "same",
            },
            {
                "name": "dup2",
                "type": "local",
                "command": ["python", "b.py"],
                "id": "same",
            },
        ]
    }
    res = await client.put("/api/mcp", json=payload)
    servers = res.json()["servers"]
    assert [server["name"] for server in servers] == ["dup", "dup2"]
    assert servers[0]["id"] != servers[1]["id"]


async def test_connect_and_disconnect_endpoints(mcp_app, client):
    await client.put(
        "/api/mcp",
        json={
            "servers": [
                {
                    "id": "s1",
                    "name": "demo",
                    "type": "local",
                    "command": ["python", "srv.py"],
                    "enabled": False,
                }
            ]
        },
    )
    res = await client.post("/api/mcp/s1/connect")
    assert res.status_code == 200
    assert res.json()["server"]["status"] == "connected"

    res = await client.post("/api/mcp/s1/disconnect")
    assert res.status_code == 200
    assert res.json()["server"]["status"] == "disabled"


async def test_connect_unknown_returns_404(mcp_app, client):
    res = await client.post("/api/mcp/nope/connect")
    assert res.status_code == 404
    res = await client.post("/api/mcp/nope/disconnect")
    assert res.status_code == 404


async def test_chat_request_includes_mcp_tools(mcp_app, client):
    """Подключённый MCP-сервер добавляет tools в запрос к модели."""
    await client.put(
        "/api/mcp",
        json={
            "servers": [
                {
                    "id": "s1",
                    "name": "jsonplaceholder",
                    "type": "local",
                    "command": ["python", "srv.py"],
                    "enabled": True,
                }
            ]
        },
    )
    created = await client.post("/api/sessions", json={"kind": "chat"})
    sid = created.json()["id"]

    res = await client.post(
        f"/api/sessions/{sid}/messages", json={"content": "покажи пост 1"}
    )
    assert res.status_code == 200

    body = json.loads(mcp_app.state.mock_transport.requests[0].content)
    names = [tool["function"]["name"] for tool in body["tools"]]
    assert names == ["mcp_jsonplaceholder_get_item"]


async def test_chat_without_mcp_servers_sends_no_tools(client, app):
    """Без подключённых MCP-серверов параметр tools не уходит в апстрим."""
    created = await client.post("/api/sessions", json={"kind": "chat"})
    sid = created.json()["id"]
    res = await client.post(
        f"/api/sessions/{sid}/messages", json={"content": "привет"}
    )
    assert res.status_code == 200
    body = json.loads(app.state.mock_transport.requests[0].content)
    assert "tools" not in body

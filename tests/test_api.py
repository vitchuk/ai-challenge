"""Интеграционные тесты HTTP API (через ASGI-клиент с мок-апстримом)."""

import json

from server.services.storage import SessionStore


def parse_sse(text: str) -> list[dict]:
    """Разбирает SSE-поток в список событий."""
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                events.append(json.loads(payload))
    return events


async def test_full_session_lifecycle(client):
    r = await client.post("/api/sessions", json={"model": "deepseek-chat"})
    assert r.status_code == 201
    sid = r.json()["id"]

    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "вопрос"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    types = [e["type"] for e in events]
    assert types == ["session", "done"]
    assert events[-1]["content"] == "Ответ"
    assert events[-1]["meta"]["completion_tokens"] == 233
    assert events[-1]["meta"]["reasoning_tokens"] == 100

    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    state = r.json()
    history = state["history"]
    # первое сообщение чата стало системным промптом (сид)
    assert state["system_prompt"] == "вопрос"
    assert [m["role"] for m in history] == ["system", "assistant"]
    assert history[1]["meta"]["prompt_tokens"] == 202

    r = await client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 204
    r = await client.get(f"/api/sessions/{sid}")
    assert r.status_code == 404


async def test_unknown_session_404(client):
    assert (await client.post("/api/sessions/nope/messages", json={"content": "x"})).status_code == 404
    assert (await client.get("/api/sessions/nope")).status_code == 404
    assert (await client.delete("/api/sessions/nope")).status_code == 404


async def test_busy_session_409(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    svc = app.state.registry.get(sid)
    svc.busy = True
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "x"})
    assert r.status_code == 409


async def test_settings_encapsulated_per_chat(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={
        "content": "инструкция",
        "settings": {"temperature": 0.3, "top_p": 0.5, "max_tokens": 500},
    })
    assert parse_sse(r.text)[-1]["type"] == "done"
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert state["settings"]["temperature"] == 0.3
    assert state["settings"]["top_p"] == 0.5
    assert state["settings"]["max_tokens"] == 500
    # следующий запрос без настроек использует настройки чата
    svc = app.state.registry.get(sid)
    assert svc.settings.temperature == 0.3


async def test_first_message_binds_model_and_params(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={
        "content": "инструкция",
        "model": "opencode/glm-5.3",
        "settings": {"temperature": 0.3, "top_p": 0.5, "top_k": 40},
    })
    assert parse_sse(r.text)[-1]["type"] == "done"
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert state["model"] == "opencode/glm-5.3"
    assert state["settings"]["temperature"] == 0.3
    assert state["settings"]["top_p"] == 0.5
    assert state["settings"]["top_k"] == 40


async def test_binding_immutable_but_mutable_on_the_fly(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    # первое сообщение привязывает модель и temperature/top_p/top_k
    await client.post(f"/api/sessions/{sid}/messages", json={
        "content": "инструкция",
        "model": "deepseek-v4-flash",
        "settings": {"temperature": 0.3, "top_p": 0.5, "top_k": 40,
                     "max_tokens": 100, "stop": ["x"]},
    })
    # второе сообщение пытается сменить всё, но привязываемые — игнорируются
    r = await client.post(f"/api/sessions/{sid}/messages", json={
        "content": "вопрос",
        "model": "opencode/glm-5.3",
        "settings": {"temperature": 0.9, "top_p": 0.9, "top_k": 90,
                     "max_tokens": 500, "stop": ["y"],
                     "response_format": {"type": "json_object"}},
    })
    assert parse_sse(r.text)[-1]["type"] == "done"

    svc = app.state.registry.get(sid)
    # привязываемые — прежние
    assert svc.model == "deepseek-v4-flash"
    assert svc.settings.temperature == 0.3
    assert svc.settings.top_p == 0.5
    assert svc.settings.top_k == 40
    # гибкие — обновились
    assert svc.settings.max_tokens == 500
    assert svc.settings.stop == ["y"]
    assert svc.settings.response_format == {"type": "json_object"}

    # в апстрим ушло: прежние temperature/top_p + новые max_tokens/stop
    body = json.loads(app.state.mock_transport.requests[-1].content)
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.5
    assert body["max_tokens"] == 500
    assert body["stop"] == ["y"]


CONTEXT_ERROR_BODY = json.dumps({
    "error": {
        "message": (
            "This model's maximum context length is 1048576 tokens. However, you "
            "requested 1293247 tokens (900031 in the messages, 393216 in the completion). "
            "Please reduce the length of the messages or completion."
        ),
        "type": "invalid_request_error",
        "code": "invalid_request_error",
    }
})


async def _error_app(monkeypatch, status, error_text):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test"
    ))
    application = create_app()
    application.state.http_client = httpx.AsyncClient(
        transport=MockTransport([], status=status, error_text=error_text)
    )
    application.state.opencode_session_id = "s"
    application.state.registry = SessionRegistry(config.get_settings())
    return application


async def test_context_limit_error_has_code(monkeypatch):
    import httpx

    application = await _error_app(monkeypatch, 400, CONTEXT_ERROR_BODY)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://t") as c:
        sid = (await c.post("/api/sessions", json={})).json()["id"]
        r = await c.post(f"/api/sessions/{sid}/messages", json={"content": "большой запрос"})
        events = parse_sse(r.text)
        err = next(e for e in events if e["type"] == "error")
        assert err["code"] == "context_length_exceeded"
        assert "Лимит контекста" in err["error"]
        assert "1 048 576" in err["error"]
        # user-сообщение откатилось — в истории ничего
        state = (await c.get(f"/api/sessions/{sid}")).json()
        assert state["history"] == []
    await application.state.http_client.aclose()


async def test_generic_error_has_no_code(monkeypatch):
    import httpx

    application = await _error_app(monkeypatch, 402, json.dumps(
        {"error": {"message": "Insufficient balance", "type": "invalid_request_error"}}
    ))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://t") as c:
        sid = (await c.post("/api/sessions", json={})).json()["id"]
        r = await c.post(f"/api/sessions/{sid}/messages", json={"content": "вопрос"})
        err = next(e for e in parse_sse(r.text) if e["type"] == "error")
        assert "code" not in err
        assert "Insufficient balance" in err["error"]
    await application.state.http_client.aclose()


async def test_models_include_context(monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk", opencode_api_key="zen"
    ))
    application = create_app()

    class CatalogTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            url = str(request.url)
            if "deepseek.com/models" in url:
                data = {"data": [{"id": "deepseek-v4-flash"}]}
            elif "/zen/go/v1/models" in url:
                data = {"data": [{"id": "glm-5.3"}]}
            elif "/zen/v1/models" in url:
                data = {"data": [{"id": "big-pickle"}]}
            else:
                data = {"data": []}
            return httpx.Response(200, json=data, request=request)

    application.state.http_client = httpx.AsyncClient(transport=CatalogTransport())
    application.state.opencode_session_id = "s"
    application.state.registry = SessionRegistry(config.get_settings())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://t") as c:
        r = await c.get("/api/models")
        by_id = {m["id"]: m for m in r.json()["data"]}
        assert by_id["deepseek-v4-flash"]["context"] == 1048576
        # лимиты этих моделей апстримом не раскрываются — context отсутствует
        assert by_id["opencode/glm-5.3"]["context"] is None
        assert by_id["opencode/big-pickle"]["context"] is None
    await application.state.http_client.aclose()


async def test_optimize_prompt_session_parameters(app, client):
    """Сценарий команды /optimize-prompt: изолированная сессия с нужными настройками."""
    user_text = "Напиши приветствие для блога"
    system_prompt = (
        "Действуй как профессиональный промт-инженер. Создай детальный промт "
        f"для языковой модели по запросу: {user_text}. Промпт должен быть на "
        "языке запроса, лаконичен и структурирован."
    )
    r = await client.post(
        "/api/sessions",
        json={
            "kind": "ephemeral",
            "model": "deepseek-chat",
            "settings": {"temperature": 0.1, "top_p": 0.01},
            "system_prompt": system_prompt,
        },
    )
    assert r.status_code == 201
    sid = r.json()["id"]
    svc = app.state.registry.get(sid)
    assert svc.kind.value == "ephemeral"
    assert svc.settings.temperature == 0.1
    assert svc.settings.top_p == 0.01
    assert svc.system_prompt == system_prompt

    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": user_text})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[-1]["type"] == "done"

    # закрытие модалки удаляет сессию
    assert (await client.delete(f"/api/sessions/{sid}")).status_code == 204
    assert app.state.registry.get(sid) is None


async def test_optimize_prompt_ephemeral_with_store(tmp_path, monkeypatch):
    """Эфемерная сессия с реальным SQLite-store не падает на FK и не персистится."""
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test",
        opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    application = create_app()
    transport = MockTransport(make_chat_chunks(content="готовый промпт", usage=USAGE))
    application.state.http_client = httpx.AsyncClient(transport=transport)
    application.state.opencode_session_id = "sess"
    application.state.registry = SessionRegistry(config.get_settings(), store=SessionStore(db))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://t") as c:
        r = await c.post("/api/sessions", json={
            "kind": "ephemeral",
            "model": "deepseek-chat",
            "settings": {"temperature": 0.1, "top_p": 0.01},
            "system_prompt": "Действуй как промт-инженер",
        })
        sid = r.json()["id"]
        r = await c.post(f"/api/sessions/{sid}/messages", json={"content": "запрос"})
        events = parse_sse(r.text)
        # нет ошибки FOREIGN KEY constraint failed — только session + done
        assert events[-1]["type"] == "done"
        assert "error" not in [e["type"] for e in events]
        assert events[-1]["content"] == "готовый промпт"

        # эфемерная сессия не попала в БД и в список восстановления
        verify = SessionStore(db)
        try:
            assert verify.load_all() == []
        finally:
            verify.close()
        assert sid not in {s["id"] for s in (await c.get("/api/sessions")).json()["data"]}

        # закрытие модалки удаляет сессию из памяти без ошибок
        assert (await c.delete(f"/api/sessions/{sid}")).status_code == 204
    await application.state.http_client.aclose()


async def test_summary_session_rebuilds_toon_context(app, client):
    other = (await client.post("/api/sessions", json={"model": "deepseek-chat"})).json()["id"]
    await client.post(f"/api/sessions/{other}/messages", json={"content": "привет"})

    summary = (await client.post("/api/sessions", json={"kind": "summary"})).json()["id"]
    r = await client.post(f"/api/sessions/{summary}/messages", json={"content": "подведи итоги"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[-1]["type"] == "done"

    svc = app.state.registry.get(summary)
    assert "У тебя есть доступ" in svc.system_prompt
    assert "привет" in svc.system_prompt


async def test_invalid_content_422(client):
    r = await client.post("/api/sessions/chat-1/messages", json={"content": ""})
    assert r.status_code == 422


async def test_list_sessions_excludes_empty_and_ephemeral(app, client):
    empty = (await client.post("/api/sessions", json={})).json()["id"]
    ephemeral = (
        await client.post("/api/sessions", json={"kind": "ephemeral"})
    ).json()["id"]
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    ids = {s["id"] for s in r.json()["data"]}
    assert empty in ids
    assert ephemeral not in ids


async def test_activate_session(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    assert (await client.post(f"/api/sessions/{sid}/activate")).status_code == 204
    assert (await client.get("/api/sessions")).json()["active_id"] == sid
    # несуществующая сессия
    assert (await client.post("/api/sessions/nope/activate")).status_code == 404


async def test_restart_restores_full_history(tmp_path, monkeypatch):
    """Сценарий рестарта: после пересоздания приложения на той же БД контекст сохраняется."""
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test",
        opencode_api_key="zen-test",
    ))

    db = str(tmp_path / "chats.db")

    def build():
        application = create_app()
        transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "sess"
        application.state.registry = SessionRegistry(
            config.get_settings(),
            store=SessionStore(db),
        )
        application.state.registry.restore()
        return application, transport

    # ── «сеанс 1»: создаём чат и отправляем сообщение
    app1, transport1 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        sid = (await c1.post("/api/sessions", json={})).json()["id"]
        r = await c1.post(f"/api/sessions/{sid}/messages", json={
            "content": "первый вопрос",
            "model": "opencode/glm-5.3",
            "settings": {"temperature": 0.3, "top_p": 0.5},
        })
        assert parse_sse(r.text)[-1]["type"] == "done"
        # первое сообщение стало системным промптом: первый запрос уходит как [system]
        first_body = json.loads(transport1.requests[-1].content)
        assert first_body["messages"] == [{"role": "system", "content": "первый вопрос"}]
        # сессия запомнила модель обращения; открытая вкладка отмечена как активная
        assert (await c1.post(f"/api/sessions/{sid}/activate")).status_code == 204
        lst1 = (await c1.get("/api/sessions")).json()
        assert lst1["active_id"] == sid
        assert next(s for s in lst1["data"] if s["id"] == sid)["model"] == "opencode/glm-5.3"
        # метаданные ответа содержат клиентский id модели (с префиксом)
        assert lst1["data"][0]["history"][1]["meta"]["model"] == "opencode/glm-5.3"
        # системный промпт (первое сообщение) сохранён в сессии
        assert lst1["data"][0]["system_prompt"] == "первый вопрос"
    await app1.state.http_client.aclose()

    # ── «рестарт»: новое приложение на той же БД
    app2, transport2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        # список восстановлен
        lst_resp = (await c2.get("/api/sessions")).json()
        assert lst_resp["active_id"] == sid  # открытая вкладка восстановлена
        lst = lst_resp["data"]
        assert len(lst) == 1
        assert lst[0]["id"] == sid
        assert lst[0]["model"] == "opencode/glm-5.3"
        assert lst[0]["system_prompt"] == "первый вопрос"
        assert lst[0]["settings"]["temperature"] == 0.3
        assert lst[0]["settings"]["top_p"] == 0.5
        assert [m["role"] for m in lst[0]["history"]] == ["system", "assistant"]
        assert [m["content"] for m in lst[0]["history"]] == ["первый вопрос", "Ответ"]
        assert lst[0]["history"][1]["meta"]["reasoning_tokens"] == 100

        # follow-up: контекст полностью восстановлен (сид в системе, без дубля)
        r = await c2.post(f"/api/sessions/{sid}/messages", json={"content": "второй вопрос"})
        assert parse_sse(r.text)[-1]["type"] == "done"
        second_body = json.loads(transport2.requests[-1].content)
        assert [m["content"] for m in second_body["messages"]] == [
            "первый вопрос",
            "Ответ",
            "второй вопрос",
        ]
        assert [m["role"] for m in second_body["messages"]] == [
            "system",
            "assistant",
            "user",
        ]

        # после «рестарта» новый чат получает хеш-id (не сквозную нумерацию)
        sid2 = (await c2.post("/api/sessions", json={})).json()["id"]
        assert sid2 != sid
        assert not sid2.startswith("chat-")
    await app2.state.http_client.aclose()

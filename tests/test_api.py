"""Интеграционные тесты HTTP API (через ASGI-клиент с мок-апстримом)."""

import json

from server.services.storage import SessionStore
from tests.conftest import seed_test_profile, strip_profile


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
    assert types == ["session", "request_log", "done"]
    assert events[1]["record"]["kind"] == "main"
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
    seed_test_profile(application.state.registry)
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
        seed_test_profile(application.state.registry)
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
        assert strip_profile(first_body["messages"]) == [
            {"role": "system", "content": "первый вопрос"}
        ]
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
        second_msgs = strip_profile(second_body["messages"])
        assert [m["content"] for m in second_msgs] == [
            "первый вопрос",
            "Ответ",
            "второй вопрос",
        ]
        assert [m["role"] for m in second_msgs] == [
            "system",
            "assistant",
            "user",
        ]

        # после «рестарта» новый чат получает хеш-id (не сквозную нумерацию)
        sid2 = (await c2.post("/api/sessions", json={})).json()["id"]
        assert sid2 != sid
        assert not sid2.startswith("chat-")
    await app2.state.http_client.aclose()


# ── Саммаризация контекста ──────────────────────────────────────────────────

SUMMARY_SETTINGS = {"context_strategy": {"strategy": "summarize", "n": 3}}


async def test_context_summarization_flow(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    # 3 запроса (сид + 2 обмена) накапливаются без саммари: чанк не полон
    for i in range(3):
        r = await client.post(
            f"/api/sessions/{sid}/messages",
            json={"content": f"вопрос {i}", "settings": SUMMARY_SETTINGS},
        )
        assert [e["type"] for e in parse_sse(r.text)] == [
            "session", "request_log", "done",
        ]

    # 4-й запрос: первый чанк завершён -> скрытая саммаризация + основной запрос
    transport = app.state.mock_transport
    r = await client.post(
        f"/api/sessions/{sid}/messages",
        json={"content": "вопрос 3", "settings": SUMMARY_SETTINGS},
    )
    events = parse_sse(r.text)
    assert [e["type"] for e in events] == [
        "session", "request_log", "request_log", "done",
    ]
    assert events[1]["record"]["kind"] == "summary"
    assert events[2]["record"]["kind"] == "main"
    assert events[2]["record"]["reasoning_tokens"] == 100
    assert events[-1]["meta"]["summarized"] is True

    summary_body = json.loads(transport.requests[-2].content)
    main_body = json.loads(transport.requests[-1].content)
    # саммаризационный запрос: суммаризатор + первый чанк (ответ на сид + 2 обмена)
    assert summary_body["messages"][0]["role"] == "system"
    assert "суммаризатор" in summary_body["messages"][0]["content"].lower()
    assert [m["content"] for m in summary_body["messages"][1:]] == [
        "Ответ", "вопрос 1", "Ответ", "вопрос 2", "Ответ",
    ]
    # основной запрос: [системный сид] + [саммари-рамка] + вербатим-хвост
    main_msgs = strip_profile(main_body["messages"])
    assert main_msgs[0] == {"role": "system", "content": "вопрос 0"}
    assert "Саммари начала диалога" in main_msgs[1]["content"]
    assert "Саммари 1:" in main_msgs[1]["content"]
    assert [m["content"] for m in main_msgs[2:]] == ["вопрос 3"]

    # записи о всех запросах отдаются в состоянии сессии
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert [r["kind"] for r in state["requests"]] == [
        "main", "main", "main", "summary", "main",
    ]


async def test_context_strategy_bound_by_first_message(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    await client.post(
        f"/api/sessions/{sid}/messages",
        json={"content": "инструкция", "settings": SUMMARY_SETTINGS},
    )
    r = await client.post(
        f"/api/sessions/{sid}/messages",
        json={
            "content": "вопрос",
            "settings": {
                "context_strategy": {"strategy": "sliding", "n": 10, "k": 10}
            },
        },
    )
    assert parse_sse(r.text)[-1]["type"] == "done"
    svc = app.state.registry.get(sid)
    assert svc.settings.context_strategy.strategy == "summarize"
    assert svc.settings.context_strategy.n == 3


async def test_restart_restores_summary_state(tmp_path, monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    def build():
        application = create_app()
        transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "sess"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        seed_test_profile(application.state.registry)
        return application, transport

    app1, _ = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        sid = (await c1.post("/api/sessions", json={})).json()["id"]
        # 4 запроса -> первый чанк сжат (курсор = 1)
        for i in range(4):
            r = await c1.post(
                f"/api/sessions/{sid}/messages",
                json={"content": f"вопрос {i}", "settings": SUMMARY_SETTINGS},
            )
            assert parse_sse(r.text)[-1]["type"] == "done"
    await app1.state.http_client.aclose()

    app2, transport2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        state = (await c2.get(f"/api/sessions/{sid}")).json()
        assert [r["kind"] for r in state["requests"]] == [
            "main", "main", "main", "summary", "main",
        ]
        assert state["requests"][3]["reasoning_tokens"] == 100
        assert state["settings"]["context_strategy"] == {
            "strategy": "summarize", "n": 3, "k": 10,
        }

        # после рестарта готовый чанк не пересчитывается: курсор восстановлен
        r = await c2.post(
            f"/api/sessions/{sid}/messages",
            json={"content": "вопрос 4", "settings": SUMMARY_SETTINGS},
        )
        assert [e["type"] for e in parse_sse(r.text)] == [
            "session", "request_log", "done",
        ]
        assert len(transport2.requests) == 1  # только основной запрос, без саммари
    await app2.state.http_client.aclose()


# ── Стратегии: sliding, facts, branching ────────────────────────────────────

SLIDING_SETTINGS = {"context_strategy": {"strategy": "sliding", "n": 3}}


async def test_sliding_window_context(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    # сид + 4 завершённые реплики
    for i in range(5):
        r = await client.post(
            f"/api/sessions/{sid}/messages",
            json={"content": f"вопрос {i}", "settings": SLIDING_SETTINGS},
        )
        assert parse_sse(r.text)[-1]["type"] == "done"

    transport = app.state.mock_transport
    r = await client.post(
        f"/api/sessions/{sid}/messages",
        json={"content": "вопрос 5", "settings": SLIDING_SETTINGS},
    )
    assert parse_sse(r.text)[-1]["type"] == "done"
    body = json.loads(transport.requests[-1].content)
    # сид (обмен №1) выпал из окна; уходят последние 3 обмена + новое
    body_msgs = strip_profile(body["messages"])
    assert [m["content"] for m in body_msgs] == [
        "вопрос 2", "Ответ", "вопрос 3", "Ответ", "вопрос 4", "Ответ", "вопрос 5",
    ]
    assert all(m["role"] != "system" for m in body_msgs)


async def test_facts_strategy_flow_and_restart(tmp_path, monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import FakeStream, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    class FactsTransport(httpx.AsyncBaseTransport):
        """Основной ответ — «Ответ»; запрос экстрактора — JSON-массив."""

        def __init__(self):
            self.requests = []

        async def handle_async_request(self, request):
            self.requests.append(request)
            body = json.loads(request.content)
            is_facts = any(
                "экстрактор фактов" in m.get("content", "")
                for m in body.get("messages", [])
            )
            text = '{"Имя": "Аня", "Город": "Москва"}' if is_facts else "Ответ"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FakeStream(make_chat_chunks(content=text, usage=USAGE)),
                request=request,
            )

    def build():
        application = create_app()
        transport = FactsTransport()
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        seed_test_profile(application.state.registry)
        return application, transport

    settings = {"context_strategy": {"strategy": "facts", "n": 5, "k": 2}}
    app1, _ = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        sid = (await c1.post("/api/sessions", json={})).json()["id"]
        r = await c1.post(
            f"/api/sessions/{sid}/messages",
            json={"content": "меня зовут Аня", "settings": settings},
        )
        events = parse_sse(r.text)
        facts_events = [e for e in events if e["type"] == "facts"]
        assert facts_events and facts_events[0]["items"] == [
            ["Имя", "Аня"], ["Город", "Москва"],
        ]
        kinds = [e["record"]["kind"] for e in events if e["type"] == "request_log"]
        assert kinds == ["main", "facts"]
        state = (await c1.get(f"/api/sessions/{sid}")).json()
        assert state["facts"] == [["Имя", "Аня"], ["Город", "Москва"]]
    await app1.state.http_client.aclose()

    # рестарт сервера: факты восстановлены
    app2, _ = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        state = (await c2.get(f"/api/sessions/{sid}")).json()
        assert state["facts"] == [["Имя", "Аня"], ["Город", "Москва"]]
    await app2.state.http_client.aclose()


async def test_branch_session(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    branch_settings = {
        "temperature": 0.4,
        "context_strategy": {"strategy": "branching"},
    }
    for i in range(2):
        r = await client.post(
            f"/api/sessions/{sid}/messages",
            json={"content": f"вопрос {i}", "settings": branch_settings},
        )
        assert parse_sse(r.text)[-1]["type"] == "done"

    r = await client.post(f"/api/sessions/{sid}/branch", json={"title": "Ветка"})
    assert r.status_code == 201
    bid = r.json()["id"]
    assert bid != sid

    state = (await client.get(f"/api/sessions/{bid}")).json()
    assert state["title"] == "Ветка"
    assert state["parent_id"] == sid
    assert state["model"] == "opencode/glm-5.3"
    assert state["settings"]["context_strategy"]["strategy"] == "branching"
    assert state["settings"]["temperature"] == 0.4
    # история скопирована, журнал запросов — нет
    assert [m["content"] for m in state["history"]] == [
        "вопрос 0", "Ответ", "вопрос 1", "Ответ",
    ]
    assert state["requests"] == []

    # оригинал не изменён
    orig = (await client.get(f"/api/sessions/{sid}")).json()
    assert orig["title"] is None
    assert orig["parent_id"] is None

    # ветвиться можно и от ветки (цепочки)
    r2 = await client.post(f"/api/sessions/{bid}/branch", json={})
    assert r2.status_code == 201
    child = (await client.get(f"/api/sessions/{r2.json()['id']}")).json()
    assert child["parent_id"] == bid


async def test_branch_errors(app, client):
    assert (await client.post("/api/sessions/nope/branch", json={})).status_code == 404
    summary = (
        await client.post("/api/sessions", json={"kind": "summary"})
    ).json()["id"]
    assert (
        await client.post(f"/api/sessions/{summary}/branch", json={})
    ).status_code == 400


# ── Память чата ─────────────────────────────────────────────────────────────

MEMORY_STORES = [
    {
        "id": "m1",
        "name": "Профиль",
        "persistent": True,
        "items": [["Имя", "Иван"], ["Город", "Москва"]],
    },
    {"id": "m2", "name": "Черновик", "persistent": False, "items": [["Тема", "API"]]},
]


async def test_memory_sync_and_injection(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    # первое сообщение — сид
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "инструкция"})

    r = await client.put(f"/api/sessions/{sid}/memory", json={"stores": MEMORY_STORES})
    assert r.status_code == 200
    assert [s["name"] for s in r.json()["memory"]] == ["Профиль", "Черновик"]

    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert [s["name"] for s in state["memory"]] == ["Профиль", "Черновик"]

    # память из всех вкладок попадает в основной запрос
    transport = app.state.mock_transport
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "вопрос"})
    body = json.loads(transport.requests[-1].content)
    msgs = strip_profile(body["messages"])
    memory_msg = msgs[1]
    assert msgs[0]["role"] == "system"
    assert "[Память чата]" in memory_msg["content"]
    assert "Имя: Иван" in memory_msg["content"]
    assert "Тема: API" in memory_msg["content"]


async def test_memory_errors(app, client):
    assert (
        await client.put("/api/sessions/nope/memory", json={"stores": []})
    ).status_code == 404
    summary = (
        await client.post("/api/sessions", json={"kind": "summary"})
    ).json()["id"]
    assert (
        await client.put(f"/api/sessions/{summary}/memory", json={"stores": []})
    ).status_code == 400


async def test_memory_state_replaced_on_full_sync(app, client):
    """Полный синк заменяет состояние: устаревшие неперсистентные вкладки уходят."""
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    r = await client.put(f"/api/sessions/{sid}/memory", json={"stores": MEMORY_STORES})
    assert r.status_code == 200
    assert len(r.json()["memory"]) == 2

    # клиент пересинхронизирует состояние без неперсистентной вкладки
    persistent_only = [s for s in MEMORY_STORES if s["persistent"]]
    r = await client.put(
        f"/api/sessions/{sid}/memory", json={"stores": persistent_only}
    )
    assert [s["name"] for s in r.json()["memory"]] == ["Профиль"]
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert [s["name"] for s in state["memory"]] == ["Профиль"]


async def test_memory_persistent_only_after_restart(tmp_path, monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    def build():
        application = create_app()
        transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        return application

    app1 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        sid = (await c1.post("/api/sessions", json={})).json()["id"]
        r = await c1.put(f"/api/sessions/{sid}/memory", json={"stores": MEMORY_STORES})
        assert r.status_code == 200
    await app1.state.http_client.aclose()

    app2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        state = (await c2.get(f"/api/sessions/{sid}")).json()
        # восстановилась только персистентная вкладка
        assert [s["name"] for s in state["memory"]] == ["Профиль"]
        assert state["memory"][0]["persistent"] is True
    await app2.state.http_client.aclose()


async def test_branch_copies_memory(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    await client.put(f"/api/sessions/{sid}/memory", json={"stores": MEMORY_STORES})
    r = await client.post(f"/api/sessions/{sid}/branch", json={"title": "Ветка"})
    assert r.status_code == 201
    state = (await client.get(f"/api/sessions/{r.json()['id']}")).json()
    assert [s["name"] for s in state["memory"]] == ["Профиль", "Черновик"]


# ── Профили пользователя (глобальная сущность) ──────────────────────────────

PROFILE_PAYLOAD = {
    "profiles": [
        {
            "id": "prf-1",
            "name": "Основной",
            "fields": {
                "address": "Иван",
                "style": "кратко",
                "language": "русский",
                "format": "текст",
                "limit": "5 предложений",
            },
        }
    ],
    "active_id": "prf-1",
}


async def test_profiles_sync_and_list(client):
    # снимаем профиль тестовой фикстуры
    await client.put("/api/profiles", json={"profiles": [], "active_id": None})
    r = await client.get("/api/profiles")
    assert r.status_code == 200
    assert r.json() == {"profiles": [], "active_id": None}

    r = await client.put("/api/profiles", json=PROFILE_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert [p["name"] for p in data["profiles"]] == ["Основной"]
    assert data["active_id"] == "prf-1"
    assert data["profiles"][0]["fields"]["address"] == "Иван"

    assert (await client.get("/api/profiles")).json()["active_id"] == "prf-1"


async def test_incomplete_profile_is_dropped(client):
    """Профиль без заполненных полей не сохраняется; активный снимается."""
    r = await client.put("/api/profiles", json={
        "profiles": [
            {"id": "a", "name": "Пустой", "fields": {"address": "Иван"}},
            {"id": "b", "name": "Полный", "fields": {
                "address": "И", "style": "с", "language": "р",
                "format": "т", "limit": "о",
            }},
        ],
        "active_id": "a",
    })
    data = r.json()
    assert [p["id"] for p in data["profiles"]] == ["b"]
    assert data["active_id"] is None


async def test_message_blocked_without_profile(app, client):
    # снимаем профиль тестовой фикстуры
    await client.put("/api/profiles", json={"profiles": [], "active_id": None})
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "привет"})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "profile_required"
    assert body["error"] == "Необходимо создать и установить профиль."
    # сообщение в историю не попало
    state = (await client.get(f"/api/sessions/{sid}")).json()
    assert state["history"] == []


async def test_deleting_active_profile_blocks_again(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "первый"})
    assert parse_sse(r.text)[-1]["type"] == "done"
    # профиль снят — теперь любое сообщение блокируется
    await client.put("/api/profiles", json={"profiles": [], "active_id": None})
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "второй"})
    assert r.status_code == 400
    assert r.json()["code"] == "profile_required"


async def test_profile_injected_into_system_prompt(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "вопрос"})
    assert parse_sse(r.text)[-1]["type"] == "done"
    body = json.loads(app.state.mock_transport.requests[-1].content)
    profile_msg = body["messages"][0]
    assert profile_msg["role"] == "system"
    assert "следует профилю пользователя" in profile_msg["content"]
    assert "[Профиль пользователя]" in profile_msg["content"]
    assert "Как ко мне обращаться: Иван" in profile_msg["content"]
    assert "Стиль общения: кратко и по делу" in profile_msg["content"]
    # без необходимости имя пользователя не упоминается
    assert "Не обращайся к пользователю по имени" in profile_msg["content"]


async def test_summary_chat_not_blocked_without_profile(app, client):
    await client.put("/api/profiles", json={"profiles": [], "active_id": None})
    sid = (await client.post("/api/sessions", json={"kind": "summary"})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "итоги"})
    assert parse_sse(r.text)[-1]["type"] == "done"


async def test_profiles_persist_after_restart(tmp_path, monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    def build():
        application = create_app()
        transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        return application

    app1 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        assert (await c1.put("/api/profiles", json=PROFILE_PAYLOAD)).status_code == 200
    await app1.state.http_client.aclose()

    app2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        data = (await c2.get("/api/profiles")).json()
        assert [p["name"] for p in data["profiles"]] == ["Основной"]
        assert data["active_id"] == "prf-1"
        # после рестарта профиль подставляется (запрос не блокируется)
        sid = (await c2.post("/api/sessions", json={})).json()["id"]
        r = await c2.post(f"/api/sessions/{sid}/messages", json={"content": "вопрос"})
        assert parse_sse(r.text)[-1]["type"] == "done"
    await app2.state.http_client.aclose()


# ── Задачи (протокол этапов) ────────────────────────────────────────────────

async def test_task_lifecycle(app, client):
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]

    # шаг 1→2: описание задачи → план
    r = await client.post(f"/api/tasks/{tid}/advance", json={
        "action": "describe", "content": "Сделай отчёт",
        "model": "opencode/glm-5.3",
    })
    events = parse_sse(r.text)
    assert [e["type"] for e in events] == ["session", "request_log", "done", "stage"]
    assert events[-1]["stage"] == "plan_review"
    assert events[-2]["content"] == "Ответ"

    task = (await client.get("/api/tasks")).json()["data"][0]
    assert task["stage"] == "plan_review"
    assert task["plan"] == "Ответ"
    assert task["title"] == "Сделай отчёт"
    assert task["model"] == "opencode/glm-5.3"
    assert [m["role"] for m in task["history"]] == ["user", "assistant"]

    # шаг 2→3: подтверждение плана → выбор режима (без LLM)
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
    assert r.status_code == 200
    assert r.json()["stage"] == "mode_select"

    # режим «всё сразу» → пошаговое выполнение всего плана
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
    events = parse_sse(r.text)
    assert [e["type"] for e in events if e["type"] == "step_run"] == ["step_run"]
    assert events[-1]["stage"] == "review"
    task = (await client.get("/api/tasks")).json()["data"][0]
    assert task["stage"] == "review"
    assert task["steps"] == ["Ответ"]
    assert task["result"] == "Шаг 1: Ответ\nОтвет"

    # шаг 4: одобрение → done (без LLM)
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "approve"})
    assert r.status_code == 200
    assert r.json()["stage"] == "done"
    task = (await client.get("/api/tasks")).json()["data"][0]
    assert task["stage"] == "done"

    # задачи не попадают в список обычных чатов
    assert (await client.get("/api/sessions")).json()["data"] == []


async def test_task_revise_loop(app, client):
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})

    # доработка плана остаётся на plan_review
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "revise", "content": "Доработай план"})
    assert parse_sse(r.text)[-1]["stage"] == "plan_review"

    # подтверждение → mode_select → выполнение → review, затем доработка → plan_review
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "revise", "content": "Переделай"})
    assert parse_sse(r.text)[-1]["stage"] == "plan_review"
    task = (await client.get("/api/tasks")).json()["data"][0]
    assert task["stage"] == "plan_review"
    # замечания и исправленный план в истории
    assert task["history"][-2]["content"] == "Переделай"


async def test_task_stage_validation(app, client):
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]

    # действия не по этапу → 409 с текущим этапом
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
    assert r.status_code == 409
    assert r.json()["stage"] == "input"
    assert (await client.post(f"/api/tasks/{tid}/advance", json={"action": "approve"})).status_code == 409
    # неизвестное действие → 400
    assert (await client.post(f"/api/tasks/{tid}/advance", json={"action": "nope"})).status_code == 400
    # пустой текст → 400
    assert (await client.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "  "})).status_code == 400
    # на этапе плана режимы выполнения ещё недоступны
    tid2 = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    await client.post(f"/api/tasks/{tid2}/advance", json={"action": "describe", "content": "Задача"})
    assert (await client.post(f"/api/tasks/{tid2}/advance", json={"action": "run_all"})).status_code == 409
    assert (await client.post(f"/api/tasks/{tid2}/advance", json={"action": "start_steps"})).status_code == 409
    assert (await client.post(f"/api/tasks/{tid2}/advance", json={"action": "revise_step", "content": "x"})).status_code == 409
    # run_step не существует — неизвестное действие
    assert (await client.post(f"/api/tasks/{tid2}/advance", json={"action": "run_step"})).status_code == 400
    # не задача → 400; нет задачи → 404
    chat = (await client.post("/api/sessions", json={})).json()["id"]
    assert (await client.post(f"/api/tasks/{chat}/advance", json={"action": "describe", "content": "x"})).status_code == 400
    assert (await client.post("/api/tasks/nope/advance", json={"action": "describe", "content": "x"})).status_code == 404
    # обычный чат нельзя вести через messages-эндпоинт задачи
    assert (await client.post(f"/api/sessions/{tid}/messages", json={"content": "x"})).status_code == 400


async def test_task_requires_profile(app, client):
    await client.put("/api/profiles", json={"profiles": [], "active_id": None})
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    r = await client.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
    assert r.status_code == 400
    assert r.json()["code"] == "profile_required"


async def test_task_prompts_and_profile(app, client):
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    transport = app.state.mock_transport

    await client.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Сделай X"})
    planner = json.loads(transport.requests[-1].content)
    planner_system = next(
        m for m in planner["messages"] if "планировщик" in m.get("content", "")
    )
    # профиль активен и подмешан в системный промпт
    assert "[Профиль пользователя]" in planner_system["content"]
    assert any("Сделай X" in m.get("content", "") for m in planner["messages"])

    # подтверждение режима не обращается к LLM; исполнитель — при run_all
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
    executor = json.loads(transport.requests[-1].content)
    assert any("исполнитель" in m.get("content", "") for m in executor["messages"])
    # план из ответа планировщика уходит исполнителю
    assert any("Ответ" in m.get("content", "") for m in executor["messages"])


async def test_task_restart_resume(tmp_path, monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    def build():
        application = create_app()
        transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        seed_test_profile(application.state.registry)
        return application

    app1 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        tid = (await c1.post("/api/sessions", json={"kind": "task"})).json()["id"]
        r = await c1.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        assert parse_sse(r.text)[-1]["stage"] == "plan_review"
    await app1.state.http_client.aclose()

    # после рестарта задача продолжается с того же этапа
    app2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        data = (await c2.get("/api/tasks")).json()["data"]
        assert len(data) == 1
        assert data[0]["id"] == tid
        assert data[0]["stage"] == "plan_review"
        assert data[0]["plan"] == "Ответ"
        r = await c2.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        assert r.json()["stage"] == "mode_select"
        r = await c2.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
        assert parse_sse(r.text)[-1]["stage"] == "review"
    await app2.state.http_client.aclose()


async def test_task_delete(app, client):
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    assert (await client.delete(f"/api/sessions/{tid}")).status_code == 204
    assert (await client.get("/api/tasks")).json()["data"] == []


# ── Задачи: пошаговый режим ─────────────────────────────────────────────────

PLAN_TEXT = "1. Первый шаг.\n2. Второй шаг.\n3. Третий шаг."


def _task_flow_app(monkeypatch):
    """Приложение с транспортом: нумерованный план + результаты по шагам."""
    import re

    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import FakeStream, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))

    class TaskFlowTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.requests = []

        async def handle_async_request(self, request):
            self.requests.append(request)
            body = json.loads(request.content)
            joined = "\n".join(m.get("content", "") for m in body.get("messages", []))
            if "планировщик" in joined:
                text = PLAN_TEXT
            elif "Переделай шаг" in joined:
                match = re.search(r"Переделай шаг (\d+):", joined)
                text = f"Исправленный шаг {match.group(1) if match else '?'}"
            elif "ТОЛЬКО указанный шаг" in joined:
                match = re.search(r"Выполни шаг (\d+):", joined)
                text = f"Результат шага {match.group(1) if match else '?'}"
            else:
                text = "Результат всего плана"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FakeStream(make_chat_chunks(content=text, usage=USAGE)),
                request=request,
            )

    application = create_app()
    transport = TaskFlowTransport()
    application.state.http_client = httpx.AsyncClient(transport=transport)
    application.state.opencode_session_id = "s"
    application.state.registry = SessionRegistry(config.get_settings())
    seed_test_profile(application.state.registry)
    return application, transport


async def test_task_step_mode(monkeypatch):
    import httpx

    application, transport = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Сделай X"})
        assert (
            await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        ).json()["stage"] == "mode_select"

        # по шагам: шаг 1 выполняется сразу
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})
        ev = parse_sse(r.text)[-1]
        assert ev["type"] == "stage"
        assert ev["stage"] == "step_review"
        assert ev["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
        assert ev["step_results"] == ["Результат шага 1"]

        # принять шаг → сразу выполняется следующий
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert parse_sse(r.text)[-1]["step_results"] == ["Результат шага 1", "Результат шага 2"]
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert parse_sse(r.text)[-1]["step_results"] == [
            "Результат шага 1", "Результат шага 2", "Результат шага 3",
        ]

        # последний confirm_step — JSON: склейка и переход в review
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert r.headers["content-type"].startswith("application/json")
        progress = r.json()
        assert progress["stage"] == "review"
        assert "Шаг 1: Первый шаг." in progress["result"]
        assert "Результат шага 3" in progress["result"]

        assert (
            await c.post(f"/api/tasks/{tid}/advance", json={"action": "approve"})
        ).json()["stage"] == "done"

        task = (await c.get("/api/tasks")).json()["data"][0]
        assert task["stage"] == "done"
        assert [m["role"] for m in task["history"]] == [
            "user", "assistant", "user", "assistant",
            "user", "assistant", "user", "assistant",
        ]
        assert task["history"][2]["content"] == "Выполни шаг 1."
        assert task["history"][4]["content"] == "Выполни шаг 2."

        # исполнителю шага уходил только текущий шаг (+план и прошлые результаты)
        step_body = json.loads(transport.requests[2].content)
        joined = "\n".join(m.get("content", "") for m in step_body["messages"])
        assert "Выполни шаг 2: Второй шаг." in joined
        assert "Выполни шаг 3:" not in joined
        assert "Результат шага 1" in joined
    await application.state.http_client.aclose()


async def test_task_step_revise_fixes_step(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})
        assert parse_sse(r.text)[-1]["step_results"] == ["Результат шага 1"]

        # доработка шага: результат текущего шага заменяется, план и прогресс целы
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "revise_step", "content": "Поправь шаг"})
        ev = parse_sse(r.text)[-1]
        assert ev["stage"] == "step_review"
        assert ev["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
        assert ev["step_results"] == ["Исправленный шаг 1"]

        # продолжаем со второго шага с сохранённым исправленным результатом
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert parse_sse(r.text)[-1]["step_results"] == ["Исправленный шаг 1", "Результат шага 2"]

        # перепись всего плана с этапа шага запрещена; пустой revise_step → 400
        assert (await c.post(f"/api/tasks/{tid}/advance", json={"action": "revise", "content": "x"})).status_code == 409
        assert (await c.post(f"/api/tasks/{tid}/advance", json={"action": "revise_step", "content": "  "})).status_code == 400

        task = (await c.get("/api/tasks")).json()["data"][0]
        assert task["stage"] == "step_review"
        assert task["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
        assert task["step_results"] == ["Исправленный шаг 1", "Результат шага 2"]
        # в историю попали замечания и исправленный результат
        contents = [m["content"] for m in task["history"]]
        assert "Поправь шаг" in contents
        assert "Исправленный шаг 1" in contents
    await application.state.http_client.aclose()


async def test_task_step_restart_resume(tmp_path, monkeypatch):
    import re

    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import FakeStream, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    class TaskFlowTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.requests = []

        async def handle_async_request(self, request):
            self.requests.append(request)
            body = json.loads(request.content)
            joined = "\n".join(m.get("content", "") for m in body.get("messages", []))
            if "планировщик" in joined:
                text = PLAN_TEXT
            elif "Переделай шаг" in joined:
                match = re.search(r"Переделай шаг (\d+):", joined)
                text = f"Исправленный шаг {match.group(1) if match else '?'}"
            elif "ТОЛЬКО указанный шаг" in joined:
                match = re.search(r"Выполни шаг (\d+):", joined)
                text = f"Результат шага {match.group(1) if match else '?'}"
            else:
                text = "Результат всего плана"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FakeStream(make_chat_chunks(content=text, usage=USAGE)),
                request=request,
            )

    def build():
        application = create_app()
        application.state.http_client = httpx.AsyncClient(transport=TaskFlowTransport())
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        seed_test_profile(application.state.registry)
        return application

    app1 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        tid = (await c1.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c1.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c1.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        r = await c1.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})
        assert parse_sse(r.text)[-1]["stage"] == "step_review"
    await app1.state.http_client.aclose()

    # после рестарта шаговый прогресс восстановлен, продолжаем со второго шага
    app2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        data = (await c2.get("/api/tasks")).json()["data"]
        assert len(data) == 1
        assert data[0]["stage"] == "step_review"
        assert data[0]["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
        assert data[0]["step_results"] == ["Результат шага 1"]
        r = await c2.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert parse_sse(r.text)[-1]["step_results"] == ["Результат шага 1", "Результат шага 2"]
    await app2.state.http_client.aclose()


# ── Правила (глобальные ограничения) ────────────────────────────────────────

RULES_PAYLOAD = {
    "rules": [
        {
            "id": "rl-1",
            "name": "Безопасность",
            "items": [["Пароли", "никогда не выводить"], ["Код", "без секретов"]],
        }
    ]
}


async def test_rules_sync_and_list(client):
    assert (await client.get("/api/rules")).json() == {"rules": []}

    r = await client.put("/api/rules", json=RULES_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert [x["name"] for x in data["rules"]] == ["Безопасность"]
    assert data["rules"][0]["id"] == "rl-1"
    assert data["rules"][0]["items"] == [
        ["Пароли", "никогда не выводить"],
        ["Код", "без секретов"],
    ]
    assert (await client.get("/api/rules")).json()["rules"][0]["name"] == "Безопасность"


async def test_rules_sanitize(client):
    r = await client.put("/api/rules", json={"rules": [
        {"id": "", "name": "", "items": []},
        {"id": "a", "name": "   ", "items": [["k", "v"]]},
        {"id": "b", "name": "Правила", "items": [["", "v"], ["k", ""], ["ok", "v"], "bad"]},
    ]})
    data = r.json()
    assert len(data["rules"]) == 1
    assert data["rules"][0]["name"] == "Правила"
    assert data["rules"][0]["items"] == [["ok", "v"]]
    # отсутствие поля rules очищает правила
    assert (await client.put("/api/rules", json={})).json() == {"rules": []}


async def test_rules_injected_into_chat(app, client):
    await client.put("/api/rules", json=RULES_PAYLOAD)
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    await client.put(f"/api/sessions/{sid}/memory", json={"stores": MEMORY_STORES})

    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "привет"})
    assert parse_sse(r.text)[-1]["type"] == "done"
    body = json.loads(app.state.mock_transport.requests[-1].content)
    # правила — первым элементом ведущего system-сообщения, профиль — после
    system0 = body["messages"][0]
    assert system0["role"] == "system"
    assert "[Правила]" in system0["content"]
    assert "## Безопасность" in system0["content"]
    assert "Пароли: никогда не выводить" in system0["content"]
    assert "Никогда не нарушай" in system0["content"]
    assert system0["content"].index("[Правила]") < system0["content"].index("[Профиль пользователя]")
    # память — отдельным user-сообщением; правила отдельным не дублируются
    assert any(
        m["role"] == "user" and "[Память чата]" in m["content"] for m in body["messages"]
    )
    assert not any(
        m["role"] == "user" and m["content"].startswith("[Правила]")
        for m in body["messages"]
    )


async def test_rules_priority_texts(app, client):
    from server.services.chat_service import PROFILE_SYSTEM_PREFIX, RULES_INSTRUCTION

    assert "приоритет над профилем" in RULES_INSTRUCTION
    assert "приоритет у правил" in PROFILE_SYSTEM_PREFIX
    assert "по имени" in PROFILE_SYSTEM_PREFIX


async def test_rules_in_summary_and_optimize(app, client):
    await client.put("/api/rules", json=RULES_PAYLOAD)

    sid = (await client.post("/api/sessions", json={"kind": "summary"})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "итоги"})
    body = json.loads(app.state.mock_transport.requests[-1].content)
    systems = [m["content"] for m in body["messages"] if m["role"] == "system"]
    assert any("[Правила]" in t for t in systems)

    eid = (await client.post("/api/sessions", json={"kind": "ephemeral"})).json()["id"]
    await client.post(f"/api/sessions/{eid}/messages", json={"content": "запрос"})
    body = json.loads(app.state.mock_transport.requests[-1].content)
    systems = [m["content"] for m in body["messages"] if m["role"] == "system"]
    assert any("[Правила]" in t for t in systems)


async def test_rules_in_tasks(monkeypatch):
    import httpx

    application, transport = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        await c.put("/api/rules", json=RULES_PAYLOAD)
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]

        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        planner = json.loads(transport.requests[-1].content)
        planner_systems = [
            m["content"] for m in planner["messages"] if m["role"] == "system"
        ]
        assert any("планировщик" in t and "[Правила]" in t for t in planner_systems)

        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
        executor = json.loads(transport.requests[-1].content)
        executor_systems = [
            m["content"] for m in executor["messages"] if m["role"] == "system"
        ]
        assert any("исполнитель" in t and "[Правила]" in t for t in executor_systems)
    await application.state.http_client.aclose()


async def test_rules_absent_in_hidden_requests(app, client):
    await client.put("/api/rules", json=RULES_PAYLOAD)
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    transport = app.state.mock_transport
    r = await client.post(f"/api/sessions/{sid}/messages", json={
        "content": "меня зовут Аня",
        "settings": {"context_strategy": {"strategy": "facts", "n": 5, "k": 2}},
    })
    assert any(e["type"] == "done" for e in parse_sse(r.text))
    # основной запрос — с правилами; скрытый экстрактор фактов — без
    main_body = json.loads(transport.requests[-2].content)
    facts_body = json.loads(transport.requests[-1].content)
    assert any("[Правила]" in m["content"] for m in main_body["messages"])
    assert not any("[Правила]" in m["content"] for m in facts_body["messages"])


async def test_rules_empty_no_block(app, client):
    sid = (await client.post("/api/sessions", json={})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "привет"})
    assert parse_sse(r.text)[-1]["type"] == "done"
    body = json.loads(app.state.mock_transport.requests[-1].content)
    # отдельного блока правил нет (упоминание «[Правила]» в профиле — это
    # примечание о приоритете, а не сам блок с ограничениями)
    assert not any(m["content"].startswith("[Правила]") for m in body["messages"])
    assert not any("Никогда не нарушай" in m["content"] for m in body["messages"])


async def test_rules_persist_after_restart(tmp_path, monkeypatch):
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import MockTransport, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    def build():
        application = create_app()
        transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
        application.state.http_client = httpx.AsyncClient(transport=transport)
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        seed_test_profile(application.state.registry)
        return application, transport

    app1, _ = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        assert (await c1.put("/api/rules", json=RULES_PAYLOAD)).status_code == 200
    await app1.state.http_client.aclose()

    app2, transport2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        data = (await c2.get("/api/rules")).json()["rules"]
        assert [x["name"] for x in data] == ["Безопасность"]
        sid = (await c2.post("/api/sessions", json={})).json()["id"]
        r = await c2.post(f"/api/sessions/{sid}/messages", json={"content": "привет"})
        assert parse_sse(r.text)[-1]["type"] == "done"
        body = json.loads(transport2.requests[-1].content)
        assert any("[Правила]" in m["content"] for m in body["messages"])
    await app2.state.http_client.aclose()


# ── Журнал промптов (вкладка «Логи») ────────────────────────────────────────

async def test_prompt_logs_chat(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "привет"})

    logs = (await client.get("/api/logs")).json()["logs"]
    assert len(logs) == 1
    entry = logs[0]
    assert entry["kind"] == "main"
    assert entry["source"] == "Чат"
    assert entry["response"] == "Ответ"
    assert entry["model"] == "opencode/glm-5.3"
    assert entry["prompt_tokens"] == 202
    assert entry["completion_tokens"] == 233
    assert entry["messages"][0]["role"] == "system"
    assert any("привет" in m["content"] for m in entry["messages"])

    assert (await client.delete("/api/logs")).status_code == 204
    assert (await client.get("/api/logs")).json()["logs"] == []


async def test_prompt_logs_include_hidden(app, client):
    sid = (await client.post(
        "/api/sessions", json={"model": "opencode/glm-5.3"}
    )).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={
        "content": "меня зовут Аня",
        "settings": {"context_strategy": {"strategy": "facts", "n": 5, "k": 2}},
    })
    assert any(e["type"] == "done" for e in parse_sse(r.text))

    logs = (await client.get("/api/logs")).json()["logs"]
    kinds = [e["kind"] for e in logs]
    assert kinds == ["facts", "main"]  # сначала новые
    facts = next(e for e in logs if e["kind"] == "facts")
    assert any("экстрактор" in m["content"].lower() for m in facts["messages"])
    assert facts["response"] == "Ответ"


async def test_prompt_logs_tasks(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        logs = (await c.get("/api/logs")).json()["logs"]
        assert logs and logs[0]["kind"] == "describe"
        assert logs[0]["source"] == "Задача"

        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
        kinds = [e["kind"] for e in (await c.get("/api/logs")).json()["logs"]]
        # план из 3 шагов → три запроса исполнителя шагов + планировщик
        assert kinds == ["run_all", "run_all", "run_all", "describe"]
    await application.state.http_client.aclose()


# ── Задачи: прямое редактирование шагов плана ──────────────────────────────

async def test_task_steps_parsed_at_plan_review(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        task = (await c.get("/api/tasks")).json()["data"][0]
        assert task["stage"] == "plan_review"
        assert task["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
    await application.state.http_client.aclose()


async def test_task_edit_future_step(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        r = await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 1, "content": "Новый второй шаг",
        })
        assert r.headers["content-type"].startswith("application/json")
        task = (await c.get("/api/tasks")).json()["data"][0]
        assert task["stage"] == "plan_review"
        assert task["steps"] == ["Первый шаг.", "Новый второй шаг", "Третий шаг."]
        assert task["plan"] == "1. Первый шаг.\n2. Новый второй шаг\n3. Третий шаг."

        # правка сохраняется при старте пошагового режима (не перепарсивается)
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})
        assert parse_sse(r.text)[-1]["steps"] == [
            "Первый шаг.", "Новый второй шаг", "Третий шаг.",
        ]
    await application.state.http_client.aclose()


async def test_task_edit_executed_step_rolls_back(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})   # шаг 1
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})  # шаг 2
        assert parse_sse(r.text)[-1]["step_results"] == ["Результат шага 1", "Результат шага 2"]

        # правим выполненный шаг 1 → откат (результат шага 2 отброшен) и авто-запуск шага 1
        r = await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0, "content": "Первый шаг (правка)",
        })
        ev = parse_sse(r.text)[-1]
        assert ev["stage"] == "step_review"
        assert ev["steps"][0] == "Первый шаг (правка)"
        assert ev["step_results"] == ["Результат шага 1"]

        # продолжаем — снова выполняется шаг 2
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert parse_sse(r.text)[-1]["step_results"] == ["Результат шага 1", "Результат шага 2"]
    await application.state.http_client.aclose()


async def test_task_edit_step_validation_and_noop(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        # на этапе input плана ещё нет — правки недоступны
        assert (await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0, "content": "x",
        })).status_code == 409

        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        # индекс вне диапазона / пустой текст
        assert (await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 9, "content": "x",
        })).status_code == 400
        assert (await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0, "content": "  ",
        })).status_code == 400

        # no-op: тот же текст ничего не меняет
        before = (await c.get("/api/tasks")).json()["data"][0]["steps"]
        r = await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0, "content": before[0],
        })
        assert r.headers["content-type"].startswith("application/json")
        assert (await c.get("/api/tasks")).json()["data"][0]["steps"] == before

        # no-op: изменились только пробелы — смысл тот же (без отката)
        r = await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0,
            "content": before[0].replace(" ", "  "),
        })
        assert r.headers["content-type"].startswith("application/json")
        task = (await c.get("/api/tasks")).json()["data"][0]
        assert task["steps"] == before
        assert task["step_results"] == []
    await application.state.http_client.aclose()


async def test_task_edit_step_failure_restores_progress(monkeypatch):
    """Сбой перезапуска правленого шага возвращает откаченный прогресс."""
    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import FakeStream, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))

    class FailingStepTransport(httpx.AsyncBaseTransport):
        """План и обычные шаги — ок; повторный запуск правленого шага — 500."""

        def __init__(self):
            self.requests = []

        async def handle_async_request(self, request):
            self.requests.append(request)
            body = json.loads(request.content)
            joined = "\n".join(m.get("content", "") for m in body.get("messages", []))
            if "Первый шаг (правка)" in joined:
                return httpx.Response(500, text="boom", request=request)
            if "планировщик" in joined:
                text = PLAN_TEXT
            else:
                text = "Результат шага"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FakeStream(make_chat_chunks(content=text, usage=USAGE)),
                request=request,
            )

    application = create_app()
    application.state.http_client = httpx.AsyncClient(transport=FailingStepTransport())
    application.state.opencode_session_id = "s"
    application.state.registry = SessionRegistry(config.get_settings())
    seed_test_profile(application.state.registry)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})

        before = (await c.get("/api/tasks")).json()["data"][0]
        assert len(before["step_results"]) == 2

        # правка выполненного шага: сервер откатывает результаты, но
        # запрос исполнителя падает → прогресс должен восстановиться
        r = await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0, "content": "Первый шаг (правка)",
        })
        assert any(e["type"] == "error" for e in parse_sse(r.text))

        after = (await c.get("/api/tasks")).json()["data"][0]
        assert after["step_results"] == before["step_results"]
        assert after["stage"] == before["stage"]
    await application.state.http_client.aclose()


async def test_task_edit_step_at_review_rolls_back(monkeypatch):
    import httpx

    application, _ = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})  # 1
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})  # 2
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})  # 3
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm_step"})
        assert r.json()["stage"] == "review"  # все шаги выполнены

        # правка первого шага с этапа review → откат и возврат к выполнению
        r = await c.post(f"/api/tasks/{tid}/advance", json={
            "action": "edit_step", "index": 0, "content": "Первый шаг (правка)",
        })
        ev = parse_sse(r.text)[-1]
        assert ev["stage"] == "step_review"
        assert ev["step_results"] == ["Результат шага 1"]  # результаты 2 и 3 отброшены
        assert ev["result"] is None
    await application.state.http_client.aclose()


async def test_task_run_all_stepwise(monkeypatch):
    import httpx

    application, transport = _task_flow_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    ) as c:
        tid = (await c.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        r = await c.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
        events = parse_sse(r.text)

        # по одному запросу исполнителя шага на каждый шаг плана
        step_runs = [e for e in events if e["type"] == "step_run"]
        assert [(e["index"], e["total"]) for e in step_runs] == [(0, 3), (1, 3), (2, 3)]
        # после каждого шага — step_review, в конце — review
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        assert stages == ["step_review", "step_review", "step_review", "review"]

        task = (await c.get("/api/tasks")).json()["data"][0]
        assert task["stage"] == "review"
        assert task["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
        assert task["step_results"] == [
            "Результат шага 1", "Результат шага 2", "Результат шага 3",
        ]
        assert "Шаг 1: Первый шаг." in task["result"]
        assert "Результат шага 3" in task["result"]
        # история: задача + план + три ответа ассистента (без user-маркеров —
        # «как будто LLM сама берёт следующий шаг»)
        assert [m["role"] for m in task["history"]] == [
            "user", "assistant", "assistant", "assistant", "assistant",
        ]
        assert not any(
            "Выполни шаг" in m["content"] for m in task["history"]
        )
        assert task["history"][2]["content"] == "Результат шага 1"
        assert task["history"][4]["content"] == "Результат шага 3"

        # каждому исполнителю уходил только его шаг
        step_body = json.loads(transport.requests[-1].content)
        joined = "\n".join(m.get("content", "") for m in step_body["messages"])
        assert "Выполни шаг 3: Третий шаг." in joined
    await application.state.http_client.aclose()


async def test_task_run_all_restart_resume(tmp_path, monkeypatch):
    import re

    import httpx

    from server import config
    from server.main import create_app
    from server.services.registry import SessionRegistry
    from tests.conftest import FakeStream, USAGE, make_chat_chunks

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test", opencode_api_key="zen-test",
    ))
    db = str(tmp_path / "chats.db")

    class TaskFlowTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.requests = []

        async def handle_async_request(self, request):
            self.requests.append(request)
            body = json.loads(request.content)
            joined = "\n".join(m.get("content", "") for m in body.get("messages", []))
            if "планировщик" in joined:
                text = PLAN_TEXT
            elif "ТОЛЬКО указанный шаг" in joined:
                match = re.search(r"Выполни шаг (\d+):", joined)
                text = f"Результат шага {match.group(1) if match else '?'}"
            else:
                text = "Результат всего плана"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FakeStream(make_chat_chunks(content=text, usage=USAGE)),
                request=request,
            )

    def build():
        application = create_app()
        application.state.http_client = httpx.AsyncClient(transport=TaskFlowTransport())
        application.state.opencode_session_id = "s"
        application.state.registry = SessionRegistry(
            config.get_settings(), store=SessionStore(db)
        )
        application.state.registry.restore()
        seed_test_profile(application.state.registry)
        return application

    app1 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app1), base_url="http://t") as c1:
        tid = (await c1.post("/api/sessions", json={"kind": "task"})).json()["id"]
        await c1.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
        await c1.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
        r = await c1.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
        assert parse_sse(r.text)[-1]["stage"] == "review"
    await app1.state.http_client.aclose()

    # после рестарта восстановлены одиночные ответы шагов (без user-маркеров)
    app2 = build()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://t") as c2:
        task = (await c2.get("/api/tasks")).json()["data"][0]
        assert task["stage"] == "review"
        assert task["step_results"] == [
            "Результат шага 1", "Результат шага 2", "Результат шага 3",
        ]
        assert [m["role"] for m in task["history"]] == [
            "user", "assistant", "assistant", "assistant", "assistant",
        ]
        assert "Результат шага 3" in task["result"]
    await app2.state.http_client.aclose()


# ── Задачи: переходы по шагам через замечания ──────────────────────────────

async def _task_at(monkeypatch, step_actions):
    """Приложение + задача, доведённая до нужного шага пошагового режима."""
    import httpx

    application, transport = _task_flow_app(monkeypatch)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    )
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "start_steps"})
    for action in step_actions:
        await client.post(f"/api/tasks/{tid}/advance", json={"action": action})
    return application, transport, client, tid


async def test_task_remark_returns_to_previous_step(monkeypatch):
    # доводим до шага 2 (results = [r1, r2])
    application, transport, c, tid = await _task_at(monkeypatch, ["confirm_step"])
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise_step", "content": "вернись к шагу 1 и переделай его",
    })
    ev = parse_sse(r.text)[-1]
    assert ev["stage"] == "step_review"
    assert ev["step_results"] == ["Результат шага 1"]  # результаты шага 2 отброшены
    # исполнителю ушёл именно шаг 1 с указанием пользователя
    body = json.loads(transport.requests[-1].content)
    joined = "\n".join(m.get("content", "") for m in body["messages"])
    assert "Выполни шаг 1:" in joined
    assert "Указание пользователя: вернись к шагу 1" in joined
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_remark_moves_to_next_step(monkeypatch):
    # шаг 1 выполнен, results = [r1], current = 1
    application, transport, c, tid = await _task_at(monkeypatch, [])
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise_step", "content": "перейди к следующему шагу",
    })
    ev = parse_sse(r.text)[-1]
    assert ev["stage"] == "step_review"
    assert ev["step_results"] == ["Результат шага 1", "Результат шага 2"]
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_remark_skip_forward_refused(monkeypatch):
    application, _, c, tid = await _task_at(monkeypatch, [])
    # пропуск нескольких шагов вперёд — отказ, прогресс не тронут
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise_step", "content": "перейди к шагу 3",
    })
    assert r.status_code == 400
    assert r.json()["code"] == "step_skip_forbidden"
    state = (await c.get(f"/api/tasks")).json()["data"][0]
    assert state["step_results"] == ["Результат шага 1"]
    assert state["stage"] == "step_review"
    # шага вне плана нет
    assert (await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise_step", "content": "вернись к шагу 9",
    })).status_code == 400
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_remark_next_on_last_step_refused(monkeypatch):
    # доводим до последнего шага (results = 3)
    application, _, c, tid = await _task_at(monkeypatch, ["confirm_step", "confirm_step"])
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise_step", "content": "перейди к следующему шагу",
    })
    assert r.status_code == 400
    assert r.json()["code"] == "no_next_step"
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_remark_without_directive_reworks_current(monkeypatch):
    # упоминание шага без глагола перехода — обычная доработка текущего шага
    application, _, c, tid = await _task_at(monkeypatch, ["confirm_step"])
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise_step", "content": "в шаге 1 мы используем X, учти это",
    })
    ev = parse_sse(r.text)[-1]
    assert ev["step_results"] == ["Результат шага 1", "Исправленный шаг 2"]  # заменён, без отката
    await application.state.http_client.aclose()
    await c.aclose()


# ── Задачи: точечная доработка шага на валидации ───────────────────────────

async def _task_at_review(monkeypatch):
    """Приложение + задача, доведённая до валидации (все 3 шага выполнены)."""
    import httpx

    application, transport = _task_flow_app(monkeypatch)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://t"
    )
    tid = (await client.post("/api/sessions", json={"kind": "task"})).json()["id"]
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "describe", "content": "Задача"})
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "confirm"})
    await client.post(f"/api/tasks/{tid}/advance", json={"action": "run_all"})
    return application, transport, client, tid


async def test_task_review_remark_reworks_single_step(monkeypatch):
    application, transport, c, tid = await _task_at_review(monkeypatch)
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise", "content": "доработай шаг 2 и учти X",
    })
    ev = parse_sse(r.text)[-1]
    # переработан только шаг 2, остальные результаты сохранены, возврат на review
    assert ev["stage"] == "review"
    assert ev["step_results"] == [
        "Результат шага 1", "Исправленный шаг 2", "Результат шага 3",
    ]
    assert "Исправленный шаг 2" in ev["result"]
    state = (await c.get("/api/tasks")).json()["data"][0]
    assert state["steps"] == ["Первый шаг.", "Второй шаг.", "Третий шаг."]
    # в апстрим ушла доработка именно шага 2 с его прежним результатом
    body = json.loads(transport.requests[-1].content)
    joined = "\n".join(m.get("content", "") for m in body["messages"])
    assert "Переделай шаг 2" in joined
    assert "Прежний результат шага 2" in joined
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_review_remark_last_step(monkeypatch):
    application, _, c, tid = await _task_at_review(monkeypatch)
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise", "content": "доработай последний шаг",
    })
    ev = parse_sse(r.text)[-1]
    assert ev["stage"] == "review"
    assert ev["step_results"] == [
        "Результат шага 1", "Результат шага 2", "Исправленный шаг 3",
    ]
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_review_remark_without_step_replans(monkeypatch):
    application, _, c, tid = await _task_at_review(monkeypatch)
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise", "content": "сделай результат короче",
    })
    ev = parse_sse(r.text)[-1]
    # без ссылки на шаг — прежнее поведение: полный пересбор плана
    assert ev["stage"] == "plan_review"
    assert ev["step_results"] == []
    await application.state.http_client.aclose()
    await c.aclose()


async def test_task_review_remark_step_out_of_range(monkeypatch):
    application, _, c, tid = await _task_at_review(monkeypatch)
    r = await c.post(f"/api/tasks/{tid}/advance", json={
        "action": "revise", "content": "доработай шаг 9",
    })
    assert r.status_code == 400
    state = (await c.get("/api/tasks")).json()["data"][0]
    assert state["stage"] == "review"
    assert len(state["step_results"]) == 3
    await application.state.http_client.aclose()
    await c.aclose()

"""Интеграционные тесты HTTP API (через ASGI-клиент с мок-апстримом)."""

import json


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
    history = r.json()["history"]
    assert [m["role"] for m in history] == ["user", "assistant"]
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

"""Тесты ChatService: история, метаданные, busy, откат, TOON-контекст."""

import pytest

from server.pricing import message_cost
from server.services.chat_service import ChatService, SessionKind
from server.services.generation import GenerationSettings


class FakeRunner:
    """Имитация StreamedCompletion с предзаданными событиями."""

    def __init__(self, events):
        self.events = events

    async def run(self, spec, messages, settings):
        for e in self.events:
            yield e
        self.seen_messages = messages


def make_events(content="Ответ"):
    from server.providers.base import ChatEvent, Usage, UsageDetails

    return [
        ChatEvent(kind="reasoning_start"),
        ChatEvent(kind="reasoning_end", content="рассуждение"),
        ChatEvent(kind="delta", content=content),
        ChatEvent(kind="usage", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
        ChatEvent(kind="done", finish_reason="stop",
                  usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                              details=UsageDetails(reasoning_tokens=2))),
    ]


def test_create_with_defaults():
    svc = ChatService("c1")
    assert svc.kind == SessionKind.CHAT
    assert svc.history == []
    assert svc.busy is False


def test_add_user_message():
    svc = ChatService("c1")
    svc.add_user_message("привет")
    assert svc.history[-1].role == "user"
    assert svc.history[-1].content == "привет"


def test_stream_completion_records_meta_and_events():
    svc = ChatService("c1", model="deepseek-chat")
    svc.add_user_message("вопрос")
    runner = FakeRunner(make_events())
    spec = type("Spec", (), {"model": "deepseek-chat"})()

    events = []
    async def consume():
        async for e in svc.stream_completion(runner, spec):
            events.append(e)
    asyncio_run(consume())

    kinds = [e["type"] for e in events]
    assert kinds == ["reasoning_start", "reasoning_end", "done"]
    assert not svc.busy
    # весь текст ответа приходит целиком в финальном done-событии
    assert events[-1]["content"] == "Ответ"
    assert "delta" not in kinds

    # история: user + assistant с метаданными
    assert svc.history[0].role == "user"
    assistant = svc.history[1]
    assert assistant.role == "assistant"
    assert assistant.content == "Ответ"
    assert assistant.meta.prompt_tokens == 10
    assert assistant.meta.completion_tokens == 5
    assert assistant.meta.reasoning_tokens == 2
    assert assistant.meta.cost_usd == message_cost("deepseek-chat", 10, 5)
    assert assistant.meta.finish_reason == "stop"
    assert assistant.meta.elapsed_s >= 0


def test_stream_completion_sends_system_prompt_and_history():
    svc = ChatService("c1", model="m", system_prompt="sys", settings=GenerationSettings())
    svc.add_user_message("q")
    runner = FakeRunner([])
    spec = type("Spec", (), {"model": "m"})()
    asyncio_run(consume_stream(svc, runner, spec))
    assert runner.seen_messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]


def test_stream_completion_prepends_extra_system():
    svc = ChatService("c1", model="m", system_prompt="sys", settings=GenerationSettings())
    svc.add_user_message("q")
    runner = FakeRunner([])
    spec = type("Spec", (), {"model": "m"})()
    asyncio_run(consume_stream(svc, runner, spec, extra_system="JSON"))
    assert runner.seen_messages[0] == {"role": "system", "content": "JSON"}


def test_rollback_user_message():
    svc = ChatService("c1")
    svc.add_user_message("q")
    svc.rollback_user_message()
    assert svc.history == []


def test_to_toon_context_roundtrip():
    from server import toon_codec
    from server.services.chat_service import MessageMeta

    svc = ChatService("c1", model="m")
    svc.add_user_message("вопрос")
    svc.append_assistant_message(
        "ответ",
        MessageMeta(
            model="m",
            elapsed_s=1.2,
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=2,
            cost_usd=0.0001,
            finish_reason="stop",
        ),
    )
    ctx = svc.to_toon_context(title="Чат 1")
    decoded = toon_codec.decode(ctx)
    assert decoded["title"] == "Чат 1"
    assert len(decoded["messages"]) == 2
    assert decoded["messages"][1]["meta"]["completion_tokens"] == 5


def asyncio_run(coro):
    import asyncio

    asyncio.run(coro)


async def consume_stream(svc, runner, spec, extra_system=None):
    async for _ in svc.stream_completion(runner, spec, extra_system=extra_system):
        pass

"""Тесты ChatService: история, метаданные, busy, откат, TOON-контекст."""

import pytest

from server.pricing import message_cost
from server.services.chat_service import (
    FACTS_MESSAGE_PREFIX,
    FACTS_SYSTEM_PROMPT,
    MEMORY_INSTRUCTION,
    MEMORY_MESSAGE_PREFIX,
    SUMMARY_MESSAGE_PREFIX,
    SUMMARIZER_SYSTEM_PROMPT,
    ChatService,
    MemoryStore,
    MessageMeta,
    SessionKind,
)
from server.services.generation import ContextStrategySettings, GenerationSettings


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
    spec = type("Spec", (), {"model": "deepseek-chat", "model_label": "deepseek-chat"})()

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
    # запись о запросе несёт рассуждения (часть выхода)
    assert svc.requests[-1].kind == "main"
    assert svc.requests[-1].reasoning_tokens == 2


def test_stream_completion_sends_system_prompt_and_history():
    svc = ChatService("c1", model="m", system_prompt="sys", settings=GenerationSettings())
    svc.add_user_message("q")
    runner = FakeRunner([])
    spec = type("Spec", (), {"model": "m", "model_label": "opencode/m"})()
    asyncio_run(consume_stream(svc, runner, spec))
    assert runner.seen_messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]


def test_stream_completion_prepends_extra_system():
    svc = ChatService("c1", model="m", system_prompt="sys", settings=GenerationSettings())
    svc.add_user_message("q")
    runner = FakeRunner([])
    spec = type("Spec", (), {"model": "m", "model_label": "opencode/m"})()
    asyncio_run(consume_stream(svc, runner, spec, extra_system="JSON"))
    assert runner.seen_messages[0] == {"role": "system", "content": "JSON"}


def test_stream_completion_uses_messages_override():
    """Переданные сообщения (протокол «Задачи») заменяют сборку по истории."""
    svc = ChatService("c1", model="m", system_prompt="sys", settings=GenerationSettings())
    svc.add_user_message("q")
    runner = FakeRunner([])
    spec = type("Spec", (), {"model": "m", "model_label": "opencode/m"})()
    custom = [
        {"role": "system", "content": "PROTOCOL"},
        {"role": "user", "content": "PLAN"},
    ]
    asyncio_run(consume_stream(svc, runner, spec, messages=custom))
    assert runner.seen_messages == custom


def test_task_session_starts_at_input_stage():
    task = ChatService("t1", kind=SessionKind.TASK)
    assert task.task_stage == "input"
    assert task.task_plan is None
    assert task.task_result is None
    # у обычного чата этапов нет
    assert ChatService("c1").task_stage is None


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


def test_seed_system_message():
    svc = ChatService("c1")
    svc.seed_system_message("Ты — переводчик")
    assert svc.system_prompt == "Ты — переводчик"
    assert svc.history[0].role == "system"
    # первый запрос уходит как [system] без дубля
    assert svc.to_openai_messages() == [{"role": "system", "content": "Ты — переводчик"}]


def test_seed_then_conversation_no_duplicate_system():
    svc = ChatService("c1")
    svc.seed_system_message("инструкция")
    svc.append_assistant_message("ок", MessageMeta(model="m", elapsed_s=1))
    svc.add_user_message("вопрос")
    assert svc.to_openai_messages() == [
        {"role": "system", "content": "инструкция"},
        {"role": "assistant", "content": "ок"},
        {"role": "user", "content": "вопрос"},
    ]


def test_rollback_seed_clears_system_prompt():
    svc = ChatService("c1")
    svc.seed_system_message("инструкция")
    svc.rollback_user_message()
    assert svc.history == []
    assert svc.system_prompt is None


def test_rollback_user_keeps_seed():
    svc = ChatService("c1")
    svc.seed_system_message("инструкция")
    svc.append_assistant_message("ок", MessageMeta(model="m", elapsed_s=1))
    svc.add_user_message("вопрос")
    svc.rollback_user_message()
    assert [m.role for m in svc.history] == ["system", "assistant"]
    assert svc.system_prompt == "инструкция"


def asyncio_run(coro):
    import asyncio

    asyncio.run(coro)


async def consume_stream(svc, runner, spec, extra_system=None, messages=None):
    async for _ in svc.stream_completion(
        runner, spec, extra_system=extra_system, messages=messages
    ):
        pass


# ── Саммаризация контекста ──────────────────────────────────────────────────


class ScriptedRunner:
    """Раннер, отдающий разные ответы на последовательные вызовы."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    async def run(self, spec, messages, settings):
        self.calls.append({"messages": messages, "settings": settings})
        script = self.scripts.pop(0) if self.scripts else []
        if isinstance(script, Exception):
            raise script
        for event in script:
            yield event


def summary_events(text="САММАРИ"):
    from server.providers.base import ChatEvent, Usage, UsageDetails

    return [
        ChatEvent(kind="delta", content=text),
        ChatEvent(kind="done", finish_reason="stop",
                  usage=Usage(prompt_tokens=50, completion_tokens=7,
                              details=UsageDetails(reasoning_tokens=3))),
    ]


def text_events(text="Ответ", prompt=100, completion=10):
    from server.providers.base import ChatEvent, Usage

    return [
        ChatEvent(kind="delta", content=text),
        ChatEvent(kind="done", finish_reason="stop",
                  usage=Usage(prompt_tokens=prompt, completion_tokens=completion)),
    ]


def run_coro(coro):
    import asyncio

    return asyncio.run(coro)


def spec_obj(model="m"):
    return type("Spec", (), {"model": model, "model_label": model})()


def fill_chat(svc, exchanges, pending_user=None):
    """Реалистичная история чата с сидом.

    Сид (первое сообщение) + ответ на него + ``exchanges`` обменов
    «вопрос+ответ» (+ висящий вопрос ``pending_user``). Ответ на сид — это
    вводное сообщение, примыкающее к первому чанку.
    """
    svc.seed_system_message("инструкция")
    svc.append_assistant_message("a0", MessageMeta(model="m", elapsed_s=0.1))
    for i in range(1, exchanges + 1):
        svc.add_user_message(f"u{i}")
        svc.append_assistant_message(f"a{i}", MessageMeta(model="m", elapsed_s=0.1))
    if pending_user is not None:
        svc.add_user_message(pending_user)


def cs_settings(n):
    return GenerationSettings(
        context_strategy=ContextStrategySettings(strategy="summarize", n=n)
    )


def strategy_settings(strategy, n=5, k=10):
    return GenerationSettings(
        context_strategy=ContextStrategySettings(strategy=strategy, n=n, k=k)
    )


def test_ensure_summaries_below_threshold():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 1, pending_user="u2")  # сид-ответ + 1 обмен = 2 запроса
    runner = ScriptedRunner([])
    assert run_coro(svc.ensure_summaries(runner, spec_obj(), settings)) == []
    assert runner.calls == []
    assert svc.summarized_chunks == 0


def test_ensure_summaries_first_chunk_includes_seed_reply():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    # 3 запроса: ответ на сид + 2 обмена; ждём саммари на 4-й отправке
    fill_chat(svc, 2, pending_user="u3")
    runner = ScriptedRunner([summary_events("S1")])
    items = run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    assert items == ["S1"]
    assert svc.summarized_chunks == 1
    # в скрытый запрос ушёл первый чанк: ответ на сид + 2 обмена
    sent = runner.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT}
    assert [m["content"] for m in sent[1:]] == ["a0", "u1", "a1", "u2", "a2"]
    assert svc.requests[0].kind == "summary"


def test_ensure_summaries_does_not_recount_ready_chunks():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 2, pending_user="u3")
    runner = ScriptedRunner([summary_events("S1")])
    run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    # ответили на u3 (запросов стало 4) и задали u4 — второй чанк ещё не полон
    svc.append_assistant_message("a3", MessageMeta(model="m", elapsed_s=0.1))
    svc.add_user_message("u4")
    assert run_coro(svc.ensure_summaries(runner, spec_obj(), settings)) == ["S1"]
    assert len(runner.calls) == 1  # готовый чанк не пересчитывается


def test_ensure_summaries_two_chunks_and_verbatim_tail():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 5, pending_user="u6")  # 6 запросов -> 2 чанка
    runner = ScriptedRunner([summary_events("S1"), summary_events("S2")])
    items = run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    assert items == ["S1", "S2"]
    assert svc.summarized_chunks == 2

    msgs = svc.build_request_messages(items)
    assert msgs[0] == {"role": "system", "content": "инструкция"}
    assert msgs[1]["content"] == (
        f"{SUMMARY_MESSAGE_PREFIX}\nСаммари 1:\nS1\n\nСаммари 2:\nS2"
    )
    # хвост — только висящее сообщение
    assert [m["content"] for m in msgs[2:]] == ["u6"]


def test_ensure_summaries_tail_verbatim_inside_chunk():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 3, pending_user="u4")  # 4 запроса: 1 чанк + незавершённый обмен
    runner = ScriptedRunner([summary_events("S1")])
    items = run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    msgs = svc.build_request_messages(items)
    assert msgs[1]["content"].startswith(SUMMARY_MESSAGE_PREFIX)
    # после первого чанка (сид-ответ + обмены 1-2) вербатим: обмен 3 + u4
    assert [m["content"] for m in msgs[2:]] == ["u3", "a3", "u4"]


def test_ensure_summaries_collapses_at_three():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 8, pending_user="u9")  # 9 запросов -> 3 чанка -> свёртка
    runner = ScriptedRunner([
        summary_events("S1"),
        summary_events("S2"),
        summary_events("S3"),
        summary_events("META"),
    ])
    items = run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    assert items == ["META"]
    assert svc.summarized_chunks == 3
    assert len(runner.calls) == 4  # 3 чанка + метасаммари
    # метасаммари получило тексты всех саммари
    meta_input = runner.calls[3]["messages"][1]["content"]
    assert "Саммари 1:\nS1" in meta_input
    assert "Саммари 3:\nS3" in meta_input

    msgs = svc.build_request_messages(items)
    assert msgs[1]["content"] == f"{SUMMARY_MESSAGE_PREFIX}\nСаммари 1:\nMETA"
    assert [m["content"] for m in msgs[2:]] == ["u9"]


def test_ensure_summaries_degrades_on_chunk_error():
    from server.providers.base import ProviderError

    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 2, pending_user="u3")
    runner = ScriptedRunner([ProviderError(500, "boom")])
    items = run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    assert items == []
    assert svc.summarized_chunks == 0
    assert svc.requests == []
    # основной запрос уходит с полной историей
    msgs = svc.build_request_messages(items)
    assert [m["content"] for m in msgs] == [
        "инструкция", "a0", "u1", "a1", "u2", "a2", "u3",
    ]


def test_ensure_summaries_collapse_failure_keeps_items():
    from server.providers.base import ProviderError

    settings = cs_settings(3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 8, pending_user="u9")
    runner = ScriptedRunner([
        summary_events("S1"),
        summary_events("S2"),
        summary_events("S3"),
        ProviderError(500, "boom"),
    ])
    items = run_coro(svc.ensure_summaries(runner, spec_obj(), settings))
    assert items == ["S1", "S2", "S3"]
    assert svc.summarized_chunks == 3


def test_ensure_summaries_disabled_returns_empty():
    svc = ChatService("c1", model="m")
    runner = ScriptedRunner([])
    assert run_coro(svc.ensure_summaries(runner, spec_obj(), GenerationSettings())) == []


def test_build_request_messages_without_summary_is_full_history():
    settings = GenerationSettings()
    svc = ChatService("c1", model="m", system_prompt="sys", settings=settings)
    svc.add_user_message("q")
    assert svc.build_request_messages() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]


def test_stream_completion_with_summary_records_main_and_marks_meta():
    settings = cs_settings(2)
    svc = ChatService("c1", model="m", settings=settings)
    svc.add_user_message("q")
    runner = ScriptedRunner([text_events()])
    events = []

    async def consume():
        async for e in svc.stream_completion(
            runner, spec_obj(), settings, summary_items=["КРАТКО"]
        ):
            events.append(e)

    run_coro(consume())
    assert events[-1]["meta"]["summarized"] is True
    assert svc.requests[-1].kind == "main"
    # в апстрим ушли: наша история с саммари-рамкой
    sent = runner.calls[0]["messages"]
    assert sent[0] == {
        "role": "user",
        "content": f"{SUMMARY_MESSAGE_PREFIX}\nСаммари 1:\nКРАТКО",
    }
    assert sent[-1]["content"] == "q"


# ── Стратегии скользящего окна и фактов ─────────────────────────────────────


def test_strategy_ignored_for_non_chat_kinds():
    settings = strategy_settings("sliding", n=1)
    svc = ChatService("c1", kind=SessionKind.SUMMARY, model="m", settings=settings)
    fill_chat(svc, 3, pending_user="u4")
    msgs = svc.build_request_messages()
    # стратегия не применяется: уходит вся история
    assert [m["content"] for m in msgs] == [
        "инструкция", "a0", "u1", "a1", "u2", "a2", "u3", "a3", "u4",
    ]


def test_sliding_window_truncates_old_turns():
    settings = strategy_settings("sliding", n=3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 4, pending_user="u5")
    msgs = svc.build_request_messages()
    # сид — равноправный обмен №1 и выпал из окна; только последние 3 обмена + новое
    assert [m["content"] for m in msgs] == [
        "u2", "a2", "u3", "a3", "u4", "a4", "u5",
    ]
    assert all(m["role"] != "system" for m in msgs)


def test_sliding_seed_drops_out_of_window():
    # Регрессия: N=2, сид «меня зовут Иван» + 2 вопроса + «Как меня зовут?»
    settings = strategy_settings("sliding", n=2)
    svc = ChatService("c1", model="m", settings=settings)
    svc.seed_system_message("меня зовут Иван")
    svc.append_assistant_message("a0", MessageMeta(model="m", elapsed_s=0.1))
    svc.add_user_message("q1")
    svc.append_assistant_message("a1", MessageMeta(model="m", elapsed_s=0.1))
    svc.add_user_message("q2")
    svc.append_assistant_message("a2", MessageMeta(model="m", elapsed_s=0.1))
    svc.add_user_message("Как меня зовут?")
    msgs = svc.build_request_messages()
    # ни сида, ни ответа на него — модель имени не знает
    assert [m["content"] for m in msgs] == ["q1", "a1", "q2", "a2", "Как меня зовут?"]
    assert all(m["role"] != "system" for m in msgs)


def test_sliding_window_no_truncation_when_few_turns():
    settings = strategy_settings("sliding", n=3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 2, pending_user="u3")
    msgs = svc.build_request_messages()
    # обменов ≤ N — вся история (сид на месте, роль system)
    assert [m["content"] for m in msgs] == [
        "инструкция", "a0", "u1", "a1", "u2", "a2", "u3",
    ]
    assert msgs[0]["role"] == "system"


def test_facts_context_includes_facts_and_window():
    settings = strategy_settings("facts", k=2)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 3, pending_user="u4")
    svc.facts = [["Имя", "Аня"], ["Хобби", "Python"]]
    msgs = svc.build_request_messages()
    # сид выпал (обменов 4 > K=2) → рамка фактов первая
    assert msgs[0]["role"] == "user"
    assert FACTS_MESSAGE_PREFIX in msgs[0]["content"]
    assert "1. Имя: Аня" in msgs[0]["content"]
    assert "2. Хобби: Python" in msgs[0]["content"]
    # окно K=2: последние 2 обмена (u2,a2,u3,a3) + новое u4
    assert [m["content"] for m in msgs[1:]] == ["u2", "a2", "u3", "a3", "u4"]
    assert all(m["role"] != "system" for m in msgs)


def test_facts_context_without_facts_has_no_frame():
    settings = strategy_settings("facts", k=2)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 1, pending_user="u2")
    msgs = svc.build_request_messages()
    assert all(FACTS_MESSAGE_PREFIX not in m["content"] for m in msgs)


def test_facts_block_renders_pairs_and_legacy():
    block = ChatService._facts_block([["Имя", "Иван"], "легаси строка"])
    assert FACTS_MESSAGE_PREFIX in block
    assert "1. Имя: Иван" in block
    assert "2. легаси строка" in block


def test_branching_uses_full_history_without_limit():
    # Регрессия: branching не использует n и не ограничивает историю
    settings = strategy_settings("branching", n=2)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 5, pending_user="u6")
    msgs = svc.build_request_messages()
    assert [m["content"] for m in msgs] == [
        "инструкция", "a0", "u1", "a1", "u2", "a2",
        "u3", "a3", "u4", "a4", "u5", "a5", "u6",
    ]
    assert msgs == svc.to_openai_messages()


def test_extract_facts_success_and_record():
    settings = strategy_settings("facts", k=3)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 1, pending_user="u2")
    runner = ScriptedRunner([summary_events('{"Имя": "Иван", "Город": "Москва"}')])
    result = run_coro(svc.extract_facts(runner, spec_obj(), settings, "u2"))
    assert result == [["Имя", "Иван"], ["Город", "Москва"]]
    assert svc.facts == [["Имя", "Иван"], ["Город", "Москва"]]
    assert svc.requests[-1].kind == "facts"
    # запрос к экстрактору: системный промпт + список фактов и сообщение
    sent = runner.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": FACTS_SYSTEM_PROMPT}
    assert "u2" in sent[1]["content"]


def test_extract_facts_parses_code_fences():
    settings = strategy_settings("facts")
    svc = ChatService("c1", model="m", settings=settings)
    svc.add_user_message("q")
    runner = ScriptedRunner([summary_events('```json\n{"Имя": "Иван"}\n```')])
    assert run_coro(svc.extract_facts(runner, spec_obj(), settings, "q")) == [
        ["Имя", "Иван"]
    ]


def test_parse_facts_object_array_variants_and_dedup():
    # JSON-объект
    assert ChatService._parse_facts('{"Имя": "Иван", "Город": "Москва"}') == [
        ["Имя", "Иван"], ["Город", "Москва"]
    ]
    # массив объектов {"key","value"}
    assert ChatService._parse_facts('[{"key": "Имя", "value": "Иван"}]') == [
        ["Имя", "Иван"]
    ]
    # массив пар
    assert ChatService._parse_facts('[["Имя", "Иван"], ["Город", "Москва"]]') == [
        ["Имя", "Иван"], ["Город", "Москва"]
    ]
    # мусор/легаси-массив строк — деградация
    assert ChatService._parse_facts('["Пользователя зовут Иван"]') is None
    assert ChatService._parse_facts("не json вовсе") is None
    # дедупликация по ключу: позиция первого, значение последнего
    assert ChatService._parse_facts('[["Имя", "Иван"], ["Имя", "Пётр"]]') == [
        ["Имя", "Пётр"]
    ]


def test_extract_facts_overwrites_legacy_string_list():
    settings = strategy_settings("facts")
    svc = ChatService("c1", model="m", settings=settings)
    svc.facts = ["Пользователя зовут Иван"]  # легаси-строка
    svc.add_user_message("q")
    runner = ScriptedRunner([summary_events('{"Имя": "Иван"}')])
    # старый список отдаётся экстрактору текстом, ответ перезаписывает список
    result = run_coro(svc.extract_facts(runner, spec_obj(), settings, "q"))
    assert result == [["Имя", "Иван"]]
    assert "Пользователя зовут Иван" in runner.calls[0]["messages"][1]["content"]


def test_extract_facts_invalid_json_keeps_old():
    settings = strategy_settings("facts")
    svc = ChatService("c1", model="m", settings=settings)
    svc.facts = ["старый"]
    svc.add_user_message("q")
    runner = ScriptedRunner([summary_events("не json вовсе")])
    assert run_coro(svc.extract_facts(runner, spec_obj(), settings, "q")) is None
    assert svc.facts == ["старый"]
    # запрос всё равно учтён (токены сожжены)
    assert svc.requests[-1].kind == "facts"


def test_extract_facts_degrades_on_error():
    from server.providers.base import ProviderError

    settings = strategy_settings("facts")
    svc = ChatService("c1", model="m", settings=settings)
    svc.facts = ["старый"]
    runner = ScriptedRunner([ProviderError(500, "boom")])
    assert run_coro(svc.extract_facts(runner, spec_obj(), settings, "q")) is None
    assert svc.facts == ["старый"]
    assert svc.requests == []


def test_extract_facts_noop_for_other_strategy():
    svc = ChatService("c1", model="m", settings=GenerationSettings())
    runner = ScriptedRunner([])
    assert run_coro(svc.extract_facts(runner, spec_obj(), GenerationSettings(), "q")) is None
    assert runner.calls == []


# ── Память чата ─────────────────────────────────────────────────────────────

def test_memory_frame_empty_returns_none():
    svc = ChatService("c1", model="m")
    assert svc.memory_frame() is None
    svc.memory_stores = [MemoryStore("m1", "Профиль", True, [])]
    assert svc.memory_frame() is None


def test_memory_frame_format():
    svc = ChatService("c1", model="m")
    svc.memory_stores = [
        MemoryStore("m1", "Профиль", True, [["Имя", "Иван"], ["Город", "Москва"]]),
        MemoryStore("m2", "Прочее", False, [["Тема", "API"]]),
    ]
    frame = svc.memory_frame()
    assert MEMORY_MESSAGE_PREFIX in frame
    assert "## Профиль" in frame
    assert "Имя: Иван" in frame
    assert "## Прочее" in frame
    assert "Тема: API" in frame
    # инструкция требует использовать память только по необходимости
    assert MEMORY_INSTRUCTION in frame


def test_memory_frame_injected_after_system():
    settings = GenerationSettings()
    svc = ChatService("c1", model="m", system_prompt="sys", settings=settings)
    svc.add_user_message("q")
    svc.memory_stores = [MemoryStore("m1", "Профиль", True, [["Имя", "Иван"]])]
    msgs = svc.build_request_messages()
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert MEMORY_MESSAGE_PREFIX in msgs[1]["content"]
    assert msgs[-1] == {"role": "user", "content": "q"}


def test_memory_frame_top_when_no_system_in_window():
    # sliding без system в окне -> рамка памяти идёт первой
    settings = strategy_settings("sliding", n=1)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 3, pending_user="u4")
    svc.memory_stores = [MemoryStore("m1", "Профиль", True, [["Имя", "Иван"]])]
    msgs = svc.build_request_messages()
    assert msgs[0]["role"] == "user"
    assert MEMORY_MESSAGE_PREFIX in msgs[0]["content"]
    assert all(m["content"] != "инструкция" for m in msgs)


def test_memory_frame_and_facts_frame_together():
    settings = strategy_settings("facts", k=2)
    svc = ChatService("c1", model="m", settings=settings)
    fill_chat(svc, 3, pending_user="u4")
    svc.facts = [["Имя", "Аня"]]
    svc.memory_stores = [MemoryStore("m1", "Профиль", True, [["Роль", "QA"]])]
    msgs = svc.build_request_messages()
    assert MEMORY_MESSAGE_PREFIX in msgs[0]["content"]
    assert FACTS_MESSAGE_PREFIX in msgs[1]["content"]


def test_memory_frame_in_summarize_path():
    settings = cs_settings(3)
    svc = ChatService("c1", model="m", system_prompt="sys", settings=settings)
    svc.add_user_message("q")
    svc.memory_stores = [MemoryStore("m1", "Профиль", True, [["Имя", "Иван"]])]
    msgs = svc.build_request_messages(summary_items=["S1"])
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert MEMORY_MESSAGE_PREFIX in msgs[1]["content"]
    assert SUMMARY_MESSAGE_PREFIX in msgs[2]["content"]


def test_memory_frame_absent_for_non_chat():
    svc = ChatService("c1", kind=SessionKind.SUMMARY, model="m")
    svc.memory_stores = [MemoryStore("m1", "Профиль", True, [["Имя", "Иван"]])]
    assert svc.memory_frame() is None


def test_task_step_prompts_forbid_skipping():
    """Промпты исполнителя/переделки шага запрещают пропуск шагов вперёд."""
    from server.services.chat_service import (
        TASK_STEP_EXECUTOR_SYSTEM_PROMPT,
        TASK_STEP_REVISE_SYSTEM_PROMPT,
    )

    for prompt in (TASK_STEP_EXECUTOR_SYSTEM_PROMPT, TASK_STEP_REVISE_SYSTEM_PROMPT):
        text = prompt.lower()
        assert "перепрыг" in text  # пропуск нескольких шагов вперёд
        assert "откажи" in text
    # возврат назад и переход к следующему шагу — легитимны (не запрещены)
    revise = TASK_STEP_REVISE_SYSTEM_PROMPT.lower()
    assert "вернуться" in revise
    assert "следующему" in revise

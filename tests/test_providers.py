"""Тесты маршрутизации провайдеров."""

import pytest

from server.providers import UnsupportedModelError, resolve_provider
from server.providers.engine import _classify_upstream_error, _parse_event_data
from server.providers.base import ChatEvent


def test_deepseek_default():
    spec = resolve_provider(None, "sk", None, "sess")
    assert spec.provider_name == "DeepSeek"
    assert spec.model == "deepseek-chat"
    assert spec.api_key == "sk"
    assert spec.endpoint.endswith("/chat/completions")


def test_opencode_go():
    spec = resolve_provider("opencode/glm-5.3", None, "zen", "sess")
    assert spec.provider_name == "OpenCode"
    assert spec.model == "glm-5.3"
    assert "/zen/go/" in spec.endpoint
    assert spec.headers["x-opencode-session"] == "sess"


def test_opencode_free():
    spec = resolve_provider("opencode/big-pickle", None, "zen", "sess")
    assert "/zen/v1/" in spec.endpoint


def test_model_label_preserved():
    # opencode-модель сохраняет клиентский id с префиксом (для метаданных/восстановления)
    spec = resolve_provider("opencode/glm-5.3", None, "zen", "sess")
    assert spec.model_label == "opencode/glm-5.3"
    assert spec.model == "glm-5.3"
    # deepseek — как есть
    spec = resolve_provider("deepseek-v4-flash", "sk", None, "sess")
    assert spec.model_label == "deepseek-v4-flash"


def test_opencode_unsupported():
    with pytest.raises(UnsupportedModelError):
        resolve_provider("opencode/nope", None, "zen", "sess")


def test_missing_key_raises():
    with pytest.raises(ValueError):
        resolve_provider("deepseek-chat", None, "zen", "sess")
    with pytest.raises(ValueError):
        resolve_provider("opencode/glm-5.3", "sk", None, "sess")


def test_parse_delta():
    e = _parse_event_data('{"choices":[{"delta":{"content":"hi"}}]}')
    assert e.kind == "delta" and e.content == "hi"


def test_parse_reasoning():
    e = _parse_event_data('{"choices":[{"delta":{"reasoning_content":"думаю"}}]}')
    assert e.kind == "reasoning" and e.content == "думаю"


def test_parse_usage():
    e = _parse_event_data('{"usage":{"prompt_tokens":1,"completion_tokens":2}}')
    assert e.kind == "usage"
    assert e.usage.prompt_tokens == 1


def test_parse_done_with_reasoning_details():
    raw = ('{"choices":[{"delta":{},"finish_reason":"stop",'
           '"usage":{"completion_tokens_details":{"reasoning_tokens":7}}}]}')
    e = _parse_event_data(raw)
    assert e.kind == "done"
    assert e.finish_reason == "stop"
    assert e.usage.details.reasoning_tokens == 7


def test_stream_buffers_reasoning_into_start_end():
    """Reasoning приходит частями, а наружу уходит парой start/end с полным текстом."""
    import json

    import httpx

    from server.providers.engine import stream_completion
    from server.providers.routing import resolve_provider
    from server.services.generation import GenerationSettings
    from tests.conftest import MockTransport

    def d(payload):
        return f"data: {json.dumps(payload)}\n\n".encode()

    chunks = [
        d({"choices": [{"delta": {"reasoning_content": "дума"}}]}),
        d({"choices": [{"delta": {"reasoning_content": "ю про"}}]}),
        d({"choices": [{"delta": {"reasoning_content": "блему"}}]}),
        d({"choices": [{"delta": {"content": "Ответ"}}]}),
        b"data: [DONE]\n\n",
    ]
    client = httpx.AsyncClient(transport=MockTransport(chunks))
    spec = resolve_provider("deepseek-chat", "sk", "zen", "sess")

    async def run():
        kinds = []
        contents = []
        async for e in stream_completion(client, spec, [{"role": "user", "content": "x"}], GenerationSettings()):
            kinds.append(e.kind)
            contents.append(e.content)
        await client.aclose()
        return kinds, contents

    import asyncio

    kinds, contents = asyncio.run(run())
    assert kinds == ["reasoning_start", "reasoning_end", "delta"]
    assert contents[1] == "думаю проблему"  # полный текст рассуждения целиком


def test_deepseek_identity_prompt_prepended():
    """DeepSeek-модели получают первым системное сообщение с фиксацией идентичности."""
    import asyncio
    import json

    import httpx

    from server.providers.engine import DEEPSEEK_IDENTITY_PROMPT, stream_completion
    from server.providers.routing import resolve_provider
    from server.services.generation import GenerationSettings
    from tests.conftest import MockTransport

    transport = MockTransport([b"data: [DONE]\n\n"])
    client = httpx.AsyncClient(transport=transport)
    spec = resolve_provider("deepseek-v4-flash", "sk", "zen", "sess")

    async def run():
        async for _ in stream_completion(
            client, spec, [{"role": "user", "content": "кто ты?"}], GenerationSettings()
        ):
            pass
        await client.aclose()

    asyncio.run(run())
    body = json.loads(transport.requests[0].content)
    messages = body["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DEEPSEEK_IDENTITY_PROMPT
    assert messages[1] == {"role": "user", "content": "кто ты?"}


def test_opencode_identity_prompt_not_prepended():
    """OpenCode-модели не получают промпт идентичности DeepSeek."""
    import asyncio
    import json

    import httpx

    from server.providers.engine import stream_completion
    from server.providers.routing import resolve_provider
    from server.services.generation import GenerationSettings
    from tests.conftest import MockTransport

    transport = MockTransport([b"data: [DONE]\n\n"])
    client = httpx.AsyncClient(transport=transport)
    spec = resolve_provider("opencode/glm-5.3", "sk", "zen", "sess")

    async def run():
        async for _ in stream_completion(
            client, spec, [{"role": "user", "content": "hi"}], GenerationSettings()
        ):
            pass
        await client.aclose()

    asyncio.run(run())
    body = json.loads(transport.requests[0].content)
    messages = body["messages"]
    assert messages[0] == {"role": "user", "content": "hi"}


def test_classify_context_error_deepseek():
    # дословно снято с api.deepseek.com
    msg = (
        "This model's maximum context length is 1048576 tokens. However, you "
        "requested 1293247 tokens (900031 in the messages, 393216 in the completion). "
        "Please reduce the length of the messages or completion."
    )
    code, details = _classify_upstream_error(400, msg)
    assert code == "context_length_exceeded"
    assert details["max_context"] == 1048576
    assert details["requested"] == 1293247


def test_classify_context_error_longcat():
    # иной формат: context length (N) / The input (M tokens)
    msg = ("The input (1800007 tokens) is longer than the model's context "
           "length (1048580 tokens)")
    code, details = _classify_upstream_error(400, msg)
    assert code == "context_length_exceeded"
    assert details["max_context"] == 1048580
    assert details["requested"] == 1800007


def test_classify_context_error_kimi_endpoint():
    msg = ("This endpoint's maximum context length is 262144 tokens. However, "
           "you requested about 100000000 tokens (1 of text)")
    code, details = _classify_upstream_error(400, msg)
    assert code == "context_length_exceeded"
    assert details["max_context"] == 262144


def test_classify_generic_error_has_no_code():
    code, details = _classify_upstream_error(402, "Insufficient balance")
    assert code is None
    assert details == {}
    # код возвращается только для 400 с признаком контекста
    code, _ = _classify_upstream_error(500, "maximum context length is 1 tokens")
    assert code is None


def test_parse_event_data_full_extracts_tool_call_fragments():
    from server.providers.engine import _parse_event_data_full

    raw = (
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"get_item","arguments":"{\\"id\\""}}]}}]}'
    )
    event, fragments = _parse_event_data_full(raw)
    assert event is None
    assert fragments[0]["name"] == "get_item"
    assert fragments[0]["arguments"] == '{"id"'


def test_stream_accumulates_tool_calls_into_done():
    """Фрагменты tool_calls склеиваются, а done несёт готовые вызовы."""
    import asyncio
    import json

    import httpx

    from server.providers.engine import stream_completion
    from server.providers.routing import resolve_provider
    from server.services.generation import GenerationSettings
    from tests.conftest import MockTransport

    def d(payload):
        return f"data: {json.dumps(payload)}\n\n".encode()

    chunks = [
        d(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_item",
                                        "arguments": '{"resource"',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        d(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": ': "posts", "id": 1}'
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        d({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        b"data: [DONE]\n\n",
    ]
    client = httpx.AsyncClient(transport=MockTransport(chunks))
    spec = resolve_provider("deepseek-chat", "sk", "zen", "sess")

    async def run():
        events = []
        async for event in stream_completion(
            client, spec, [{"role": "user", "content": "x"}], GenerationSettings()
        ):
            events.append(event)
        await client.aclose()
        return events

    events = asyncio.run(run())
    done = [event for event in events if event.kind == "done"][0]
    assert done.finish_reason == "tool_calls"
    assert done.tool_calls == [
        {
            "id": "call_1",
            "name": "get_item",
            "arguments": '{"resource": "posts", "id": 1}',
        }
    ]


def test_stream_synthesizes_done_for_tool_calls_without_finish():
    """Даже без финального чанка tool_calls не теряются при [DONE]."""
    import asyncio
    import json

    import httpx

    from server.providers.engine import stream_completion
    from server.providers.routing import resolve_provider
    from server.services.generation import GenerationSettings
    from tests.conftest import MockTransport

    def d(payload):
        return f"data: {json.dumps(payload)}\n\n".encode()

    chunks = [
        d(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "t", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    client = httpx.AsyncClient(transport=MockTransport(chunks))
    spec = resolve_provider("deepseek-chat", "sk", "zen", "sess")

    async def run():
        events = []
        async for event in stream_completion(
            client, spec, [{"role": "user", "content": "x"}], GenerationSettings()
        ):
            events.append(event)
        await client.aclose()
        return events

    events = asyncio.run(run())
    done = [event for event in events if event.kind == "done"][0]
    assert done.finish_reason == "tool_calls"
    assert done.tool_calls[0]["name"] == "t"


def test_stream_includes_tools_in_payload():
    """Переданные tools уходят в тело запроса к апстриму."""
    import asyncio
    import json

    import httpx

    from server.providers.engine import stream_completion
    from server.providers.routing import resolve_provider
    from server.services.generation import GenerationSettings
    from tests.conftest import MockTransport

    transport = MockTransport([b"data: [DONE]\n\n"])
    client = httpx.AsyncClient(transport=transport)
    spec = resolve_provider("deepseek-chat", "sk", "zen", "sess")
    tools = [{"type": "function", "function": {"name": "t"}}]

    async def run():
        async for _ in stream_completion(
            client,
            spec,
            [{"role": "user", "content": "x"}],
            GenerationSettings(),
            tools,
        ):
            pass
        await client.aclose()

    asyncio.run(run())
    body = json.loads(transport.requests[0].content)
    assert body["tools"] == tools

"""Тесты маршрутизации провайдеров."""

import pytest

from server.providers import UnsupportedModelError, resolve_provider
from server.providers.engine import _parse_event_data
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

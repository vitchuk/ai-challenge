"""Реализация стриминга SSE апстрима и его нормализация в события.

Совместно используется провайдерами DeepSeek и OpenCode: единый код
разбирает OpenAI-совместимый поток и отдаёт события :class:`ChatEvent`.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from ..services.generation import GenerationSettings
from .base import ChatEvent, ProviderError, ProviderSpec, Usage, UsageDetails

# Промпт идентичности для моделей DeepSeek: модель ошибочно представляется
# «ассистентом OpenAI», поэтому ей явно фиксируем, кто она такая.
DEEPSEEK_IDENTITY_PROMPT = (
    "Ты — DeepSeek V4 Flash, языковая модель, созданная компанией DeepSeek "
    "(深度求索). Никогда не называй себя ChatGPT, GPT, OpenAI или "
    "«ассистентом OpenAI». Ты — DeepSeek."
)


def _upstream_error_message(raw_text: str) -> str:
    """Извлекает человекочитаемое сообщение ошибки из тела апстрима."""
    if not raw_text:
        return "no details"
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, str) and err:
                return err
            if isinstance(err, dict) and isinstance(err.get("message"), str) and err["message"]:
                return err["message"]
    except json.JSONDecodeError:
        pass
    trimmed = raw_text.strip()
    return trimmed[:200] + "…" if len(trimmed) > 200 else trimmed


def _parse_event_data(data_str: str) -> Optional[ChatEvent]:
    """Превращает одну ``data:``-строку апстрима в событие (или ``None``)."""
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    usage = data.get("usage")
    if isinstance(usage, dict):
        details_raw = usage.get("completion_tokens_details") or {}
        details = UsageDetails(
            reasoning_tokens=details_raw.get("reasoning_tokens")
            if isinstance(details_raw, dict)
            else None
        )
        return ChatEvent(
            kind="usage",
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                details=details,
            ),
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None

    finish_reason = choice.get("finish_reason")
    if finish_reason:
        usage = choice.get("usage")
        if not isinstance(usage, dict):
            usage = None
        details_raw = usage.get("completion_tokens_details") if usage else None
        details = (
            UsageDetails(reasoning_tokens=details_raw.get("reasoning_tokens"))
            if isinstance(details_raw, dict)
            else None
        )
        return ChatEvent(
            kind="done",
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens") if usage else None,
                completion_tokens=usage.get("completion_tokens") if usage else None,
                total_tokens=usage.get("total_tokens") if usage else None,
                details=details,
            ),
        )

    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None
    reasoning = delta.get("reasoning_content")
    content = delta.get("content")
    if isinstance(reasoning, str) and reasoning:
        return ChatEvent(kind="reasoning", content=reasoning)
    if isinstance(content, str) and content:
        return ChatEvent(kind="delta", content=content)
    return None


async def stream_completion(
    client: httpx.AsyncClient,
    spec: ProviderSpec,
    messages: list[dict],
    settings: GenerationSettings,
) -> AsyncIterator[ChatEvent]:
    """Выполняет стриминговый запрос к апстриму и нормализует ответ.

    Args:
        client: инжектируемый HTTP-клиент (позволяет мокать в тестах).
        spec: описание вызова апстрима.
        messages: массив сообщений OpenAI (role + content).
        settings: параметры генерации.

    Yields:
        Нормализованные события :class:`ChatEvent`.

    Raises:
        ProviderError: если апстрим вернул HTTP-ошибку.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {spec.api_key}",
        **spec.headers,
    }
    # Для DeepSeek фиксируем идентичность модели первым системным сообщением.
    outgoing = list(messages)
    if spec.provider_name == "DeepSeek":
        outgoing.insert(0, {"role": "system", "content": DEEPSEEK_IDENTITY_PROMPT})
    payload = {
        "model": spec.model,
        "messages": outgoing,
        "stream": True,
        "stream_options": {"include_usage": True},
        **settings.to_upstream(),
    }

    try:
        async with client.stream(
            "POST", spec.endpoint, headers=headers, json=payload
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                text = body.decode("utf-8", errors="replace")
                raise ProviderError(
                    response.status_code,
                    f"{spec.provider_name} API error: {response.status_code} — "
                    f"{_upstream_error_message(text)}",
                )
            buffer = ""
            reasoning_buf: list[str] = []
            reasoning_open = False

            def take_reasoning() -> str:
                """Забирает и сбрасывает накопленное рассуждение.

                Returns:
                    Полный текст рассуждения (или пустую строку).
                """
                nonlocal reasoning_buf, reasoning_open
                if not reasoning_open:
                    return ""
                reasoning_open = False
                text = "".join(reasoning_buf)
                reasoning_buf = []
                return text

            async for raw_chunk in response.aiter_bytes():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        text = take_reasoning()
                        if text:
                            yield ChatEvent(kind="reasoning_end", content=text)
                        return
                    event = _parse_event_data(data_str)
                    if event is None:
                        continue
                    if event.kind == "reasoning":
                        if not reasoning_open:
                            reasoning_open = True
                            yield ChatEvent(kind="reasoning_start")
                        reasoning_buf.append(event.content)
                    elif event.kind == "delta":
                        text = take_reasoning()
                        if text:
                            yield ChatEvent(kind="reasoning_end", content=text)
                        yield event
                    else:
                        yield event
            text = take_reasoning()
            if text:
                yield ChatEvent(kind="reasoning_end", content=text)
    except httpx.HTTPError as exc:
        raise ProviderError(502, f"{spec.provider_name} connection error: {exc}") from exc

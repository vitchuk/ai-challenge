"""Реализация стриминга SSE апстрима и его нормализация в события.

Совместно используется провайдерами DeepSeek и OpenCode: единый код
разбирает OpenAI-совместимый поток и отдаёт события :class:`ChatEvent`.
"""

from __future__ import annotations

import json
import re
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


_CONTEXT_ERROR_RE = re.compile(
    r"(maximum context length|context length|context window|reduce the length|"
    r"longer than the model's context)",
    re.IGNORECASE,
)
_MAX_RE = re.compile(
    r"(?:maximum context length is|context length \()\s*(\d+)", re.IGNORECASE
)
_REQUESTED_RE = re.compile(r"(?:requested|input)\s*\(?\s*(?:about\s*)?(\d+)", re.IGNORECASE)


def _classify_upstream_error(status: int, message: str) -> tuple[Optional[str], dict]:
    """Классифицирует ошибку апстрима.

    Args:
        status: HTTP-статус.
        message: извлечённое сообщение ошибки.

    Returns:
        Кортеж ``(code, details)``: для ошибки лимита контекста —
        ``("context_length_exceeded", {"max_context": int|None, "requested": int|None})``,
        иначе ``(None, {})``.
    """
    if status != 400 or not _CONTEXT_ERROR_RE.search(message):
        return None, {}
    details: dict = {}
    m = _MAX_RE.search(message)
    if m:
        details["max_context"] = int(m.group(1))
    r = _REQUESTED_RE.search(message)
    if r:
        details["requested"] = int(r.group(1))
    return "context_length_exceeded", details


def _parse_tool_calls(delta: dict) -> list[dict]:
    """Извлекает фрагменты ``tool_calls`` из дельты апстрима.

    Провайдеры шлют вызовы инструментов частями: ``id`` и ``name`` приходят
    целиком, ``arguments`` — фрагментами JSON, которые затем склеиваются.
    """
    raw = delta.get("tool_calls")
    if not isinstance(raw, list):
        return []
    fragments: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int):
            index = 0
        function = item.get("function")
        function = function if isinstance(function, dict) else {}
        fragments.append(
            {
                "index": index,
                "id": item.get("id") if isinstance(item.get("id"), str) else None,
                "name": function.get("name")
                if isinstance(function.get("name"), str)
                else None,
                "arguments": function.get("arguments")
                if isinstance(function.get("arguments"), str)
                else None,
            }
        )
    return fragments


def _accumulate_tool_calls(acc: dict, fragments: list[dict]) -> None:
    """Накапливает фрагменты вызовов инструментов по индексу."""
    for fragment in fragments:
        slot = acc.setdefault(
            fragment["index"], {"id": None, "name": None, "arguments": ""}
        )
        if fragment["id"]:
            slot["id"] = fragment["id"]
        if fragment["name"]:
            slot["name"] = (slot["name"] or "") + fragment["name"]
        if fragment["arguments"]:
            slot["arguments"] += fragment["arguments"]


def _build_tool_calls(acc: dict) -> list[dict]:
    """Собирает завершённые вызовы инструментов из накопленных фрагментов."""
    calls = []
    for index in sorted(acc):
        slot = acc[index]
        if not slot["name"]:
            continue
        calls.append(
            {
                "id": slot["id"] or f"call_{index}",
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            }
        )
    return calls


def _parse_event_data(data_str: str) -> Optional[ChatEvent]:
    """Превращает одну ``data:``-строку апстрима в событие (или ``None``)."""
    event, _ = _parse_event_data_full(data_str)
    return event


def _parse_event_data_full(data_str: str) -> tuple[Optional[ChatEvent], list[dict]]:
    """Разбирает ``data:``-строку на событие и фрагменты ``tool_calls``."""
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None, []
    if not isinstance(data, dict):
        return None, []

    usage = data.get("usage")
    if isinstance(usage, dict):
        details_raw = usage.get("completion_tokens_details") or {}
        details = UsageDetails(
            reasoning_tokens=details_raw.get("reasoning_tokens")
            if isinstance(details_raw, dict)
            else None
        )
        return (
            ChatEvent(
                kind="usage",
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    details=details,
                ),
            ),
            [],
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, []
    choice = choices[0]
    if not isinstance(choice, dict):
        return None, []

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
        delta = choice.get("delta")
        fragments = _parse_tool_calls(delta) if isinstance(delta, dict) else []
        return (
            ChatEvent(
                kind="done",
                finish_reason=finish_reason,
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens") if usage else None,
                    completion_tokens=usage.get("completion_tokens") if usage else None,
                    total_tokens=usage.get("total_tokens") if usage else None,
                    details=details,
                ),
            ),
            fragments,
        )

    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None, []
    fragments = _parse_tool_calls(delta)
    reasoning = delta.get("reasoning_content")
    content = delta.get("content")
    if isinstance(reasoning, str) and reasoning:
        return ChatEvent(kind="reasoning", content=reasoning), fragments
    if isinstance(content, str) and content:
        return ChatEvent(kind="delta", content=content), fragments
    return None, fragments


async def stream_completion(
    client: httpx.AsyncClient,
    spec: ProviderSpec,
    messages: list[dict],
    settings: GenerationSettings,
    tools: Optional[list[dict]] = None,
) -> AsyncIterator[ChatEvent]:
    """Выполняет стриминговый запрос к апстриму и нормализует ответ.

    Args:
        client: инжектируемый HTTP-клиент (позволяет мокать в тестах).
        spec: описание вызова апстрима.
        messages: массив сообщений OpenAI (role + content).
        settings: параметры генерации.
        tools: OpenAI-описания доступных инструментов (или ``None``).

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
    if tools:
        payload["tools"] = tools

    try:
        async with client.stream(
            "POST", spec.endpoint, headers=headers, json=payload
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                text = body.decode("utf-8", errors="replace")
                upstream_msg = _upstream_error_message(text)
                code, details = _classify_upstream_error(response.status_code, upstream_msg)
                raise ProviderError(
                    response.status_code,
                    f"{spec.provider_name} API error: {response.status_code} — {upstream_msg}",
                    code=code,
                    details=details,
                )
            buffer = ""
            reasoning_buf: list[str] = []
            reasoning_open = False
            tool_acc: dict = {}
            done_seen = False

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
                        if not done_seen and tool_acc:
                            calls = _build_tool_calls(tool_acc)
                            if calls:
                                yield ChatEvent(
                                    kind="done",
                                    finish_reason="tool_calls",
                                    tool_calls=calls,
                                )
                        return
                    event, fragments = _parse_event_data_full(data_str)
                    if fragments:
                        _accumulate_tool_calls(tool_acc, fragments)
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
                        if event.kind == "done":
                            calls = _build_tool_calls(tool_acc)
                            if calls:
                                event.tool_calls = calls
                            done_seen = True
                        yield event
            text = take_reasoning()
            if text:
                yield ChatEvent(kind="reasoning_end", content=text)
    except httpx.HTTPError as exc:
        detail = f"{type(exc).__name__}: {exc}".strip()
        raise ProviderError(502, f"{spec.provider_name} connection error: {detail}") from exc

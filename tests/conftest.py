"""Общие фикстуры pytest: приложение, реестр, мок-провайдер."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from server.main import create_app


class FakeStream(httpx.AsyncByteStream):
    """Мок-ответ апстрима: раздаёт предзаданные SSE-чанки."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class MockTransport(httpx.AsyncBaseTransport):
    """HTTP-транспорт, возвращающий предзаданные SSE-чанки для chat/completions."""

    def __init__(self, chunks: list[bytes], status: int = 200, error_text: str = "") -> None:
        self.chunks = chunks
        self.status = status
        self.error_text = error_text
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, text=self.error_text, request=request)
        return httpx.Response(
            self.status,
            headers={"content-type": "text/event-stream"},
            stream=FakeStream(self.chunks),
            request=request,
        )


def sse_data(payload: dict) -> bytes:
    """Формирует одну ``data:``-строку SSE апстрима."""
    import json

    return f"data: {json.dumps(payload)}\n\n".encode()


def make_chat_chunks(reasoning: str = "", content: str = "", usage: dict | None = None):
    """Генерирует стандартную последовательность SSE-чанков Chat Completions."""
    chunks = []
    if reasoning:
        chunks.append(sse_data({"choices": [{"delta": {"reasoning_content": reasoning}}]}))
    if content:
        chunks.append(sse_data({"choices": [{"delta": {"content": content}}]}))
    if usage:
        chunks.append(sse_data({"usage": usage}))
    chunks.append(sse_data({"choices": [{"delta": {}, "finish_reason": "stop", "usage": usage}]}))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


USAGE = {
    "prompt_tokens": 202,
    "completion_tokens": 233,
    "total_tokens": 435,
    "completion_tokens_details": {"reasoning_tokens": 100},
}


@pytest_asyncio.fixture
async def app(monkeypatch) -> FastAPI:
    """Приложение с замоканным HTTP-клиентом (не уходит в сеть)."""
    from server import config
    from server.services.registry import SessionRegistry

    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(
        deepseek_api_key="sk-test",
        opencode_api_key="zen-test",
    ))

    application = create_app()
    transport = MockTransport(make_chat_chunks(content="Ответ", usage=USAGE))
    application.state.http_client = httpx.AsyncClient(transport=transport)
    application.state.mock_transport = transport
    application.state.opencode_session_id = "test-session"
    application.state.registry = SessionRegistry(config.get_settings())
    yield application
    await application.state.http_client.aclose()


@pytest_asyncio.fixture
async def client(app):
    """ASGI-клиент поверх приложения."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

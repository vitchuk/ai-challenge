"""Составной стриминговый результат: исполнение запроса к апстриму.

Провайдеры DeepSeek и OpenCode различаются только маршрутизацией
(см. :mod:`server.providers.routing`) и разделяют общий движок нормализации
SSE (:mod:`server.providers.engine`). :class:`StreamedCompletion` —
единая точка исполнения с возможностью инжекции HTTP-клиента для тестов.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from ..services.generation import GenerationSettings
from .base import ChatEvent, ProviderSpec
from .engine import stream_completion


class StreamedCompletion:
    """Исполнение запроса к апстриму с нормализацией SSE в события.

    Args:
        client: HTTP-клиент; если ``None``, создаётся собственный.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def run(
        self,
        spec: ProviderSpec,
        messages: list[dict],
        settings: GenerationSettings,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Итерирует по событиям ответа модели.

        Args:
            spec: описание вызова апстрима (endpoint/ключ/модель/заголовки).
            messages: массив сообщений OpenAI (role + content).
            settings: параметры генерации.
            tools: OpenAI-описания доступных инструментов (или ``None``).

        Yields:
            События :class:`ChatEvent`.

        Raises:
            ProviderError: если апстрим вернул ошибку.
        """
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))
        try:
            async for event in stream_completion(client, spec, messages, settings, tools):
                yield event
        finally:
            if owns_client:
                await client.aclose()

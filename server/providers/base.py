"""Базовые типы и контракт провайдера LLM.

Провайдер нормализует поток SSE апстрима (OpenAI-совместимый Chat
Completions) в последовательность событий :class:`ChatEvent`. События
используются сервисным слоем для стриминга клиенту и сбора метаданных.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from ..services.generation import GenerationSettings


@dataclass
class UsageDetails:
    """Детализация выходных токенов (для reasoning-моделей)."""

    reasoning_tokens: Optional[int] = None


@dataclass
class Usage:
    """Сводка по токенам запроса из финального чанка апстрима."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    details: Optional[UsageDetails] = None


@dataclass
class ChatEvent:
    """Нормализованное событие стриминга от провайдера.

    ``kind`` принимает значения: ``"delta"``, ``"reasoning_start"``,
    ``"reasoning_end"``, ``"usage"``, ``"done"``. Событие ``done`` несёт
    финальный ``finish_reason`` и ``usage``.
    """

    kind: str
    content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None


class ProviderError(Exception):
    """Ошибка апстрима (передаётся клиенту с кодом HTTP).

    Args:
        status: HTTP-статус ошибки.
        message: человекочитаемое сообщение.
        code: машинный код (например, ``context_length_exceeded``).
        details: дополнительные данные (например, лимит контекста).
    """

    def __init__(
        self,
        status: int,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.details = details or {}


@dataclass
class ProviderSpec:
    """Описание того, как вызывать апстрим для конкретной модели."""

    endpoint: str
    api_key: str
    model: str
    provider_name: str
    headers: dict[str, str] = field(default_factory=dict)
    model_label: str = ""


class LLMProvider(abc.ABC):
    """Контракт провайдера: превращает список сообщений в поток событий."""

    @abc.abstractmethod
    async def stream(
        self,
        spec: ProviderSpec,
        messages: list[dict],
        settings: GenerationSettings,
    ) -> AsyncIterator[ChatEvent]:
        """Итерирует по нормализованным событиям ответа модели.

        Args:
            spec: описание вызова апстрима (endpoint/ключ/модель/заголовки).
            messages: массив сообщений в формате OpenAI (role + content).
            settings: параметры генерации.

        Yields:
            События :class:`ChatEvent` по мере поступления данных.

        Raises:
            ProviderError: если апстрим вернул ошибку.
        """
        raise NotImplementedError

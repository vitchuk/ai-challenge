"""Сервис, инкапсулирующий отдельный чат/агента.

:class:`ChatService` хранит системный промпт, параметры генерации, модель
и историю сообщений (с метаданными каждого ответа LLM: время, токены,
стоимость). Он не зависит от HTTP и UI, поэтому в будущем может выступать
как «агент» в мультиагентной оркестрации (модератор + субагенты).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

from ..pricing import message_cost
from ..toon_codec import encode as toon_encode
from .generation import GenerationSettings


class SessionKind(str, Enum):
    """Тип сессии (чата)."""

    CHAT = "chat"
    SUMMARY = "summary"
    EPHEMERAL = "ephemeral"


@dataclass
class MessageMeta:
    """Метаданные ответа LLM.

    Заполняются сервером по завершении стрима и сохраняются в истории,
    чтобы впоследствии вкладываться (в TOON) в контекст других чатов.
    """

    model: str
    elapsed_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Представляет метаданные как словарь (для TOON/API)."""
        return {
            "model": self.model,
            "time_s": round(self.elapsed_s, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["MessageMeta"]:
        """Восстанавливает метаданные из словаря (из БД/API).

        Args:
            data: словарь метаданных (может быть ``None``).

        Returns:
            Экземпляр :class:`MessageMeta` или ``None``.
        """
        if not isinstance(data, dict):
            return None
        return cls(
            model=data.get("model", ""),
            elapsed_s=data.get("time_s", 0.0),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            reasoning_tokens=data.get("reasoning_tokens"),
            cost_usd=data.get("cost_usd"),
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class MessageRecord:
    """Одно сообщение истории чата.

    ``meta`` заполняется только для ответов ассистента.
    """

    role: str
    content: str
    meta: Optional[MessageMeta] = None
    created_at: float = 0.0

    def to_dict(self) -> dict:
        """Представляет сообщение как словарь (для API)."""
        out: dict = {"role": self.role, "content": self.content}
        if self.meta is not None:
            out["meta"] = self.meta.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "MessageRecord":
        """Восстанавливает сообщение из словаря (из БД/API).

        Args:
            data: словарь ``{"role", "content", "meta"?}``.

        Returns:
            Экземпляр :class:`MessageRecord`.
        """
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            meta=MessageMeta.from_dict(data.get("meta")),
        )


class ChatService:
    """Инкапсуляция одного чата/агента.

    Args:
        chat_id: уникальный идентификатор чата.
        kind: тип сессии (обычный чат / итоги / изолированный).
        model: идентификатор модели (может быть переопределён запросом).
        settings: параметры генерации по умолчанию.
        system_prompt: системный промпт чата (если есть).
    """

    def __init__(
        self,
        chat_id: str,
        kind: SessionKind = SessionKind.CHAT,
        model: Optional[str] = None,
        settings: Optional[GenerationSettings] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.id = chat_id
        self.kind = kind
        self.model = model
        self.settings = settings or GenerationSettings()
        self.system_prompt = system_prompt
        self.history: list[MessageRecord] = []
        self.busy = False
        self.created_at = time.time()
        self.last_active = time.time()

    def add_user_message(self, content: str) -> MessageRecord:
        """Добавляет пользовательское сообщение в историю.

        Args:
            content: текст сообщения.

        Returns:
            Добавленная запись :class:`MessageRecord`.
        """
        record = MessageRecord(role="user", content=content, created_at=time.time())
        self.history.append(record)
        self.last_active = time.time()
        return record

    def append_assistant_message(self, content: str, meta: MessageMeta) -> MessageRecord:
        """Добавляет ответ ассистента с метаданными в историю.

        Args:
            content: итоговый текст ответа.
            meta: метаданные ответа (время/токены/стоимость).

        Returns:
            Добавленная запись :class:`MessageRecord`.
        """
        record = MessageRecord(
            role="assistant",
            content=content,
            meta=meta,
            created_at=time.time(),
        )
        self.history.append(record)
        self.last_active = time.time()
        return record

    def seed_system_message(self, content: str) -> MessageRecord:
        """Делает первое сообщение чата системным промптом.

        Устанавливает ``system_prompt`` чата и добавляет запись истории
        с ролью ``system`` (для отображения/восстановления). Первый запрос
        к модели уходит как ``[system]``.

        Args:
            content: текст первого сообщения.

        Returns:
            Добавленная запись :class:`MessageRecord`.
        """
        self.system_prompt = content
        record = MessageRecord(role="system", content=content, created_at=time.time())
        self.history.append(record)
        self.last_active = time.time()
        return record

    def rollback_user_message(self) -> None:
        """Откатывает последнее пользовательское сообщение (при ошибке).

        Если откатывается первое сообщение-сид (``system``-запись),
        системный промпт чата также сбрасывается — повторная попытка
        снова станет системным промптом.
        """
        while self.history and self.history[-1].role == "user":
            self.history.pop()
        if len(self.history) == 1 and self.history[-1].role == "system":
            self.history.pop()
            self.system_prompt = None

    def to_openai_messages(self) -> list[dict]:
        """Собирает массив сообщений OpenAI из системного промпта и истории.

        Если первая запись истории — системный промпт (сид), он не
        вставляется повторно (уже присутствует в истории). Иначе промпт
        вставляется первым сообщением (summary/ephemeral).

        Returns:
            Список ``{"role", "content"}`` для запроса к апстриму.
        """
        messages: list[dict] = []
        if self.system_prompt and not (self.history and self.history[0].role == "system"):
            messages.append({"role": "system", "content": self.system_prompt})
        for record in self.history:
            messages.append({"role": record.role, "content": record.content})
        return messages

    def to_toon_context(self, title: Optional[str] = None) -> str:
        """Сериализует историю с метаданными в строку TOON.

        Используется для вложения содержимого чата (в т.ч. метаданных
        ответов) в промпты других агентов/чатов — например, в контекст
        чата «Подвести итоги» или при передаче состояния между агентами.

        Args:
            title: заголовок чата для идентификации в контексте.

        Returns:
            Строка TOON с историей и метаданными.
        """
        data: dict = {
            "title": title or self.id,
            "messages": [m.to_dict() for m in self.history],
        }
        return toon_encode(data)

    def build_meta(
        self,
        model: str,
        start_time: float,
        usage,
        finish_reason: Optional[str],
    ) -> MessageMeta:
        """Собирает метаданные ответа из счётчиков апстрима.

        Args:
            model: использованная модель.
            start_time: метка времени (time.time()) старта запроса.
            usage: объект :class:`Usage` из финального события апстрима.
            finish_reason: причина завершения (``stop``/``length``/…).

        Returns:
            Заполненные метаданные :class:`MessageMeta`.
        """
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        reasoning_tokens = usage.details.reasoning_tokens if usage and usage.details else None
        cost = (
            message_cost(model, prompt_tokens, completion_tokens)
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        return MessageMeta(
            model=model,
            elapsed_s=time.time() - start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            finish_reason=finish_reason,
        )

    async def stream_completion(
        self,
        runner,
        spec,
        generation_settings: Optional[GenerationSettings] = None,
        extra_system: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Выполняет стрим ответа и накапливает события для клиента.

        События отдаются как словари (для сериализации в SSE). Пофрагментные
        ``delta``-события движка наружу **не** передаются — текст накапливается,
        а наружу уходит только ``reasoning_start``/``reasoning_end`` и финальное
        ``done`` с полным ``content`` и метаданными.

        Args:
            runner: исполнитель :class:`StreamedCompletion`.
            spec: описание вызова апстрима.
            generation_settings: параметры для этого запроса (иначе — дефолтные).
            extra_system: дополнительный системный промпт, вставляемый
                перед остальными сообщениями (например, инструкция JSON-режима).

        Yields:
            Словари событий: ``{"type": ...}``.
        """
        settings = generation_settings or self.settings
        self.busy = True
        start_time = time.time()
        messages = self.to_openai_messages()
        if extra_system:
            messages.insert(0, {"role": "system", "content": extra_system})
        full = ""
        usage = None
        finish_reason = None

        try:
            async for event in runner.run(spec, messages, settings):
                if event.kind == "delta":
                    full += event.content
                elif event.kind == "reasoning_start":
                    yield {"type": "reasoning_start"}
                elif event.kind == "reasoning_end":
                    yield {"type": "reasoning_end", "content": event.content}
                elif event.kind == "usage":
                    usage = event.usage
                elif event.kind == "done":
                    finish_reason = event.finish_reason
                    if event.usage is not None:
                        usage = event.usage

            meta = self.build_meta(
                spec.model_label or spec.model, start_time, usage, finish_reason
            )
            self.append_assistant_message(full, meta)
            yield {
                "type": "done",
                "meta": meta.to_dict(),
                "content": full,
            }
        finally:
            self.busy = False

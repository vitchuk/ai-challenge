"""Сервис, инкапсулирующий отдельный чат/агента.

:class:`ChatService` хранит системный промпт, параметры генерации, модель
и историю сообщений (с метаданными каждого ответа LLM: время, токены,
стоимость). Он не зависит от HTTP и UI, поэтому в будущем может выступать
как «агент» в мультиагентной оркестрации (модератор + субагенты).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import AsyncIterator, Optional

from ..pricing import message_cost
from ..toon_codec import encode as toon_encode
from .generation import GenerationSettings

logger = logging.getLogger(__name__)

#: Системный промпт скрытого запроса саммаризации истории.
SUMMARIZER_SYSTEM_PROMPT = (
    "Ты — суммаризатор диалога. Сожми нижеприведённую переписку, сохранив "
    "все факты, решения, имена и незавершённые вопросы. Отвечай только саммари."
)

#: Рамка служебного сообщения с саммари в основном запросе.
SUMMARY_MESSAGE_PREFIX = "[Саммари начала диалога]"

#: Сколько накопленных саммари сворачивать в одно метасаммари.
COLLAPSE_SUMMARIES_AT = 3


class SessionKind(str, Enum):
    """Тип сессии (чата)."""

    CHAT = "chat"
    SUMMARY = "summary"
    EPHEMERAL = "ephemeral"


@dataclass
class RequestRecord:
    """Один запрос к LLM (основной ответ или скрытая саммаризация).

    Используется для графика расхода токенов по сообщениям
    (``prompt_tokens``/``completion_tokens``/``reasoning_tokens``) и
    счётчика «Сожжено токенов» (``prompt + completion``).
    """

    index: int
    kind: str  # "main" | "summary"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    created_at: float = 0.0
    persisted: bool = False

    def to_dict(self) -> dict:
        """Представляет запись как словарь (для API)."""
        return {
            "index": self.index,
            "kind": self.kind,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict, persisted: bool = True) -> "RequestRecord":
        """Восстанавливает запись из словаря (из БД).

        Args:
            data: словарь ``{"index", "kind", "prompt_tokens",
                "completion_tokens", "reasoning_tokens"}``.
            persisted: считать ли запись уже сохранённой в БД.

        Returns:
            Экземпляр :class:`RequestRecord`.
        """
        return cls(
            index=data.get("index", 0),
            kind=data.get("kind", "main"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            reasoning_tokens=data.get("reasoning_tokens"),
            created_at=data.get("created_at", 0.0),
            persisted=persisted,
        )


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
    summarized: Optional[bool] = None

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
            "summarized": self.summarized,
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
            summarized=data.get("summarized"),
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
        self.requests: list[RequestRecord] = []
        # Состояние чанковой саммаризации: накопленные саммари и число
        # уже сжатых чанков (по `requests_per_summary` обменов каждый).
        self.summary_items: list[str] = []
        self.summarized_chunks: int = 0
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

    def non_seed_messages(self) -> list[MessageRecord]:
        """Возвращает историю без начального системного сида.

        Под «сообщениями» для саммаризации понимаются именно эти записи.
        """
        if self.history and self.history[0].role == "system":
            return self.history[1:]
        return list(self.history)

    def _record_request(self, kind: str, usage) -> RequestRecord:
        """Добавляет запись о запросе к LLM (для графика/счётчика)."""
        details = usage.details if usage else None
        record = RequestRecord(
            index=len(self.requests) + 1,
            kind=kind,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            reasoning_tokens=details.reasoning_tokens if details else None,
            created_at=time.time(),
        )
        self.requests.append(record)
        return record

    async def _run_summarizer(
        self, runner, spec, settings, messages: list[dict]
    ) -> Optional[str]:
        """Выполняет один скрытый запрос саммаризации.

        Args:
            runner: исполнитель :class:`StreamedCompletion`.
            spec: описание вызова апстрима (та же модель, что и чат).
            settings: настройки генерации чата.
            messages: сообщения для суммаризатора (без системного промпта).

        Returns:
            Текст саммари или ``None`` при ошибке/пустом ответе (деградация).
        """
        # Служебный запрос идёт с параметрами чата, но без JSON-режима:
        # response_format управляет форматом ответа пользователю, а не саммари.
        summary_settings = replace(settings, response_format=None)
        request_messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            *messages,
        ]
        parts: list[str] = []
        usage = None
        try:
            async for event in runner.run(spec, request_messages, summary_settings):
                if event.kind == "delta":
                    parts.append(event.content)
                elif event.kind == "usage":
                    usage = event.usage
                elif event.kind == "done" and event.usage is not None:
                    usage = event.usage
        except Exception as exc:  # noqa: BLE001 - деградация вместо обрыва чата
            logger.warning("Саммаризация не удалась (деградация): %s", exc)
            return None

        text = "".join(parts).strip()
        if not text:
            logger.warning("Саммаризация вернула пустой текст (деградация)")
            return None

        self._record_request("summary", usage)
        return text

    def _chunk_size(self) -> int:
        """Размер чанка в запросах (из настроек саммаризации)."""
        context_summary = self.settings.context_summary
        return context_summary.requests_per_summary if context_summary else 0

    def _request_units(self) -> list[list[MessageRecord]]:
        """Делит переписку без сида на «запросы» — единицы чанкования.

        Запрос №1 — первое сообщение чата (системный сид): в переписке без
        сида ему соответствует только ответ ассистента (вводное сообщение),
        примыкающее к первому чанку; сам текст сида остаётся системным
        промптом и в саммари не дублируется. Каждый следующий запрос — это
        обмен «вопрос+ответ». Висящее пользовательское сообщение (ответ ещё
        не получен) в единицы не входит.

        Returns:
            Список единиц, каждая — список сообщений (1 для сид-ответа,
            2 для обычного обмена).
        """
        rest = self.non_seed_messages()
        start = 0
        while start < len(rest) and rest[start].role != "user":
            start += 1
        units: list[list[MessageRecord]] = [[m] for m in rest[:start]]
        i = start
        while (
            i + 1 < len(rest)
            and rest[i].role == "user"
            and rest[i + 1].role == "assistant"
        ):
            units.append([rest[i], rest[i + 1]])
            i += 2
        return units

    def _chunk_messages(self, chunk_index: int, size: int) -> list[dict]:
        """Сообщения чанка: ``size`` запросов (единиц) начиная с ``chunk_index``."""
        units = self._request_units()
        chunk = [
            m
            for unit in units[chunk_index * size : (chunk_index + 1) * size]
            for m in unit
        ]
        return [{"role": m.role, "content": m.content} for m in chunk]

    @staticmethod
    def _summaries_block(items: list[str]) -> str:
        """Пронумерованный блок саммари (для метасаммари и основного запроса)."""
        return "\n\n".join(
            f"Саммари {i + 1}:\n{text}" for i, text in enumerate(items)
        )

    async def ensure_summaries(self, runner, spec, settings) -> list[str]:
        """Досчитывает чанковые саммари и сворачивает их при накоплении.

        История делится на чанки по ``requests_per_summary`` завершённых
        обменов «вопрос+ответ». Перед основным запросом каждый ещё не сжатый
        завершённый чанк сжимается скрытым запросом, а при накоплении
        ``COLLAPSE_SUMMARIES_AT`` саммари они сворачиваются в одно
        метасаммари (цикл повторяется). При сбое деградирует: курсор не
        двигается, и сообщения соответствующего чанка уйдут в основной
        запрос без изменений.

        Args:
            runner: исполнитель :class:`StreamedCompletion`.
            spec: описание вызова апстрима (та же модель, что и чат).
            settings: настройки генерации чата.

        Returns:
            Текущий список саммари (пустой, если саммаризация выключена).
        """
        context_summary = settings.context_summary
        if context_summary is None or not context_summary.enabled:
            return []
        size = context_summary.requests_per_summary

        # Единица чанкования — «запрос» (сид-ответ либо обмен «вопрос+ответ»).
        units = self._request_units()
        complete_chunks = len(units) // size

        while self.summarized_chunks < complete_chunks:
            chunk_index = self.summarized_chunks
            text = await self._run_summarizer(
                runner, spec, settings, self._chunk_messages(chunk_index, size)
            )
            if text is None:
                break  # деградация: чанк и последующие уйдут вербатим
            self.summarized_chunks += 1
            self.summary_items.append(text)
            if len(self.summary_items) >= COLLAPSE_SUMMARIES_AT:
                meta = await self._run_summarizer(
                    runner,
                    spec,
                    settings,
                    [{"role": "user", "content": self._summaries_block(self.summary_items)}],
                )
                if meta is not None:
                    self.summary_items = [meta]
                # при сбое свёртки саммари остаются и будут свёрнуты позже

        return list(self.summary_items)

    def build_request_messages(
        self,
        summary_items: Optional[list[str]] = None,
        extra_system: Optional[str] = None,
    ) -> list[dict]:
        """Собирает массив сообщений для основного запроса.

        Без саммаризации — обычная история (см. :meth:`to_openai_messages`).
        С саммаризацией — ``[системный промпт] + [рамка со всеми саммари] +
        [сообщения после курсора сжатых чанков без изменений]``.

        Args:
            summary_items: текущие саммари (``None`` — саммаризация выключена).
            extra_system: дополнительный системный промпт (JSON-режим),
                вставляется самым первым.

        Returns:
            Список ``{"role", "content"}`` для запроса к апстриму.
        """
        if summary_items is None:
            messages = self.to_openai_messages()
        else:
            messages = []
            if self.history and self.history[0].role == "system":
                messages.append(
                    {"role": "system", "content": self.history[0].content}
                )
            elif self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            if summary_items:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{SUMMARY_MESSAGE_PREFIX}\n"
                            f"{self._summaries_block(summary_items)}"
                        ),
                    }
                )
            # Курсор: сколько сообщений покрыто сжатыми единицами (запросами).
            units = self._request_units()
            covered = sum(
                len(unit)
                for unit in units[: self.summarized_chunks * self._chunk_size()]
            )
            messages.extend(
                {"role": m.role, "content": m.content}
                for m in self.non_seed_messages()[covered:]
            )

        if extra_system:
            messages.insert(0, {"role": "system", "content": extra_system})
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
        summary_items: Optional[list[str]] = None,
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
            summary_items: накопленные саммари (если включена саммаризация).

        Yields:
            Словари событий: ``{"type": ...}``.
        """
        settings = generation_settings or self.settings
        self.busy = True
        start_time = time.time()
        messages = self.build_request_messages(
            summary_items=summary_items, extra_system=extra_system
        )
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
            if summary_items:
                meta.summarized = True
            self.append_assistant_message(full, meta)
            self._record_request("main", usage)
            yield {
                "type": "done",
                "meta": meta.to_dict(),
                "content": full,
            }
        finally:
            self.busy = False

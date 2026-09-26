"""Сервис, инкапсулирующий отдельный чат/агента.

:class:`ChatService` хранит системный промпт, параметры генерации, модель
и историю сообщений (с метаданными каждого ответа LLM: время, токены,
стоимость). Он не зависит от HTTP и UI, поэтому в будущем может выступать
как «агент» в мультиагентной оркестрации (модератор + субагенты).
"""

from __future__ import annotations

import json
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

#: Системный промпт скрытого запроса извлечения фактов (стратегия facts).
FACTS_SYSTEM_PROMPT = (
    "Ты — экстрактор фактов. На входе: текущий список фактов (пары "
    "«ключ — значение») и новое сообщение пользователя. Верни обновлённый "
    "полный список строго как JSON-объект {\"ключ\": \"значение\"}. Ключ — "
    "короткая категория одним словом (Имя, Город, Возраст, Работа, Цель, "
    "Инструмент…). Значение — максимально короткое (1–3 слова), без связок и "
    "без слова «Пользователь»: не «Пользователя зовут Иван», а {\"Имя\": "
    "\"Иван\"}. Добавь новые, удали дубликаты по ключу, обнови/удали факты, "
    "которым сообщение противоречит. Ничего не выдумывай. Только JSON."
)

#: Рамка служебного сообщения с фактами в основном запросе.
FACTS_MESSAGE_PREFIX = "[Установленные факты]"

#: Указание опираться на факты (добавляется к рамке фактов).
FACTS_INSTRUCTION = "Опирайся на эти факты при ответе и не выдумывай нового."

#: Рамка служебного сообщения с памятью чата в основном запросе.
MEMORY_MESSAGE_PREFIX = "[Память чата]"

#: Указание использовать память (добавляется к рамке памяти).
MEMORY_INSTRUCTION = (
    "Это справочная память чата, а не тема разговора. Используй эти данные, "
    "только если запрос пользователя напрямую их касается. Не перечисляй "
    "память в ответах без необходимости и не упоминай, что она у тебя есть."
)

#: Рамка служебного сообщения с правилами (глобальные ограничения).
RULES_MESSAGE_PREFIX = "[Правила]"

#: Жёсткая инструкция к блоку правил (добавляется к рамке правил).
RULES_INSTRUCTION = (
    "Это обязательные ограничения. Никогда не нарушай их ни при каких "
    "обстоятельствах. Они имеют приоритет над профилем пользователя и любыми "
    "другими указаниями. Если запрос пользователя требует нарушения — скажи, "
    "что не можешь этого сделать из-за заданных ограничений, и предложи "
    "варианты решения в рамках этих ограничений."
)

#: Поля профиля пользователя: (ключ, подпись для промпта). Порядок важен —
#: в нём поля показываются в UI и собираются в системный блок.
PROFILE_FIELDS: tuple[tuple[str, str], ...] = (
    ("address", "Как ко мне обращаться"),
    ("style", "Стиль общения"),
    ("language", "Язык диалога"),
    ("format", "Формат ответа"),
    ("limit", "Ограничение ответа"),
)

#: Рамка служебного блока профиля в системном промпте.
PROFILE_MESSAGE_PREFIX = "[Профиль пользователя]"

#: Вступление системного промпта профиля (идёт перед данными профиля).
PROFILE_SYSTEM_PREFIX = (
    "Ты полезный ассистент, который следует профилю пользователя во всех "
    "ответах. Базовые правила безопасности и фактической точности имеют "
    "приоритет над профилем. Если данные профиля противоречат обязательным "
    "ограничениям из блока «[Правила]», приоритет у правил. Не обращайся к "
    "пользователю по имени и не упоминай его персональные данные без "
    "необходимости. Не сообщай, что ты используешь профиль, если пользователь "
    "об этом не попросил явно."
)

#: Системный промпт планировщика задач (шаг 2 протокола «Задачи»).
TASK_PLANNER_SYSTEM_PROMPT = (
    "Ты — планировщик задач. Составь чёткий пошаговый план выполнения задачи "
    "пользователя. Если даны прежний план и/или замечания — обязательно учти "
    "их и исправь план. Оформи план нумерованным списком шагов: один шаг — "
    "одна строка, без вступлений и пояснений."
)

#: Системный промпт исполнителя задач (шаг 3 — выполнение всего плана).
TASK_EXECUTOR_SYSTEM_PROMPT = (
    "Ты — исполнитель задачи. Выполни задачу пользователя строго по "
    "подтверждённому плану. Отвечай только результатом, без вступлений."
)

#: Системный промпт исполнителя одного шага (пошаговый режим).
TASK_STEP_EXECUTOR_SYSTEM_PROMPT = (
    "Ты — исполнитель задачи. Выполни ТОЛЬКО указанный шаг плана, не выполняя "
    "остальные шаги. Опирайся на результаты уже выполненных шагов. Отвечай "
    "только результатом этого шага, без вступлений. Перепрыгивать шаги вперёд "
    "нельзя: если просят выполнить сразу несколько шагов или шаг не по "
    "порядку — откажи. Переход к следующему шагу и возврат к предыдущим "
    "управляются системой — выполняй только указанный шаг."
)

#: Системный промпт переделки одного шага (доработка на этапе шага).
TASK_STEP_REVISE_SYSTEM_PROMPT = (
    "Ты — исполнитель задачи. Пользователь дал замечания к результату "
    "указанного шага. Переделай ТОЛЬКО этот шаг с учётом замечаний, не "
    "выполняя остальные шаги. Опирайся на результаты предыдущих шагов. "
    "Отвечай только исправленным результатом этого шага, без вступлений. "
    "Если просят перепрыгнуть несколько шагов вперёд — откажи. Просьбы "
    "вернуться к предыдущему шагу или перейти к следующему легитимны (ими "
    "управляет система); если шаг не указан явно — уточни его номер и "
    "переделай текущий."
)

#: Этапы задачи (машина состояний протокола «Задачи»).
TASK_STAGES = (
    "input",
    "plan_review",
    "mode_select",
    "step_review",
    "review",
    "done",
)

#: Максимум раундов tool calling в одном запросе (последний — без инструментов).
MAX_TOOL_ROUNDS = 8


class SessionKind(str, Enum):
    """Тип сессии (чата)."""

    CHAT = "chat"
    TASK = "task"
    SUMMARY = "summary"
    EPHEMERAL = "ephemeral"


@dataclass
class RequestRecord:
    """Один запрос к LLM (основной ответ, скрытая саммаризация или факты).

    Используется для графика расхода токенов по сообщениям
    (``prompt_tokens``/``completion_tokens``/``reasoning_tokens``) и
    счётчика «Сожжено токенов» (``prompt + completion``).
    """

    index: int
    kind: str  # "main" | "summary" | "facts"
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
class MemoryStore:
    """Вкладка «Памяти» чата: именованное хранилище пар «ключ — значение».

    Attributes:
        id: идентификатор вкладки (клиентский или сгенерированный сервером).
        name: название вкладки.
        persistent: сохраняется ли содержимое вкладки в БД.
        items: пары ``[[ключ, значение], …]``.
    """

    id: str
    name: str
    persistent: bool = False
    items: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Представляет вкладку как словарь (для API/БД)."""
        return {
            "id": self.id,
            "name": self.name,
            "persistent": self.persistent,
            "items": [list(item) for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryStore":
        """Восстанавливает вкладку из словаря (из БД/API)."""
        items = data.get("items")
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            persistent=bool(data.get("persistent", False)),
            items=[list(item) for item in items] if isinstance(items, list) else [],
        )


@dataclass
class RuleStore:
    """Вкладка «Правил» — именованное хранилище пар «ключ — значение».

    Глобальная сущность (не привязана к чату): все вкладки всегда
    персистентны и действуют во всех запросах к LLM.

    Attributes:
        id: идентификатор вкладки (клиентский или сгенерированный сервером).
        name: название вкладки.
        items: пары ``[[ключ, значение], …]``.
    """

    id: str
    name: str
    items: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Представляет вкладку как словарь (для API/БД)."""
        return {
            "id": self.id,
            "name": self.name,
            "items": [list(item) for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuleStore":
        """Восстанавливает вкладку из словаря (из БД/API)."""
        items = data.get("items")
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            items=[list(item) for item in items] if isinstance(items, list) else [],
        )


@dataclass
class Profile:
    """Профиль пользователя — глобальная сущность (не привязана к чату).

    Хранит название и фиксированный набор полей (см. :data:`PROFILE_FIELDS`),
    подставляемых в системный промпт основного запроса. Активный профиль
    один на всё приложение.

    Attributes:
        id: идентификатор профиля.
        name: название профиля.
        fields: словарь ``{ключ: значение}`` (только ключи из
            :data:`PROFILE_FIELDS`).
    """

    id: str
    name: str
    fields: dict = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Заполнены ли название и все поля профиля."""
        if not self.name.strip():
            return False
        return all(str(self.fields.get(key, "")).strip() for key, _ in PROFILE_FIELDS)

    def to_dict(self) -> dict:
        """Представляет профиль как словарь (для API/БД)."""
        return {
            "id": self.id,
            "name": self.name,
            "fields": {key: str(self.fields.get(key, "")) for key, _ in PROFILE_FIELDS},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        """Восстанавливает профиль из словаря (из БД/API)."""
        fields = data.get("fields")
        fields = fields if isinstance(fields, dict) else {}
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            fields={key: str(fields.get(key, "")) for key, _ in PROFILE_FIELDS},
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
        # уже сжатых чанков (по `context_strategy.n` запросов каждый).
        self.summary_items: list[str] = []
        self.summarized_chunks: int = 0
        # Канонические факты (стратегия facts).
        self.facts: list[str] = []
        # Память чата: именованные хранилища пар «ключ — значение».
        self.memory_stores: list[MemoryStore] = []
        # Состояние задачи (протокол «Задачи»): этап, текущий план и результат.
        self.task_stage: Optional[str] = "input" if kind == SessionKind.TASK else None
        self.task_plan: Optional[str] = None
        self.task_result: Optional[str] = None
        # Пошаговый режим: разобранные шаги плана и результаты выполненных.
        self.task_steps: list[str] = []
        self.task_step_results: list[str] = []
        # Журнал промптов (не персистится): выполненные запросы к LLM для
        # вкладки «Логи» — итоговые сообщения и ответ модели.
        self.prompt_log_queue: list[dict] = []
        # Метаданные ответвлённого чата (снапшот): родитель и своё название.
        self.parent_id: Optional[str] = None
        self.title: Optional[str] = None
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

    async def _run_hidden(
        self,
        kind: str,
        system_prompt: str,
        runner,
        spec,
        settings,
        messages: list[dict],
    ) -> Optional[str]:
        """Выполняет один скрытый служебный запрос к модели.

        Служебные запросы (саммаризация, извлечение фактов) идут с
        параметрами чата, но без JSON-режима: ``response_format`` управляет
        форматом ответа пользователю. При ошибке апстрима деградирует
        (возвращает ``None``), не обрывая чат; при успехе пишет запись
        ``kind`` в журнал запросов.

        Args:
            kind: тип записи в журнале (``summary``/``facts``).
            system_prompt: системный промпт служебного запроса.
            runner: исполнитель :class:`StreamedCompletion`.
            spec: описание вызова апстрима (та же модель, что и чат).
            settings: настройки генерации чата.
            messages: сообщения служебного запроса (без системного промпта).

        Returns:
            Текст ответа или ``None`` при ошибке/пустом ответе.
        """
        hidden_settings = replace(settings, response_format=None)
        request_messages = [{"role": "system", "content": system_prompt}, *messages]
        parts: list[str] = []
        usage = None
        try:
            async for event in runner.run(spec, request_messages, hidden_settings):
                if event.kind == "delta":
                    parts.append(event.content)
                elif event.kind == "usage":
                    usage = event.usage
                elif event.kind == "done" and event.usage is not None:
                    usage = event.usage
        except Exception as exc:  # noqa: BLE001 - деградация вместо обрыва чата
            logger.warning("Служебный запрос (%s) не удался (деградация): %s", kind, exc)
            return None

        text = "".join(parts).strip()
        if not text:
            logger.warning("Служебный запрос (%s) вернул пустой текст (деградация)", kind)
            return None

        self._record_request(kind, usage)
        self._queue_prompt(kind, request_messages, text, spec, usage)
        return text

    async def _run_summarizer(self, runner, spec, settings, messages: list[dict]) -> Optional[str]:
        """Скрытый запрос саммаризации (обёртка :meth:`_run_hidden`)."""
        return await self._run_hidden(
            "summary", SUMMARIZER_SYSTEM_PROMPT, runner, spec, settings, messages
        )

    def _strategy(self):
        """Настройки стратегии управления контекстом (только для chat-чатов)."""
        if self.kind != SessionKind.CHAT:
            return None
        return self.settings.context_strategy

    def _chunk_size(self) -> int:
        """Размер чанка в запросах (из настроек стратегии summarize)."""
        strategy = self._strategy()
        if strategy is None or strategy.strategy != "summarize":
            return 0
        return strategy.n

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

    @staticmethod
    def _facts_block(items: list) -> str:
        """Пронумерованный блок фактов (пары «ключ: значение») с указанием."""
        lines = "\n".join(
            f"{i + 1}. {ChatService._fact_line(item)}"
            for i, item in enumerate(items)
        )
        return f"{FACTS_MESSAGE_PREFIX}\n{lines}\n\n{FACTS_INSTRUCTION}"

    @staticmethod
    def _fact_line(item) -> str:
        """Строка факта: пара → «ключ: значение», легаси-строка → как есть."""
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return f"{item[0]}: {item[1]}"
        return str(item)

    def memory_frame(self) -> Optional[str]:
        """Служебный блок «Память чата» (или ``None``, если данных нет).

        Только для обычных чатов: секции по вкладкам памяти с парами
        «ключ: значение» и указанием использовать данные.

        Returns:
            Текст служебного сообщения или ``None``.
        """
        if self.kind != SessionKind.CHAT:
            return None
        sections = []
        for store in self.memory_stores:
            lines = "\n".join(
                f"{item[0]}: {item[1]}"
                for item in store.items
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
            if lines:
                sections.append(f"## {store.name}\n{lines}")
        if not sections:
            return None
        return (
            f"{MEMORY_MESSAGE_PREFIX}\n"
            + "\n\n".join(sections)
            + f"\n\n{MEMORY_INSTRUCTION}"
        )

    def _history_exchanges(self) -> list[list[MessageRecord]]:
        """Обмены по всей истории, включая сид.

        Обмен — реплика пользователя (или системный сид) и следующий за ней
        ответ ассистента. Обмен №1 — сид и ответ на него. Висящее (ещё не
        отвеченное) сообщение в обмены не входит.

        Returns:
            Список обменов, каждый — список сообщений (всегда 2).
        """
        exchanges: list[list[MessageRecord]] = []
        i = 0
        history = self.history
        while (
            i + 1 < len(history)
            and history[i].role in ("user", "system")
            and history[i + 1].role == "assistant"
        ):
            exchanges.append([history[i], history[i + 1]])
            i += 2
        return exchanges

    def _window_messages(self, count: int) -> list[MessageRecord]:
        """Последние ``count`` обменов истории (сид — равноправный участник).

        Обмен №1 — системный сид и ответ на него. Если обменов не больше
        ``count`` — возвращается вся история целиком; иначе — только последние
        ``count`` обменов (выпавшие, включая сид, не уходят вовсе) плюс висящее
        сообщение. Роли записей сохраняются как в истории.

        Args:
            count: размер окна в обменах.

        Returns:
            Список записей истории для контекста основного запроса.
        """
        exchanges = self._history_exchanges()
        covered = len(exchanges) * 2
        pending = self.history[covered:]
        if len(exchanges) <= count:
            return list(self.history)
        window = [m for exchange in exchanges[-count:] for m in exchange]
        return window + pending

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
        strategy = self._strategy()
        if strategy is None or strategy.strategy != "summarize":
            return []
        size = strategy.n

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

    async def extract_facts(
        self, runner, spec, settings, user_message: str
    ) -> Optional[list]:
        """Обновляет канонические факты скрытым запросом (стратегия facts).

        На входе — текущий список фактов (пары «ключ — значение») и новое
        сообщение пользователя; модель возвращает полный обновлённый
        JSON-объект. Парсинг терпим к Markdown-оградам; при сбое парсинга
        старый список сохраняется (деградация). Запрос идёт с параметрами
        чата, кроме ``response_format``.

        Args:
            runner: исполнитель :class:`StreamedCompletion`.
            spec: описание вызова апстрима (та же модель, что и чат).
            settings: настройки генерации чата.
            user_message: новое сообщение пользователя.

        Returns:
            Обновлённый список фактов или ``None`` (не facts / ошибка / сбой).
        """
        strategy = self._strategy()
        if strategy is None or strategy.strategy != "facts":
            return None

        facts_text = (
            "\n".join(f"- {self._fact_line(fact)}" for fact in self.facts)
            if self.facts
            else "(список пуст)"
        )
        content = (
            f"Установленные факты:\n{facts_text}\n\n"
            f"Новое сообщение пользователя:\n{user_message}"
        )
        text = await self._run_hidden(
            "facts",
            FACTS_SYSTEM_PROMPT,
            runner,
            spec,
            settings,
            [{"role": "user", "content": content}],
        )
        if text is None:
            return None
        parsed = self._parse_facts(text)
        if parsed is None:
            logger.warning("Не удалось разобрать факты (деградация): %r", text[:200])
            return None
        self.facts = parsed
        return list(self.facts)

    @staticmethod
    def _load_json(text: str):
        """Пытается распарсить JSON: целиком, затем первую ``{…}``/``[…]``."""
        candidates = [text, ChatService._extract_json(text)]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Подстрока от первой ``{``/``[`` до последней ``}``/``]``."""
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
        if not starts or not ends:
            return ""
        start, end = min(starts), max(ends)
        return text[start : end + 1] if end > start else ""

    @staticmethod
    def _parse_facts(text: str) -> Optional[list[list[str]]]:
        """Разбирает ответ модели в список пар «ключ — значение».

        Принимает JSON-объект ``{"ключ": "значение"}``, массив объектов
        ``{"key","value"}`` и массив пар ``[ключ, значение]``; терпимо к
        Markdown-оградам. Дедупликация по ключу (позиция первого вхождения,
        значение последнего). Пустой список — валидный результат. ``None`` —
        формат не распознан (деградация; в т.ч. прежний массив строк).

        Args:
            text: ответ модели.

        Returns:
            Список пар ``[[ключ, значение], …]`` или ``None``.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
        data = ChatService._load_json(cleaned)
        if data is None:
            return None

        if isinstance(data, dict):
            raw_items = list(data.items())
        elif isinstance(data, list):
            raw_items = []
            for item in data:
                if isinstance(item, dict):
                    raw_items.append((item.get("key"), item.get("value")))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    raw_items.append((item[0], item[1]))
                else:
                    return None
        else:
            return None

        pairs: list[list[str]] = []
        positions: dict[str, int] = {}
        for key, value in raw_items:
            if key is None or value is None:
                continue
            key_str, value_str = str(key).strip(), str(value).strip()
            if not key_str or not value_str:
                continue
            if key_str in positions:
                pairs[positions[key_str]][1] = value_str
            else:
                positions[key_str] = len(pairs)
                pairs.append([key_str, value_str])
        return pairs

    def _system_message(self) -> Optional[dict]:
        """Системное сообщение чата (сид из истории или ``system_prompt``)."""
        if self.history and self.history[0].role == "system":
            return {"role": "system", "content": self.history[0].content}
        if self.system_prompt:
            return {"role": "system", "content": self.system_prompt}
        return None

    def _summarize_messages(self, summary_items: list[str]) -> list[dict]:
        """Основной запрос для стратегии summarize (саммари + вербатим-хвост)."""
        messages: list[dict] = []
        system = self._system_message()
        if system:
            messages.append(system)
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
        units = self._request_units()
        covered = sum(
            len(unit) for unit in units[: self.summarized_chunks * self._chunk_size()]
        )
        messages.extend(
            {"role": m.role, "content": m.content}
            for m in self.non_seed_messages()[covered:]
        )
        return messages

    def _windowed_messages(self, count: int, facts: bool = False) -> list[dict]:
        """Основной запрос для стратегий sliding/facts (окно + новое).

        Сид не добавляется принудительно: он подчиняется окну и уходит как
        обычная запись истории (с ролью ``system``), а если выпал — не уходит
        вовсе. Блок фактов стратегии ``facts`` присутствует всегда и ставится
        после ведущего system-сообщения (если оно есть в окне), иначе первым.
        """
        messages = [
            {"role": m.role, "content": m.content}
            for m in self._window_messages(count)
        ]
        if facts and self.facts:
            index = 1 if messages and messages[0]["role"] == "system" else 0
            messages.insert(
                index, {"role": "user", "content": self._facts_block(self.facts)}
            )
        return messages

    def build_request_messages(
        self,
        summary_items: Optional[list[str]] = None,
        extra_system: Optional[str] = None,
    ) -> list[dict]:
        """Собирает массив сообщений для основного запроса.

        Способ сборки зависит от стратегии контекста чата (только
        ``kind=chat``): ``none``/``branching`` — вся история, ``summarize`` —
        саммари чанков, ``sliding`` — окно последних реплик, ``facts`` —
        служебный блок фактов + окно реплик. Дополнительно (для обычных
        чатов) в начало после system-сообщения подставляется служебный блок
        «Память чата». Блок правил (глобальные ограничения) приходит внутри
        ``extra_system`` (ведущее system-сообщение).

        Args:
            summary_items: текущие саммари (для стратегии summarize).
            extra_system: дополнительный системный промпт (профиль, правила,
                JSON-режим), вставляется самым первым.

        Returns:
            Список ``{"role", "content"}`` для запроса к апстриму.
        """
        strategy = self._strategy()
        name = strategy.strategy if strategy else "none"

        if name == "summarize" and summary_items is not None:
            messages = self._summarize_messages(summary_items)
        elif name == "sliding" and strategy is not None:
            messages = self._windowed_messages(strategy.n)
        elif name == "facts" and strategy is not None:
            messages = self._windowed_messages(strategy.k, facts=True)
        else:
            messages = self.to_openai_messages()

        # Память чата — служебный блок сразу после system-сообщения.
        memory = self.memory_frame()
        if memory:
            index = 1 if messages and messages[0]["role"] == "system" else 0
            messages.insert(index, {"role": "user", "content": memory})

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
        messages: Optional[list[dict]] = None,
        prompt_kind: str = "main",
        tools: Optional[list[dict]] = None,
        tool_executor=None,
    ) -> AsyncIterator[dict]:
        """Выполняет стрим ответа и накапливает события для клиента.

        События отдаются как словари (для сериализации в SSE). Пофрагментные
        ``delta``-события движка наружу **не** передаются — текст накапливается,
        а наружу уходит только ``reasoning_start``/``reasoning_end``, события
        вызова инструментов (``tool_call``/``tool_result``) и финальное ``done``
        с полным ``content`` и метаданными.

        Если переданы ``tools`` и ``tool_executor``, запрос идёт в режиме
        tool calling: модель может запросить вызовы инструментов, они
        исполняются, результаты возвращаются модели, и так до
        :data:`MAX_TOOL_ROUNDS` раундов. Токены всех раундов суммируются в
        метаданные ответа; каждый раунд пишется в журнал промптов и график.
        Обмены с инструментами в историю чата не персистятся — там остаётся
        только итоговый текст.

        Args:
            runner: исполнитель :class:`StreamedCompletion`.
            spec: описание вызова апстрима.
            generation_settings: параметры для этого запроса (иначе — дефолтные).
            extra_system: дополнительный системный промпт (профиль, правила,
                JSON-режим), вставляемый перед остальными сообщениями.
            summary_items: накопленные саммари (если включена саммаризация).
            messages: готовый список сообщений запроса (протокол «Задачи»);
                при передаче заменяет сборку по истории/стратегии.
            prompt_kind: метка запроса для журнала промптов (``main`` или
                действие задачи).
            tools: OpenAI-описания доступных инструментов (или ``None``).
            tool_executor: async-исполнитель инструмента
                ``(name, args) -> (text, is_error)``.

        Yields:
            Словари событий: ``{"type": ...}``.
        """
        # Локальный импорт: providers импортирует services (цикл на уровне модулей).
        from ..providers.base import Usage, UsageDetails

        settings = generation_settings or self.settings
        self.busy = True
        start_time = time.time()
        base_messages = (
            messages
            if messages is not None
            else self.build_request_messages(
                summary_items=summary_items, extra_system=extra_system
            )
        )
        # Инструменты доступны только с исполнителем и без строгого JSON-режима.
        active_tools = (
            tools
            if (tools and tool_executor and settings.response_format is None)
            else None
        )

        full_parts: list[str] = []
        total_prompt = 0
        total_completion = 0
        total_reasoning = 0
        usage_seen = False
        finish_reason = None
        working = list(base_messages)

        try:
            for round_index in range(MAX_TOOL_ROUNDS + 1):
                # Последний раунд идёт без инструментов — вынуждаем текстовый ответ.
                last_round = round_index >= MAX_TOOL_ROUNDS
                round_tools = None if last_round else active_tools
                round_text = ""
                round_usage = None
                round_finish = None
                round_calls = None

                async for event in runner.run(spec, working, settings, round_tools):
                    if event.kind == "delta":
                        round_text += event.content
                    elif event.kind == "reasoning_start":
                        yield {"type": "reasoning_start"}
                    elif event.kind == "reasoning_end":
                        yield {"type": "reasoning_end", "content": event.content}
                    elif event.kind == "usage":
                        round_usage = event.usage
                    elif event.kind == "done":
                        round_finish = event.finish_reason
                        round_calls = event.tool_calls
                        if event.usage is not None:
                            round_usage = event.usage

                if round_text.strip():
                    full_parts.append(round_text)
                if round_usage is not None:
                    usage_seen = True
                    total_prompt += round_usage.prompt_tokens or 0
                    total_completion += round_usage.completion_tokens or 0
                    details = round_usage.details
                    if details is not None and details.reasoning_tokens:
                        total_reasoning += details.reasoning_tokens
                finish_reason = round_finish
                self._record_request("main", round_usage)
                self._queue_prompt(prompt_kind, working, round_text, spec, round_usage)

                if (
                    not last_round
                    and active_tools
                    and round_calls
                    and round_finish == "tool_calls"
                ):
                    for call in round_calls:
                        yield {
                            "type": "tool_call",
                            "tool": call["name"],
                            "args": self._parse_tool_arguments(call.get("arguments")),
                        }
                    working = [
                        *working,
                        {
                            "role": "assistant",
                            "content": round_text,
                            "tool_calls": [
                                {
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["name"],
                                        "arguments": call.get("arguments") or "{}",
                                    },
                                }
                                for call in round_calls
                            ],
                        },
                    ]
                    for call in round_calls:
                        text, is_error = await self._execute_tool(tool_executor, call)
                        yield {
                            "type": "tool_result",
                            "tool": call["name"],
                            "content": text,
                            "is_error": is_error,
                        }
                        working.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": text,
                            }
                        )
                    continue
                break

            aggregate = (
                Usage(
                    prompt_tokens=total_prompt or None,
                    completion_tokens=total_completion or None,
                    details=UsageDetails(reasoning_tokens=total_reasoning or None),
                )
                if usage_seen
                else None
            )
            full = "\n\n".join(part for part in full_parts if part.strip())
            meta = self.build_meta(
                spec.model_label or spec.model, start_time, aggregate, finish_reason
            )
            if summary_items:
                meta.summarized = True
            self.append_assistant_message(full, meta)
            yield {
                "type": "done",
                "meta": meta.to_dict(),
                "content": full,
            }
        finally:
            self.busy = False

    @staticmethod
    def _parse_tool_arguments(raw) -> dict:
        """Разбирает аргументы вызова инструмента из строки/словаря JSON."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    async def _execute_tool(tool_executor, call: dict) -> tuple[str, bool]:
        """Исполняет один вызов инструмента, не обрывая чат при ошибке."""
        arguments = ChatService._parse_tool_arguments(call.get("arguments"))
        try:
            result = await tool_executor(call["name"], arguments)
        except Exception as exc:  # noqa: BLE001 - ошибка уходит модели как результат
            logger.warning("Инструмент %s завершился ошибкой: %s", call["name"], exc)
            return f"Ошибка вызова инструмента: {exc}", True
        if isinstance(result, tuple) and len(result) == 2:
            return str(result[0]), bool(result[1])
        return str(result), False

    def _queue_prompt(self, kind: str, messages: list[dict], response: str, spec, usage) -> None:
        """Ставит в журнал промптов запись о выполненном запросе к LLM."""
        details = usage.details if usage else None
        self.prompt_log_queue.append(
            {
                "type": "prompt",
                "kind": kind,
                "messages": [
                    {"role": m.get("role"), "content": m.get("content")} for m in messages
                ],
                "response": response,
                "model": spec.model_label or spec.model,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "reasoning_tokens": details.reasoning_tokens if details else None,
            }
        )

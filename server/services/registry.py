"""Реестр сессий (чатов): создание, хранение, удаление, сводный контекст.

Реестр владеет всеми живыми :class:`ChatService` сервера. При наличии
хранилища (:class:`SessionStore`) сессии и история персистентны и
восстанавливаются при перезапуске сервера. Для чата «Подвести итоги»
строит TOON-контекст из содержимого остальных сессий.
"""

from __future__ import annotations

import copy
import secrets
from typing import Optional

from ..config import Settings
from ..toon_codec import encode as toon_encode
from .chat_service import ChatService, MessageRecord, SessionKind
from .storage import SessionStore

SUMMARY_CONTEXT_PROMPT = (
    "У тебя есть доступ к содержимому всех открытых чатов. "
    "Используй его при ответе на вопрос пользователя."
)


class SessionRegistry:
    """Реестр живых сессий сервера.

    Args:
        settings: настройки сервера.
        store: хранилище SQLite; если задано, сессии персистентны.
    """

    def __init__(self, settings: Optional[Settings] = None, store: Optional[SessionStore] = None) -> None:
        self._settings = settings or Settings()
        self._store = store
        self._sessions: dict[str, ChatService] = {}
        self._active_id: Optional[str] = None
        if self._store is not None:
            self._active_id = self._store.get_state("active_session")

    def create(
        self,
        kind: SessionKind = SessionKind.CHAT,
        model: Optional[str] = None,
        settings=None,
        system_prompt: Optional[str] = None,
    ) -> ChatService:
        """Создаёт и регистрирует новую сессию.

        Args:
            kind: тип сессии.
            model: модель по умолчанию.
            settings: параметры генерации по умолчанию.
            system_prompt: системный промпт.

        Returns:
            Новый экземпляр :class:`ChatService`.
        """
        chat_id = secrets.token_hex(8)
        while chat_id in self._sessions:
            chat_id = secrets.token_hex(8)
        service = ChatService(
            chat_id=chat_id,
            kind=kind,
            model=model,
            settings=settings,
            system_prompt=system_prompt,
        )
        self._sessions[chat_id] = service
        if self._store is not None:
            self._store.save_session(service)
        return service

    def restore(self) -> int:
        """Восстанавливает сессии из хранилища (если оно задано).

        Returns:
            Число восстановленных сессий.
        """
        if self._store is None:
            return 0
        loaded = self._store.load_all()
        for service in loaded:
            self._sessions[service.id] = service
        return len(loaded)

    def close(self) -> None:
        """Закрывает хранилище (если оно есть)."""
        if self._store is not None:
            self._store.close()

    def get(self, session_id: str) -> Optional[ChatService]:
        """Возвращает сессию по идентификатору (или ``None``)."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Удаляет сессию из памяти и хранилища.

        Если удаляется активная сессия — маркер активности очищается.

        Args:
            session_id: идентификатор сессии.

        Returns:
            ``True``, если сессия существовала и удалена.
        """
        removed = self._sessions.pop(session_id, None)
        if self._store is not None:
            self._store.delete_session(session_id)
            if self.get_active() == session_id:
                self._store.delete_state("active_session")
                self._active_id = None
        return removed is not None

    def list_sessions(self) -> list[ChatService]:
        """Возвращает список всех живых сессий."""
        return list(self._sessions.values())

    def set_active(self, session_id: str) -> bool:
        """Отмечает сессию как активную (открытую у пользователя).

        Args:
            session_id: идентификатор сессии.

        Returns:
            ``True``, если сессия существует и отмечена.
        """
        if session_id not in self._sessions:
            return False
        self._active_id = session_id
        if self._store is not None:
            self._store.set_state("active_session", session_id)
        return True

    def get_active(self) -> Optional[str]:
        """Возвращает id активной сессии (или ``None``).

        Значение валидируется: если отмеченная сессия уже удалена —
        возвращается ``None``.
        """
        if self._active_id is not None and self._active_id in self._sessions:
            return self._active_id
        return None

    def remember_pair(self, chat: ChatService) -> None:
        """Персистит последнюю пару user+assistant (если есть хранилище).

        Вызывается после успешного завершения стрима ответа. Заодно
        обновляет модель сессии и метку активности (upsert строки сессии).
        Первая пара может начинаться с ``system``-записи (первое сообщение
        чата, ставшее системным промптом).

        Args:
            chat: сессия, в историю которой только что добавлены
                пользовательское сообщение и ответ ассистента.
        """
        if self._store is None:
            return
        if chat.kind == SessionKind.EPHEMERAL:
            # Временные сессии (/optimize-prompt) не персистятся целиком:
            # строки в `sessions` нет, поэтому запись сообщений нарушила бы FK.
            return
        if len(chat.history) < 2:
            return
        first_record, assistant_record = chat.history[-2], chat.history[-1]
        if first_record.role not in ("user", "system") or assistant_record.role != "assistant":
            return
        self._store.save_session(chat)
        self._store.append_pair(chat.id, first_record, assistant_record)

    def persist_new_requests(self, chat: ChatService) -> None:
        """Персистит ещё не сохранённые записи о запросах к LLM.

        Вызывается после саммаризации (токены уже сожжены) и после
        завершения основного стрима. Для ``ephemeral``-сессий — no-op.

        Args:
            chat: сессия, чьи записи о запросах нужно сохранить.
        """
        if self._store is None or chat.kind == SessionKind.EPHEMERAL:
            return
        for record in chat.requests:
            if not record.persisted:
                self._store.append_request(chat.id, record)
                record.persisted = True

    def persist_summary_state(self, chat: ChatService) -> None:
        """Персистит состояние чанковой саммаризации (курсор + саммари).

        Args:
            chat: сессия с заполненными ``summarized_chunks``/``summary_items``.
        """
        if self._store is None or chat.kind == SessionKind.EPHEMERAL:
            return
        if chat.summarized_chunks > 0 or chat.summary_items:
            self._store.save_summary_state(
                chat.id, chat.summarized_chunks, chat.summary_items
            )

    def persist_facts(self, chat: ChatService) -> None:
        """Персистит канонические факты чата (стратегия facts).

        Args:
            chat: сессия со списком ``facts``.
        """
        if self._store is None or chat.kind == SessionKind.EPHEMERAL:
            return
        self._store.save_facts(chat.id, chat.facts)

    def persist_memory(self, chat: ChatService) -> None:
        """Персистит память чата (только персистентные вкладки).

        Args:
            chat: сессия со списком ``memory_stores``.
        """
        if self._store is None or chat.kind == SessionKind.EPHEMERAL:
            return
        self._store.save_memory(chat.id, chat.memory_stores)

    def branch(self, chat: ChatService, title: Optional[str] = None) -> ChatService:
        """Создаёт чат-снапшот (ветку) от существующего чата.

        Копируются история (включая сид), системный промпт, модель, настройки,
        состояние чанковой саммаризации и факты. Журнал запросов **не**
        копируется — график и «Сожжено токенов» нового чата начинаются с нуля.
        Исходный чат не изменяется.

        Args:
            chat: исходный чат.
            title: название новой вкладки (иначе — из первого сообщения).

        Returns:
            Новый :class:`ChatService`.
        """
        chat_id = secrets.token_hex(8)
        while chat_id in self._sessions:
            chat_id = secrets.token_hex(8)
        new = ChatService(
            chat_id=chat_id,
            kind=SessionKind.CHAT,
            model=chat.model,
            settings=copy.deepcopy(chat.settings),
            system_prompt=chat.system_prompt,
        )
        new.parent_id = chat.id
        new.title = title or self._derive_title(chat)
        new.history = [
            MessageRecord(
                role=record.role,
                content=record.content,
                meta=record.meta,
                created_at=record.created_at,
            )
            for record in chat.history
        ]
        new.summarized_chunks = chat.summarized_chunks
        new.summary_items = list(chat.summary_items)
        new.facts = list(chat.facts)
        new.memory_stores = copy.deepcopy(chat.memory_stores)

        self._sessions[chat_id] = new
        if self._store is not None:
            self._store.save_session(new)
            self._store.save_history(new.id, new.history)
            if new.summarized_chunks > 0 or new.summary_items:
                self._store.save_summary_state(
                    new.id, new.summarized_chunks, new.summary_items
                )
            if new.facts:
                self._store.save_facts(new.id, new.facts)
            if any(store.persistent for store in new.memory_stores):
                self._store.save_memory(new.id, new.memory_stores)
        return new

    @staticmethod
    def _derive_title(chat: ChatService) -> str:
        """Название для ветки из первого сообщения истории."""
        for record in chat.history:
            if record.role in ("user", "system"):
                title = " ".join(record.content.split()).strip()
                if title:
                    return title
        return "Новый чат"

    def build_summary_context(self, exclude_id: str) -> str:
        """Собирает TOON-контекст из всех сессий, кроме ``exclude_id``.

        Args:
            exclude_id: идентификатор сессии «итогов» (исключается).

        Returns:
            Строка TOON с содержимым и метаданными остальных чатов.
        """
        others = [
            s for s in self._sessions.values() if s.id != exclude_id and s.history
        ]
        if not others:
            return toon_encode({"empty": True})
        data = [s.to_toon_context(title=s.id) for s in others]
        return toon_encode({"chats": data})

    def summary_system_prompt(self, exclude_id: str) -> str:
        """Возвращает системный промпт для сессии «итоги».

        Args:
            exclude_id: идентификатор сессии «итогов».

        Returns:
            Промпт с вложенным TOON-контекстом остальных чатов.
        """
        return f"{SUMMARY_CONTEXT_PROMPT}\n\n{self.build_summary_context(exclude_id)}"
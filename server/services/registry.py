"""Реестр сессий (чатов): создание, хранение, удаление, сводный контекст.

Реестр владеет всеми живыми :class:`ChatService` сервера. Для чата
«Подвести итоги» строит TOON-контекст из содержимого остальных сессий.
Периодически удаляет осиротевшие сессии (например, после перезагрузки
страницы, когда клиент не вызвал DELETE).
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Optional

from ..config import Settings
from ..toon_codec import encode as toon_encode
from .chat_service import ChatService, SessionKind

SUMMARY_CONTEXT_PROMPT = (
    "У тебя есть доступ к содержимому всех открытых чатов. "
    "Используй его при ответе на вопрос пользователя."
)

_ids = itertools.count(1)


class SessionRegistry:
    """Реестр живых сессий сервера.

    Args:
        settings: настройки сервера (для TTL-очистки).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._sessions: dict[str, ChatService] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

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
        chat_id = f"chat-{next(_ids)}"
        service = ChatService(
            chat_id=chat_id,
            kind=kind,
            model=model,
            settings=settings,
            system_prompt=system_prompt,
        )
        self._sessions[chat_id] = service
        return service

    def get(self, session_id: str) -> Optional[ChatService]:
        """Возвращает сессию по идентификатору (или ``None``)."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Удаляет сессию из памяти сервера.

        Args:
            session_id: идентификатор сессии.

        Returns:
            ``True``, если сессия существовала и удалена.
        """
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[ChatService]:
        """Возвращает список всех живых сессий."""
        return list(self._sessions.values())

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

    async def start_cleanup(self) -> asyncio.Task:
        """Запускает фоновую задачу удаления осиротевших сессий.

        Returns:
            Объект задачи :class:`asyncio.Task`.
        """
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        return self._cleanup_task

    async def _cleanup_loop(self) -> None:
        ttl = self._settings.session_ttl_hours * 3600
        while True:
            await asyncio.sleep(3600)
            now = time.time()
            stale = [
                sid
                for sid, s in self._sessions.items()
                if now - s.last_active > ttl and not s.busy
            ]
            for sid in stale:
                self._sessions.pop(sid, None)

    async def stop_cleanup(self) -> None:
        """Останавливает фоновую задачу очистки."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None

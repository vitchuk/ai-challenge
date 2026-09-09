"""Хранилище сессий в SQLite.

Персистентность чатов для одного пользователя: сессии и история сообщений
сохраняются в файл SQLite и восстанавливаются при перезапуске сервера.

Изолированные (``ephemeral``) сессии команды ``/optimize-prompt`` в БД
не сохраняются — они временные по спецификации.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .chat_service import ChatService, MessageRecord, SessionKind
from .generation import GenerationSettings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    model        TEXT,
    system_prompt TEXT,
    settings     TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL,
    last_active  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta        TEXT,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class SessionStore:
    """Прокси к SQLite-файлу: сохранение/загрузка сессий и сообщений.

    Args:
        db_path: путь к файлу БД (``:memory:`` — временная БД для тестов).
    """

    def __init__(self, db_path: str = "chats.db") -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Закрывает соединение с БД."""
        self._conn.close()

    def save_session(self, chat: ChatService) -> None:
        """Сохраняет (upsert) сессию без сообщений.

        Args:
            chat: сессия для сохранения.
        """
        if chat.kind == SessionKind.EPHEMERAL:
            return
        self._conn.execute(
            """
            INSERT INTO sessions (id, kind, model, system_prompt, settings, created_at, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                model = excluded.model,
                system_prompt = excluded.system_prompt,
                settings = excluded.settings,
                last_active = excluded.last_active
            """,
            (
                chat.id,
                chat.kind.value,
                chat.model,
                chat.system_prompt,
                json.dumps(chat.settings.to_dict(), ensure_ascii=False),
                getattr(chat, "created_at", chat.last_active),
                chat.last_active,
            ),
        )
        self._conn.commit()

    def append_pair(
        self,
        session_id: str,
        user_record: MessageRecord,
        assistant_record: MessageRecord,
    ) -> None:
        """Атомарно дописывает пару user+assistant в историю сессии.

        Args:
            session_id: идентификатор сессии.
            user_record: пользовательское сообщение.
            assistant_record: ответ ассистента (с метаданными).
        """
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
                (
                    session_id,
                    user_record.role,
                    user_record.content,
                    None,
                    user_record.created_at,
                ),
            )
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
                (
                    session_id,
                    assistant_record.role,
                    assistant_record.content,
                    json.dumps(assistant_record.meta.to_dict(), ensure_ascii=False)
                    if assistant_record.meta is not None
                    else None,
                    assistant_record.created_at,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?",
                (max(user_record.created_at, assistant_record.created_at), session_id),
            )

    def delete_session(self, session_id: str) -> bool:
        """Удаляет сессию вместе с сообщениями.

        Args:
            session_id: идентификатор сессии.

        Returns:
            ``True``, если сессия существовала и удалена.
        """
        cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def get_state(self, key: str) -> Optional[str]:
        """Возвращает значение из таблицы ``app_state``.

        Args:
            key: ключ состояния.

        Returns:
            Значение или ``None``.
        """
        row = self._conn.execute(
            "SELECT value FROM app_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        """Сохраняет значение в таблицу ``app_state``.

        Args:
            key: ключ состояния.
            value: значение.
        """
        self._conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def delete_state(self, key: str) -> None:
        """Удаляет значение из таблицы ``app_state``.

        Args:
            key: ключ состояния.
        """
        self._conn.execute("DELETE FROM app_state WHERE key = ?", (key,))
        self._conn.commit()

    def load_all(self) -> list[ChatService]:
        """Загружает все сессии с полной историей.

        Returns:
            Список :class:`ChatService` в порядке создания.
        """
        sessions = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at, id"
        ).fetchall()
        result: list[ChatService] = []
        for row in sessions:
            settings = GenerationSettings.from_dict(json.loads(row["settings"] or "{}"))
            chat = ChatService(
                chat_id=row["id"],
                kind=SessionKind(row["kind"]),
                model=row["model"],
                settings=settings,
                system_prompt=row["system_prompt"],
            )
            chat.last_active = row["last_active"]
            messages = self._conn.execute(
                "SELECT role, content, meta, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
            for msg in messages:
                record = MessageRecord.from_dict(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "meta": json.loads(msg["meta"]) if msg["meta"] else None,
                    }
                )
                record.created_at = msg["created_at"]
                chat.history.append(record)
            result.append(chat)
        return result
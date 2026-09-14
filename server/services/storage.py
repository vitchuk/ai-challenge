"""Хранилище сессий в SQLite.

Персистентность чатов для одного пользователя: сессии и история сообщений
сохраняются в файл SQLite и восстанавливаются при перезапуске сервера.

Изолированные (``ephemeral``) сессии команды ``/optimize-prompt`` в БД
не сохраняются — они временные по спецификации.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .chat_service import (
    ChatService,
    MessageRecord,
    RequestRecord,
    SessionKind,
)
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

CREATE TABLE IF NOT EXISTS llm_requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx               INTEGER NOT NULL,
    kind              TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens  INTEGER,
    created_at        REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_requests_session ON llm_requests(session_id, id);

CREATE TABLE IF NOT EXISTS session_summary_state (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    chunks      INTEGER NOT NULL DEFAULT 0,
    items       TEXT NOT NULL DEFAULT '[]',
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_facts (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    items       TEXT NOT NULL DEFAULT '[]',
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Дополнительные колонки, добавленные после первой версии схемы: для уже
# созданных БД они досоздаются через ALTER TABLE (`CREATE TABLE IF NOT
# EXISTS` существующую таблицу не изменяет). Формат: (таблица, колонка, тип).
MIGRATIONS: list[tuple[str, str, str]] = [
    ("llm_requests", "reasoning_tokens", "INTEGER"),
    ("sessions", "parent_id", "TEXT"),
    ("sessions", "title", "TEXT"),
]

# Устаревшие таблицы, которые удаляются при открытии БД.
DROPS: list[str] = ["session_summaries"]


class SessionStore:
    """Прокси к SQLite-файлу: сохранение/загрузка сессий и сообщений.

    При открытии существующей БД недостающие колонки (см. :data:`MIGRATIONS`)
    добавляются автоматически, чтобы не терять историю при обновлении версии.

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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Досоздаёт недостающие колонки и удаляет устаревшие таблицы."""
        for table, column, decl in MIGRATIONS:
            existing = {
                row["name"]
                for row in self._conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        for table in DROPS:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")

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
            INSERT INTO sessions
                (id, kind, model, system_prompt, settings, created_at,
                 last_active, parent_id, title)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                model = excluded.model,
                system_prompt = excluded.system_prompt,
                settings = excluded.settings,
                last_active = excluded.last_active,
                parent_id = excluded.parent_id,
                title = excluded.title
            """,
            (
                chat.id,
                chat.kind.value,
                chat.model,
                chat.system_prompt,
                json.dumps(chat.settings.to_dict(), ensure_ascii=False),
                getattr(chat, "created_at", chat.last_active),
                chat.last_active,
                getattr(chat, "parent_id", None),
                getattr(chat, "title", None),
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

    def append_request(self, session_id: str, record: RequestRecord) -> None:
        """Дописывает запись о запросе к LLM (для графика/счётчика).

        Для ``ephemeral``-сессий не вызывается (у них нет строки в ``sessions``,
        и вставка нарушила бы FK) — фильтрация в ``SessionRegistry``.

        Args:
            session_id: идентификатор сессии.
            record: запись о запросе.
        """
        self._conn.execute(
            """
            INSERT INTO llm_requests
                (session_id, idx, kind, prompt_tokens, completion_tokens,
                 reasoning_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                record.index,
                record.kind,
                record.prompt_tokens,
                record.completion_tokens,
                record.reasoning_tokens,
                record.created_at or time.time(),
            ),
        )
        self._conn.commit()

    def save_summary_state(self, session_id: str, chunks: int, items: list[str]) -> None:
        """Сохраняет (upsert) состояние чанковой саммаризации сессии.

        Args:
            session_id: идентификатор сессии.
            chunks: сколько чанков уже сжато.
            items: накопленные саммари.
        """
        self._conn.execute(
            """
            INSERT INTO session_summary_state (session_id, chunks, items, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                chunks = excluded.chunks,
                items = excluded.items,
                updated_at = excluded.updated_at
            """,
            (session_id, chunks, json.dumps(items, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def save_facts(self, session_id: str, items: list[str]) -> None:
        """Сохраняет (upsert) список фактов сессии (стратегия facts).

        Args:
            session_id: идентификатор сессии.
            items: канонические факты.
        """
        self._conn.execute(
            """
            INSERT INTO session_facts (session_id, items, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                items = excluded.items,
                updated_at = excluded.updated_at
            """,
            (session_id, json.dumps(items, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def save_history(self, session_id: str, records: list[MessageRecord]) -> None:
        """Перезаписывает историю сессии целиком (для снапшота при ветвлении).

        Args:
            session_id: идентификатор сессии.
            records: полная история сообщений.
        """
        with self._conn:
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            for record in records:
                self._conn.execute(
                    "INSERT INTO messages (session_id, role, content, meta, created_at) "
                    "VALUES (?,?,?,?,?)",
                    (
                        session_id,
                        record.role,
                        record.content,
                        json.dumps(record.meta.to_dict(), ensure_ascii=False)
                        if record.meta is not None
                        else None,
                        record.created_at,
                    ),
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
            chat.parent_id = row["parent_id"]
            chat.title = row["title"]
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
            requests = self._conn.execute(
                "SELECT idx, kind, prompt_tokens, completion_tokens, "
                "reasoning_tokens, created_at "
                "FROM llm_requests WHERE session_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
            for req in requests:
                chat.requests.append(
                    RequestRecord(
                        index=req["idx"],
                        kind=req["kind"],
                        prompt_tokens=req["prompt_tokens"],
                        completion_tokens=req["completion_tokens"],
                        reasoning_tokens=req["reasoning_tokens"],
                        created_at=req["created_at"],
                        persisted=True,
                    )
                )
            summary = self._conn.execute(
                "SELECT chunks, items FROM session_summary_state WHERE session_id = ?",
                (row["id"],),
            ).fetchone()
            if summary is not None:
                chat.summarized_chunks = summary["chunks"]
                loaded_items = json.loads(summary["items"] or "[]")
                if isinstance(loaded_items, list):
                    chat.summary_items = [str(item) for item in loaded_items]
            facts_row = self._conn.execute(
                "SELECT items FROM session_facts WHERE session_id = ?",
                (row["id"],),
            ).fetchone()
            if facts_row is not None:
                loaded_facts = json.loads(facts_row["items"] or "[]")
                if isinstance(loaded_facts, list):
                    chat.facts = [
                        item if isinstance(item, str) else [str(x) for x in item]
                        for item in loaded_facts
                        if isinstance(item, str)
                        or (isinstance(item, (list, tuple)) and len(item) == 2)
                    ]
            result.append(chat)
        return result
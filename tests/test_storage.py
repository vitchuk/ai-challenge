"""Тесты слоя хранения сессий (SQLite)."""

from server.services.chat_service import (
    PROFILE_FIELDS,
    ChatService,
    MessageMeta,
    Profile,
    SessionKind,
)
from server.services.generation import GenerationSettings
from server.services.storage import SessionStore


def make_chat(**kw):
    defaults = dict(chat_id="chat-1", kind=SessionKind.CHAT, model="deepseek-v4-flash")
    defaults.update(kw)
    return ChatService(**defaults)


def test_save_load_roundtrip(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat(
        system_prompt="sys prompt",
        settings=GenerationSettings(temperature=0.1, top_p=0.01, top_k=5, max_tokens=100, stop=["stop1"], response_format={"type": "json_object"}),
    )
    chat.add_user_message("привет")
    chat.append_assistant_message(
        "ответ",
        MessageMeta(model="deepseek-v4-flash", elapsed_s=1.2, prompt_tokens=10, completion_tokens=5, reasoning_tokens=2, cost_usd=0.0001, finish_reason="stop"),
    )
    store.save_session(chat)
    store.append_pair(chat.id, chat.history[0], chat.history[1])

    loaded = store.load_all()
    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.id == chat.id
    assert restored.kind == SessionKind.CHAT
    assert restored.model == "deepseek-v4-flash"
    assert restored.system_prompt == "sys prompt"
    assert restored.settings.temperature == 0.1
    assert restored.settings.top_p == 0.01
    assert restored.settings.top_k == 5
    assert restored.settings.max_tokens == 100
    assert restored.settings.stop == ["stop1"]
    assert restored.settings.response_format == {"type": "json_object"}
    assert len(restored.history) == 2
    assert restored.history[0].role == "user"
    assert restored.history[0].content == "привет"
    assert restored.history[1].role == "assistant"
    assert restored.history[1].content == "ответ"
    assert restored.history[1].meta.prompt_tokens == 10
    assert restored.history[1].meta.reasoning_tokens == 2
    assert restored.history[1].meta.cost_usd == 0.0001
    assert restored.history[1].meta.finish_reason == "stop"
    store.close()


def test_ephemeral_not_saved(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat(kind=SessionKind.EPHEMERAL)
    store.save_session(chat)
    assert store.load_all() == []
    store.close()


def test_delete_cascades_messages(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat()
    chat.add_user_message("q")
    chat.append_assistant_message("a", MessageMeta(model="m", elapsed_s=0.5))
    store.save_session(chat)
    store.append_pair(chat.id, chat.history[0], chat.history[1])

    assert store.delete_session(chat.id) is True
    assert store.delete_session(chat.id) is False
    assert store.load_all() == []
    store.close()


def test_app_state_roundtrip(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    assert store.get_state("active_session") is None
    store.set_state("active_session", "chat-3")
    assert store.get_state("active_session") == "chat-3"
    # перезапись
    store.set_state("active_session", "chat-5")
    assert store.get_state("active_session") == "chat-5"
    store.delete_state("active_session")
    assert store.get_state("active_session") is None
    store.close()


def test_upsert_session_updates_model(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat(model="deepseek-v4-flash")
    store.save_session(chat)
    chat.model = "opencode/glm-5.3"
    store.save_session(chat)
    loaded = store.load_all()
    assert loaded[0].model == "opencode/glm-5.3"
    store.close()


def test_requests_and_summary_roundtrip(tmp_path):
    from server.services.chat_service import RequestRecord

    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat()
    store.save_session(chat)
    store.append_request(
        chat.id,
        RequestRecord(index=1, kind="summary", prompt_tokens=50, completion_tokens=7, created_at=1.0),
    )
    store.append_request(
        chat.id,
        RequestRecord(index=2, kind="main", prompt_tokens=100, completion_tokens=10, reasoning_tokens=4, created_at=2.0),
    )
    store.save_summary_state(chat.id, chunks=2, items=["S1", "S2"])

    loaded = store.load_all()[0]
    assert [r.kind for r in loaded.requests] == ["summary", "main"]
    assert loaded.requests[1].prompt_tokens == 100
    assert loaded.requests[1].reasoning_tokens == 4
    assert loaded.requests[0].persisted is True
    assert loaded.summarized_chunks == 2
    assert loaded.summary_items == ["S1", "S2"]

    # upsert состояния перезаписывает
    store.save_summary_state(chat.id, chunks=3, items=["META"])
    reloaded = store.load_all()[0]
    assert reloaded.summarized_chunks == 3
    assert reloaded.summary_items == ["META"]
    store.close()


def test_delete_cascades_requests_and_summary(tmp_path):
    from server.services.chat_service import RequestRecord

    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat()
    store.save_session(chat)
    store.append_request(chat.id, RequestRecord(index=1, kind="main"))
    store.save_summary_state(chat.id, chunks=1, items=["S"])
    store.delete_session(chat.id)
    assert store.load_all() == []
    store.close()


def test_facts_parent_and_history_roundtrip(tmp_path):
    from server.services.chat_service import MessageMeta

    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat(model="m")
    chat.parent_id = "parent1"
    chat.title = "Ветка"
    chat.facts = [["Имя", "Иван"], ["Город", "Москва"], "легаси строка"]
    store.save_session(chat)
    store.save_facts(chat.id, chat.facts)
    chat.add_user_message("q")
    chat.append_assistant_message("a", MessageMeta(model="m", elapsed_s=0.1))
    store.save_history(chat.id, chat.history)

    loaded = store.load_all()[0]
    assert loaded.parent_id == "parent1"
    assert loaded.title == "Ветка"
    # пары сохраняются как пары, легаси-строки — как строки
    assert loaded.facts == [["Имя", "Иван"], ["Город", "Москва"], "легаси строка"]
    assert [m.content for m in loaded.history] == ["q", "a"]

    # save_history перезаписывает историю целиком
    store.save_history(chat.id, chat.history[:1])
    assert [m.content for m in store.load_all()[0].history] == ["q"]
    store.close()


def test_memory_roundtrip_persistent_only(tmp_path):
    from server.services.chat_service import MemoryStore

    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    chat = make_chat()
    store.save_session(chat)
    stores = [
        MemoryStore("m1", "Профиль", True, [["Имя", "Иван"]]),
        MemoryStore("m2", "Временная", False, [["x", "y"]]),
    ]
    store.save_memory(chat.id, stores)

    loaded = store.load_all()[0]
    # сохраняются только персистентные вкладки
    assert len(loaded.memory_stores) == 1
    assert loaded.memory_stores[0].name == "Профиль"
    assert loaded.memory_stores[0].persistent is True
    assert loaded.memory_stores[0].items == [["Имя", "Иван"]]

    # повторный синк с пустым списком очищает память
    store.save_memory(chat.id, [])
    assert store.load_all()[0].memory_stores == []
    store.close()


def test_migrates_missing_reasoning_column(tmp_path):
    """Ранее созданная БД без llm_requests.reasoning_tokens досоздаётся."""
    import sqlite3

    from server.services.chat_service import RequestRecord

    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            model TEXT,
            system_prompt TEXT,
            settings TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            last_active REAL NOT NULL
        );
        CREATE TABLE llm_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            kind TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            created_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, kind, model, settings, created_at, last_active) "
        "VALUES ('old1', 'chat', 'm', '{}', 1.0, 1.0)"
    )
    conn.commit()
    conn.close()

    store = SessionStore(db)
    verify = sqlite3.connect(db)
    columns = {row[1] for row in verify.execute("PRAGMA table_info(llm_requests)")}
    verify.close()
    assert "reasoning_tokens" in columns

    store.append_request(
        "old1",
        RequestRecord(
            index=1, kind="main", prompt_tokens=10, completion_tokens=5,
            reasoning_tokens=3, created_at=1.0,
        ),
    )
    assert store.load_all()[0].requests[0].reasoning_tokens == 3
    store.close()


# ── Профили пользователя ────────────────────────────────────────────────────

def profile_fields(value: str = "v") -> dict:
    return {key: f"{value}-{key}" for key, _ in PROFILE_FIELDS}


def test_profiles_roundtrip(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    fields = profile_fields()
    store.save_profiles([Profile(id="p1", name="Основной", fields=fields)], "p1")

    loaded = store.load_profiles()
    assert [x.id for x in loaded] == ["p1"]
    assert loaded[0].name == "Основной"
    assert loaded[0].fields == fields
    assert store.get_state("active_profile") == "p1"
    store.close()


def test_profiles_full_replace_clears_active(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    fields = profile_fields()
    store.save_profiles([Profile(id="a", name="A", fields=fields)], "a")
    store.save_profiles([Profile(id="b", name="B", fields=fields)], None)

    assert [x.id for x in store.load_profiles()] == ["b"]
    assert store.get_state("active_profile") is None
    store.close()


def test_profiles_persist_across_reopen(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    store.save_profiles(
        [Profile(id="p", name="P", fields=profile_fields())], "p"
    )
    store.close()

    store2 = SessionStore(db)
    assert [x.id for x in store2.load_profiles()] == ["p"]
    assert store2.get_state("active_profile") == "p"
    store2.close()


# ── Состояние задачи (протокол «Задачи») ────────────────────────────────────

def test_task_state_roundtrip(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    task = make_chat(chat_id="t1", kind=SessionKind.TASK)
    store.save_session(task)
    store.save_task_state(task.id, "plan_review", "план", None)

    loaded = store.load_all()[0]
    assert loaded.task_stage == "plan_review"
    assert loaded.task_plan == "план"
    assert loaded.task_result is None
    store.close()


def test_task_state_persists_across_reopen(tmp_path):
    db = str(tmp_path / "test.db")
    store = SessionStore(db)
    task = make_chat(chat_id="t1", kind=SessionKind.TASK)
    store.save_session(task)
    store.save_task_state(
        task.id, "step_review", "план", None,
        steps=["Шаг один", "Шаг два"], step_results=["результат 1"],
    )
    store.close()

    store2 = SessionStore(db)
    loaded = store2.load_all()[0]
    assert loaded.task_stage == "step_review"
    assert loaded.task_plan == "план"
    assert loaded.task_result is None
    assert loaded.task_steps == ["Шаг один", "Шаг два"]
    assert loaded.task_step_results == ["результат 1"]
    store2.close()
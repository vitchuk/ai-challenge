"""Тесты слоя хранения сессий (SQLite)."""

from server.services.chat_service import ChatService, MessageMeta, SessionKind
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
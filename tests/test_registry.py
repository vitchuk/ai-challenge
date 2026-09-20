"""Тесты реестра сессий."""

from server import toon_codec
from server.services.chat_service import (
    PROFILE_FIELDS,
    MessageMeta,
    Profile,
    RuleStore,
    SessionKind,
)
from server.services.generation import GenerationSettings
from server.services.registry import SessionRegistry
from server.services.storage import SessionStore


def make_registry_with_store(db_path: str) -> SessionRegistry:
    store = SessionStore(db_path)
    return SessionRegistry(store=store)


def test_create_get_delete():
    reg = SessionRegistry()
    svc = reg.create()
    assert reg.get(svc.id) is svc
    assert reg.delete(svc.id) is True
    assert reg.get(svc.id) is None
    assert reg.delete(svc.id) is False


def test_ids_are_unique():
    reg = SessionRegistry()
    a = reg.create()
    b = reg.create()
    assert a.id != b.id


def test_summary_context_excludes_itself():
    reg = SessionRegistry()
    other = reg.create()
    other.add_user_message("вопрос")
    other.append_assistant_message("ответ", None)
    summary = reg.create(kind=SessionKind.SUMMARY)

    ctx = reg.build_summary_context(exclude_id=summary.id)
    decoded = toon_codec.decode(ctx)
    # чат «итоги» исключается из собственного контекста, но другие чаты — нет
    assert "chats" in decoded
    assert "вопрос" in ctx

    empty = SessionRegistry().build_summary_context(exclude_id="x")
    assert toon_codec.decode(empty)["empty"] is True


def test_summary_system_prompt_contains_context():
    reg = SessionRegistry()
    other = reg.create()
    other.add_user_message("привет")
    other.append_assistant_message("мир", None)
    summary = reg.create(kind=SessionKind.SUMMARY)
    prompt = reg.summary_system_prompt(summary.id)
    assert "У тебя есть доступ" in prompt
    assert "привет" in prompt


def test_create_persists_with_store(tmp_path):
    reg = make_registry_with_store(str(tmp_path / "t.db"))
    svc = reg.create(model="deepseek-v4-flash", system_prompt="sys",
                     settings=GenerationSettings(temperature=0.1))
    assert len(reg.list_sessions()) == 1
    # повторное открытие БД тем же реестром (создание без сохранения) не теряет сессию
    reg2 = make_registry_with_store(str(tmp_path / "t.db"))
    assert reg2.restore() == 1
    restored = reg2.get(svc.id)
    assert restored is not None
    assert restored.model == "deepseek-v4-flash"
    assert restored.settings.temperature == 0.1
    reg.close()


def test_delete_removes_from_store(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    svc = reg.create()
    assert reg.delete(svc.id) is True
    reg2 = make_registry_with_store(db)
    assert reg2.restore() == 0
    reg.close()
    reg2.close()


def test_hash_ids_unique_and_opaque(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    a = reg.create()
    b = reg.create()
    # id — случайные 16 hex-символов, без сквозной нумерации
    assert len(a.id) == 16 and all(c in "0123456789abcdef" for c in a.id)
    assert a.id != b.id
    reg.close()

    reg2 = make_registry_with_store(db)
    assert reg2.restore() == 2
    c = reg2.create()
    assert len(c.id) == 16  # новые id не коллизируют с восстановленными
    reg2.close()


def test_remember_pair_persists(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    svc = reg.create()
    svc.add_user_message("вопрос")
    svc.append_assistant_message("ответ", MessageMeta(model="m", elapsed_s=1.0))
    reg.remember_pair(svc)
    reg.close()

    reg2 = make_registry_with_store(db)
    assert reg2.restore() == 1
    restored = reg2.get(svc.id)
    assert [m.content for m in restored.history] == ["вопрос", "ответ"]
    assert restored.history[1].meta.model == "m"
    reg2.close()


def test_remember_pair_skips_invalid_tail(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    svc = reg.create()
    svc.add_user_message("один")
    reg.remember_pair(svc)  # в конце только user — ничего не пишем
    reg.close()

    reg2 = make_registry_with_store(db)
    assert reg2.restore() == 1
    assert reg2.get(svc.id).history == []
    reg2.close()


def test_remember_pair_accepts_system_seed(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    svc = reg.create()
    svc.seed_system_message("инструкция")
    svc.append_assistant_message("ок", MessageMeta(model="m", elapsed_s=1.0))
    reg.remember_pair(svc)
    reg.close()

    reg2 = make_registry_with_store(db)
    assert reg2.restore() == 1
    restored = reg2.get(svc.id)
    assert restored.system_prompt == "инструкция"
    assert [m.role for m in restored.history] == ["system", "assistant"]
    reg2.close()


def test_active_session_set_get_clear(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    a = reg.create()
    b = reg.create()
    assert reg.get_active() is None
    assert reg.set_active(a.id) is True
    assert reg.get_active() == a.id
    assert reg.set_active("chat-999") is False  # нет такой сессии
    assert reg.get_active() == a.id
    # удаление активной сессии очищает маркер
    assert reg.delete(a.id) is True
    assert reg.get_active() is None
    reg.close()


def test_active_persists_across_restart(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    a = reg.create()
    reg.create()
    reg.set_active(a.id)
    reg.close()

    reg2 = make_registry_with_store(db)
    reg2.restore()
    assert reg2.get_active() == a.id
    reg2.close()


# ── Профили пользователя ────────────────────────────────────────────────────

def full_fields(value: str = "v") -> dict:
    return {key: value for key, _ in PROFILE_FIELDS}


def test_profiles_replace_and_active():
    reg = SessionRegistry()
    p = Profile(id="p1", name="Основной", fields=full_fields())
    reg.replace_profiles([p], "p1")
    assert [x.id for x in reg.list_profiles()] == ["p1"]
    assert reg.get_active_profile_id() == "p1"
    assert reg.get_active_profile().name == "Основной"
    # активный снимается, если такого профиля нет
    reg.replace_profiles([p], "nope")
    assert reg.get_active_profile_id() is None


def test_active_profile_block_only_when_complete():
    reg = SessionRegistry()
    assert reg.active_profile_block() is None

    incomplete = Profile(id="x", name="Неполный", fields={"address": "Иван"})
    reg.replace_profiles([incomplete], "x")
    assert reg.active_profile_block() is None

    complete = Profile(id="y", name="Полный", fields=full_fields("значение"))
    reg.replace_profiles([complete], "y")
    block = reg.active_profile_block()
    assert block is not None
    assert "следует профилю пользователя" in block
    assert "[Профиль пользователя]" in block
    assert "Как ко мне обращаться: значение" in block


def test_profiles_persist_across_restart(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    p = Profile(id="p1", name="Основной", fields=full_fields("значение"))
    reg.replace_profiles([p], "p1")
    reg.close()

    reg2 = make_registry_with_store(db)
    reg2.restore()
    assert [x.id for x in reg2.list_profiles()] == ["p1"]
    assert reg2.get_active_profile_id() == "p1"
    assert reg2.get_active_profile().fields["address"] == "значение"
    reg2.close()


# ── Правила приложения ──────────────────────────────────────────────────────

def test_rules_frame_none_and_format():
    reg = SessionRegistry()
    assert reg.rules_frame() is None

    reg.replace_rules([
        RuleStore(id="a", name="Первое", items=[["K1", "V1"]]),
        RuleStore(id="b", name="Пустое", items=[]),
    ])
    frame = reg.rules_frame()
    assert frame is not None
    assert "[Правила]" in frame
    assert "## Первое" in frame
    assert "K1: V1" in frame
    assert "## Пустое" not in frame
    assert "Никогда не нарушай" in frame

    reg.replace_rules([])
    assert reg.rules_frame() is None


def test_rules_persist_across_restart(tmp_path):
    db = str(tmp_path / "t.db")
    reg = make_registry_with_store(db)
    reg.replace_rules([RuleStore(id="rl-1", name="Правила", items=[["K", "V"]])])
    reg.close()

    reg2 = make_registry_with_store(db)
    reg2.restore()
    assert [r.id for r in reg2.list_rules()] == ["rl-1"]
    assert "K: V" in reg2.rules_frame()
    reg2.close()

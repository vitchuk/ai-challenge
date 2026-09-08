"""Тесты реестра сессий."""

from server import toon_codec
from server.services.chat_service import SessionKind
from server.services.generation import GenerationSettings
from server.services.registry import SessionRegistry


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

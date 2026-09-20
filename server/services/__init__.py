"""Сервисный слой: инкапсуляция чатов, параметры генерации, реестр сессий."""

from .chat_service import (
    ChatService,
    MemoryStore,
    MessageMeta,
    MessageRecord,
    Profile,
    RuleStore,
    SessionKind,
)
from .generation import GenerationSettings, sanitize_settings
from .registry import SessionRegistry
from .storage import SessionStore

__all__ = [
    "ChatService",
    "GenerationSettings",
    "MemoryStore",
    "MessageMeta",
    "MessageRecord",
    "Profile",
    "RuleStore",
    "SessionKind",
    "SessionRegistry",
    "SessionStore",
    "sanitize_settings",
]

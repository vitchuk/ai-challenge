"""Сервисный слой: инкапсуляция чатов, параметры генерации, реестр сессий."""

from .chat_service import ChatService, MessageMeta, MessageRecord, SessionKind
from .generation import GenerationSettings, sanitize_settings
from .registry import SessionRegistry

__all__ = [
    "ChatService",
    "GenerationSettings",
    "MessageMeta",
    "MessageRecord",
    "SessionKind",
    "SessionRegistry",
    "sanitize_settings",
]

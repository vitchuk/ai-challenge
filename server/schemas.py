"""Pydantic-схемы запросов/ответов HTTP API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    """Тело ``POST /api/sessions``."""

    model: Optional[str] = None
    settings: Optional[dict[str, Any]] = None
    system_prompt: Optional[str] = None
    kind: str = "chat"


class SessionCreateResponse(BaseModel):
    """Ответ ``POST /api/sessions``."""

    id: str
    model: Optional[str] = None


class BranchRequest(BaseModel):
    """Тело ``POST /api/sessions/{id}/branch``."""

    title: Optional[str] = None


class MemorySyncRequest(BaseModel):
    """Тело ``PUT /api/sessions/{id}/memory`` — полное состояние памяти чата."""

    stores: list[dict[str, Any]] = []


class ProfilesSyncRequest(BaseModel):
    """Тело ``PUT /api/profiles`` — полное состояние профилей пользователя."""

    profiles: list[dict[str, Any]] = []
    active_id: Optional[str] = None


class RulesSyncRequest(BaseModel):
    """Тело ``PUT /api/rules`` — полное состояние правил приложения."""

    rules: list[dict[str, Any]] = []


class TaskAdvanceRequest(BaseModel):
    """Тело ``POST /api/tasks/{id}/advance`` — действие протокола «Задачи»."""

    action: str
    content: Optional[str] = None
    model: Optional[str] = None
    settings: Optional[dict[str, Any]] = None


class MessageCreateRequest(BaseModel):
    """Тело ``POST /api/sessions/{id}/messages``."""

    content: str = Field(..., min_length=1)
    model: Optional[str] = None
    settings: Optional[dict[str, Any]] = None
    system_prompt: Optional[str] = None


class SessionMeta(BaseModel):
    """Метаданные сессии для ``GET /api/sessions/{id}``."""

    id: str
    kind: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    history: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

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

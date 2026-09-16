"""Маршруты профилей пользователя (глобальная сущность).

Профиль не привязан к чату: активный профиль один на всё приложение и
подставляется в системный промпт основного запроса обычных чатов.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Request

from ..schemas import ProfilesSyncRequest
from ..services.chat_service import PROFILE_FIELDS, Profile

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

#: Лимиты профилей (невалидное молча отбрасывается).
MAX_PROFILES = 50
MAX_PROFILE_NAME = 100
MAX_PROFILE_FIELD = 2000


def _sanitize_profiles(raw_profiles) -> list[Profile]:
    """Валидирует и нормализует список профилей.

    Профиль сохраняется, только если заполнены название и все пять полей
    (обязательны по спецификации). Невалидные профили отбрасываются,
    дубликаты идентификаторов перегенерируются.

    Args:
        raw_profiles: «сырой» список профилей из запроса.

    Returns:
        Список :class:`Profile` (только полные).
    """
    if not isinstance(raw_profiles, list):
        return []
    profiles: list[Profile] = []
    seen: set[str] = set()
    for raw in raw_profiles[:MAX_PROFILES]:
        if not isinstance(raw, dict):
            continue
        profile = Profile.from_dict(raw)
        profile.name = profile.name.strip()[:MAX_PROFILE_NAME]
        profile.fields = {
            key: str(profile.fields.get(key, "")).strip()[:MAX_PROFILE_FIELD]
            for key, _ in PROFILE_FIELDS
        }
        if not profile.is_complete():
            continue
        if not profile.id or profile.id in seen:
            profile.id = f"prf-{secrets.token_hex(6)}"
        seen.add(profile.id)
        profiles.append(profile)
    return profiles


def _state(registry) -> dict:
    """Текущее состояние профилей: ``{"profiles": [...], "active_id": …}``."""
    return {
        "profiles": [profile.to_dict() for profile in registry.list_profiles()],
        "active_id": registry.get_active_profile_id(),
    }


@router.get("")
async def list_profiles(request: Request) -> dict:
    """Возвращает все профили пользователя и активный.

    Args:
        request: HTTP-запрос (реестр из состояния приложения).

    Returns:
        ``{"profiles": [{id, name, fields}], "active_id": "…"|null}``.
    """
    return _state(request.app.state.registry)


@router.put("")
async def sync_profiles(body: ProfilesSyncRequest, request: Request) -> dict:
    """Синхронизирует профили (полное состояние) и активный профиль.

    Клиент присылает все профили; сервер заменяет состояние, сохраняет
    в БД и отмечает активный (если он есть среди валидных).

    Args:
        body: список профилей и id активного.
        request: HTTP-запрос.

    Returns:
        Санированное состояние ``{"profiles": [...], "active_id": …}``.
    """
    registry = request.app.state.registry
    profiles = _sanitize_profiles(body.profiles)
    registry.replace_profiles(profiles, body.active_id)
    return _state(registry)

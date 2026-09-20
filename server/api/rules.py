"""Маршруты правил (глобальные ограничения для LLM).

Правила не привязаны к чату и не имеют «активного»: все вкладки действуют
всегда, во всех чатах и видах, подмешиваясь в основные запросы к LLM.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Request

from ..schemas import RulesSyncRequest
from ..services.chat_service import RuleStore

router = APIRouter(prefix="/api/rules", tags=["rules"])

#: Лимиты правил (невалидное молча отбрасывается).
MAX_RULES = 50
MAX_RULES_ITEMS = 500
MAX_RULES_NAME = 100
MAX_RULES_KEY = 200
MAX_RULES_VALUE = 2000


def _sanitize_rules(raw_rules) -> list[RuleStore]:
    """Валидирует и нормализует список вкладок правил.

    Вкладка без названия и пары с пустым ключом/значением отбрасываются;
    длины обрезаются по лимитам, отсутствующие/дублирующиеся идентификаторы
    перегенерируются (правила всегда персистентны — флага нет).

    Args:
        raw_rules: «сырой» список вкладок из запроса.

    Returns:
        Список :class:`RuleStore` (невалидное отброшено).
    """
    if not isinstance(raw_rules, list):
        return []
    rules: list[RuleStore] = []
    seen: set[str] = set()
    for raw in raw_rules[:MAX_RULES]:
        if not isinstance(raw, dict):
            continue
        rule = RuleStore.from_dict(raw)
        rule.name = rule.name.strip()[:MAX_RULES_NAME]
        if not rule.name:
            continue
        items = []
        for item in rule.items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            key, value = item[0], item[1]
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            key, value = key.strip(), value.strip()
            if key and value:
                items.append([key[:MAX_RULES_KEY], value[:MAX_RULES_VALUE]])
            if len(items) >= MAX_RULES_ITEMS:
                break
        if not rule.id or rule.id in seen:
            rule.id = f"rl-{secrets.token_hex(6)}"
        seen.add(rule.id)
        rule.items = items
        rules.append(rule)
    return rules


def _state(registry) -> dict:
    """Текущее состояние правил: ``{"rules": [...]}``."""
    return {"rules": [rule.to_dict() for rule in registry.list_rules()]}


@router.get("")
async def list_rules(request: Request) -> dict:
    """Возвращает все вкладки правил.

    Args:
        request: HTTP-запрос (реестр из состояния приложения).

    Returns:
        ``{"rules": [{id, name, items}]}``.
    """
    return _state(request.app.state.registry)


@router.put("")
async def sync_rules(body: RulesSyncRequest, request: Request) -> dict:
    """Синхронизирует правила (полное состояние).

    Клиент присылает все вкладки; сервер заменяет состояние и сохраняет в БД.

    Args:
        body: список вкладок правил.
        request: HTTP-запрос.

    Returns:
        Санированное состояние ``{"rules": [...]}``.
    """
    registry = request.app.state.registry
    rules = _sanitize_rules(body.rules)
    registry.replace_rules(rules)
    return _state(registry)

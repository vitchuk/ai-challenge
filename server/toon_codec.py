"""TOON-сериализация структурированных данных для промптов LLM.

TOON (Token-Oriented Object Notation) — компактный, человекочитаемый формат,
на 30–60% сокращающий число токенов по сравнению с JSON. Здесь он применяется
для вложения структурированных данных (метаданные истории, контекст чата
«Подвести итоги», будущие передачи состояния между агентами) в текст промптов.

Этот модуль — единственная точка зависимости от сторонней библиотеки
``toon_format``; при необходимости его легко заменить на другую реализацию
или собственный кодек без правки остального кода.
"""

from __future__ import annotations

from typing import Any

from toon_format import decode as _decode
from toon_format import encode as _encode


def encode(data: Any) -> str:
    """Сериализует данные Python в строку TOON.

    Args:
        data: произвольная структура данных (dict/list/примитивы).

    Returns:
        Строка в формате TOON, готовая для вложения в промпт.
    """
    return _encode(data)


def decode(data: str) -> Any:
    """Разбирает строку TOON обратно в структуру Python.

    Args:
        data: строка в формате TOON.

    Returns:
        Восстановленная структура данных.
    """
    return _decode(data)

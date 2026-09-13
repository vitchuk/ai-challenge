"""Параметры генерации ответа LLM и их валидация/нормализация.

Инкапсулирует набор настроек агента/чата. Невалидные значения
молча игнорируются (запрос не отклоняется), как и в прежней Node-версии.

Параметры делятся на две группы:
- **привязываемые** к чату первым сообщением (`temperature`, `top_p`,
  `top_k`, `context_summary`) — далее не меняются;
- **гибкие** (`max_tokens`, `stop`, `response_format`) — можно менять на лету.

Поле ``top_k`` хранится для будущих провайдеров, но в текущие апстримы
(DeepSeek / OpenCode Chat Completions) не прокидывается — их API его
не поддерживают.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

MIN_TOP_P = 0.01
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_STOP_WORDS = 16
MIN_REQUESTS_PER_SUMMARY = 1
MAX_REQUESTS_PER_SUMMARY = 20
DEFAULT_REQUESTS_PER_SUMMARY = 5


@dataclass
class ContextSummarySettings:
    """Настройки чанковой саммаризации истории чата.

    При включении история делится на чанки по ``requests_per_summary``
    завершённых обменов «вопрос+ответ»; каждый чанк сжимается в саммари
    отдельным скрытым запросом к той же модели. Накопленные саммари
    вкладываются в основной запрос.

    Attributes:
        enabled: включена ли саммаризация.
        requests_per_summary: сколько запросов (обменов) покрывает одно
            саммари (1–20).
    """

    enabled: bool = False
    requests_per_summary: int = DEFAULT_REQUESTS_PER_SUMMARY

    def to_dict(self) -> dict[str, Any]:
        """Представляет настройки как словарь (для персистентности/API)."""
        return {"enabled": self.enabled, "requests_per_summary": self.requests_per_summary}


@dataclass
class GenerationSettings:
    """Набор параметров генерации для одного запроса к модели.

    Все поля необязательны: ``None`` означает «использовать дефолт модели».
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: list[str] = field(default_factory=list)
    response_format: Optional[dict[str, Any]] = None
    context_summary: Optional[ContextSummarySettings] = None

    def to_upstream(self) -> dict[str, Any]:
        """Возвращает словарь параметров, безопасный для передачи в апстрим.

        Top_k и context_summary намеренно не включаются: первый не
        поддерживается DeepSeek/OpenCode, второй — серверная механика
        саммаризации, а не параметр генерации.
        Пустые/незначимые поля опускаются.
        """
        out: dict[str, Any] = {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.stop:
            out["stop"] = self.stop
        if self.response_format and self.response_format.get("type"):
            out["response_format"] = {"type": self.response_format["type"]}
        return out

    def to_dict(self) -> dict[str, Any]:
        """Представляет настройки как словарь (для персистентности/API).

        ``None``-значения и пустой ``stop`` включаются как есть, чтобы
        сохранить полное состояние настроек.
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "response_format": self.response_format,
            "context_summary": self.context_summary.to_dict()
            if self.context_summary is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "GenerationSettings":
        """Восстанавливает настройки из словаря.

        Args:
            data: словарь настроек (например, загруженный из БД).

        Returns:
            Экземпляр :class:`GenerationSettings` (значения проходят
            через :func:`sanitize_settings`).
        """
        if not isinstance(data, dict):
            return cls()
        return sanitize_settings(data)


def _sanitize_temperature(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    if not (MIN_TEMPERATURE <= value <= MAX_TEMPERATURE):
        return None
    return value


def _sanitize_top_p(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    if not (0.0 <= value <= 1.0):
        return None
    return max(value, MIN_TOP_P)  # 0 нормализуется в 0.01 (диапазон DeepSeek (0, 1.0])


def _sanitize_top_k(value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value <= 0:
        return None
    return value


def _sanitize_max_tokens(value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value <= 0:
        return None
    return value


def _sanitize_stop(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s for s in value if isinstance(s, str) and s.strip()][:MAX_STOP_WORDS]


def _sanitize_response_format(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    rtype = value.get("type")
    if not isinstance(rtype, str) or not rtype:
        return None
    return {"type": rtype}


def _sanitize_context_summary(value: Any) -> Optional[ContextSummarySettings]:
    """Валидирует настройку чанковой саммаризации истории.

    Невалидный ``enabled`` трактуется как выключенная саммаризация,
    невалидный ``requests_per_summary`` заменяется дефолтом, выход за
    диапазон — клэмпится. ``None``/не-словарь — настройка отсутствует.
    """
    if not isinstance(value, dict):
        return None
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        enabled = False
    requests = value.get("requests_per_summary")
    if not isinstance(requests, int) or isinstance(requests, bool):
        requests = DEFAULT_REQUESTS_PER_SUMMARY
    requests = max(MIN_REQUESTS_PER_SUMMARY, min(MAX_REQUESTS_PER_SUMMARY, requests))
    return ContextSummarySettings(enabled=enabled, requests_per_summary=requests)


def sanitize_settings(raw: Any) -> GenerationSettings:
    """Валидирует и нормализует «сырой» словарь настроек.

    Args:
        raw: объект настроек из запроса (dict или ``None``).

    Returns:
        Экземпляр :class:`GenerationSettings` с очищенными значениями.
    """
    if not isinstance(raw, dict):
        return GenerationSettings()
    return GenerationSettings(
        temperature=_sanitize_temperature(raw.get("temperature")),
        top_p=_sanitize_top_p(raw.get("top_p")),
        top_k=_sanitize_top_k(raw.get("top_k")),
        max_tokens=_sanitize_max_tokens(raw.get("max_tokens")),
        stop=_sanitize_stop(raw.get("stop")),
        response_format=_sanitize_response_format(raw.get("response_format")),
        context_summary=_sanitize_context_summary(raw.get("context_summary")),
    )

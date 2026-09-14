"""Параметры генерации ответа LLM и их валидация/нормализация.

Инкапсулирует набор настроек агента/чата. Невалидные значения
молча игнорируются (запрос не отклоняется), как и в прежней Node-версии.

Параметры делятся на две группы:
- **привязываемые** к чату первым сообщением (`temperature`, `top_p`,
  `top_k`, `context_strategy`) — далее не меняются;
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
MIN_STRATEGY_PARAM = 1
MAX_STRATEGY_PARAM = 20
DEFAULT_SUMMARIZE_N = 5
DEFAULT_SLIDING_N = 10
DEFAULT_FACTS_K = 10

#: Допустимые стратегии управления контекстом.
STRATEGIES = ("none", "summarize", "sliding", "facts", "branching")


@dataclass
class ContextStrategySettings:
    """Стратегия управления контекстом чата.

    Attributes:
        strategy: одна из ``none`` (вся история), ``summarize`` (чанковая
            саммаризация), ``sliding`` (скользящее окно), ``facts``
            (извлекаемые факты + окно), ``branching`` (ветвление чатов).
        n: параметр стратегий ``summarize``/``sliding`` — сколько запросов
            покрывает суммаризация / размер окна в репликах (1–20).
        k: параметр стратегии ``facts`` — размер окна в репликах (1–20).
    """

    strategy: str = "none"
    n: int = DEFAULT_SUMMARIZE_N
    k: int = DEFAULT_FACTS_K

    def to_dict(self) -> dict[str, Any]:
        """Представляет настройки как словарь (для персистентности/API)."""
        return {"strategy": self.strategy, "n": self.n, "k": self.k}


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
    context_strategy: Optional[ContextStrategySettings] = None

    def to_upstream(self) -> dict[str, Any]:
        """Возвращает словарь параметров, безопасный для передачи в апстрим.

        Top_k и context_strategy намеренно не включаются: первый не
        поддерживается DeepSeek/OpenCode, второй — серверная механика
        управления контекстом, а не параметр генерации.
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
            "context_strategy": self.context_strategy.to_dict()
            if self.context_strategy is not None
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


def _sanitize_strategy_param(value: Any, default: int) -> int:
    """Нормализует числовой параметр стратегии (целое 1–20)."""
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return max(MIN_STRATEGY_PARAM, min(MAX_STRATEGY_PARAM, value))


def _sanitize_context_strategy(raw: dict) -> Optional[ContextStrategySettings]:
    """Валидирует настройку стратегии управления контекстом.

    Поддерживает новый ключ ``context_strategy`` и **мигрирует легаси**
    ``context_summary`` (формат прежней чанковой саммаризации):
    ``enabled`` → ``summarize`` с ``n`` из ``requests_per_summary``/
    ``keep_recent``, иначе ``none``. При наличии обоих ключей приоритет
    у ``context_strategy``. ``None`` — если ключей нет.

    Args:
        raw: словарь настроек чата.

    Returns:
        Экземпляр :class:`ContextStrategySettings` или ``None``.
    """
    value = raw.get("context_strategy")
    if isinstance(value, dict):
        strategy = value.get("strategy")
        if strategy not in STRATEGIES:
            strategy = "none"
        default_n = DEFAULT_SLIDING_N if strategy == "sliding" else DEFAULT_SUMMARIZE_N
        n = _sanitize_strategy_param(value.get("n"), default_n)
        k = _sanitize_strategy_param(value.get("k"), DEFAULT_FACTS_K)
        return ContextStrategySettings(strategy=strategy, n=n, k=k)

    legacy = raw.get("context_summary")
    if isinstance(legacy, dict):
        enabled = legacy.get("enabled")
        if not isinstance(enabled, bool):
            enabled = False
        if not enabled:
            return ContextStrategySettings(strategy="none")
        requests = legacy.get("requests_per_summary")
        if not isinstance(requests, int) or isinstance(requests, bool):
            requests = legacy.get("keep_recent")
        n = _sanitize_strategy_param(requests, DEFAULT_SUMMARIZE_N)
        return ContextStrategySettings(strategy="summarize", n=n, k=DEFAULT_FACTS_K)
    return None


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
        context_strategy=_sanitize_context_strategy(raw),
    )

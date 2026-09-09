"""Слой провайдеров апстрима (DeepSeek / OpenCode)."""

from .base import ChatEvent, LLMProvider, ProviderError, Usage, UsageDetails
from .client import StreamedCompletion
from .routing import (
    ProviderSpec,
    UnsupportedModelError,
    resolve_provider,
)

__all__ = [
    "ChatEvent",
    "LLMProvider",
    "ProviderError",
    "ProviderSpec",
    "StreamedCompletion",
    "UnsupportedModelError",
    "Usage",
    "UsageDetails",
    "resolve_provider",
]

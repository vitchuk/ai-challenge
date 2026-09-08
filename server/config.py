"""Конфигурация сервера: чтение переменных окружения и `.env`.

Настройки загружаются через pydantic-settings (поддерживает `.env`-файл
и переменные окружения; приоритет — у переменных окружения).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Параметры сервера.

    Все поля необязательны: достаточно задать хотя бы один API-ключ —
    подключённый ключ активирует соответствующий провайдер.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str | None = None
    opencode_api_key: str | None = None
    port: int = 3000
    session_ttl_hours: float = 12.0


@lru_cache
def get_settings() -> Settings:
    """Возвращает (кэшированный) объект настроек сервера."""
    return Settings()

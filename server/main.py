"""Точка входа приложения: сборка FastAPI-приложения и запуск uvicorn."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.models import router as models_router
from .api.profiles import router as profiles_router
from .api.sessions import router as sessions_router
from .config import Settings, get_settings
from .services.registry import SessionRegistry
from .services.storage import SessionStore

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


def build_registry(settings: Settings) -> SessionRegistry:
    """Создаёт реестр сессий с SQLite-хранилищем и восстанавливает чаты.

    Args:
        settings: настройки сервера (путь к БД).

    Returns:
        Реестр с уже восстановленными из БД сессиями.
    """
    store = SessionStore(settings.chats_db_path)
    registry = SessionRegistry(settings, store=store)
    registry.restore()
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управляет временем жизни HTTP-клиента, реестра и хранилища сессий."""
    # Большие запросы (длинный контекст) и длинная генерация не должны
    # упираться в дефолтный 5-секундный таймаут httpx.
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0))
    app.state.registry = build_registry(get_settings())
    app.state.opencode_session_id = str(uuid.uuid4())
    yield
    app.state.registry.close()
    await app.state.http_client.aclose()


def create_app() -> FastAPI:
    """Собирает и возвращает FastAPI-приложение.

    Returns:
        Сконфигурированное приложение (маршруты API + раздача статики).
    """
    app = FastAPI(title="Помогатор2К", lifespan=lifespan)

    @app.middleware("http")
    async def no_cache_static(request, call_next):
        """Принудительная ревалидация статики (HTML/JS/CSS) у браузера.

        Иначе браузер может держать устаревшие версии index.html/app.js
        (например, без инфоблока ``.chat-info``).
        """
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.include_router(sessions_router)
    app.include_router(models_router)
    app.include_router(profiles_router)
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("server.main:app", host="127.0.0.1", port=settings.port, reload=False)
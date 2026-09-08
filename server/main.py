"""Точка входа приложения: сборка FastAPI-приложения и запуск uvicorn."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api.models import router as models_router
from .api.sessions import router as sessions_router
from .config import get_settings
from .services.registry import SessionRegistry

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управляет временем жизни общего HTTP-клиента и реестра сессий."""
    app.state.http_client = httpx.AsyncClient()
    app.state.registry = SessionRegistry(get_settings())
    app.state.opencode_session_id = str(uuid.uuid4())
    await app.state.registry.start_cleanup()
    yield
    await app.state.registry.stop_cleanup()
    await app.state.http_client.aclose()


def create_app() -> FastAPI:
    """Собирает и возвращает FastAPI-приложение.

    Returns:
        Сконфигурированное приложение (маршруты API + раздача статики).
    """
    app = FastAPI(title="Помогатор2К", lifespan=lifespan)
    app.include_router(sessions_router)
    app.include_router(models_router)
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("server.main:app", host="127.0.0.1", port=settings.port, reload=False)

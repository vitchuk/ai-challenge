# AGENTS.md

## Что это

`mcp_demo/jsonplaceholder_server.py` — встроенный MCP-сервер на FastMCP:
прослойка между LLM и тестовым API
[jsonplaceholder.typicode.com](https://jsonplaceholder.typicode.com).
Отдаёт агенту **только чтение** данных. Копия из репозитория `ai-challenge-mcp`.

## Стек и ограничения

- Python 3.10+ (проект — 3.14), Windows/Linux.
- SDK `mcp` версии **1.x** (пин `mcp<2`). В `mcp` 2.x `FastMCP` переименован
  в `MCPServer` — не обновлять без переписывания сервера.
- HTTP-запросы — `httpx` (идёт зависимостью `mcp`, отдельно не ставить).
- Запускается общим `.venv` проекта (отдельного окружения нет).

## Инструменты (только чтение)

- `list_items(resource, filters?, limit=20)` — `GET /{resource}`;
  `filters` → query-параметры (например `{"postId": 1}`), `limit<=0` → все.
- `get_item(resource, id)` — `GET /{resource}/{id}`.

`resource` — enum: `posts`, `comments`, `albums`, `photos`, `todos`, `users`.

## Правила

- **Не добавлять модифицирующие тулы** (`POST`/`PATCH`/`DELETE`): сервер
  спроектирован как read-only.
- Ошибки API/сети поднимать через `ValueError` → MCP вернёт `isError`.
- Держать ответы компактными (по умолчанию `limit=20`; в `photos` 5000 записей).

## Запуск и проверка

```powershell
# stdio (так сервер запускает менеджер MCP приложения)
.venv\Scripts\python.exe mcp_demo\jsonplaceholder_server.py

# streamable-http на 127.0.0.1:8001
.venv\Scripts\python.exe mcp_demo\jsonplaceholder_server.py http

# проверка синтаксиса
.venv\Scripts\python.exe -m py_compile mcp_demo\jsonplaceholder_server.py
```

## Подключение в приложении

Сервер настраивается во виде **«MCP»** приложения (тип `local`, stdio) и
хранится в БД. После правки конфига/кода сервера переподключите его кнопкой
«Отключить»/«Подключить» в карточке (или перезапустите приложение).
Подробности и вариант remote — в `mcp_demo/README.md`.

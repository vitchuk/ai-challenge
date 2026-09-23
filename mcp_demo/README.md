# mcp_demo

Встроенный MCP-сервер на FastMCP — прослойка между LLM и тестовым API
[jsonplaceholder.typicode.com](https://jsonplaceholder.typicode.com).
Сервер **read-only**: отдаёт только чтение данных (список и один объект).

Копия сервера из репозитория `ai-challenge-mcp` живёт внутри проекта, чтобы
ездить вместе с приложением (в т.ч. на будущий VPS). Подключается к приложению
через вид **«MCP»** (менеджер MCP на сервере), а не через `opencode.json`.

## Инструменты

| Инструмент                                  | HTTP                     | Возврат                                        |
|---------------------------------------------|--------------------------|------------------------------------------------|
| `list_items(resource, filters?, limit=20)`  | `GET /{resource}`        | список; `filters` → query-параметры, `limit<=0` → все |
| `get_item(resource, id)`                    | `GET /{resource}/{id}`   | объект; 404 → ошибка                            |

`resource` — enum: `posts`, `comments`, `albums`, `photos`, `todos`, `users`.
Вложенные выборки делаются фильтром: комментарии поста — `list_items comments` +
`{"postId": 1}`.

## Требования

- Python 3.10+ (проект — 3.14), Windows/Linux.
- Пакет `mcp` версии **1.x** (пин `mcp<2` в `requirements.txt` проекта; он же
  используется MCP-клиентом приложения). HTTP-запросы — `httpx` (зависимость
  `mcp`).
- Запускается общим окружением проекта — `.venv` (отдельный venv не нужен).

> Важно: сервер написан на `FastMCP`, который существует только в SDK 1.x.
> В `mcp` 2.x `FastMCP` переименован в `MCPServer`, поэтому проект пинится
> на `mcp<2` (проверено на `mcp==1.30.0`).

## Запуск вручную

```powershell
# stdio (так сервер запускает менеджер MCP приложения)
.venv\Scripts\python.exe mcp_demo\jsonplaceholder_server.py

# streamable-http на 127.0.0.1:8001 (endpoint /mcp)
.venv\Scripts\python.exe mcp_demo\jsonplaceholder_server.py http

# проверка синтаксиса
.venv\Scripts\python.exe -m py_compile mcp_demo\jsonplaceholder_server.py
```

## Подключение в приложении

Вид **«MCP»** → сервер `jsonplaceholder`, тип `local` (stdio), команда:

```json
{
  "type": "local",
  "command": [".venv/Scripts/python.exe", "mcp_demo/jsonplaceholder_server.py"],
  "enabled": true,
  "timeout": 15000
}
```

Конфиг хранится в БД (`chats.db`) и переживает рестарт сервера; `enabled`-серверы
подключаются автоматически при старте приложения. На Linux/VPS путь к
интерпретатору в карточке меняется на `.venv/bin/python`.

### Вариант remote (streamable-http)

Если сервер запущен отдельно в режиме `http`, его можно подключить как `remote`:

```powershell
# терминал 1
.venv\Scripts\python.exe mcp_demo\jsonplaceholder_server.py http
```

```json
{
  "type": "remote",
  "url": "http://127.0.0.1:8001/mcp",
  "enabled": true
}
```

## Использование

Просто попросите агента в чате, например:

- «покажи пост 1 из jsonplaceholder»;
- «выведи комментарии поста 3»;
- «первые 5 задач пользователя 2»;
- «данные пользователя 1».

## Обработка ошибок

Ошибки API/сети (`404`, недоступный хост) поднимаются как `ValueError` → MCP
возвращает `isError` с текстом вида `API GET /posts/9999 -> HTTP 404`; сервер
при этом не падает. Строки `Processing request ...` и `HTTP Request: GET ...`
в stderr — штатные логи сервера и `httpx`.

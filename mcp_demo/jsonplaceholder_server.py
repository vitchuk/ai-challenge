import sys
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = "https://jsonplaceholder.typicode.com"
Resource = Literal["posts", "comments", "albums", "photos", "todos", "users"]

mcp = FastMCP("jsonplaceholder", host="127.0.0.1", port=8001)


async def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        response = await client.get(path, params=params)
    if response.status_code >= 400:
        raise ValueError(f"API GET {path} -> HTTP {response.status_code}")
    return response.json() if response.content else None


@mcp.tool()
async def list_items(
    resource: Resource,
    filters: dict[str, Any] | None = None,
    limit: int = 20,
) -> list:
    """List items of a resource. filters -> query params, e.g. {"postId": 1}. limit <= 0 returns all items."""
    items = await _get(f"/{resource}", params=filters)
    if not isinstance(items, list):
        raise ValueError(f"unexpected response for /{resource}")
    return items[:limit] if limit > 0 else items


@mcp.tool()
async def get_item(resource: Resource, id: int) -> dict:
    """Get a single item by id."""
    return await _get(f"/{resource}/{id}")


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport in ("http", "streamable-http"):
        mcp.run(transport="streamable-http")
    else:
        raise SystemExit(f"unknown transport: {transport!r} (expected 'stdio' or 'http')")

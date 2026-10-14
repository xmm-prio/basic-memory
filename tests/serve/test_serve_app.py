"""Tests for the combined MCP + v2 REST app served by `basic-memory serve`.

The app runs under a real uvicorn server on a free loopback port, with its own
config directory and knowledge base under tmp_path, so requests travel the same
path an external MCP client and an HTTP client would.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import uvicorn

from basic_memory.config import BasicMemoryConfig, ConfigManager, ProjectEntry
from basic_memory.serve import create_app

PROJECT_NAME = "kb"
STARTUP_TIMEOUT_SECONDS = 60


@pytest.fixture
def knowledge_base(config_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate config + index in tmp and register a single knowledge base."""
    project_dir = config_home / PROJECT_NAME
    project_dir.mkdir()
    monkeypatch.setenv("BASIC_MEMORY_CONFIG_DIR", str(config_home / "config"))
    monkeypatch.setenv("BASIC_MEMORY_HOME", str(project_dir))
    ConfigManager().save_config(
        BasicMemoryConfig(
            env="test",
            projects={PROJECT_NAME: ProjectEntry(path=str(project_dir))},
            default_project=PROJECT_NAME,
        )
    )
    return project_dir


@pytest_asyncio.fixture
async def client(knowledge_base: Path) -> AsyncIterator[httpx.AsyncClient]:
    """Serve the combined app on a free loopback port and yield a client for it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), lifespan="on", log_level="warning"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(STARTUP_TIMEOUT_SECONDS):
            while not server.started:
                if serving.done():
                    serving.result()
                    pytest.fail("server stopped before it started")
                await asyncio.sleep(0.05)

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as http:
            yield http
    finally:
        server.should_exit = True
        await serving
        sock.close()


async def mcp_request(
    client: httpx.AsyncClient, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Send one stateless JSON-RPC request to /mcp and return its result."""
    response = await client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers={"Accept": "application/json, text/event-stream"},
    )
    response.raise_for_status()
    body = response.json()
    assert "error" not in body, body
    return body["result"]


@pytest.mark.asyncio
async def test_one_port_serves_mcp_tools_and_rest_projects(client: httpx.AsyncClient) -> None:
    tools = await mcp_request(client, "tools/list")
    projects = await client.get("/v2/projects/")

    assert {"write_note", "read_note", "search_notes"} <= {tool["name"] for tool in tools["tools"]}
    assert projects.status_code == 200
    assert [project["name"] for project in projects.json()["projects"]] == [PROJECT_NAME]


@pytest.mark.asyncio
async def test_note_written_over_rest_is_readable_over_mcp(client: httpx.AsyncClient) -> None:
    projects = (await client.get("/v2/projects/")).json()["projects"]
    written = await client.post(
        f"/v2/projects/{projects[0]['external_id']}/knowledge/write",
        json={
            "note": {
                "title": "Shared note",
                "directory": "notes",
                "content": "Written over REST, read over MCP.",
            }
        },
    )

    result = await mcp_request(
        client,
        "tools/call",
        {"name": "read_note", "arguments": {"identifier": "Shared note", "project": PROJECT_NAME}},
    )

    assert written.json()["kind"] == "created"
    assert "Written over REST, read over MCP." in result["content"][0]["text"]

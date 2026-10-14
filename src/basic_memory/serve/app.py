"""ASGI app that serves MCP streamable HTTP and the v2 REST API from one process."""

from starlette.applications import Starlette

import basic_memory.mcp.prompts  # noqa: F401  # registers prompts on the MCP server
import basic_memory.mcp.resources  # noqa: F401  # registers resources on the MCP server
import basic_memory.mcp.tools  # noqa: F401  # registers tools on the MCP server
from basic_memory.api.app import app as api_app
from basic_memory.mcp.server import mcp

MCP_PATH = "/mcp"


def create_app() -> Starlette:
    """Build the combined ASGI app.

    MCP streamable HTTP is served at ``/mcp`` in stateless JSON-response mode, so a
    client can POST a JSON-RPC request without an initialize handshake or session id.
    Every other path falls through to the v2 REST API.

    Only the MCP lifespan runs. It initializes the database, starts file watching and
    installs the API container that both external REST requests and the in-process
    client used by MCP tools resolve, so both surfaces read and write the same data.
    The REST app's own lifespan stays unused because it would initialize and watch a
    second time. The Host/Origin guard installed by the MCP app covers both surfaces.
    """
    app = mcp.http_app(
        path=MCP_PATH,
        transport="streamable-http",
        json_response=True,
        stateless_http=True,
        host_origin_protection="auto",
    )
    app.mount("/", api_app)
    return app

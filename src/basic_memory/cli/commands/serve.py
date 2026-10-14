"""Serve MCP and the v2 REST API from one local HTTP process."""

import os

import typer
from loguru import logger

from basic_memory.cli.app import app
from basic_memory.cli.commands.mcp import is_loopback_host
from basic_memory.config import init_mcp_logging


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        help="Host to bind. Defaults to loopback; the server has no authentication",
    ),
    port: int = typer.Option(8000, help="Port to bind"),
) -> None:
    """Run one HTTP server exposing MCP at /mcp and the v2 REST API.

    MCP uses streamable HTTP in stateless JSON-response mode. Initialization, file
    indexing, watching and cleanup are handled by the MCP lifespan, as with
    `basic-memory mcp`.
    """
    # Trigger: serve command invocation.
    # Why: MCP tools must reach the same in-process REST app that HTTP clients use,
    #      never a cloud project.
    # Outcome: same explicit local routing as `basic-memory mcp` over HTTP.
    os.environ["BASIC_MEMORY_FORCE_LOCAL"] = "true"
    os.environ.pop("BASIC_MEMORY_FORCE_CLOUD", None)
    os.environ["BASIC_MEMORY_EXPLICIT_ROUTING"] = "true"

    init_mcp_logging()

    if not is_loopback_host(host):
        warning = (
            f"Server is bound to {host}, which is reachable from other machines. "
            "It has no authentication; bind to 127.0.0.1 unless the network is trusted."
        )
        typer.echo(f"Warning: {warning}", err=True)
        logger.warning(warning)

    # Deferred: building the app imports the MCP server and the REST API.
    import uvicorn

    from basic_memory.serve import create_app

    logger.info(f"Starting MCP + REST server on {host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, lifespan="on", timeout_graceful_shutdown=2)

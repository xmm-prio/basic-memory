"""Single-process HTTP server combining MCP and the v2 REST API."""

from .app import create_app

__all__ = ["create_app"]

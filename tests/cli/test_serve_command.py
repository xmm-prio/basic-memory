"""Tests for the `basic-memory serve` command's bind options.

The server process boundary (uvicorn) is replaced by a recorder so the command
returns immediately; what the combined app serves is covered in tests/serve.
"""

from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

import basic_memory.cli.commands.serve as serve_mod
from basic_memory.cli.main import app as cli_app

runner = CliRunner()


@pytest.fixture
def captured_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the uvicorn.run call instead of binding a socket."""
    captured: dict[str, Any] = {}

    def record_run(app: Any, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", record_run)
    monkeypatch.setattr(serve_mod, "init_mcp_logging", lambda: None)
    return captured


def test_serve_binds_loopback_on_requested_port(captured_run: dict[str, Any]) -> None:
    result = runner.invoke(cli_app, ["serve", "--port", "8765"])

    assert result.exit_code == 0, result.output
    assert captured_run["host"] == "127.0.0.1"
    assert captured_run["port"] == 8765
    assert "no authentication" not in result.output


def test_serve_warns_when_bound_beyond_loopback(captured_run: dict[str, Any]) -> None:
    result = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert captured_run["host"] == "0.0.0.0"
    assert "no authentication" in result.output

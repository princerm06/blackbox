"""Alexa+ / MCP adapter for BlackBox.

Run:
    python -m pip install -e ".[amazon]"
    python -m blackbox.amazon_mcp

The server exposes Streamable HTTP at /mcp and uses MCP SDK v2, which serves
both the 2026-07-28 protocol revision and legacy 2025-era clients.
"""

from __future__ import annotations

import os

from mcp.server import MCPServer

from blackbox.amazon_memory import (
    recommended_check,
    what_changed,
    what_do_you_know,
    where_are_my_keys,
)


mcp = MCPServer("BlackBox Visual Memory")


@mcp.tool()
def where_are_my_keys_tool() -> dict[str, object]:
    """Ask BlackBox where the missing keys are likely to be."""
    return where_are_my_keys()


@mcp.tool()
def what_changed_tool() -> dict[str, object]:
    """Ask BlackBox what changed in the observed scene over time."""
    return what_changed()


@mcp.tool()
def what_do_you_know_tool() -> dict[str, object]:
    """Separate observed facts, inferences, and unknowns about the scene."""
    return what_do_you_know()


@mcp.tool()
def recommended_check_tool() -> dict[str, object]:
    """Ask BlackBox what the user should inspect next to reduce uncertainty."""
    return recommended_check()


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        json_response=True,
        stateless_http=True,
    )

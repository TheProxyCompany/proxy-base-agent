from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool


class MCPClient:
    """
    A client for the MCP protocol.
    """

    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, server: str):
        is_python = server.endswith(".py")
        is_js = server.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server], env=None)

        # Enter context with a single exit stack
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport

        # Use the same exit stack consistently
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        assert self.session is not None
        await self.session.initialize()

    async def get_tools(self) -> list[Tool]:
        """
        Get the tools available from the MCP server.
        """
        assert self.session is not None
        tools = await self.session.list_tools()
        return tools.tools

    async def use_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Use a tool on the MCP server.
        """
        assert self.session is not None
        return await self.session.call_tool(name, arguments)

    async def disconnect(self):
        """
        Disconnect from the MCP server.
        """
        if self.session:
            self.session = None
        await self.exit_stack.aclose()

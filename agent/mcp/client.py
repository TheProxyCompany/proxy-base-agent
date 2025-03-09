from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from agent.mcp.server import MCPServer


class MCPClient:
    """
    A client for the MCP protocol.
    """

    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(
        self,
        server: MCPServer,
    ):
        server_params = StdioServerParameters(
            command=server.command,
            args=server.args,
        )
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

    async def use_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Use a tool on the MCP server.
        """
        assert self.session is not None
        tool_result = await self.session.call_tool(name, arguments)
        for content in tool_result.content:
            if isinstance(content, TextContent):
                return content.text
            elif isinstance(content, ImageContent):
                return str(content)
            elif isinstance(content, EmbeddedResource):
                return str(content)
        raise ValueError("No text content found in tool result")

    async def disconnect(self):
        """
        Disconnect from the MCP server.
        """
        if self.session:
            await self.exit_stack.aclose()
            self.session = None

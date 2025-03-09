import logging

from agent.mcp.client import MCPClient
from agent.mcp.server import MCPServer
from agent.tools import Tool, ToolCall

logger = logging.getLogger(__name__)

class MCPHost:
    """
    A class that handles MCP servers and clients.
    """
    def __init__(self):
        self.mcp_clients: dict[str, MCPClient] = {}
        available_servers = MCPServer.load_available_servers_from_json()
        self.available_servers = {server.name: server for server in available_servers}

    async def connect_to_server(
        self,
        server_name: str,
        command: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        Connect to the MCP server and get the tools.
        """
        self.mcp_clients[server_name] = MCPClient()
        await self.mcp_clients[server_name].connect(server_name, command, env)

    async def get_tools(self, server_name: str) -> list[Tool]:
        """
        Get the tools from the given MCP server.
        """
        if server_name not in self.mcp_clients:
            raise ValueError(f"MCP server {server_name} not found")

        new_tools = []
        for tool in await self.mcp_clients[server_name].get_tools():
            new_tool = Tool.from_mcp_tool(tool, server_name)
            new_tools.append(new_tool)

        return new_tools

    async def use_tool(self, server_name: str, tool_call: ToolCall) -> str:
        """
        Use a tool from the given MCP server.

        Raises:
            ValueError: If the MCP server is not found.
        """
        if server_name not in self.mcp_clients:
            raise ValueError(f"MCP server {server_name} not found")

        return await self.mcp_clients[server_name].use_tool(tool_call.name, tool_call.arguments)


    async def cleanup(self):
        """
        Cleanup the MCP clients.
        """
        for client in self.mcp_clients.values():
            await client.disconnect()

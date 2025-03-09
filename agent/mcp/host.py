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
        self.available_servers = {server.identifier: server for server in available_servers}

    async def connect_to_server(self, server_id: str) -> list[Tool]:
        """
        Connect to the MCP server and get the tools.
        """
        if server_id not in self.available_servers:
            raise ValueError(f"MCP server {server_id} not found")

        requested_server = self.available_servers[server_id]
        requested_server.download_server()

        new_client = MCPClient()
        await new_client.connect(requested_server)
        self.mcp_clients[server_id] = new_client

        return await self.get_tools(server_id)

    async def get_tools(self, server_id: str) -> list[Tool]:
        """
        Get the tools from the given MCP server.
        """
        if server_id not in self.mcp_clients:
            raise ValueError(f"MCP server {server_id} not found")

        new_tools = []
        for tool in await self.mcp_clients[server_id].get_tools():
            new_tool = Tool.from_mcp_tool(tool, server_id)
            new_tools.append(new_tool)

        return new_tools

    async def use_tool(self, server_id: str, tool_call: ToolCall) -> str:
        """
        Use a tool from the given MCP server.

        Raises:
            ValueError: If the MCP server is not found.
        """
        if server_id not in self.mcp_clients:
            raise ValueError(f"MCP server {server_id} not found")

        return await self.mcp_clients[server_id].use_tool(tool_call.name, tool_call.arguments or {})


    async def cleanup(self):
        """
        Cleanup the MCP clients.
        """
        for client in self.mcp_clients.values():
            await client.disconnect()

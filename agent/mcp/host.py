import base64
import logging
import uuid
from io import BytesIO

from PIL import Image

from agent.mcp.client import MCPClient
from agent.mcp.server import MCPServer
from agent.system.interaction import Interaction
from agent.tools import Tool, ToolCall
from mcp.types import ImageContent

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
        # requested_server.download_server()

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

    async def use_tool(self, server_id: str, tool_call: ToolCall) -> Interaction:
        """
        Use a tool from the given MCP server.

        Raises:
            ValueError: If the MCP server is not found.
        """
        if server_id not in self.mcp_clients:
            raise ValueError(f"MCP server {server_id} not found")

        result = await self.mcp_clients[server_id].use_tool(tool_call.name, tool_call.arguments or {})
        if isinstance(result, ImageContent):
            base64_data = result.data
            image = Image.open(BytesIO(base64.b64decode(base64_data)))
            image_path = f"/tmp/image_{uuid.uuid4().hex}.png"
            image.save(image_path)
            result = image_path
            interaction = Interaction(
                role=Interaction.Role.TOOL,
                image_url=image_path,
            )
            return interaction
        else:
            interaction = Interaction(
                role=Interaction.Role.TOOL,
                content=str(result),
            )
            return interaction


    async def cleanup(self):
        """
        Cleanup the MCP clients.
        """
        for client in self.mcp_clients.values():
            await client.disconnect()

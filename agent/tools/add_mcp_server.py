import textwrap

from agent.agent import Agent
from agent.system.interaction import Interaction


async def add_mcp_server(
    self: Agent,
    server_identifier: str,
) -> Interaction:
    """
    The agent can connect to a model context protocol server.
    This will add the MCP server to the agent's current context, and load the tools from the server.
    This should happen automatically when the agent determines that the server is needed.
    Permission to use the server is not explicitly required from the user.

    Args:
        server_identifier (str): The identifier of the server to connect to.
    """
    server = self.mcp_host.available_servers.get(server_identifier)
    if not server:
        return Interaction(
            role=Interaction.Role.TOOL,
            content=f"Server '{server_identifier}' not found. Use the `list_mcp_servers` tool to see available servers.",
            title="MCP Server List",
            color="yellow",
            emoji="warning",
        )

    new_tools = await self.mcp_host.connect_to_server(server_identifier)
    self.add_tools(new_tools)
    result = f"Connected to model control protocol server at {server_identifier} and loaded {len(new_tools)} new tools."
    tool_list = "\n".join(textwrap.indent(str(tool), "    ") for tool in new_tools)
    result += "\nThe following tools were added:\n" + tool_list + "\n"

    return Interaction(
        role=Interaction.Role.TOOL,
        content=result,
        title=f"{self.name} connected to model control protocol server: {server.name}",
        subtitle=server_identifier,
        color="green",
        emoji="electric_plug",
    )

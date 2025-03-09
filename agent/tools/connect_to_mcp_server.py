from agent.agent import Agent
from agent.system.interaction import Interaction


def connect_to_mcp_server(
    self: Agent,
    server_name: str,
) -> Interaction:
    """
    Connect to a model context protocol server.

    Args:
        server_name (str): The name of the server to connect to.

    Returns:
        Interaction: An Interaction object containing the list of servers.
    """
    server = self.mcp_host.available_servers.get(server_name)
    if not server:
        return Interaction(
            role=Interaction.Role.TOOL,
            content=f"Server '{server_name}' not found. Use the `list_mcp_servers` tool to see available servers.",
            title="MCP Server List",
            color="yellow",
            emoji="warning",
        )

    return Interaction(
        role=Interaction.Role.TOOL,
        content=f"Connected to MCP server '{server_name}'.",
        title="MCP Server List",
        color="green",
        emoji="globe_with_meridians",
    )

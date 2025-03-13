from agent.agent import Agent
from agent.system.interaction import Interaction


def list_mcp_servers(
    self: Agent,
    limit: int | None = None,
) -> Interaction:
    """
    List available model context protocol servers, returning their names, descriptions, and runtime environments.
    Model Context Protocol (MCP) is a protocol for integrating external services and APIs. These are the only MCP servers available to the agent.
    If the user requires capabilities that are outside your core functionality, use this tool to see what servers are available.

    Args:
        limit (int, optional): Limit the number of servers returned. Defaults to None (all servers).
    """
    available_servers = list(self.mcp_host.available_servers.values())

    if limit:
        available_servers = available_servers[:limit]

    if not available_servers:
        return Interaction(
            role=Interaction.Role.TOOL,
            content="No MCP servers found.",
            title="MCP Server List",
            color="yellow",
            emoji="warning",
        )

    server_list = [str(server) for server in available_servers]
    joined_servers = "\n".join(server_list)
    content = f"Available MCP servers:\n\n{joined_servers}\n\n"
    content += "Use the `add_mcp_server` tool to connect to a server."

    return Interaction(
        role=Interaction.Role.TOOL,
        content=content,
        title="Model Context Protocol Servers",
        color="green",
        emoji="globe_with_meridians",
    )

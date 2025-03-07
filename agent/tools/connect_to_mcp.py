import asyncio

from agent.agent import Agent
from agent.system.interaction import Interaction


def connect_to_mcp(
    self: Agent,
    mcp_server: str,
) -> Interaction:
    """
    Connect to an Model Control Protocol server.

    This tool will connect to an MCP server and load the tools from the server.
    """
    try:
        new_tools = asyncio.run(self.connect_to_mcp_and_get_tools(mcp_server))
        result = f"Connected to MCP server and loaded {len(new_tools)} tools."
        result += "\n\nThe following tools were added:" + "\n---".join(str(tool) for tool in new_tools)
    except Exception as e:
        return Interaction(
            role=Interaction.Role.TOOL,
            content=f"Error connecting to MCP server: {e}",
        )

    return Interaction(
        role=Interaction.Role.TOOL,
        content=result,
        title=self.name + " connected to a model control protocol server",
        subtitle=mcp_server,
        color="green",
        emoji="plug",
    )

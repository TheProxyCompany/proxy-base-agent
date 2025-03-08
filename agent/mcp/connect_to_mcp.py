import textwrap

from agent.agent import Agent
from agent.system.interaction import Interaction


async def connect_to_mcp(
    self: Agent,
    mcp_server: str,
    command: str | None = None,
    env: dict[str, str] | None = None,
) -> Interaction:
    new_tools = await self.connect_to_mcp_and_get_tools(mcp_server, command, env)

    result = f"Connected to model control protocol server at {mcp_server} and loaded {len(new_tools)} new tools."
    tool_list = "\n".join(textwrap.indent(str(tool), "    ") for tool in new_tools)
    result += "\nThe following tools were added:\n" + tool_list

    return Interaction(
        role=Interaction.Role.TOOL,
        content=result,
        title=f"{self.name} connected to a model control protocol server",
        subtitle=mcp_server,
        color="green",
        emoji="electric_plug",
    )

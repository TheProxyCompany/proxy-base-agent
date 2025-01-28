from agent.agent import Agent
from agent.event import Event, EventState


def list_tools(self: Agent, name: str | None = None) -> Event:
    """
    List all tools available to the agent.

    Arguments:
        name (str | None):
            Optional tool name to get specific information about. If None (default), returns information about all available tools.
    """

    tools: list[str] = []
    content = self.tool_list
    if not name:
        for tool in self.state.tools_map.values():
            tools.append(f"{tool}")
            content = "\n".join(tools)

    return Event(
        state=EventState.TOOL,
        name=self.state.name + "'s tools",
        content=content,
    )

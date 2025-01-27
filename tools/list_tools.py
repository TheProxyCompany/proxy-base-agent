from agent.agent import Agent
from agent.event import Event, EventState


def list_tools(self: Agent, query: str | list[str] | None = None) -> Event:
    """
    List all tools available to the agent.

    Arguments:
        query (str | list[str] | None):
            If provided, only tools matching the query are returned.
            Default is None, and all tools are returned.
    """

    tools: list[str] = []
    if not query:
        for tool in self.state.tools_map.values():
            tools.append(f"{tool}")
    else:
        for tool in self.state.tools_map.values():
            if tool.name in query:
                tools.append(f"{tool}")

    return Event(
        state=EventState.TOOL,
        name=self.state.name + "'s tools",
        content=tools,
    )

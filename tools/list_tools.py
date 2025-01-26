from agent.agent import Agent
from agent.event import Event, State


def list_tools(self: Agent, short: bool = True) -> Event:
    """
    List all tools available to the agent.

    Arguments:
        short (bool): If True, only the tool name and description are shown. Default is True.
    """

    tools: list[str] = []
    for tool in self.state.tools_map.values():
        if short:
            tools.append(f"{tool}")
        else:
            tools.append(f"{tool!r}")

    return Event(
        role="ipython",
        state=State.TOOL_RESULT,
        name=self.state.name + "'s tools",
        content=tools,
    )

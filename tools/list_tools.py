from agent.agent import Agent
from agent.event import Event, State


def list_tools(self: Agent, brief: bool = True) -> Event:
    """
    List all tools available to the agent.

    If brief is True, only the tool name and description are shown.

    If brief is False, the tool name, description, and full invocation schema are shown.

    Arguments:
        brief (bool): Boolean flag to determine the level of detail in the tool list. Default is True.
    """

    tools: list[str] = []
    for tool in self.state.tools_map.values():
        if brief:
            tools.append(f"{tool}")
        else:
            tools.append(f"{tool!r}")

    return Event(
        role="ipython",
        state=State.TOOL_RESULT,
        name=self.state.name + "'s tools",
        content=tools,
    )

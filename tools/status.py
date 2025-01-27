from agent.agent import Agent, AgentStatus
from agent.event import Event, EventState


def status(self: Agent, new_status: AgentStatus | None = None) -> Event:
    """
    Get the current state of the agent.

    Arguments:
        new_status (AgentStatus | None): If provided, the agent's status is set to this new status.
    """

    name = self.state.name + "'s status"
    if new_status:
        self.status = new_status
        name = self.state.name + "'s new status"

    return Event(
        state=EventState.TOOL,
        name=name,
        content=f"*{self.status}*",
    )

from agent.agent import Agent
from agent.interaction import Interaction


def list_tools(self: Agent) -> Interaction:
    """
    List all tools available to the agent.
    """

    return Interaction(
        role=Interaction.Role.TOOL,
        name=self.name + "'s tools",
        content=self.tool_reminder,
    )

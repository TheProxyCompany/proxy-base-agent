from agent.agent import Agent
from agent.interaction import Interaction


def send_message(self: Agent, message: str) -> Interaction:
    """
    This tool is the only way for the agent to interact with the user.
    Use this tool to summarize the agent's thoughts to the user.

    Arguments:
        message (str):
            The message content to be sent to the recipient.
            This is the only text that the recipient will receive from the agent.
    """

    self.status = Agent.Status.SUCCESS

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        content=message,
        title=self.name,
        color="green",
        emoji="speech_balloon",
        last=True,
        silent=True,
    )

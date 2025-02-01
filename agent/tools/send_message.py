from agent.core.agent import Agent
from agent.core.interaction import Interaction


def send_message(self: Agent, message: str) -> Interaction:
    """
    This tool is the only way to interact with the user.
    Use this tool to summarize your latent thoughts to the user.

    Arguments:
        message (str):
            The message content to be sent to the recipient.
            This is the message that the agent will send to the recipient.
    """

    self.status = Agent.Status.SUCCESS

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        content=message,
        title=self.name,
        color="green",
        emoji="speech_balloon",
        last=True,
    )

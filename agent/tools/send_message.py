from agent.agent import Agent
from agent.interaction import Interaction


def send_message(self: Agent, message: str) -> Interaction:
    """
    Send a formatted message to the user - this is the primary communication channel.

    This tool formats and delivers messages.
    It's designed to be the sole method of direct user communication.

    Args:
        message: The content to send to the user. Should be clear, concise,
            and self-contained as this will be the only visible part of your
            thought process.
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

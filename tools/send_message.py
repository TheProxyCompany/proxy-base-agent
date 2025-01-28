from agent.agent import Agent
from agent.interaction import Interaction


def send_message(self: Agent, message: str, recipient: str = "user") -> Interaction:
    """
    Sends a message to the recipient.

    Arguments:
        message (str):
            The message content to be sent to the recipient.
            Supports all Unicode characters, including emojis.

        recipient (str):
            The recipient of the message. Default is "user".
    """

    self.status = Agent.Status.AWAITING_INPUT

    return Interaction(
        role=Interaction.Role.TOOL,
        content=message,
        name=self.name + " sent a message",
        buffer=recipient,
        color="green",
        emoji="speech_balloon",
    )

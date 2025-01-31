from agent.agent import Agent
from agent.interaction import Interaction


def send_message(self: Agent, message: str) -> Interaction:
    """
    Sends a message to the recipient.

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

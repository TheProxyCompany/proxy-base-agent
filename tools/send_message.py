from agent.agent import Agent, AgentStatus
from agent.event import Event, EventState


def send_message(self: Agent, message: str, recipient: str = "user") -> Event:
    """
    This function sends a message to the recipient.
    Effectively conveys information to the recipient.

    Arguments:
        message (str):
            The message content to be sent to the recipient.
            Supports all Unicode characters, including emojis.

        recipient (str):
            The recipient of the message. Default is "user".
    """

    self.state.step_number = 0
    self.status = AgentStatus.AWAITING_INPUT

    return Event(
        state=EventState.ASSISTANT,
        content=message,
        name=self.state.name + " sent a message to " + recipient,
    )

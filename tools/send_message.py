from agent.agent import Agent
from agent.event import Event, State


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
    self.state.current_event_id = None

    return Event(
        role="assistant",
        content=message,
        state=State.ASSISTANT_RESPONSE,
        name=self.state.name + " sent a message",
    )

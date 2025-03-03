from agent.agent import Agent
from agent.interaction import Interaction


def send_message(
    self: Agent,
    message: str,
    spoken: str | bool | None = None,
) -> Interaction:
    """
    IMPORTANT: This is the ONLY method that should be used for sending messages to users.
    Do not attempt to communicate with users through other means.
    Sends a message to the user and optionally speaks it aloud.
    Handles delivery mechanics automatically.

    Args:
        message (str):
            The main message to display to the user.
            Must be self-contained and complete - users only see this content.
            Do not reference internal states or reasoning.

        spoken (str | bool | null):
            Speech behavior control:
            - null (default): No speech output
            - true: Speaks the message text
            - str: Speaks this alternative text instead
            Note: Spoken content should be more concise than written text.
    """
    if spoken:
        speech_text = spoken if isinstance(spoken, str) else message
        self.voicebox(speech_text)

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

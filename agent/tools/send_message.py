from agent.agent import Agent
from agent.system.interaction import Interaction


def send_message(
    self: Agent,
    message: str,
    spoken: str | bool | None = None,
    wait_for_response: bool = True,
) -> Interaction:
    """
    IMPORTANT: This is the ONLY method that should be used for sending messages to users.
    Do not attempt to communicate with users through other means.
    Sends a message to the user and optionally speaks it aloud.
    Handles delivery mechanics automatically.
    When `wait_for_response` is True, the agent pauses its activity, awaiting the user's reply before proceeding.

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
        wait_for_response (bool):
            If True, the agent will wait for a response from the user before continuing.
            If False, the agent will continue acting immediately.
    """

    if isinstance(spoken, str):
        spoken_lower = spoken.lower()
        if spoken_lower == "true":
            spoken = True
        elif spoken_lower in ("none", "null", "false"):
            spoken = None

    if spoken:
        speech_text = spoken if isinstance(spoken, str) else message
        self.voicebox(speech_text)

    self.status = (
        Agent.Status.SUCCESS
        if wait_for_response
        else Agent.Status.PROCESSING
    )
    return Interaction(
        role=Interaction.Role.ASSISTANT,
        content=message,
        title=self.name,
        color="cyan",
        emoji="alien",
        last=wait_for_response,
        silent=True,
    )

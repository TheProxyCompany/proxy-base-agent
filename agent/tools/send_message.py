from agent.agent import Agent
from agent.system.interaction import Interaction


def send_message(
    self: Agent,
    message: str,
    spoken: str | bool | None = None,
    wait_for_response: bool = True,
) -> Interaction:
    """
    This tool sends a text-based message to the user and optionally speaks it aloud using a text-to-speech engine.
    By default, the agent will wait for a response from the user before proceeding.
    Multiple messages can be sent in a row by setting `wait_for_response` to False.

    IMPORTANT:
        This is the ONLY method that should be used for sending messages to users.
        Do not attempt to communicate with users through other means.

    Args:
        message (str): The message to display to the user. This message should concisely summarize all previous actions, states, and reasoning that led to this point. Be self-contained and complete; users only see this message content, and will not see internal states or processes.
        spoken (str | bool | None): Controls the text-to-speech behavior. Default is no speech output. True will speak the message text aloud. A string will speak this alternative text aloud. Note: Spoken content should generally be more concise than the written `message`.
        wait_for_response (bool): This parameter determines whether the agent waits for user response after sending the message.
    """

    if isinstance(spoken, str):
        spoken_lower = spoken.lower()
        if spoken_lower == "true":
            spoken = True
        elif spoken_lower in ("none", "null", "false"):
            spoken = None

    if spoken and self.voicebox is not None and self.enable_voice:
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

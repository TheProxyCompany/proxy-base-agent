from agent.agent import Agent
from agent.interaction import Interaction


def send_message(
    self: Agent,
    message: str,
    spoken: str | bool | None = False,
) -> Interaction:
    """
    Primary communication tool for interacting with users.
    This is the only method that should be used for direct user interaction.

    Args:
        message (str):
            Message content to send to the user. Must be self-contained
            and coherent, as this will be the only content the user will see.
            Keep messages clear and focused, written knowing that the user
            will not see the internal reasoning process.

        spoken (str | bool | None):
            Controls text-to-speech output:
            - If a string is provided: Speaks that text aloud instead of the written message
            - If True: Speaks the written message aloud
            - If None/False: No speech output
            For better user experience, spoken text should be more concise than written.
    """
    if spoken:
        if isinstance(spoken, bool) or bool(spoken):
            spoken = message
        self.voicebox(spoken)

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

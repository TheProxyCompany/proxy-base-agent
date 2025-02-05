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
        message (str): Message content to send to the user. Must be self-contained
            and coherent since users will not see the internal reasoning process.
            Keep messages clear and focused.
        spoken (str | bool | None):
            If provided, the spoken message is spoken to the user as well as written.
            If True, the written message is spoken.
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

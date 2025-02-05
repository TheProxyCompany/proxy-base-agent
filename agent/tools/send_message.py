from f5_tts_mlx.generate import generate

from agent.agent import Agent
from agent.interaction import Interaction


def send_message(self: Agent, message: str, expect_response: bool = True, speak: bool = False) -> Interaction:
    """
    Primary communication tool for interacting with users.

    Formats and sends messages through a dedicated channel.
    This is the only method that should be used for direct user interaction.

    Text based communication is preferred, but spoken communication is allowed.
    Spoken communication is not recommended for important messages.

    Args:
        message (str): Message content to send to the user. Must be self-contained
            and coherent since users will not see the internal reasoning process.
            Keep messages clear and focused.
        expect_response (bool): Controls conversation flow.
            - True: Pauses execution to await user input.
            - False: Continues execution without waiting (fire and forget).
        spoken (bool):
            If True, the message is spoken to the user as well as written.
            5 second duration. Default is False.
    """
    if speak:
        generate(message)

    if expect_response:
        self.status = Agent.Status.SUCCESS

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        content=message,
        title=self.name,
        color="green",
        emoji="speech_balloon",
        last=expect_response,
        silent=True,
    )

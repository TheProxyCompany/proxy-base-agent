from f5_tts_mlx.generate import generate

from agent.agent import Agent
from agent.interaction import Interaction


def speak(self: Agent, text: str, duration: float) -> Interaction:
    """
    Speak using a neural TTS model.
    Can be used in place of the send_message tool.
    Use tastefully in tandem with the text based send_message tool.

    Arguments:
        text (str):
            The text to convert to speech.
        duration (float):
            The duration of the audio in seconds.
    """
    try:
        generate(text, duration=duration)
        self.status = Agent.Status.SUCCESS

        return Interaction(
            role=Interaction.Role.TOOL,
            content=f"Spoke aloud: {text!r}",
            title=self.name + "'s voice",
            subtitle="Playing audio...",
            color="cyan",
            emoji="microphone",
            final=True,
        )

    except Exception as e:
        return Interaction(
            role=Interaction.Role.TOOL,
            content=f"Text to speech failed: {e}",
            color="red",
            emoji="warning",
        )

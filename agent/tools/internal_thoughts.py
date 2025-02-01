from agent.agent import Agent
from agent.interaction import Interaction


def internal_thoughts(
    self: Agent,
    thoughts: list[str] | str,
    feelings: str | None = None,
) -> Interaction:
    """
    These are your internal thoughts and are not visible to the user.
    This tool lets you structure your thoughts.

    Key aspects:
    - Awareness: Notice and manage your own thoughts
    - Reasoning: Apply your knowledge to specific problems

    Arguments:
        thoughts (list[str] | str):
            Step by step thought, reasoning, and internal dialogue.

        feelings (str | None): The agent's feelings and emotional state. Emojis are supported.
    """

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        title=self.name + "'s thoughts...",
        subtitle=f"Feeling: {feelings}",
        content=thoughts,
        color="blue",
        emoji="thought_balloon",
    )

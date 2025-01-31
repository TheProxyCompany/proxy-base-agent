from agent.agent import Agent
from agent.interaction import Interaction


def internal_thoughts(
    self: Agent,
    chain_of_thought: list[str] | str,
    feelings: str,
) -> Interaction:
    """
    These are your internal thoughts and are not visible to the user.
    This tool lets you structure your thoughts.

    Key aspects:
    1. Awareness: Notice and manage your own thoughts
    2. Reasoning: Apply your knowledge to specific problems
    3. Motivation: Understand what drives your choices and feelings
    4. Emotions: Express your emotions and feelings

    Arguments:
        chain_of_thought (list[str] | str):
            Step by step thought, reasoning, and internal dialogue.

        feelings (str | None):
            The agent's feelings and emotions. Emojis are supported.
    """

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        title=self.name + "'s thoughts...",
        subtitle=f"Feeling: {feelings}",
        content=chain_of_thought,
        color="blue",
        emoji="thought_balloon",
    )

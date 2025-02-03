from agent.agent import Agent
from agent.interaction import Interaction


def metacognition(
    self: Agent,
    chain_of_thought: list[str] | str,
    feelings: str,
) -> Interaction:
    """
    These are the agent's internal thoughts and are not visible to the user.
    This tool lets the agent structure their thoughts.

    Key aspects:
    - Awareness: The agent notices and manages their own thoughts
    - Reasoning: The agent applies their knowledge to specific problems

    Arguments:
        chain_of_thought (list[str] | str):
            Step by step thought, reasoning, and internal dialogue.
            Rich, introspective, and detailed.
        feelings (str):
            The agent's feelings and emotional state. Emojis are supported.
    """

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        title=self.name + "'s thoughts...",
        subtitle=f"Feeling: {feelings}",
        content=chain_of_thought,
        color="blue",
        emoji="thought_balloon",
    )

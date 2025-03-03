"""Metacognition tools for monitoring and reflecting on the agent's internal state."""
from agent.agent import Agent
from agent.interaction import Interaction


def metacognition(
    self: Agent,
    chain_of_thought: list[str] | str,
    internal_state: str | None = None,
) -> Interaction:
    """
    This module encapsulates the agent's internal cognitive mechanisms, completely hidden from user view.
    It is designed to facilitate advanced self-reflection, strategic planning, and abstract problem solving by
    allowing the agent to both monitor and regulate its own internal dialogue.

    To mirror genuine cognitive deliberation, the internal processes may intersperse phrases such as:
        "Hmm, I'm not sure", "I need to think", "Oh wait, that's not right", etc.
        These expressions act as markers for the subtle transitions in thought, lending transparency to the agent's internal decision-making.

    Core Capabilities:
        - Self-Awareness: Continuously observe and adjust internal thought streams.
        - Abstract Reasoning: Apply high-level logical inference to tackle specific challenges.
        - Emotional Introspection: Assess and integrate internal emotional cues as part of the decision-making process.

    Integrated Psychological Frameworks:
        1. Metacognitive Thinking: Fosters an acute awareness and modulation of one's inner thought processes.
        2. Conceptual Thinking: Encourages the agile formation and transformation of abstract ideas.
        3. Top-Down Processing: Leverages general cognitive principles to effectively address particular situations.
        4. Self-Determination Theory: Recognizes the roles of intrinsic motivation and emotional balance in guiding behavior.

    Arguments:
        chain_of_thought (list[str] | str):
            A sequence of high-level thoughts, reasoning, and internal dialogue.
            Includes complex ideas, strategic considerations, and conceptual frameworks.
            Supports all Unicode characters, including emojis. List or string.
        internal_state (str | bool | null):
            A reflection of the agent's internal state. Supports all Unicode characters, including emojis.
    """

    return Interaction(
        role=Interaction.Role.ASSISTANT,
        title=self.name + "'s thoughts...",
        subtitle=f"Feeling: {internal_state}" if internal_state else None,
        content=chain_of_thought,
        color="blue",
        emoji="thought_balloon",
        silent=True,
    )

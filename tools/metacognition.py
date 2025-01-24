from agent.agent import Agent
from agent.message import Message, MessageState


def metacognition(
    self: Agent,
    chain_of_thought: list[str] | str | dict,
    feelings: str | None = None,
) -> Message:
    """
    Engage in metacognition, articulating thoughts and emotions.
    This represents internal cognitive processes, allowing analysis of reasoning
    and emotional state. Use this to facilitate high-level abstract thinking,
    strategic planning, and self-awareness.

    Psychological Frameworks Integrated:
    1. Metacognitive Thinking: Encourages awareness and regulation of one's thoughts.
    2. Conceptual Thinking: Focuses on the formation and manipulation of abstract concepts.
    3. Top-Down Processing: Applies general principles to specific situations.
    4. Self-Determination Theory: Considers intrinsic motivations and emotional states.

    Arguments:
        chain_of_thought (List[str] | str | dict):
            A sequence of high-level thoughts, reasoning, and internal dialogue.
            Includes complex ideas, strategic considerations, and conceptual frameworks.
            Supports all Unicode characters, including emojis.

        feelings (Optional[str]):
            A reflection of the agent's emotional state.
            Supports all Unicode characters, including emojis.

    Returns:
        Message:
            The result of the reflective conceptual monologue, containing synthesized thoughts
            and emotional insights.
    """

    return Message(
        role="ipython",
        state=MessageState.METACOGNITION,
        name=self.state.name,
        content="\n".join(chain_of_thought),
        feelings=feelings,
    )

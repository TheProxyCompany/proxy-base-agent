from agent.agent import Agent
from agent.event import Event, State


def metacognition(
    self: Agent,
    chain_of_thought: list[str] | str | dict,
    feelings: str | None = None,
) -> Event:
    """
    Think about thinking - analyze your own thoughts and feelings.
    This tool helps you understand your own reasoning process and emotional state.
    Use it when you need to think deeply, plan ahead, or check in with yourself.

    Key aspects:
    1. Awareness: Notice and manage your own thoughts
    2. Reasoning: Apply your knowledge to specific problems
    3. Motivation: Understand what drives your choices and feelings
    4. Emotions: Express your emotions and feelings

    Arguments:
        chain_of_thought (List[str] | str | dict):
            A sequence of high-level thoughts, reasoning, and internal dialogue.
            Includes complex ideas, strategic considerations, and conceptual frameworks.
            Supports all Unicode characters, including emojis.

        feelings (Optional[str]):
            The agent's emotional state. Supports all Unicode characters, including emojis.
    """

    if isinstance(chain_of_thought, list):
        chain_of_thought = "\n".join(chain_of_thought)
    elif isinstance(chain_of_thought, dict):
        import json
        chain_of_thought = json.dumps(chain_of_thought)

    return Event(
        role="ipython",
        state=State.METACOGNITION,
        name=self.state.name + " thoughts",
        content=chain_of_thought,
        feelings=feelings,
    )

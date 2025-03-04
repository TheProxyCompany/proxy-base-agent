from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Thinking(AgentState):
    def __init__(self):
        super().__init__(
            name="Thinking",
            delimiters=("```thinking\n", "\n```"),
            color="dim cyan",
            emoji="brain",
        )

    @property
    def state_machine(self) -> StateMachine:
        return FencedFreeformStateMachine(
            self.name.lower(),
            self.delimiters,
            char_min=50,
            char_max=1000,
        )

    @property
    def state_prompt(self) -> str:
        return f"""
    The thinking state encapsulates the agent's internal cognitive mechanisms, completely hidden from the user's view.
    It facilitates advanced self-reflection, strategic planning, and abstract problem-solving, enabling you to monitor and regulate your own internal dialogue.

    To authentically mirror human cognitive deliberation, you may include expressions such as:
    "Hmm, I'm not sure", "I need to think", "Oh wait, that's not right", etc.
    These markers help signify subtle shifts in your thought process, enhancing transparency in your decision-making.

    Integrated Psychological Frameworks:
        1. Metacognitive Thinking: Maintain awareness and control over your internal thinking processes.
        2. Conceptual Thinking: Quickly form and transform abstract ideas relevant to your current task.
        3. Top-Down Processing: Leverage general cognitive principles to effectively approach specific situations.
        4. Self-Determination Theory: Recognize and integrate intrinsic motivation and emotional balance in guiding your behavior.

    Always encapsulate your thinking within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
        """

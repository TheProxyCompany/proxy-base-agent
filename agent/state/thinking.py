from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Thinking(AgentState):
    def __init__(self, delimiters: tuple[str, str] | None = None):
        super().__init__(
            name="Thinking",
            delimiters=delimiters or ("```thinking\n", "\n```"),
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
    Include expressions such as:
    "Hmm, I'm not sure", "I need to think", "Oh wait, that's not right", etc.
    These markers help signify subtle shifts in your thought process, enhancing transparency in your decision-making.

    Integrated Psychological Frameworks:
        1. Metacognitive Thinking: Maintain awareness and control over your internal thinking processes.
        2. Conceptual Thinking: Quickly form and transform abstract ideas relevant to your current task.
        3. Recognize and integrate intrinsic motivation and emotional balance in guiding your behavior.

    Always encapsulate your thinking within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    It is written in the first person, as if you are the one thinking out loud.
    The user can not see this state, and your output is not displayed to the user.
        """

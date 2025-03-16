from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Thinking(AgentState):
    def __init__(self, delimiters: tuple[str, str] | None = None, character_max: int = 1000):
        super().__init__(
            identifier="thinking",
            readable_name="Thinking",
            delimiters=delimiters or ("```thinking\n", "\n```"),
            color="dim cyan",
            emoji="brain",
        )
        self.character_max = character_max

    @property
    def state_machine(self) -> StateMachine:
        return FencedFreeformStateMachine(
            self.identifier,
            self.delimiters,
            char_min=50,
            char_max=self.character_max,
        )

    @property
    def state_prompt(self) -> str:
        return f"""
    The Thinking state is for deliberate, System 2 thought processes.
    When using this state, you should simulate a conscious, effortful, and analytical mode of thinking, akin to a human pausing to reflect with "Wait, am I sure about that?".
    It's about self-awareness of your cognitive process, identifying potential flaws or biases in your reasoning, and actively correcting them.
    This state is not for directly answering questions, but for introspective thinking and modeling your own cognitive processes.

    Always encapsulate your thinking within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    This state is private and hidden from the user.
        """

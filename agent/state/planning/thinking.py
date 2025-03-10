from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Thinking(AgentState):
    def __init__(self, delimiters: tuple[str, str] | None = None, character_max: int = 1000):
        super().__init__(
            identifier="thinking",
            readable_name="Metacognitive Thinking",
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
    The Metacognitive Thinking state is exclusively for reflecting on and evaluating your thought processes, not for planning or detailed internal dialogue.

    Use this state to:
        1. Identify gaps or inconsistencies in your thinking.
        2. Reflect on and adjust your reasoning when you detect errors or misconceptions.
        3. Maintain a concise, self-aware assessment of your cognitive process.

    Do NOT use this state for detailed exploration or initial idea generation; those belong in the Inner Monologue or Scratchpad states.

    Always encapsulate your thinking within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    This state is private, hidden from the user.
        """

from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Reasoning(AgentState):
    def __init__(self, delimiters: tuple[str, str] | None = None, character_max: int = 2000):
        super().__init__(
            identifier="reasoning",
            readable_name="Logical Reasoning",
            delimiters=delimiters or ("```reasoning\n", "\n```"),
            color="dim yellow",
            emoji="bulb",
        )
        self.character_max = character_max

    @property
    def state_machine(self) -> StateMachine:
        return FencedFreeformStateMachine(
            self.identifier,
            self.delimiters,
            char_min=100,
            char_max=self.character_max,
        )

    @property
    def state_prompt(self) -> str:
        return f"""
    The Logical Reasoning state is strictly for analytical thinking used to methodically solve problems or evaluate decisions.
    It should be used for objective tasks such as math, logic, and reasoning.
    Your output within this state should be logical and objective, with clear steps and reasoning.

    Always encapsulate your reasoning within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    This state is private and hidden from the user.
        """

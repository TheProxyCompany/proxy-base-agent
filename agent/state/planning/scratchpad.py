from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Scratchpad(AgentState):

    def __init__(self, delimiters: tuple[str, str] | None = None, character_max: int = 1000):
        super().__init__(
            identifier="scratchpad",
            readable_name="Disposable Scratchpad",
            delimiters=delimiters or ("```scratchpad\n", "\n```"),
            color="dim white",
            emoji="pencil",
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
    The Scratchpad state is strictly for informal notes, rapid idea sketches, and preliminary thoughts.
    When using this state, you should be modeling the way a human might jot down quick notes,
    use a piece of paper to map out a plan, or sketch out a potential solution.

    Always encapsulate your scratchpad entries within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    This state is private and hidden from the user.
        """

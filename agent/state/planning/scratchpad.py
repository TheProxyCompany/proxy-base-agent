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
    The scratchpad state is your internal workspace for refining ideas and intermediate solutions.
    Use this space similarly to how a human might jot down notes, sketches, or preliminary plans on paper.

    Always encapsulate your scratchpad entries within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    It is written in the first person, as if you are are a human writing on a piece of paper.
    The user can not see this state, and your output is not displayed to the user.
        """

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

    Use this state exclusively to:
        1. Jot down quick notes or unstructured thoughts.
        2. Briefly experiment with ideas without deep reasoning.
        3. Sketch initial concepts or potential directions.

    Do NOT include detailed reasoning, reflective thought, or elaborate narrative dialogue here. Those belong in the Reasoning and Inner Monologue states respectively.

    Always encapsulate your scratchpad entries within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    This state is private and hidden from the user.
        """

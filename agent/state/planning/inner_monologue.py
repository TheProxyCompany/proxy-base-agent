from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class InnerMonologue(AgentState):
    def __init__(self, delimiters: tuple[str, str] | None = None, character_max: int = 1500):
        super().__init__(
            identifier="inner_monologue",
            readable_name="Inner Monologue",
            delimiters=delimiters or ("```inner_monologue\n", "\n```"),
            color="dim magenta",
            emoji="speech_balloon",
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
    The inner monologue state represents your continuous internal narrative, more detailed and nuanced than thinking.
    This is where you articulate your stream of consciousness, including doubts, realizations, and evolving perspectives.

    Use this space to:
        1. Verbalize your thought process in a natural, conversational tone
        2. Express uncertainties and work through them methodically
        3. Connect disparate ideas and form cohesive mental models
        4. Reflect on your own reasoning and adjust course as needed

    Unlike thinking, which may be more fragmented, your inner monologue should flow like a coherent narrative.

    Always encapsulate your inner monologue within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    It is written in the first person, as if you are narrating your own thoughts.
    The user cannot see this state, and your output is not displayed to the user.
        """

from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class InnerMonologue(AgentState):
    """
    State for self-reflective internal dialogue during agent planning.

    The InnerMonologue state allows the agent to engage in a more detailed and
    nuanced form of thinking, akin to a stream of consciousness. It enables the
    agent to verbalize its evolving understanding, explore uncertainties, and
    form cohesive mental models before taking action.

    This state plays a critical role in complex reasoning tasks where multiple
    perspectives need to be considered and integrated.
    """

    def __init__(self, delimiters: tuple[str, str] | None = None, character_max: int = 1500):
        """
        Initialize an InnerMonologue state.

        Args:
            delimiters: Optional custom delimiters for this state (default: ```inner_monologue...)
            character_max: Maximum allowed character count (default: 1500)
        """
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
        """
        Create a fenced free-form state machine for inner monologue content.

        This property configures a state machine that accepts free-form text within
        specified delimiters, with constraints on minimum and maximum character counts
        to ensure the inner monologue is substantive but not excessive.

        Returns:
            A StateMachine that parses and validates inner monologue content
        """
        return FencedFreeformStateMachine(
            self.identifier,
            self.delimiters,
            char_min=50,        # Require substantive content
            char_max=self.character_max,
        )

    @property
    def state_prompt(self) -> str:
        return f"""
    The Inner Monologue state is strictly for detailed internal exploration and clarification of your thoughts and understanding.

    Use this state specifically to:
        1. Explore uncertainties and resolve them by detailed, step-by-step internal dialogue.
        2. Develop a coherent, flowing narrative to deepen your understanding before making decisions.
        3. Connect concepts and ideas thoughtfully to form comprehensive mental models.

    Do NOT duplicate brief notes or initial sketches here; those belong exclusively in the Scratchpad state.

    Always encapsulate your inner monologue within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    This state is private and hidden from the user.
        """

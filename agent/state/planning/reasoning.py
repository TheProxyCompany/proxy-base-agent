from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Reasoning(AgentState):
    def __init__(self, delimiters: tuple[str, str] | None = None):
        super().__init__(
            identifier="reasoning",
            readable_name="Logical Reasoning",
            delimiters=delimiters or ("```reasoning\n", "\n```"),
            color="dim yellow",
            emoji="bulb",
        )

    @property
    def state_machine(self) -> StateMachine:
        return FencedFreeformStateMachine(
            self.identifier,
            self.delimiters,
            char_min=100,
            char_max=2000,
        )

    @property
    def state_prompt(self) -> str:
        return f"""
    The reasoning state is where you engage in structured, logical analysis to solve problems or make decisions.
    This state emphasizes rigorous thinking patterns and explicit logical steps.

    In this state, you should:
        1. Break down complex problems into manageable components
        2. Apply formal reasoning techniques (deduction, induction, abduction)
        3. Consider multiple hypotheses and evaluate evidence systematically
        4. Identify and mitigate cognitive biases in your thinking
        5. Use techniques like reductio ad absurdum when appropriate

    Your reasoning should be methodical, transparent, and follow clear logical progressions.
    When appropriate, use mathematical notation, logical symbols, or structured formats to clarify your thinking.

    Always encapsulate your reasoning within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
    It is written in a formal, analytical style that emphasizes logical connections.
    The user cannot see this state, and your output is not displayed to the user.
        """

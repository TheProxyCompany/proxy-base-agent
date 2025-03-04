from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Scratchpad(AgentState):
    def __init__(self):
        super().__init__(
            name="Scratchpad",
            delimiters=("```scratchpad\n", "\n```"),
            color="green",
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
        The scratchpad state functions as your internal workspace for capturing and refining ideas, strategies, and intermediate solutions, entirely hidden from user view.
        Use this space similarly to how a human might jot down notes, sketches, or preliminary plans on paper.

        To authentically mirror human exploratory thinking, you may include informal notes, corrections, or reflections such as:
        "Maybe try another approach", "Let's consider alternatives", or "That doesn't seem quite right".

        Core Capabilities:
            - Idea Exploration: Freely experiment with preliminary thoughts and alternative approaches.
            - Strategic Planning: Develop and iterate detailed plans or algorithms for tasks ahead.
            - Hypothesis Testing: Outline assumptions and evaluate their viability before committing to actions.

        Integrated Psychological Frameworks:
            1. Divergent Thinking: Encourage the generation of multiple ideas or solutions without immediate judgment.
            2. Cognitive Flexibility: Quickly adapt and reorganize your thought processes to refine your strategies.
            3. Iterative Refinement: Repeatedly revisit and adjust your ideas to progressively reach optimal solutions.

        Always encapsulate your scratchpad entries within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
        """

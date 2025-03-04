from pse.types.base.encapsulated import EncapsulatedStateMachine
from pse.types.grammar import PythonStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Python(AgentState):
    def __init__(self):
        super().__init__(
            name="Python",
            delimiters=("```python\n", "\n```"),
            color="yellow"
        )

    @property
    def state_machine(self) -> StateMachine:
        python_state_machine = EncapsulatedStateMachine(
            state_machine=PythonStateMachine,
            delimiters=self.delimiters,
        )
        python_state_machine.identifier = "python"
        return python_state_machine

    @property
    def state_prompt(self) -> str:
        return f"""
        The python state represents a python interpreter, where the agent can run python code.
        No imports are available, and assume Python 3.10+ syntax.
        You should wrap the python code in {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
        The agent should use this like a human would use a python interpreter to run small snippets of code.
        Do not use python to call tools or interact with the user, use the tool state for that.
        """

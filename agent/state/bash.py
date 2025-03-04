from pse.types.base.encapsulated import EncapsulatedStateMachine
from pse.types.grammar import BashStateMachine
from pse_core.state_machine import StateMachine

from agent.state import AgentState


class Bash(AgentState):
    def __init__(self):
        super().__init__(
            name="Bash",
            delimiters=("```bash\n", "\n```"),
            color="yellow"
        )

    @property
    def state_machine(self) -> StateMachine:
        bash_state_machine = EncapsulatedStateMachine(
            state_machine=BashStateMachine,
            delimiters=self.delimiters,
        )
        bash_state_machine.identifier = "bash"
        return bash_state_machine

    @property
    def state_prompt(self) -> str:
        return f"""
        The bash state represents a bash terminal, where the agent can run commands.
        You should wrap the bash command in {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
        The agent should use this like a human would use a bash terminal.
        Do not use bash to call tools or interact with the user, use the tool state for that.
        """

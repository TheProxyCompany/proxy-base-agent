from abc import ABC, abstractmethod

from pse_core.state_machine import StateMachine


class AgentState(ABC):
    def __init__(self, name: str, delimiters: tuple[str, str], color: str):
        self.name = name
        self.delimiters = delimiters
        self.color = color

    @property
    @abstractmethod
    def state_machine(self) -> StateMachine:
        pass

    @property
    @abstractmethod
    def state_prompt(self) -> str:
        pass

    def __str__(self) -> str:
        return f"- {self.name.title()}:{self.state_prompt}"


from agent.state.bash import Bash  # noqa: E402
from agent.state.python import Python  # noqa: E402
from agent.state.scratchpad import Scratchpad  # noqa: E402
from agent.state.thinking import Thinking  # noqa: E402
from agent.state.tool_call import ToolCallState  # noqa: E402

__all__ = ["Bash", "Python", "Scratchpad", "Thinking", "ToolCallState"]

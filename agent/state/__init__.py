from abc import ABC, abstractmethod

from pse_core.state_machine import StateMachine


class AgentState(ABC):
    def __init__(
        self,
        identifier: str,
        readable_name: str,
        delimiters: tuple[str, str],
        color: str,
        emoji: str,
    ):
        self.identifier = identifier
        self.readable_name = readable_name
        self.delimiters = delimiters
        self.color = color
        self.emoji = emoji

    @property
    @abstractmethod
    def state_machine(self) -> StateMachine:
        pass

    @property
    @abstractmethod
    def state_prompt(self) -> str:
        pass

    def format(self, string: str) -> str:
        return f"{self.delimiters[0]}{string}{self.delimiters[1]}"

    def readable_format(self, string: str) -> str:
        return f"```markdown\n{string}\n```"

    def __str__(self) -> str:
        return f"{self.readable_name.title()}: {self.state_prompt}"


from agent.state.action.bash import Bash  # noqa: E402
from agent.state.action.python import Python  # noqa: E402
from agent.state.action.tool_call import ToolCallState  # noqa: E402
from agent.state.planning.inner_monologue import InnerMonologue  # noqa: E402
from agent.state.planning.scratchpad import Scratchpad  # noqa: E402
from agent.state.planning.thinking import Thinking  # noqa: E402

__all__ = [
    "Bash",
    "InnerMonologue",
    "Python",
    "Scratchpad",
    "Thinking",
    "ToolCallState",
]

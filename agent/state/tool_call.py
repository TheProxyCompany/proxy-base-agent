from pse.types.json import json_schema_state_machine
from pse_core.state_machine import StateMachine

from agent.state import AgentState
from agent.tools import Tool


class ToolCallState(AgentState):
    def __init__(self, tools: list[Tool]):
        super().__init__(
            name="Tool_Call",
            delimiters=("```json\n", "\n```"),
            color="dim yellow",
            emoji="wrench",
        )
        self.tools = tools

    @property
    def state_machine(self) -> StateMachine:
        _, state_machine = json_schema_state_machine(
            [tool.to_dict() for tool in self.tools],
            delimiters=self.delimiters
        )
        state_machine.identifier = "tool_call"
        return state_machine

    @property
    def state_prompt(self) -> str:
        return f"""
The tool_call state represents your interface for invoking external tools or APIs.
You should use this state to call tools or interact with the user.

Core Capabilities:
    - Purposeful Tool Selection: Thoughtfully choose the most appropriate tool based on the current task and context.
    - Structured Interaction: Clearly format tool calls as valid JSON objects matching the specified schema.
    - Outcome Anticipation: Internally anticipate the results and usefulness of invoking the chosen tool.

The following tools are available:
{"\n---\n".join(str(tool) for tool in self.tools)}
No other tools are available, and these tools are not available in any other state.
Always encapsulate your tool calls within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
        """

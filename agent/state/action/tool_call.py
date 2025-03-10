import textwrap

from pse.types.json import json_schema_state_machine
from pse_core.state_machine import StateMachine

from agent.state import AgentState
from agent.tools import Tool


class ToolCallState(AgentState):
    def __init__(
        self,
        tools: list[Tool],
        delimiters: tuple[str, str] | None = None,
        list_delimiters: tuple[str, str] | None = None,
    ):
        super().__init__(
            identifier="tool_call",
            readable_name="External Tool Use",
            delimiters=delimiters or ("```tool\n", "\n```"),
            color="dim yellow",
            emoji="wrench",
        )
        self.list_delimiters = list_delimiters or ("", "")
        self.tools = tools

    @property
    def state_machine(self) -> StateMachine:
        _, state_machine = json_schema_state_machine(
            [tool.to_dict() for tool in self.tools],
            delimiters=self.delimiters
        )
        state_machine.identifier = self.identifier
        return state_machine

    @property
    def state_prompt(self) -> str:
        tool_list_start = self.list_delimiters[0]
        tool_list_end = self.list_delimiters[1]
        if tool_list_end.startswith("\n"):
            tool_list_end = "\n    " + tool_list_end.removeprefix("\n")
        return f"""
    The tool_call state represents your interface for invoking external tools or APIs.
    You should use this state to call tools or interact with the user.

    The following tools are available:
    {tool_list_start}
    {"\n    ----------\n".join(textwrap.indent(str(tool), "    ") for tool in self.tools)}
    {tool_list_end}

    No other tools are available, and these tools are not available in any other state.
    Always encapsulate your tool calls within {self.delimiters[0]!r} and {self.delimiters[1]!r} tags.
        """

    def readable_format(self, string: str) -> str:
        return f"```json\n{string}\n```"

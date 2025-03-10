import textwrap

from pse.types.json import json_schema_state_machine
from pse_core.state_machine import StateMachine

from agent.state import AgentState
from agent.tools import Tool


class ToolCallState(AgentState):
    """
    State for handling tool calls during agent execution.
    
    The ToolCallState enables agents to interact with their environment by providing
    a structured interface for invoking external tools. It creates a JSON schema-based
    state machine that ensures tool calls have the correct format, required parameters,
    and follow the expected structure.
    
    This state is a key component of the agent's ability to take action in the world,
    allowing it to perform operations like web searches, calculations, or API calls.
    """
    
    def __init__(
        self,
        tools: list[Tool],
        delimiters: tuple[str, str] | None = None,
        list_delimiters: tuple[str, str] | None = None,
    ):
        """
        Initialize a new ToolCallState.
        
        Args:
            tools: List of Tool objects available for use in this state
            delimiters: Optional custom delimiters for the tool call state
            list_delimiters: Optional delimiters for the tool list in the prompt
        """
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
        """
        Create a JSON schema-based state machine for tool calls.
        
        This property dynamically generates a state machine that validates tool calls
        against their JSON schemas, ensuring all required parameters are provided and
        correctly formatted.
        
        Returns:
            A StateMachine instance configured for tool invocation validation
        """
        _, state_machine = json_schema_state_machine(
            [tool.to_dict() for tool in self.tools],
            delimiters=self.delimiters
        )
        state_machine.identifier = self.identifier
        return state_machine

    @property
    def state_prompt(self) -> str:
        """
        Generate instructions for using tools in the agent prompt.
        
        This property creates a part of the system prompt that explains:
        - What tools are available to the agent
        - How to format tool calls correctly 
        - The parameters each tool accepts
        
        Returns:
            Formatted string with tool usage instructions for the agent prompt
        """
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
        """
        Format tool call output for human readability.
        
        This method wraps the tool call JSON in code block format for better
        display in interfaces that support Markdown.
        
        Args:
            string: The raw tool call output string
            
        Returns:
            Formatted string with JSON syntax highlighting markers
        """
        return f"```json\n{string}\n```"

from typing import Any

from agent.agent import AgentState
from agent.event import Event, State
from tools import FunctionCall, ToolUse


class Hippocampus:
    """Central memory management system for the agent."""

    def __init__(self, agent_state: AgentState):
        """
        Initialize the Hippocampus with different memory components.

        Args:
            interface (AgentInterface): The interface for agent communication.
            agent_seed (int): Seed value for the agent.
        """
        self.agent_state = agent_state
        self.events: list[Event] = []

    def _construct_message(self, message_data: dict[str, Any]) -> Event | None:
        """
        Construct a Message object from a dictionary of message data.

        Args:
            message_data (Dict[str, Any]): The dictionary containing message data.

        Returns:
            Optional[Message]: The constructed Message object or None if an error occurs.
        """
        try:
            # Extract basic fields
            role = message_data.get("role")
            content = message_data.get("content") or ""
            message_id = None
            state: State
            if role == "assistant":
                state = State.ASSISTANT_RESPONSE
            elif role == "user":
                state = State.USER_INPUT
            elif role == "ipython" or role == "tool":
                message_id = message_data.get("tool_call_id") or message_data.get(
                    "tool_used_id"
                )
                state = message_data.get("state", State.SYSTEM_MESSAGE)
            elif role == "system":
                state = State.SYSTEM_MESSAGE
            else:
                state = State.ASSISTANT_RESPONSE

            # Handle tool_calls
            tool_calls = []
            if "tool_calls" in message_data:
                for tool_call_data in message_data["tool_calls"]:
                    tool_call_id = tool_call_data.get("id")
                    function_call = FunctionCall(
                        name=tool_call_data.get("name", ""),
                        arguments=tool_call_data.get("arguments", {}),
                    )
                    tool_calls.append(ToolUse("function", function_call, tool_call_id))

            # Construct Message object
            message = Event(
                event_id=message_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                state=state,
                name=message_data.get("name"),
            )
            return message
        except Exception as e:
            self.agent_state.interface.console.print(
                f"[bold red]Error constructing message: {e}"
            )
            return None

    def append_to_history(self, input: list[Event] | list[dict[str, Any]] | Event):
        """
        Append messages to the current message list.

        Args:
            input (list[Message] | dict[str, Any] | list[dict[str, Any]] | Message): The messages to append.
        """
        if not isinstance(input, Event) or isinstance(input, list):
            raise ValueError("Input must be a list of messages or a single message")

        if isinstance(input, list):
            for msg in input:
                if isinstance(msg, dict) and (
                    constructed_message := self._construct_message(msg)
                ):
                    self.append_to_history(constructed_message)
                elif isinstance(msg, Event):
                    self.append_to_history(msg)
            return

        if input.id == self.agent_state.current_event_id:
            current_event = self.get_event_by_id(input.id)
            if current_event:
                current_event.append_content(input.content)
                current_event.tool_calls.extend(input.tool_calls)
        else:
            self.events.append(input)

    def get_event_by_id(self, event_id: str | None) -> Event | None:
        """
        Get a message by its ID.

        Args:
            event_id (str): The ID of the event to retrieve.

        Returns:
            Optional[Message]: The message with the specified ID or None if not found.
        """
        if not event_id:
            return None
        for event in self.events:
            if event.id == event_id:
                return event
        return None

    def clear_messages(self):
        """
        Clear all messages from the current message list.
        """
        self.events = []

from agent.agent import AgentState
from agent.event import Event


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
        self.events: dict[str, Event] = {}

    def append_to_history(self, input_events: list[Event] | Event) -> None:
        """
        Append events to the history.

        Args:
            input_events: A single Event or list of Events to append to history.

        Raises:
            ValueError: If input is not an Event or list of Events.
        """
        if isinstance(input_events, list):
            if not all(isinstance(event, Event) for event in input_events):
                raise ValueError("All items in list must be Events")
            for event in input_events:
                self.events[event.event_id] = event
            return

        self.events[input_events.event_id] = input_events

    def clear_messages(self):
        """
        Clear all messages from the current message list.
        """
        self.events = {}

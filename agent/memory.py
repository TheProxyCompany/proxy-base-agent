from agent.interaction import Interaction


class Hippocampus:
    """Central memory management system for the agent."""

    def __init__(self, system: Interaction):
        """
        Initialize the Hippocampus with different memory components.
        """
        self.events: dict[str, Interaction] = {system.event_id: system}

    def append_to_history(self, input_events: list[Interaction] | Interaction) -> None:
        """
        Append events to the agent's history.

        Args:
            input_events: A single Event or list of Events to append to history.
        """
        if isinstance(input_events, list):
            if not all(isinstance(event, Interaction) for event in input_events):
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

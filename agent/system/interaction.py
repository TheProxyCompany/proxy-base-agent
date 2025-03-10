from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any


class Interaction:
    """
    Represents a single interaction within the agent's conversation history.
    
    An Interaction encapsulates all information about a message in the conversation,
    including who sent it (role), the content, metadata, and styling information.
    Interactions form the fundamental building blocks of the agent's memory and
    provide a consistent structure for different types of messages (user inputs,
    agent responses, tool results, etc.).
    
    Interactions are uniquely identified by an event_id and can be converted to
    dictionaries for serialization or LLM context building.
    """

    class Role(Enum):
        """
        Enumeration of possible roles for an interaction.
        
        These roles determine how the interaction is processed, displayed,
        and included in the context window for the language model.
        """
        ASSISTANT = "assistant"  # Messages from the agent
        SYSTEM = "system"        # System messages and instructions
        TOOL = "tool"            # Results from tool executions
        USER = "user"            # Messages from the human user

    def __init__(
        self,
        event_id: str | None = None,
        name: str | None = None,
        role: Role = Role.SYSTEM,
        content: Any = "",
        **kwargs,
    ) -> None:
        """
        Initialize a new Interaction.
        
        Args:
            event_id: Unique identifier for this interaction (auto-generated if None)
            name: Optional name or identifier for the creator of this interaction
            role: The role of this interaction (SYSTEM, USER, ASSISTANT, or TOOL)
            content: The primary content of the interaction (text, structured data, etc.)
            **kwargs: Additional metadata attributes to store with this interaction
                      Common metadata includes title, color, emoji for display styling
        """
        self.content = content        # The primary content (text, structured data, etc.)
        self.created_at = datetime.now()  # Timestamp for creation time
        self.event_id = event_id or str(uuid.uuid4())  # Unique identifier
        self.metadata = kwargs        # Additional attributes (title, color, etc.)
        self.name = name              # Name of the interaction creator
        self.role = role              # Role type (SYSTEM, USER, ASSISTANT, TOOL)

    @property
    def styling(self) -> dict[str, str]:
        """
        Get styling information for this event type.
        Returns a dict with title, color, and emoji fields.
        """
        title: str | None = self.metadata.get("title", self.name)
        color: str | None = self.metadata.get("color", None)
        emoji: str | None = self.metadata.get("emoji", None)

        match self.role:
            case Interaction.Role.ASSISTANT:
                return {
                    "title": title or "Agent",
                    "color": color or "cyan",
                    "emoji": emoji or "alien",
                }
            case Interaction.Role.SYSTEM:
                return {
                    "title": title or "System",
                    "color": color or "magenta",
                    "emoji": emoji or "gear",
                }
            case Interaction.Role.TOOL:
                return {
                    "title": title or "Tool",
                    "color": color or "yellow",
                    "emoji": emoji or "wrench",
                }
            case Interaction.Role.USER:
                return {
                    "title": title or "User",
                    "color": color or "green",
                    "emoji": emoji or "speech_balloon",
                }
            case _:
                return {
                    "title": title or "Unknown",
                    "color": color or "red",
                    "emoji": emoji or "question",
                }

    def to_dict(self) -> dict:
        """
        Convert this interaction to a dictionary representation.
        
        This method serializes the interaction into a dictionary format
        suitable for:
        - Passing to language models as context
        - Storing in memory/databases
        - Converting to JSON for APIs
        
        Returns:
            A dictionary containing all relevant interaction data
        """
        # Initialize with core attributes
        dict = {
            "event_id": self.event_id,
            "role": self.role.value,
            "content": str(self.content),
        }
        
        # Process metadata attributes
        for key, value in self.metadata.items():
            # Handle objects with their own to_dict method
            if value and hasattr(value, "to_dict"):
                dict[key] = value.to_dict()
            else:
                dict[key] = value

        # Add compatibility fields for different LLM APIs
        if self.role == Interaction.Role.TOOL:
            dict["tool_call_id"] = str(self.event_id)  # For OpenAI format
            dict["tool_used_id"] = str(self.event_id)  # For Anthropic format

        return dict

    def __str__(self) -> str:
        """Convert to a JSON string representation for debugging and logging."""
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        """Return string representation for REPL and debugging."""
        return self.__str__()

    def __eq__(self, other):
        """
        Check equality by comparing event_ids.
        
        Two interactions are considered equal if they have the same event_id,
        regardless of any other differences in their content or metadata.
        """
        if not isinstance(other, Interaction):
            return False
        return self.event_id == other.event_id

    def __hash__(self):
        """Create a hash based on event_id for use in sets and dictionaries."""
        return hash(self.event_id)

    def __getattribute__(self, name: str) -> Any:
        """
        Enhanced attribute access that transparently exposes metadata attributes.
        
        This magic method allows metadata attributes to be accessed directly as if they
        were instance attributes. For example, if an Interaction has metadata["title"],
        you can access it using interaction.title.
        
        The lookup order is:
        1. Look for actual attributes on the instance
        2. If not found, check if it exists in metadata
        3. If not in metadata, return None
        
        This creates a more convenient API for accessing metadata fields.
        """
        try:
            # First try to get the actual attribute
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, check if it's in metadata
            metadata = object.__getattribute__(self, "metadata")
            if name in metadata:
                return metadata[name]
            return None

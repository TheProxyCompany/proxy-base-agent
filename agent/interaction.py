from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import pytz


class Interaction:
    """
    An interaction in the agent's environment.
    """

    class Role(Enum):
        ASSISTANT = "assistant"
        SYSTEM = "system"
        TOOL = "tool"
        USER = "user"

    def __init__(
        self,
        event_id: str | None = None,
        name: str | None = None,
        role: Role = Role.SYSTEM,
        content: Any = "",
        **kwargs,
    ) -> None:
        self.content = content
        self.created_at = datetime.now().astimezone(pytz.timezone("US/Eastern"))
        self.event_id = event_id or str(uuid.uuid4())
        self.metadata = kwargs
        self.name = name
        self.role = role

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
        dict = {
            "event_id": self.event_id,
            "role": self.role.value,
            "content": str(self.content),
        }
        for key, value in self.metadata.items():
            if value and hasattr(value, "to_dict"):
                dict[key] = value.to_dict()
            elif isinstance(value, str):
                dict[key] = value
            else:
                dict[key] = json.dumps(value)

        if self.role == Interaction.Role.TOOL:
            dict["tool_call_id"] = str(self.event_id)  # openai
            dict["tool_used_id"] = str(self.event_id)  # anthropic

        return dict

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Interaction):
            return False
        return self.event_id == other.event_id

    def __hash__(self):
        return hash(self.event_id)

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            metadata = object.__getattribute__(self, "metadata")
            if name in metadata:
                return metadata[name]
            return None

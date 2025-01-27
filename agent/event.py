from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import pytz

from tools import ToolUse


class EventState(Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, state_string: str):
        return cls(state_string)

class Event:
    """
    Represents an event in the agent's context.
    """

    def __init__(
        self,
        event_id: str | None = None,
        content: str | list[str] = "",
        name: str | None = None,
        state: EventState = EventState.SYSTEM,
        tool_calls: list[ToolUse] | None = None,
        **kwargs,
    ) -> None:
        self.content = content
        self.created_at = datetime.now().astimezone(pytz.timezone("US/Eastern"))
        self.event_id = event_id or str(uuid.uuid4())
        self.metadata = kwargs
        self.name = name
        self.state = state
        self.tool_calls = tool_calls or []
        self.tool_results: dict[str, Event] = {}

    def append_content(self, content: str | list[str]):
        if isinstance(self.content, list) and isinstance(content, list):
            self.content.extend(content)
        elif isinstance(self.content, str) and isinstance(content, str):
            self.content += content
        else:
            self.content = str(self.content) + str(content)

    def to_dict(self) -> dict:
        dict = {
            "event_id": self.event_id,
            "state": self.state.value,
            "content": self.content,
            ""
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "metadata": self.metadata,
        }

        if len(self.tool_calls) == 0:
            dict.pop("tool_calls")

        if self.state == EventState.TOOL:
            dict["tool_call_id"] = str(self.event_id)  # openai
            dict["tool_used_id"] = str(self.event_id)  # anthropic

        return dict

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Event):
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

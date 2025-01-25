from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import pytz

from tools import ToolCall

EST = pytz.timezone("US/Eastern")


class Event:
    """
    Represents an event in the agent's context.
    """

    class State(Enum):
        USER_INPUT = "user_input"
        ASSISTANT_RESPONSE = "assistant_response"
        TOOL_CALL = "tool_call"
        TOOL_RESULT = "tool_result"
        TOOL_ERROR = "tool_error"
        SYSTEM_MESSAGE = "system_message"
        METACOGNITION = "metacognition"
        ERROR = "error"
        SCRATCHPAD = "scratchpad"

        def __str__(self):
            return self.value

        @classmethod
        def from_string(cls, state_string: str):
            return cls(state_string)

        def get_value(self) -> int:
            """
            Returns an integer score representing the importance of the message state.
            This score can be used with a predefined threshold to determine whether
            to display the message to the user.

            Returns:
                int: The score associated with the current MessageState.
            """
            score_mapping = {
                Event.State.SYSTEM_MESSAGE: 12,
                Event.State.USER_INPUT: 10,
                Event.State.ASSISTANT_RESPONSE: 9,
                Event.State.ERROR: 8,
                Event.State.TOOL_RESULT: 7,
                Event.State.METACOGNITION: 6,
                Event.State.SCRATCHPAD: 5,
                Event.State.TOOL_CALL: 4,
                Event.State.TOOL_ERROR: 3,
            }
            return score_mapping.get(self, 0)

    def __init__(
        self,
        event_id: str | None = None,
        content: str | list[str] = "",
        created_at: datetime | None = None,
        name: str | None = None,
        role: str | None = None,
        state: State = State.SYSTEM_MESSAGE,
        tool_calls: list[ToolCall] | None = None,
        **kwargs,
    ) -> None:
        self.content = content
        self.created_at = created_at or datetime.now().astimezone(EST)
        self.id = event_id or str(uuid.uuid4())
        self.metadata = kwargs
        self.name = name
        self.role = role
        self.state = state
        self.tool_calls = tool_calls or []

    def append_content(self, content: str | list[str]):
        if isinstance(self.content, list) and isinstance(content, list):
            self.content.extend(content)
        elif isinstance(self.content, str) and isinstance(content, str):
            self.content += content
        else:
            self.content = str(self.content) + str(content)

    def to_dict(self) -> dict:
        dict = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "state": self.state.value,
        }

        if len(self.tool_calls) == 0:
            dict.pop("tool_calls")

        if self.role == "tool" or self.role == "ipython":
            dict["tool_call_id"] = str(self.id)  # openai
            dict["tool_used_id"] = str(self.id)  # anthropic

        return dict

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            metadata = object.__getattribute__(self, 'metadata')
            if name in metadata:
                return metadata[name]
            return None


State = Event.State

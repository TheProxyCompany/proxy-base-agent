import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import pytz
from discord import TextChannel

from tools import ToolCall


class MessageState(Enum):
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
            MessageState.USER_INPUT: 10,
            MessageState.ASSISTANT_RESPONSE: 9,
            MessageState.ERROR: 8,
            MessageState.TOOL_RESULT: 7,
            MessageState.METACOGNITION: 6,
            MessageState.SCRATCHPAD: 5,
            MessageState.TOOL_CALL: 4,
            MessageState.TOOL_ERROR: 3,
            MessageState.SYSTEM_MESSAGE: 2,
        }
        return score_mapping.get(self, 0)


class Message:
    def __init__(
        self,
        id: str | None = str(uuid.uuid4()),
        content: str | list[str] = "",
        created_at: datetime | None = None,
        image_path: str | None = None,
        inner_thoughts: str | None = None,
        name: str | None = None,
        role: str | None = None,
        state: MessageState = MessageState.SYSTEM_MESSAGE,
        tool_calls: list[ToolCall] | None = None,
        metadata: dict[str, Any] | None = None,
        attachments: list[str] | None = None,
        feelings: str | None = None,
        channel: TextChannel | None = None,
    ) -> None:
        self.id = id
        self.content = content
        self.created_at = created_at or datetime.now(pytz.utc).astimezone(
            pytz.timezone("US/Eastern")
        )
        self.image_path = image_path
        self.inner_thoughts = inner_thoughts
        self.name = name
        self.role = role
        self.state = state
        self.metadata = metadata
        self.attachments = attachments
        self.feelings = feelings
        self.channel = channel

        if role is None:
            raise ValueError("Role must be provided")
        if tool_calls is not None:
            if not all(isinstance(tc, ToolCall) for tc in tool_calls):
                raise ValueError("All elements in tool_calls must be of type ToolCall")
        self.tool_calls = tool_calls or []

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

        if self.role == "tool":
            dict["tool_call_id"] = str(self.id)  # openai
            dict["tool_used_id"] = str(self.id)  # anthropic

        return dict

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

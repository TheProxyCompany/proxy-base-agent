import json
import os

from pydantic import BaseModel


class RoleTokens(BaseModel):
    system: str
    assistant: str
    user: str
    tool: str


class ControlTokens(BaseModel):
    template_type: str
    bos_token: str
    eos_token: str
    eom_token: str
    tool_use_token_start: str
    tool_use_token_end: str
    start_header_token: str
    end_header_token: str
    roles: RoleTokens

    def end_tokens(self) -> list[str]:
        return [self.eos_token, self.eom_token]

    def tool_use_delimiters(self) -> tuple[str, str]:
        return self.tool_use_token_start, self.tool_use_token_end


def get_control_tokens(model_type: str) -> ControlTokens:
    """Get the control tokens for the model."""
    match model_type:
        case "llama":
            return _load_control_tokens("llama")
        case "mistral":
            return _load_control_tokens("mistral")
        case "deepseek":
            return _load_control_tokens("deepseek")
        case _:
            return _load_control_tokens("chatml")


def _load_control_tokens(model_type: str) -> ControlTokens:
    """Load the control tokens for the model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{model_type}.json")
    with open(file_path) as f:
        data = json.load(f)
        return ControlTokens(**data)

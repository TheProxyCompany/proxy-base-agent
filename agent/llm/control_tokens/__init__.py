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
    begin_of_text: str
    end_of_message: str
    end_of_sequence: str
    user_start: str
    user_end: str
    assistant_header_start: str
    assistant_header_end: str
    thinking_start: str
    thinking_end: str
    scratchpad_start: str
    scratchpad_end: str
    tool_start: str
    tool_end: str
    tool_result_start: str
    tool_result_end: str
    roles: RoleTokens

    def get_primary_control_tokens(self) -> list[str]:
        return [
            self.begin_of_text,
            self.end_of_sequence,
            self.end_of_message,
            self.user_start.strip(),
            self.user_end.strip(),
            self.assistant_header_start.strip(),
            self.assistant_header_end.strip(),
        ]

    def end_tokens(self) -> list[str]:
        return [self.end_of_sequence, self.end_of_message]

    @property
    def tool_use_delimiters(self) -> tuple[str, str] | None:
        if self.tool_start and self.tool_end:
            return self.tool_start, self.tool_end
        return None

    @property
    def thinking_delimiters(self) -> tuple[str, str] | None:
        if self.thinking_start and self.thinking_end:
            return self.thinking_start, self.thinking_end
        return None

    @property
    def scratchpad_delimiters(self) -> tuple[str, str] | None:
        if self.scratchpad_start and self.scratchpad_end:
            return self.scratchpad_start, self.scratchpad_end


def get_control_tokens(model_path: str, tokenizer_config: dict) -> ControlTokens:
    """Get the control tokens for the model."""
    model_type = _determine_model_type(model_path, tokenizer_config)
    match model_type:
        case "llama":
            return _load_control_tokens("llama")
        case "llama-deepseek":
            return _load_control_tokens("llama-deepseek")
        case "mistral":
            return _load_control_tokens("mistral")
        case "deepseek":
            return _load_control_tokens("deepseek")
        case _:
            return _load_control_tokens("chatml")


def _determine_model_type(model_path: str, tokenizer_config: dict) -> str:
    """Determine the model type from the model path."""
    model_type = tokenizer_config.get("model_type", "chatml")
    eos_token = tokenizer_config.get("eos_token", "<|eot_id|>")
    if eos_token == "<|eot_id|>":
        model_type = "llama"
    elif eos_token.strip() == "<|im_end|>":
        model_type = "chatml"

    if model_type == "llama":
        if "deepseek" in model_path.lower():
            model_type = "llama-deepseek"
        # elif "hermes" in model_path.lower():
        #     model_type = "hermes"

    return model_type


def _load_control_tokens(model_type: str) -> ControlTokens:
    """Load the control tokens for the model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{model_type}.json")
    with open(file_path) as f:
        data = json.load(f)
        return ControlTokens(**data)

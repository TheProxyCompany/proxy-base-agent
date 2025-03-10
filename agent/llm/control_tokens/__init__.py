import json
import os

from pydantic import BaseModel


class Role(BaseModel):
    role_name: str
    role_start_tag: str
    role_end_tag: str
    end_of_message: str | None = None

class RoleTags(BaseModel):
    system: Role | None = None
    assistant: Role | None = None
    user: Role | None = None
    tool: Role | None = None


class ControlTokens(BaseModel):
    """Control tokens for different model templates.

    This class defines the structure and access methods for control tokens used in
    various LLM template formats.
    """

    template_type: str
    begin_of_text: str
    end_of_message: str
    end_of_sequence: str
    inner_monologue_start: str | None = None
    inner_monologue_end: str | None = None
    thinking_start: str | None = None
    thinking_end: str | None = None
    scratchpad_start: str | None = None
    scratchpad_end: str | None = None
    reasoning_start: str | None = None
    reasoning_end: str | None = None
    tool_list_start: str | None = None
    tool_list_end: str | None = None
    tool_call_start: str
    tool_call_end: str
    tool_calls_start: str | None = None
    tool_calls_end: str | None = None
    tool_result_start: str | None = None
    tool_result_end: str | None = None
    tool_results_start: str | None = None
    tool_results_end: str | None = None
    roles: RoleTags

    def delimiters(self) -> dict[str, tuple[str, str] | None]:
        """Returns a dictionary of all delimiter pairs.

        Returns:
            A dictionary mapping state names to their delimiter tuples.
        """
        return {
            "inner_monologue": self.inner_monologue_delimiters,
            "reasoning": self.reasoning_delimiters,
            "scratchpad": self.scratchpad_delimiters,
            "thinking": self.thinking_delimiters,
            "tool_call": self.tool_use_delimiters,
            "tool_list": self.tool_list_delimiters,
            "tool_result": self.tool_result_delimiters,
            "tool_results": self.tool_results_delimiters,
        }

    def end_tokens(self) -> list[str]:
        """Returns a list of tokens that indicate the end of a sequence.

        Returns:
            A list of end tokens.
        """
        return [self.end_of_sequence, self.end_of_message]

    def get_whitelist_control_tokens(self) -> list[str]:
        """Returns the control tokens used for tokenization.

        Returns:
            A list of the most essential control tokens.
        """
        tokens: list[str] = []
        for delim in self.delimiters().values():
            if delim:
                start, end = delim
                if start.strip():
                    tokens.append(start.strip())
                if end.strip():
                    tokens.append(end.strip())

        return tokens

    @property
    def inner_monologue_delimiters(self) -> tuple[str, str] | None:
        """Returns the inner monologue delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.inner_monologue_start and self.inner_monologue_end:
            return self.inner_monologue_start, self.inner_monologue_end
        return None

    @property
    def reasoning_delimiters(self) -> tuple[str, str] | None:
        """Returns the reasoning delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.reasoning_start and self.reasoning_end:
            return self.reasoning_start, self.reasoning_end
        return None

    @property
    def scratchpad_delimiters(self) -> tuple[str, str] | None:
        """Returns the scratchpad delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.scratchpad_start and self.scratchpad_end:
            return self.scratchpad_start, self.scratchpad_end
        return None

    @property
    def thinking_delimiters(self) -> tuple[str, str] | None:
        """Returns the thinking delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.thinking_start and self.thinking_end:
            return self.thinking_start, self.thinking_end
        return None

    @property
    def tool_list_delimiters(self) -> tuple[str, str] | None:
        """Returns the tool list delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.tool_list_start and self.tool_list_end:
            return self.tool_list_start, self.tool_list_end
        return None

    @property
    def tool_result_delimiters(self) -> tuple[str, str] | None:
        """Returns the tool result delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.tool_result_start and self.tool_result_end:
            return self.tool_result_start, self.tool_result_end
        return None

    @property
    def tool_results_delimiters(self) -> tuple[str, str] | None:
        """Returns the tool results delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.tool_results_start and self.tool_results_end:
            return self.tool_results_start, self.tool_results_end
        return None

    @property
    def tool_use_delimiters(self) -> tuple[str, str] | None:
        """Returns the tool use delimiter pair if defined.

        Returns:
            A tuple of start and end delimiters, or None if not defined.
        """
        if self.tool_call_start and self.tool_call_end:
            return self.tool_call_start, self.tool_call_end
        return None


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
        case "hermes":
            return _load_control_tokens("hermes")
        case _:
            return _load_control_tokens("chatml")


def _determine_model_type(model_path: str, tokenizer_config: dict) -> str:
    """Determine the model type from the model path."""
    model_type = tokenizer_config.get("model_type", "chatml")
    eos_token = tokenizer_config.get("eos_token", "<|eot_id|>")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content", "<|eot_id|>")

    if eos_token == "<|eot_id|>":
        model_type = "llama"
    elif eos_token == "<｜end▁of▁sentence｜>":  # noqa: RUF001
        if tokenizer_config.get("tokenizer_class") == "LlamaTokenizerFast":
            model_type = "llama-deepseek"
        else:
            model_type = "deepseek"
    elif isinstance(eos_token, str) and eos_token.strip() == "<|im_end|>":
        model_type = "chatml"

    if "hermes" in model_path.lower():
        model_type = "hermes"

    return model_type


def _load_control_tokens(model_type: str) -> ControlTokens:
    """Load the control tokens for the model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{model_type}.json")
    with open(file_path) as f:
        data = json.load(f)
        return ControlTokens(**data)

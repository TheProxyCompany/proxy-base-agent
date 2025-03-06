from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from agent.llm.control_tokens import ControlTokens, get_control_tokens
from agent.llm.prompts import load_template

logger = logging.getLogger(__name__)


class Tokenizer:
    """A convienience wrapper around a Hugging Face tokenizer.

    The wrapper provides convienient access to control tokens,
    encoding/decoding with templates, and vocabulary management.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        control_tokens: ControlTokens | None = None,
    ) -> None:
        """
        Args:
            tokenizer: The base Hugging Face tokenizer to wrap
            control_tokens: Optional control tokens - such as end-of-sequence or tool-use tokens
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._tokenizer = tokenizer
        self._control_tokens = control_tokens

    @property
    def control_tokens(self) -> ControlTokens:
        """
        Get the control tokens, or raise an error if they are not set.

        Control tokens such as end-of-sequence or tool-use tokens are used to control the model's behavior.
        """
        if self._control_tokens is None:
            raise ValueError("Control tokens are not set")
        return self._control_tokens

    @property
    def whitelist_control_tokens(self) -> list[str]:
        """Get the whitelist of control tokens.

        Returns:
            List of control tokens
        """
        return self.control_tokens.get_whitelist_control_tokens()

    @property
    def stop_tokens(self) -> set[int]:
        """Get the set of token IDs that indicate stopping generation.

        Returns:
            Set of token IDs for EOS and EOM tokens from control_tokens.
            Returns empty set if no control tokens configured.
        """
        if not self._control_tokens:
            return set()

        # Get all end token IDs without special tokens to avoid duplicates
        stop_tokens = set()
        for stop_token in self._control_tokens.end_tokens():
            stop_tokens.add(
                self._tokenizer.encode(stop_token, add_special_tokens=False)[0]
            )

        # Flatten and deduplicate token IDs into a set
        return stop_tokens

    @property
    def delimiters(self) -> dict[str, tuple[str, str] | None]:
        """Get the delimiters for the control tokens.

        Returns:
            Dictionary of control token names to their delimiters
        """
        return self.control_tokens.delimiters()

    def decode(self, tokens: list[int], **kwargs) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(tokens, **kwargs)

    def encode(self, prompt: str | list[dict[str, str]] | dict[str, Any], **kwargs) -> list[int]:
        """Encode text or chat messages into tokens.

        Handles both raw text and chat message formats. For raw text, supports
        template substitution of tools and date strings.

        Args:
            prompt: Text string or list of chat messages to encode
            **kwargs: Additional encoding options

        Returns:
            Token IDs or templated string depending on input format

        Raises:
            ValueError: If chat template produces unsupported format
        """
        if isinstance(prompt, str):
            tools = kwargs.pop("tools", None)
            date_string = kwargs.pop("date_string", None)
            if tools is not None or date_string is not None:
                try:
                    prompt = prompt.format(date_string=date_string, tools=tools)
                except Exception:
                    pass
            return self._tokenizer.encode(prompt, **kwargs)

        if isinstance(prompt, list) or isinstance(prompt, dict):
            kwargs["interactions"] = prompt
            if isinstance(prompt, dict):
                conversation = [event.to_dict() for event in prompt.values()]
                templated = self._tokenizer.apply_chat_template(conversation, **kwargs)
            else:
                templated = self._tokenizer.apply_chat_template(prompt, **kwargs)

            if isinstance(templated, list) and isinstance(templated[0], int):
                return templated  # type: ignore[reportReturnValue]
            raise ValueError(f"Unsupported prompt format: {templated}")

    @staticmethod
    def load(model_path: str | Path, **kwargs) -> Tokenizer:
        """Create a TokenizerWrapper by loading a Hugging Face tokenizer.

        Args:
            model_path: Path to the model/tokenizer
            **kwargs: Additional args passed to AutoTokenizer.from_pretrained()

        Returns:
            Configured TokenizerWrapper instance
        """
        # Convert string path to Path object for consistent handling
        model_path = Path(model_path) if isinstance(model_path, str) else model_path

        # Load tokenizer configuration and determine appropriate control tokens
        try:
            with open(model_path / "tokenizer_config.json") as f:
                tokenizer_config = json.load(f)
                control_tokens = get_control_tokens(str(model_path), tokenizer_config)
        except FileNotFoundError:
            logger.warning(f"Tokenizer config not found at {model_path}, using default control tokens")
            control_tokens = get_control_tokens(str(model_path), {})

        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        tokenizer.chat_template = load_template(control_tokens.template_type)
        return Tokenizer(tokenizer, control_tokens)

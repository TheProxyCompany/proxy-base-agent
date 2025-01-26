from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from agent.model_inference.chat_templates import load_chat_template
from agent.model_inference.chat_templates.control_tokens import (
    ControlTokens,
    get_control_tokens,
)

logger = logging.getLogger(__name__)
class TokenizerWrapper:
    """A wrapper around a Hugging Face tokenizer that adds control token handling and chat templating.

    The wrapper provides convenient access to control tokens, encoding/decoding with templates,
    and vocabulary management.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, control_tokens: ControlTokens | None = None) -> None:
        """Initialize the TokenizerWrapper.

        Args:
            tokenizer: The base Hugging Face tokenizer to wrap
            control_tokens: Optional control tokens - such as end-of-sequence or tool-use tokens
        """
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
    def stop_tokens(self) -> set[int]:
        """Get the set of token IDs that indicate stopping generation.

        Returns:
            Set of token IDs for EOS and EOM tokens from control_tokens.
            Returns empty set if no control tokens configured.
        """
        if not self._control_tokens:
            return set()

        # Get all end token IDs without special tokens to avoid duplicates
        token_ids = [
            self._tokenizer.encode(token, add_special_tokens=False)
            for token in self._control_tokens.end_tokens()
        ]

        # Flatten and deduplicate token IDs into a set
        return {id for ids in token_ids for id in ids}

    @property
    def eos_token(self) -> str | int:
        """Get the end-of-sequence token.

        Returns:
            Token ID for EOS token, or 0 if not set
        """
        if not self._control_tokens:
            return 0
        eos_token = self._control_tokens.eos_token
        return self._tokenizer.encode(eos_token)[0] if eos_token else 0

    @property
    def eom_token(self) -> str | int:
        """Get the end-of-message token ID.

        Returns:
            Token ID for EOM token, or 0 if not set
        """
        if not self._control_tokens:
            return 0
        eom_token = self._control_tokens.eom_token
        return self._tokenizer.encode(eom_token)[0] if eom_token else 0

    def decode(self, tokens: list[int], **kwargs) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(tokens, **kwargs)

    def encode(self, prompt: str | list[dict[str, str]], **kwargs) -> str | list[int]:
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

        if isinstance(prompt, list):
            templated = self._tokenizer.apply_chat_template(prompt, **kwargs)
            if isinstance(templated, str):
                logger.info(f"Templated prompt:\n{templated}")
                return templated
            if isinstance(templated, list) and isinstance(templated[0], int):
                return templated  # type: ignore[reportReturnValue]
            raise ValueError(f"Unsupported prompt format: {templated}")

    @staticmethod
    def load(
        model_path: str | Path,
        model_type: str,
        **kwargs,
    ) -> TokenizerWrapper:
        """Create a TokenizerWrapper by loading a Hugging Face tokenizer.

        Args:
            model_path: Path to the model/tokenizer
            model_type: Type of model for control token selection
            **kwargs: Additional args passed to AutoTokenizer.from_pretrained()

        Returns:
            Configured TokenizerWrapper instance
        """
        control_tokens = get_control_tokens(model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        tokenizer.chat_template = load_chat_template()
        logger.debug(f"Loaded tokenizer: {tokenizer}")
        return TokenizerWrapper(tokenizer, control_tokens)

    def __getattribute__(self, name: str) -> Any:
        """Forward attribute lookups to the underlying tokenizer if not found on wrapper."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_tokenizer"), name)

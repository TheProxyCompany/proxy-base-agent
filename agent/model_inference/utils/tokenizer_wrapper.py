import json
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
)
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from agent.model_inference.chat_templates.control_tokens import ControlTokens

DetokenizerType = (
    type[NaiveStreamingDetokenizer]
    | partial[NaiveStreamingDetokenizer]
    | type[SPMStreamingDetokenizer]
    | partial[SPMStreamingDetokenizer]
    | type[BPEStreamingDetokenizer]
)

T = TypeVar("T", bound=PreTrainedTokenizer | PreTrainedTokenizerFast)


class TokenizerWrapper(Generic[T]):
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(
        self,
        tokenizer: T,
        detokenizer_class: DetokenizerType,
        control_tokens: ControlTokens | None = None,
    ):
        """
        Initialize the TokenizerWrapper.

        Args:
            tokenizer (T): The Hugging Face tokenizer.
            detokenizer_class (DetokenizerType): The class of the detokenizer to use.
            control_tokens (dict[str, int] | None): The control tokens to use.
        """
        self._tokenizer: T = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)
        self._control_tokens = control_tokens

    @property
    def vocabulary(self) -> dict[str, int]:
        """
        Get the vocabulary of the tokenizer.

        Returns:
            Dict[str, int]: The vocabulary.
        """
        return self._tokenizer.get_vocab()

    @property
    def eos_token_id(self) -> int:
        """
        Get the EOS token ID of the tokenizer.

        Returns:
            int: The EOS token ID.
        """
        eos_token = self._control_tokens.eos_token if self._control_tokens else None
        return self._tokenizer.encode(eos_token)[0] if eos_token else 0

    @property
    def eom_token_id(self) -> int:
        """
        Get the EOM token ID of the tokenizer.

        Returns:
            int: The EOM token ID.
        """
        eom_token = self._control_tokens.eom_token if self._control_tokens else None
        return self._tokenizer.encode(eom_token)[0] if eom_token else 0

    @property
    def end_token_ids(self) -> list[int]:
        """
        Get the end token IDs of the tokenizer.
        """
        return [self.eos_token_id, self.eom_token_id]

    def decode(self, tokens: list[int]) -> str:
        """
        Decode the tokens, applying the detokenizer template first if the tokens
        are a series of messages instead of a straight string.
        """
        return self._tokenizer.decode(tokens)

    def encode(self, prompt: str | list[dict[str, str]], **kwargs) -> str | list[int]:
        """
        Encode the prompt, applying the tokenizer template first if the prompt
        is a series of messages instead of a straight string.
        """

        if isinstance(prompt, str):
            if "tools" in kwargs or "date_string" in kwargs:
                tools = kwargs.pop("tools", None)
                date_string = kwargs.pop("date_string", None)
                try:
                    prompt = prompt.format(date_string=date_string, tools=tools)
                except Exception:
                    pass
            return self._tokenizer.encode(prompt, **kwargs)

        if isinstance(prompt, list):
            templated_prompt = self._tokenizer.apply_chat_template(prompt, **kwargs)
            if isinstance(templated_prompt, str):
                return templated_prompt
            elif isinstance(templated_prompt, list) and isinstance(
                templated_prompt[0], int
            ):
                return templated_prompt  # type: ignore[reportReturnValue]
            else:
                raise ValueError(f"unsupported prompt format: {templated_prompt}")


def _match(a: Any, b: Any) -> bool:
    """
    Check if two objects match in structure and content.

    Args:
        a (Any): The first object.
        b (Any): The second object.

    Returns:
        bool: True if the objects match, False otherwise.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b, strict=True))

    return a == b


def _is_spm_decoder(decoder: dict) -> bool:
    """
    Check if a decoder is an SPM decoder.

    Args:
        decoder (dict): The decoder configuration.

    Returns:
        bool: True if the decoder is an SPM decoder, False otherwise.
    """
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder: dict) -> bool:
    """
    Check if a decoder is an SPM decoder without space trimming.

    Args:
        decoder (dict): The decoder configuration.

    Returns:
        bool: True if the decoder is an SPM decoder without space trimming, False otherwise.
    """
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder: dict) -> bool:
    """
    Check if a decoder is a BPE decoder.

    Args:
        decoder (dict): The decoder configuration.

    Returns:
        bool: True if the decoder is a BPE decoder, False otherwise.
    """
    _target_description = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }

    return _match(_target_description, decoder)


def load_tokenizer(
    model_path: str | Path,
    control_tokens: ControlTokens | None = None,
    tokenizer_config_extra: dict | None = None,
) -> TokenizerWrapper[PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Load a Hugging Face tokenizer and infer the type of streaming detokenizer to use.

    Args:
        model_path (Union[str, Path]): The path to the model.
        tokenizer_config_extra (dict, optional): Additional configuration for the tokenizer. Defaults to {}.

    Returns:
        TokenizerWrapper[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]: The loaded tokenizer wrapped with a detokenizer.
    """

    if tokenizer_config_extra is None:
        tokenizer_config_extra = {}
    detokenizer_class: DetokenizerType = NaiveStreamingDetokenizer
    tokenizer_file = Path(model_path) / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file) as fid:
            tokenizer_content = json.load(fid)
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    return TokenizerWrapper(
        AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
        detokenizer_class,
        control_tokens,
    )

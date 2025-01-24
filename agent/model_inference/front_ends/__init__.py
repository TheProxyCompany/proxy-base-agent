from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from model_inference.chat_templates.control_tokens import ControlTokens
from model_inference.utils.tokenizer_wrapper import TokenizerWrapper
from pse.structuring_engine import StructuringEngine


class FrontEndType(enum.Enum):
    MLX = "mlx"
    TORCH = "torch"
    ONNX = "onnx"
    JAX = "jax"
    TRITON = "triton"

    def __str__(self):
        return self.value

class FrontEnd(ABC):
    """
    Abstract base class for front-ends.
    """

    control_tokens: ControlTokens
    model_type: str
    engine: StructuringEngine
    tokenizer: TokenizerWrapper

    @staticmethod
    def from_type(model_path: str, front_end_type: FrontEndType | None = None) -> FrontEnd:
        if front_end_type is None:
            front_end_type = FrontEndType.MLX

        if front_end_type == FrontEndType.MLX:
            from model_inference.front_ends.mlx_front_end import MLXFrontEnd
            return MLXFrontEnd(model_path)
        else:
            raise ValueError(f"Invalid front-end type: {front_end_type}")

    @abstractmethod
    def __call__(self, prompt: list[int], **kwargs: Any) -> Iterator[Result]:
        pass

    @abstractmethod
    def initialize_cache(self, model: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: list[int], **kwargs: Any) -> Iterator[Result]:
        pass

    @abstractmethod
    def sample(self, prompt: list[int], **kwargs: Any) -> Result:
        pass

    @dataclass
    class Result:
        """
        Result of a generation step.

        Args:
            token (mx.array): The generated token.
            token_id (int): The id of the generated token.
            logprobs (mx.array): The log probabilities of the generated token.
            tool_called (bool): Whether a tool was called.
            inference_time (float): The time taken for inference.
            engine_time (float): The time taken for the engine.
            sampling_time (float): The time taken for sampling.
        """

        tokens: Any
        token_ids: list[int]
        logprobs: Any
        schema_complete: bool
        inference_time: float
        engine_time: float
        sampling_time: float

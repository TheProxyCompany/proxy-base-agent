from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pse.structure.engine import StructuringEngine

from agent.model_inference.control_tokens import ControlTokens
from agent.model_inference.utils.tokenizer_wrapper import TokenizerWrapper


class FrontEndType(enum.Enum):
    MLX = "mlx"
    TORCH = "torch"
    JAX = "jax"

    def __str__(self):
        return self.value


class FrontEnd(ABC):
    """
    Abstract base class for front-ends.
    """

    control_tokens: ControlTokens
    engine: StructuringEngine
    model_type: str
    computed_prompt_tokens: list[int]
    tokenizer: TokenizerWrapper

    @staticmethod
    def from_type(
        model_path: str, front_end_type: FrontEndType | None = None
    ) -> FrontEnd:
        if front_end_type is None:
            front_end_type = FrontEndType.MLX

        if front_end_type == FrontEndType.MLX:
            from agent.model_inference.inference.frontend_mlx import MLXFrontEnd

            return MLXFrontEnd(model_path)
        else:
            raise ValueError(f"Invalid front-end type: {front_end_type}")

    def __call__(self, prompt: list[int], **kwargs) -> Iterator[ModelOutput]:
        return self.inference(prompt, **kwargs)

    @abstractmethod
    def inference(self, prompt: list[int], **kwargs: Any) -> Iterator[ModelOutput]:
        pass

    @abstractmethod
    def inference_step(self, prompt: list[int], **kwargs: Any) -> ModelOutput:
        pass

    @staticmethod
    @abstractmethod
    def sample_tokens(logprobs: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def initialize_cache(self, model: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None:
        pass

    @dataclass
    class ModelOutput:
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
        inference_time: float
        engine_time: float
        sampling_time: float

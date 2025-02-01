from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pse.structure.engine import StructuringEngine

from agent.llm.util.tokenizer import Tokenizer


class Frontend(ABC):
    """
    Abstract base class for front-ends.
    """

    engine: StructuringEngine
    model_type: str
    computed_prompt_tokens: list[int]
    tokenizer: Tokenizer

    @staticmethod
    def from_path(model_path: str, frontend: str | None = "mlx") -> Frontend:
        if frontend == "mlx":
            from agent.llm.inference.mlx_frontend import MLXFrontEnd

            return MLXFrontEnd(model_path)
        else:
            raise ValueError(f"Invalid front-end type: {frontend:}")

    def __call__(self, prompt: list[int], **kwargs) -> Iterator[Output]:
        return self.inference(prompt, **kwargs)

    @abstractmethod
    def inference(self, prompt: list[int], **kwargs: Any) -> Iterator[Output]:
        pass

    @abstractmethod
    def inference_step(self, prompt: Any, **kwargs: Any) -> Output:
        pass

    @staticmethod
    @abstractmethod
    def sample_tokens(logprobs: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> None:
        pass

    @dataclass
    class Output:
        """
        Result of a generation step.

        Args:
            tokens: The generated tokens.
            token_ids: The ids of the generated tokens.
            logprobs: The log probabilities of the generated tokens.
            inference_time: The time taken for inference.
            engine_time: The time taken for the engine.
            sampling_time: The time taken for sampling.
        """

        tokens: Any
        token_ids: list[int]
        logprobs: Any
        structured: bool
        inference_time: float
        engine_time: float
        sampling_time: float

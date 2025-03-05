from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, TypeVar

from pse.structuring_engine import StructuringEngine

from agent.llm.tokenizer import Tokenizer

T = TypeVar("T")


class Frontend(ABC):
    """
    Abstract base class for front-ends.
    """

    tokenizer: Tokenizer

    @staticmethod
    def from_path(model_path: str, frontend: str | None = "mlx") -> Frontend:
        if frontend == "mlx":
            from agent.llm.frontend.mlx import MLXInference

            return MLXInference(model_path)
        elif frontend == "torch":
            from agent.llm.frontend.torch import TorchInference

            return TorchInference(model_path)
        else:
            raise ValueError(f"Invalid front-end type: {frontend!r}")

    @abstractmethod
    def inference(self, prompt: list[int], engine: StructuringEngine, **kwargs: Any) -> Iterator[Any]:
        pass

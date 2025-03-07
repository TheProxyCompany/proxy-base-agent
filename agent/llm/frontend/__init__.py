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
    cache: list[Any]
    processed_token_ids: list[int]

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

    def supports_reusing_prompt_cache(self) -> bool:
        return False

    @abstractmethod
    def inference(self, prompt: list[int], engine: StructuringEngine, **kwargs: Any) -> Iterator[Any]:
        pass

    @abstractmethod
    def load_cache_from_file(self, file_path: str) -> tuple[list[Any], list[int]]:
        """
        Load a KV cache from a file.

        Args:
            file_path (str): Path to the cache file.

        Returns:
            tuple[list[Any], list[int]]: The loaded cache and the computed token IDs.
        """
        pass

    @abstractmethod
    def save_cache_to_file(self, file_path: str, computed_ids: list[int]) -> None:
        """
        Save a KV cache to a file.

        Args:
            file_path (str): Path to the cache file.
            computed_ids (list[int]): The token IDs that have been processed.
        """
        pass

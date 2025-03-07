import threading
from collections.abc import Iterator
from typing import Any

import torch
from pse.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin
from transformers import LlamaForCausalLM, TextIteratorStreamer

from agent.llm.frontend import Frontend
from agent.llm.tokenizer import Tokenizer


class PSE_Torch(PSETorchMixin, LlamaForCausalLM):
    pass


class TorchInference(Frontend):
    """
    Front-end for PyTorch models.
    """

    def __init__(self, model_path: str):
        """
        Initialize the TorchFrontEnd.

        Args:
            model_path (str): The path to the model.
        """
        # Load the model from the specified path
        self.model = PSE_Torch.from_pretrained(model_path)
        assert isinstance(self.model, LlamaForCausalLM)

        # Initialize tokenizer with appropriate model type
        self.tokenizer = Tokenizer.load(model_path)
        self.processed_token_ids: list[int] = []
        # Configure padding token to match EOS token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Apply the same padding configuration to generation config if it exists
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id

    def inference(self, prompt: list[int], engine: StructuringEngine, **kwargs: Any) -> Iterator[str]:
        assert isinstance(self.model, PSE_Torch)
        self.model.engine = engine
        if seed := kwargs.get("seed", None):
            torch.random.manual_seed(seed)

        tensor = torch.tensor(prompt)
        tensor = tensor.unsqueeze(0).to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer._tokenizer, skip_prompt=True) # type: ignore [reportArgumentType]
        generate_kwargs = {
            "inputs": tensor,
            "do_sample": True,
            "streamer": streamer,
            "max_new_tokens": kwargs.get("max_tokens", None),
            "top_k": kwargs.get("top_k", 10),
            "top_p": kwargs.get("top_p", None),
            "temperature": kwargs.get("temp", 0.7),
        }
        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        yield from streamer

    def load_cache_from_file(self, file_path: str) -> tuple[list[Any], list[int]]:
        """
        Load a KV cache from a file.

        PyTorch implementation does not support caching yet.

        Args:
            file_path (str): Path to the cache file.

        Returns:
            tuple[list[Any], list[int]]: Empty lists as PyTorch doesn't support caching yet.
        """
        # Return empty lists as PyTorch doesn't support caching yet
        return [], []

    def save_cache_to_file(self, file_path: str, computed_ids: list[int]) -> None:
        """
        Save a KV cache to a file.

        PyTorch implementation does not support caching yet.

        Args:
            file_path (str): Path to the cache file.
            computed_ids (list[int]): The token IDs that have been processed.
        """
        # No-op as PyTorch doesn't support caching yet
        pass

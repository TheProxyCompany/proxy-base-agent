from collections.abc import Iterator
from typing import Any

import torch
from pse.structuring_engine import StructuringEngine
from pse.util.torch_mixin import PSETorchMixin
from transformers import PreTrainedModel, TextStreamer

from agent.llm.frontend import Frontend
from agent.llm.tokenizer import Tokenizer


class PSE_Torch(PSETorchMixin, PreTrainedModel):
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
        self.model = PSE_Torch.from_pretrained(model_path)
        assert isinstance(self.model, PreTrainedModel)
        self.model_type = self.model.config.model_type
        self.tokenizer = Tokenizer.load(model_path, self.model_type)
        self.model.config.pad_token_id = self.model.config.eos_token_id[0]
        if self.model.generation_config:
            self.model.generation_config.pad_token_id = self.model.config.eos_token_id[0]

    def inference(self, prompt: list[int], engine: StructuringEngine, **kwargs: Any) -> Iterator[str]:
        self.model.engine = engine
        if seed := kwargs.get("seed", None):
            torch.random.manual_seed(seed)
        tensor = torch.tensor(prompt)
        tensor = tensor.to(self.model.device)
        yield from self.model.generate(
            tensor,
            do_sample=True,
            max_new_tokens=200,
            top_k=10,
            top_p=None,
            streamer=TextStreamer(self.tokenizer._tokenizer),  # type: ignore [reportArgumentType]
        )

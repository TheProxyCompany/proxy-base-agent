import glob
import importlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import generate_step, get_model_path, load_config
from pse.structure.engine import StructuringEngine

from agent.llm.frontend import Frontend
from agent.llm.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class MLXLLM(Frontend):
    """
    Front-end for MLX models.
    """

    def __init__(self, model_path: str):
        """
        Initialize the MLXFrontEnd.

        Args:
            model_path (str): The path to the model.
        """
        self.set_device_limit()
        self.model, self.model_type = self.load_model(model_path)
        self.tokenizer = Tokenizer.load(model_path, self.model_type)
        self.engine = StructuringEngine(self.tokenizer._tokenizer)

    def inference(self, prompt: list[int], structured: bool = True, **kwargs) -> Iterator[int]:
        """
        A generator producing token ids based on the given prompt from the model.

        Args:
            prompt (list[int]): The input prompt.
            simple_sampler (bool): Whether to use simple sampling.
            **kwargs: Keyword arguments for the sampler.
        """
        if seed := kwargs.get("seed", None):
            mx.random.seed(seed)

        for tokens, _ in generate_step(
            prompt=mx.array(prompt),
            model=self.model,
            logits_processors=[self.engine.process_logits] if structured else None,
            sampler=self.make_sampler(structured, **kwargs),
            max_tokens=kwargs.get("max_tokens", 1000),
        ):
            if isinstance(tokens, int):
                tokens = [tokens]
            else:
                tokens = tokens.tolist()
            assert isinstance(tokens, list)
            for token_id in tokens:
                if token_id in self.tokenizer.stop_tokens:
                    break
                yield token_id

            if self.engine.has_reached_accept_state:
                break

    def make_sampler(self, structured: bool, **kwargs) -> Callable[..., Any]:
        """
        Return a sampler function.
        If structured is True, use the structured sampler.
        Otherwise, use the simple sampler.
        """
        temp = float(kwargs.get("temp", 1.0))
        min_p = float(kwargs.get("min_p", 0.0))
        min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))
        sampler = make_sampler(
            temp=temp,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep
        )
        if structured:
            return lambda x: self.engine.sample(x, sampler)
        else:
            return sampler

    @staticmethod
    def load_model(model_path: str) -> tuple[nn.Module, str]:
        """
        Load and initialize the model from a given path.

        Args:
            model_path (Path): The path to load the model from.
        Returns:
            nn.Module: The loaded and initialized model.
        """
        path = get_model_path(model_path)
        config = load_config(path)
        weight_files = glob.glob(str(path / "model*.safetensors"))
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model_class, model_args_class = MLXLLM.get_model_architecture(config)
        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        if (quantization := config.get("quantization", None)) is not None:
            nn.quantize(model, **quantization)

        model.load_weights(list(weights.items()))
        assert isinstance(model, nn.Module)
        mx.eval(model.parameters())
        model.eval()
        return model, config.get("model_type", "chatml")

    @staticmethod
    def get_model_architecture(config: dict[str, Any]):
        """
        Retrieve the model and model args classes based on the configuration.

        Args:
            config (dict): The model configuration.

        Returns:
            A tuple containing the Model class and the ModelArgs class.
        """
        model_type = config["model_type"]
        model_type = {
            "mistral": "llama",
            "phi-msft": "phixtral",
            "falcon_mamba": "mamba",
        }.get(model_type, model_type)

        arch = None
        try:
            try:
                arch = importlib.import_module(f"agent.llm.models.{model_type}")
            except ImportError:
                arch = importlib.import_module(f"mlx_lm.models.{model_type}")
        except ImportError:
            msg = f"Model type {model_type} not supported."
            logging.error(msg)

        if arch is None:
            raise ValueError("No model architecture found for the given model type.")

        return arch.Model, arch.ModelArgs

    def set_device_limit(self):
        device_info = mx.metal.device_info()
        safe_max_size = device_info["max_recommended_working_set_size"]
        if isinstance(safe_max_size, int):
            mx.metal.set_wired_limit(safe_max_size)
            max_rec_gb = safe_max_size / 2**30
            logger.info(f"Set wired memory limit to {max_rec_gb:.2f}GB")
        else:
            logger.warning(f"Max recommended size is not an integer: {safe_max_size}")

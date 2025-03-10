import hashlib
import json
import logging
import pathlib
from collections.abc import Iterable
from typing import Any

from pse.structuring_engine import StructuringEngine

from agent.llm.frontend import Frontend
from agent.system.interaction import Interaction

logger = logging.getLogger(__name__)


class LocalInference:
    def __init__(self, model_path: str, frontend: str | None = "mlx"):
        """
        Initialize the Inference class.

        Args:
            model_path (str): Path to the model.

        This method sets up the necessary components for inference, including:
        - Loading the model configuration
        - Initializing the tokenizer and model
        - Setting up caches and data structures for efficient inference
        """
        self.model_path = model_path
        self.front_end = Frontend.from_path(model_path, frontend)
        self.engine = StructuringEngine(
            self.front_end.tokenizer._tokenizer,
            whitelist_control_tokens=self.front_end.tokenizer.whitelist_control_tokens,
            multi_token_sampling=True,
        )

    def run_inference(
        self,
        prompt: str | list[dict[str, Any]] | list[Interaction],
        **inference_kwargs,
    ) -> Iterable[int]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str | list[dict[str, Any]] | list[Event]): The input prompt for completion.
            **inference_kwargs: Additional keyword arguments to use for inference.
        """
        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.front_end.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.front_end.tokenizer.encode(**tokenizer_config)

        # Try to load from cache first if caching is enabled
        cache_system_prompt = inference_kwargs.get("cache_system_prompt", True)
        reuse_prompt_cache = inference_kwargs.get("reuse_prompt_cache", True)

        if (
            cache_system_prompt
            and reuse_prompt_cache
            and self.front_end.supports_reusing_prompt_cache()
            and not self.front_end.processed_token_ids
        ):
            # Check if we have a cached prompt
            self._load_cached_system_prompt(encoded_prompt)

        logger.info(f"PROMPT:\n{self.front_end.tokenizer.decode(encoded_prompt)}")
        for generation_number, token_id in enumerate(
            self.front_end.inference(
                encoded_prompt,
                self.engine,
                **inference_kwargs,
            )
        ):
            yield token_id

            if self.front_end.supports_reusing_prompt_cache():
                if generation_number == 0:
                    if (
                        cache_system_prompt
                        and encoded_prompt[100:] != self.front_end.processed_token_ids[100:]
                    ):
                        self._cache_system_prompt(encoded_prompt)
                    self.front_end.processed_token_ids = encoded_prompt
                else:
                    self.front_end.processed_token_ids.append(token_id)

    def _get_cache_directory(self) -> pathlib.Path:
        """
        Get the cache directory path, creating it if it doesn't exist.

        Returns:
            pathlib.Path: The path to the cache directory.
        """
        # Get the directory where local.py is located
        module_dir = pathlib.Path(__file__).parent.absolute()
        cache_dir = module_dir / ".cache"

        # Create the cache directory if it doesn't exist
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {cache_dir}")

        return cache_dir

    def _compute_prompt_hash(self, token_ids: list[int]) -> str:
        """
        Compute a hash of the token IDs and model path to use as the cache key.

        This function creates a deterministic hash by taking the first half of the token sequence
        and converting it to a JSON string before hashing. The model path is prepended to the hash
        to ensure uniqueness across different models using the same prompts.

        Note: Only the first half of tokens is used for efficiency, as system prompts typically
        contain sufficient uniqueness in their initial portion.

        Args:
            token_ids (list[int]): The token IDs to hash.

        Returns:
            str: A composite string containing the model path and hexadecimal hash digest,
                 formatted as "{model_path}:{hash_digest}".
        """
        model_name = self.model_path.split("/")[-1]
        # Use a fixed number of tokens (first 100) for consistent hashing
        # This ensures the same system prompt always produces the same hash
        # regardless of the total prompt length
        fixed_token_count = min(100, len(token_ids))
        token_ids_str = json.dumps(token_ids[:fixed_token_count])
        hash_obj = hashlib.sha256(token_ids_str.encode())
        return f"{model_name}:{hash_obj.hexdigest()}"

    def _cache_system_prompt(self, token_ids: list[int]) -> None:
        """
        Cache the system prompt token IDs and KV cache to a file.

        Args:
            token_ids (list[int]): The token IDs to cache.
        """
        if not self.front_end.supports_reusing_prompt_cache():
            logger.debug("Frontend does not support reusing prompt cache, skipping")
            return

        try:
            # Get the cache directory and compute the hash
            cache_dir = self._get_cache_directory()
            prompt_hash = self._compute_prompt_hash(token_ids)
            cache_path = cache_dir / f"{prompt_hash}.safetensors"
            # Save the cache
            self.front_end.save_cache_to_file(str(cache_path), token_ids)
            logger.debug(f"Cached system prompt to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache system prompt: {e}")

    def _load_cached_system_prompt(self, token_ids: list[int]) -> None:
        """
        Load a cached system prompt if available.

        Args:
            token_ids (list[int]): The token IDs to look up in the cache.

        Returns:
            Optional[list[int]]: The cached token IDs if available, None otherwise.
        """
        if not self.front_end.supports_reusing_prompt_cache():
            return

        try:
            # Get the cache directory and compute the hash
            cache_dir = self._get_cache_directory()
            prompt_hash = self._compute_prompt_hash(token_ids)
            cache_path = cache_dir / f"{prompt_hash}.safetensors"

            # Check if the cache file exists
            if not cache_path.exists():
                logger.debug(f"No cache found for prompt hash {prompt_hash}")
                return

            # Load the cache
            cache, computed_ids = self.front_end.load_cache_from_file(str(cache_path))
            if cache:
                # Set the cache on the frontend
                self.front_end.cache = cache
                self.front_end.processed_token_ids = computed_ids
                logger.debug(f"Loaded cached system prompt from {cache_path}")

            return
        except Exception as e:
            logger.error(f"Failed to load cached system prompt: {e}")
            return

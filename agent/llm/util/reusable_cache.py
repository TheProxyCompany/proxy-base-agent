from typing import Self

import mlx.core as mx
import mlx.nn as nn

from agent.llm.util.kv_cache import KeyValueCache


class ReusableKVCache(KeyValueCache):
    """
    Reusable cache for a single layer.

    Useful when a prompt has a common prefix
    with the prompt stored in the cache.
    """

    def __init__(self, head_dim: int, num_heads: int):
        super().__init__()
        self.head_dim = head_dim
        self.key_head_dim = head_dim
        self.value_head_dim = head_dim
        self.num_key_value_heads = num_heads

    @classmethod
    def from_model(cls, model: nn.Module) -> list[Self]:
        """
        Factory method to create a list of ReusableKVCache instances for each layer in the model.

        Args:
            model: A transformer model that must have an attribute 'num_key_value_heads', a 'head_dim', and a list of layers accessible via 'model.layers'.

        Returns:
            List[ReusableKVCache]: A list of reusable cache instances, one for each layer.
        """
        if not hasattr(model, "args") or not isinstance(model.args, object):
            raise ValueError("Model must have an attribute 'args'")
        head_dim = model.args.head_dim
        num_kv_heads = model.args.num_key_value_heads
        num_layers = len(model.layers)
        return [cls(head_dim, num_kv_heads) for _ in range(num_layers)]

    def reuse(self, new_prompt_length: int, common_prefix_length: int) -> None:
        """
        Reuse a portion of the cache for a new prompt that shares a common prefix with the previous prompt.

        Args:
            new_prompt_length (int): The total sequence length of the new prompt.
            common_prefix_length (int): The length of the shared prefix in the cache.
        """
        if self.keys is None:
            return

        # Retain only the common prefix in the cache.
        self.offset = common_prefix_length

        # Ensure the cache has enough capacity for the new prompt.
        current_capacity = self.size
        if current_capacity < new_prompt_length:
            num_steps = (self.step_size + new_prompt_length - 1) // self.step_size
            additional_capacity = num_steps * self.step_size - current_capacity

            # Define shapes for additional allocation (assuming a batch size of 1 for reused cache).
            keys_extra_shape = (
                1,
                self.num_key_value_heads,
                additional_capacity,
                self.key_head_dim,
            )
            values_extra_shape = (
                1,
                self.num_key_value_heads,
                additional_capacity,
                self.value_head_dim,
            )

            extra_keys = mx.zeros(keys_extra_shape, self.keys.dtype)
            extra_values = mx.zeros(values_extra_shape, self.values.dtype)

            # Extend the cache with the extra allocation.
            self.keys = mx.concatenate([self.keys, extra_keys], axis=2)
            self.values = mx.concatenate([self.values, extra_values], axis=2)

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        current_offset = self.offset
        new_seq_len = new_keys.shape[2]

        if self.keys is None or (current_offset + new_seq_len) > self.keys.shape[2]:
            num_steps = (self.step_size + new_seq_len - 1) // self.step_size
            batch_size = new_keys.shape[0]
            keys_alloc_shape = (
                batch_size,
                self.num_key_value_heads,
                num_steps * self.step_size,
                self.key_head_dim,
            )
            values_alloc_shape = (
                batch_size,
                self.num_key_value_heads,
                num_steps * self.step_size,
                self.value_head_dim,
            )

            additional_keys = mx.zeros(keys_alloc_shape, new_keys.dtype)
            additional_values = mx.zeros(values_alloc_shape, new_values.dtype)

            if self.keys is not None:
                if current_offset % self.step_size != 0:
                    self.keys = self.keys[..., :current_offset, :]
                    self.values = self.values[..., :current_offset, :]
                self.keys = mx.concatenate([self.keys, additional_keys], axis=2)
                self.values = mx.concatenate([self.values, additional_values], axis=2)
            else:
                self.keys, self.values = additional_keys, additional_values

        self.offset += new_seq_len
        self.keys[..., current_offset : self.offset, :] = new_keys
        self.values[..., current_offset : self.offset, :] = new_values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

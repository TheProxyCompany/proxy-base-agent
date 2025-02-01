import mlx.core as mx


class KVCache:
    """
    A key-value cache for transformer models that accumulates key and value tensors
    across tokens or time steps. It dynamically resizes its storage in fixed-size steps
    as new data is added.

    Attributes:
        num_key_value_heads (int): Number of key-value heads.
        key_head_dim (int): Dimension of keys per head.
        value_head_dim (int): Dimension of values per head.
        keys (mx.array): Tensor storing cached keys.
        values (mx.array): Tensor storing cached values.
        offset (int): Current position in the cache (number of tokens stored).
        step (int): Fixed allocation step size used when resizing the cache.
    """

    def __init__(self, head_dim: int | tuple[int, int], num_key_value_heads: int):
        """
        Initializes the KVCache.

        Args:
            head_dim (int or tuple): If an int, both key and value head dimensions are set to this value.
                                     If a tuple of two ints, the first is used for keys and the second for values.
            num_key_value_heads (int): Number of key-value heads.
        """
        self.num_key_value_heads = num_key_value_heads
        if isinstance(head_dim, int):
            self.key_head_dim = self.value_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.key_head_dim, self.value_head_dim = head_dim

        self.keys = mx.array([])
        self.values = mx.array([])
        self.offset = 0  # Number of tokens currently stored.
        self.step = 256  # Allocation step size.

    def update_and_fetch(self, new_keys, new_values):
        """
        Updates the cache with new key and value tensors and returns the updated cache.

        Args:
            new_keys (mx.array): New key tensor with shape (batch, n_kv_heads, seq_len, k_head_dim).
            new_values (mx.array): New value tensor with shape (batch, n_kv_heads, seq_len, v_head_dim).

        Returns:
            Tuple[mx.array, mx.array]: The cached keys and values up to the current offset.
        """
        current_offset = self.offset
        new_seq_len = new_keys.shape[2]  # Sequence length of the new keys/values

        # Check if there is enough capacity in the cache; if not, allocate additional space.
        if self.keys is None or (current_offset + new_seq_len) > self.keys.shape[2]:
            batch_size = new_keys.shape[0]
            # Calculate how many allocation steps are needed to cover the new tokens.
            num_steps = (self.step + new_seq_len - 1) // self.step
            # Define shapes for the additional allocation.
            keys_alloc_shape = (
                batch_size,
                self.num_key_value_heads,
                num_steps * self.step,
                self.key_head_dim,
            )
            values_alloc_shape = (
                batch_size,
                self.num_key_value_heads,
                num_steps * self.step,
                self.value_head_dim,
            )

            # Create new allocation tensors filled with zeros.
            additional_keys = mx.zeros(keys_alloc_shape, new_keys.dtype)
            additional_values = mx.zeros(values_alloc_shape, new_values.dtype)

            if self.keys is not None:
                # If the current offset is not aligned with the allocation step,
                # trim the cached tensors to the valid portion.
                if current_offset % self.step != 0:
                    self.keys = self.keys[..., :current_offset, :]
                    self.values = self.values[..., :current_offset, :]
                # Extend the cache by concatenating the newly allocated tensors.
                self.keys = mx.concatenate([self.keys, additional_keys], axis=2)
                self.values = mx.concatenate([self.values, additional_values], axis=2)
            else:
                # First-time allocation.
                self.keys, self.values = additional_keys, additional_values

        # Update the cache offset to account for the new tokens.
        self.offset += new_seq_len

        # Insert new keys and values into the cache at the appropriate positions.
        self.keys[..., current_offset : self.offset, :] = new_keys
        self.values[..., current_offset : self.offset, :] = new_values

        # Return the cached keys and values up to the current offset.
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def state(self):
        """
        Returns the current state of the cache.

        Returns:
            Tuple[mx.array, mx.array]: The cached keys and values.
        """
        return self.keys, self.values


class ReusableKVCache(KVCache):
    """
    An enhanced KVCache that adds functionality for reusing part of the cache for new prompts
    sharing a common prefix.
    """

    @classmethod
    def for_model(cls, model):
        """
        Factory method to create a list of ReusableKVCache instances for each layer in the model.

        Args:
            model: A transformer model that must have an attribute 'num_key_value_heads',
                   a 'head_dim', and a list of layers accessible via 'model.layers'.

        Returns:
            List[ReusableKVCache]: A list of cache instances, one for each layer.
        """

        # Determine the number of KV heads for each layer.
        # If model.num_key_value_heads is an int, replicate it for each layer; otherwise, assume it's a list.
        num_kv_heads_per_layer = (
            [model.num_key_value_heads] * len(model.layers)
            if isinstance(model.num_key_value_heads, int)
            else model.num_key_value_heads
        )
        # Create a cache for each layer using the model's head_dim and the layer-specific n_kv_heads.
        return [cls(model.head_dim, num_heads) for num_heads in num_kv_heads_per_layer]

    def reuse(self, new_prompt_length, common_prefix_length):
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
        current_capacity = self.keys.shape[2]
        if current_capacity < new_prompt_length:
            num_steps = (self.step + new_prompt_length - 1) // self.step
            additional_capacity = num_steps * self.step - current_capacity

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

    def update_and_fetch(self, new_keys, new_values):
        """
        Override the base class method to support updating the cache with batches of size greater than 1.

        Args:
            new_keys (mx.array): New key tensor with shape (batch, n_kv_heads, seq_len, k_head_dim).
            new_values (mx.array): New value tensor with shape (batch, n_kv_heads, seq_len, v_head_dim).

        Returns:
            Tuple[mx.array, mx.array]: The cached keys and values up to the current offset.
        """
        current_offset = self.offset
        new_seq_len = new_keys.shape[2]

        if self.keys is None or (current_offset + new_seq_len) > self.keys.shape[2]:
            num_steps = (self.step + new_seq_len - 1) // self.step
            batch_size = new_keys.shape[0]
            keys_alloc_shape = (
                batch_size,
                self.num_key_value_heads,
                num_steps * self.step,
                self.key_head_dim,
            )
            values_alloc_shape = (
                batch_size,
                self.num_key_value_heads,
                num_steps * self.step,
                self.value_head_dim,
            )

            additional_keys = mx.zeros(keys_alloc_shape, new_keys.dtype)
            additional_values = mx.zeros(values_alloc_shape, new_values.dtype)

            if self.keys is not None:
                if current_offset % self.step != 0:
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

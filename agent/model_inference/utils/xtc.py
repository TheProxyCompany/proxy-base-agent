import mlx.core as mx
from typing import List, Optional

def xtc_sampling(
    logits: mx.array,
    threshold: float,
    probability: float,
    filter_value: float = float('-inf'),
    special_token_ids: Optional[List[int]] = None
) -> mx.array:
    """
    Apply Exclude Top Choices (XTC) sampling to the logits.

    XTC removes all except the least likely token meeting a given threshold, with a given probability.
    This ensures that at least one "viable" choice remains, retaining coherence while boosting creativity.

    Args:
        logits (mx.array): The logits from the model's output. Shape: (batch_size, vocab_size)
            This represents the unnormalized log probabilities for each token in the vocabulary.
        threshold (float): The probability threshold for token exclusion.
            Tokens with probabilities above this threshold are considered for exclusion.
        probability (float): The probability of applying XTC sampling.
            This controls how often the XTC sampling is applied. A value of 1.0 means always apply,
            while 0 means never apply.
        filter_value (float, optional): The value to assign to excluded tokens. Default: -inf
            This is typically set to negative infinity to effectively remove the token from consideration.
        special_token_ids (List[int], optional): List of special token IDs to preserve. Default: None
            These tokens will not be excluded by the XTC sampling, regardless of their probabilities.

    Returns:
        mx.array: The bias array applied to the logits.
            This array has the same shape as `logits` and contains the `filter_value` for excluded tokens
            and 0 for tokens that are not excluded.

    Raises:
        ValueError: If `threshold` or `probability` is not in the [0, 1] interval.

    Example:
        >>> logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        >>> threshold = 0.1
        >>> probability = 1.0
        >>> special_token_ids = [1, 3]
        >>> bias = xtc_sampling(logits, threshold, probability, special_token_ids=special_token_ids)
        >>> print(bias)
        [[  0.   0. -inf   0. -inf]]

    Note:
        - This function is designed to enhance the creativity of language model outputs by excluding
          the most probable tokens under certain conditions.
        - It can be used in conjunction with other sampling methods to achieve a balance between
          coherence and creativity.
    """
    if not (0 <= threshold <= 1.0) or not (0 <= probability <= 1.0):
        raise ValueError(
            f"Both `threshold` and `probability` must be floats in the [0, 1] interval, "
            f"but got threshold={threshold} and probability={probability}"
        )

    # `random.random()` returns values in the half-open range [0, 1), so setting `probability`
    # to 0 means the sampler never takes action, while setting it to 1 means the sampler
    # always takes action.
    #
    # Note that while XTC is most intuitively described as "if multiple tokens meet
    # the threshold, then with probability...", reversing the two conditions is logically
    # equivalent, and improves performance because processing can immediately be stopped
    # if the random check fails.
    if mx.random.uniform() >= probability:
        return mx.zeros_like(logits)

    probs = mx.softmax(logits, axis=-1, stream=mx.cpu)
    sorted_indices = mx.argsort(-probs, axis=-1, stream=mx.cpu)

    # Correctly create the mask
    probs_sorted = mx.take_along_axis(probs, sorted_indices, axis=-1)
    mask = probs_sorted >= threshold

    # Efficiently map the mask back to the original indices
    indices_to_remove = mx.zeros_like(probs)  # Initialize as boolean
    mx.put_along_axis(indices_to_remove, sorted_indices, mask, axis=-1)

    if special_token_ids is not None:
        for token_id in special_token_ids:
            if token_id >= 0 and token_id < logits.shape[-1]:
                indices_to_remove[..., token_id] = False

    # Ensure at least one token remains (using mx.argmax for efficiency)
    if mx.all(indices_to_remove):  # Check if all tokens are marked for removal
        # Keep the most likely token (index 0 after sorting)
        indices_to_remove[mx.arange(logits.shape[0]), sorted_indices[:, 0]] = False

    # Create the bias array
    bias = mx.where(indices_to_remove, filter_value, 0.0, stream=mx.cpu)

    return bias

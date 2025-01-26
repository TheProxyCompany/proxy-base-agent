# import mlx.core as mx

# def calculate_dynamic_temperature(
#     normalized_entropy: mx.array,
#     min_temp: float,
#     max_temp: float,
#     exponent_val: float,
#     epsilon: float = 1e-9
# ) -> mx.array:
#     """
#     Calculate dynamic temperature based on the normalized entropy.

#     Args:
#         normalized_entropy (mx.array): Normalized entropy values. Can be 1D or 2D (for batched inputs).
#         min_temp (float): Minimum temperature value.
#         max_temp (float): Maximum temperature value.
#         exponent_val (float): Exponent for entropy normalization.
#         epsilon (float): Small value to avoid numerical instability. Default is 1e-9.

#     Returns:
#         mx.array: The dynamically calculated temperature(s).

#     Raises:
#         ValueError: If input parameters are invalid or contain NaN values.
#     """
#     # Input validation
#     if min_temp < 0 or max_temp <= 0 or exponent_val <= 0:
#         raise ValueError("min_temp, max_temp, and exponent_val must all be positive values.")
#     if min_temp > max_temp:
#         raise ValueError(f"min_temp ({min_temp}) must be less than or equal to max_temp ({max_temp}).")
#     if epsilon <= 0:
#         raise ValueError(f"epsilon must be positive, but got {epsilon}")

#     # Check for NaN values in input parameters
#     assert not mx.isnan(normalized_entropy), "normalized entropy cannot be NaN"

#     # Check for NaN values in normalized_entropy
#     if mx.any(mx.isnan(normalized_entropy)):
#         raise ValueError("normalized_entropy contains NaN values")

#     # Ensure normalized_entropy is at least 2D for consistent processing
#     if normalized_entropy.ndim == 1:
#         normalized_entropy = normalized_entropy.reshape(1, -1)

#     # Map the normalized entropy to the desired temperature range using power function
#     normalized_entropy = mx.maximum(normalized_entropy, epsilon, stream=mx.cpu)
#     dyn_temp = min_temp + (max_temp - min_temp) * mx.power(normalized_entropy, exponent_val, stream=mx.cpu)

#     # Check for NaN values in the result
#     if mx.any(mx.isnan(dyn_temp)):
#         raise ValueError("NaN values detected in the calculated dynamic temperature")

#     # Squeeze results to remove singleton dimensions if input was 1D
#     if normalized_entropy.shape[0] == 1:
#         dyn_temp = mx.array(dyn_temp.item())

#     # Final check to ensure the output is within the expected range
#     assert mx.all(dyn_temp >= min_temp), "Calculated temperature is outside the expected range"

#     return dyn_temp

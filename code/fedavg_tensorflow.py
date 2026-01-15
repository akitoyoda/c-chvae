"""Utility functions to run FedAvg with the original TensorFlow-style models.

The helpers here keep the API intentionally small: given a list of client
weight structures (e.g., the output of ``model.get_weights()`` or any nested
TensorFlow structure), compute the weighted Federated Averaging result.
"""

from typing import Iterable, Sequence

import tensorflow as tf


def fedavg(
    client_weights: Sequence[Iterable],
    client_sizes: Sequence[float] | None = None,
):
    """
    Compute the (weighted) FedAvg of client weight structures.

    Args:
        client_weights: Sequence of model weights for each client. Each entry
            must share the same nested structure (lists/tuples/dicts of tensors
            or arrays).
        client_sizes: Optional sequence of example counts per client. When not
            provided, uniform weighting is used.

    Returns:
        A structure matching ``client_weights[0]`` with the averaged tensors.

    Raises:
        ValueError: if ``client_weights`` is empty or lengths mismatch.
    """

    if not client_weights:
        raise ValueError("client_weights must be a non-empty sequence")

    if client_sizes is None:
        client_sizes = [1.0] * len(client_weights)
    if len(client_sizes) != len(client_weights):
        raise ValueError("client_sizes and client_weights must have the same length")

    # Convert to TensorFlow scalars once to avoid repeated conversions in the
    # reducer below.
    client_sizes = [tf.cast(size, tf.float32) for size in client_sizes]
    total_examples = tf.add_n(client_sizes)

    def _average_one_layer(*layer_values):
        # layer_values contains the same layer from each client.
        # Keep dtype consistent with the first value.
        dtype = tf.dtypes.as_dtype(getattr(layer_values[0], "dtype", tf.float32))
        weighted = [
            tf.cast(value, dtype) * tf.cast(client_sizes[idx], dtype)
            for idx, value in enumerate(layer_values)
        ]
        return tf.math.add_n(weighted) / tf.cast(total_examples, dtype)

    return tf.nest.map_structure(_average_one_layer, *client_weights)


# Friendly alias for callers used to the longer name
federated_average = fedavg

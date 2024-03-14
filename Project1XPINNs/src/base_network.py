import jax.numpy as np
from jax import random
from type_util import Activator, Array, Shape, Params, ArrayLike, Callable


def random_layer_params(
    m: int, n: int, key: int, scale: float = 1e-2
) -> tuple[Array, Array]:
    """Initialize the weights and biases for a layer

    Adapted from the JAX example on neural networks:
    https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html#hyperparameters

    Args:
        m (int): Number of input nodes
        n (int): Number of output nodes
        key (int): Random PRNG key
        scale (float, optional): Scaling of random values. Defaults to 1e-2.

    Returns:
        tuple[Array, Array]: Initialized weights and biases
    """
    w_key, b_key = random.split(key)
    return scale * random.normal(
        w_key, (n, m), dtype=np.float64
    ), scale * random.normal(b_key, (n,), dtype=np.float64)


def init_network_params(sizes: Shape, key: int) -> Params:
    """Initialize the parameters of the network with Xavier initialization

    Adapted from the JAX example on neural networks:
    https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html#hyperparameters

    Args:
        sizes (Shape): Shape of the layers, [in_size, hidden1, ..., hiddenN, out_size]
        key (int): Intial PRNG key

    Returns:
        Params: Initialized parameters of the network
    """
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, np.sqrt(2 / (m + n)))
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def neural_network(activation: Activator) -> Callable[[Params, ArrayLike], Array]:
    """Create a neural network model

    Adapted from the JAX example on neural networks:
    https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html#auto-batching-predictions

    Args:
        activation (Activator): Activation function for the layers

    Returns:
        Callable[[Params, ArrayLike], Array]: Neural network model
    """

    def NN_model(params: Params, inputs: ArrayLike) -> Array:
        """MLP model, taking one input sample

        Note that this function should be vmapped.

        Args:
            params (Params): Parameters of the network
            inputs (ArrayLike): One input sample

        Returns:
            Array: Predicted output
        """
        for W, b in params:
            outputs = np.dot(W, inputs) + b
            inputs = activation(outputs)
        return outputs  # No activation on the last layer

    return NN_model

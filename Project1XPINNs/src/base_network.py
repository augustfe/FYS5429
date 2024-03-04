import jax.numpy as jnp
from jax import random, vmap
import numpy as np
from type_util import Activator, Array, Shape, Params, ArrayLike, Callable
from jax import lax


# ---------------------------- Jax documentation ----------------------------
# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(
    m: int, n: int, key: int, scale: float = 1e-2
) -> tuple[Array, Array]:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes: Shape, key: int) -> Params:
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


# ---------------------------------------------------------------------------


def neural_network(activation: Activator) -> Callable[[Params, ArrayLike], Array]:

    def NN_model(params: Params, input: ArrayLike) -> Array:
        z = input

        for w, b in params[:-1]:
            outputs = lax.add(lax.dot(w, z), b)
            # outputs = jnp.dot(w, z) + b
            z = activation(outputs)

        final_w, final_b = params[-1]
        z = lax.add(lax.dot(final_w, z), final_b)
        # z = jnp.dot(final_w, z) + final_b
        return z

    return NN_model


def XPINNs(activations: list[Activator]):
    pinns = []
    for activ in activations:
        pinns.append(neural_network(activ))

    return pinns


# Each PINN has interface loss


if __name__ == "__main__":
    layer_sizes = [2, 32, 3]
    params = init_network_params(layer_sizes, random.PRNGKey(0))
    activation = lambda x: jnp.tanh(x)  # noqa: E731
    predictor = neural_network(activation)
    input_size = 2
    x = jnp.arange(0, input_size)
    print(x)
    print(f"The shape is {x.shape}")
    print(predictor(params, x))
    x = np.random.randn(input_size, 4)

    v_predictor = vmap(predictor, in_axes=(None, 1))
    print("-----")
    print(x.shape)
    print(v_predictor(params, x))
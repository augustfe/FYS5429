import jax.numpy as np
from jax import jit, lax, jacobian, hessian, vmap, grad
from jax.tree_util import Partial
from typing import Callable
import numpy as onp
from tqdm import tqdm
from pathlib import Path
from jax.nn import tanh


@jit
def neural_network(
    input_layer: np.ndarray,
    hidden_layers: np.ndarray,
    output_layer: np.ndarray,
    x: np.ndarray,
    activation_func: Callable[[float], float],
    num_hidden: int,
) -> float:
    """Feedforward pass for neural network

    Args:
        input_layer (np.ndarray): Weights and biases for input layer
        hidden_layers (np.ndarray): Hidden layers, all of homogenous shape
        output_layer (np.ndarray): Weights and biases for output layer
        x (np.ndarray): Input vector
        activation_func (Callable[[float], float]): Activation function (Jax.tree_util.Partial)
        num_hidden (int): Number of hidden layers

    Returns:
        float: Output of the neural network
    """
    # x is now a point and a 1D numpy array; make it a column vector
    num_coordinates = np.size(x, 0)
    x = x.reshape(num_coordinates, -1)

    num_points = np.size(x, 1)

    # Assume that the input layer does nothing to the input x
    x_prev = np.concatenate((np.ones((1, num_points)), x), axis=0)
    z_hidden = lax.dot(input_layer, x_prev)
    x_prev = activation_func(z_hidden)

    def inner_loop(i: int, x: np.ndarray):
        # Choose correct layer
        w_hidden = hidden_layers[i, :, :]
        # Add a row of ones to include bias
        x_prev = np.concatenate((np.ones((1, num_points)), x), axis=0)

        z_hidden = lax.dot(w_hidden, x_prev)
        return activation_func(z_hidden)

    # Iterate through the hidden layers:
    x_prev = lax.fori_loop(0, num_hidden, inner_loop, x_prev)

    # Output layer:
    # Include bias:
    x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)

    z_output = lax.dot(output_layer, x_prev)
    # No activation for output layer
    x_output = z_output

    # Unpack the final layer
    return x_output


def setup_network(input_size: int, width: int, depth: int, output_size: int):
    # +1 to introduce bias
    input_layer = onp.random.randn(width, input_size + 1)
    hidden_layers = onp.random.randn(depth - 1, width, width + 1)
    output_layer = onp.random.randn(output_size, width + 1)

    x = np.arange(0, input_size)
    x = onp.random.randn(input_size, 4)
    # x = np.array([[1, 2], [1, 2], [3, 4], [3, 4]]).T
    # print(x.shape)
    # print(x)
    # x = np.vstack([x_t for i in range(3)])

    activation_func = Partial(tanh)

    test_output = neural_network(
        input_layer,
        hidden_layers,
        output_layer,
        x,
        activation_func,
        depth - 1,
    )

    print(test_output)


@jit
def update_P(P: np.ndarray, cost_grad: np.ndarray, lmb: float) -> np.ndarray:
    """Convenience function for updating the parameters of the network.

    Args:
        P (np.ndarray): Parameters of the network
        cost_grad (np.ndarray): Gradient of the cost, w.r.t. the parameters
        lmb (float): Learning rate for gradient descent

    Returns:
        np.ndarray: Updated parameters for the layer.
    """
    return lax.sub(P, lax.mul(lmb, cost_grad))


def back_prop(
    x: np.ndarray,
    P: list[np.ndarray],
    num_iter: int,
    lmb: float,
    activation_func: Callable[[float], float],
    cost_function: Callable,
) -> list[np.ndarray]:
    """Solve the chosen PDE using a physics-informed neural network.

    Args:
        x (np.ndarray): Input x values
        t (np.ndarray): Input t values
        num_neurons (list[int]): Size of each hidden layer
        num_iter (int): Number of iterations for gradient descent
        lmb (float): Learning rate for gradient descent
        activation_func (Callable[[float], float]): Activation function for the hidden layers

    Returns:
        list[np.ndarray]: Optimized parameters for the network.
    """

    print("Initial cost: ", cost_function(P, x, activation_func))

    cost_function_grad = jit(grad(cost_function, 0))

    for _ in tqdm(range(num_iter)):
        cost_grad = cost_function_grad(P, x, activation_func)

        for layer in range(3):
            P[layer] = update_P(P[layer], cost_grad[layer], lmb)

    print(f"Final cost: {cost_function(P, x, activation_func)}")

    return P


if __name__ == "__main__":
    setup_network(2, 50, 4, 1)

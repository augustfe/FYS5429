import optax
import jax.numpy as np
from base_network import init_network_params, neural_network
from jax import grad, jit, random, vmap
from type_util import (
    Activator,
    Array,
    Callable,
    GradientTransformation,
    Shape,
    Params,
    Updates,
    OptState,
)
from functools import partial

import pickle
from pathlib import Path


class PINN:
    def __init__(
        self,
        interior: Array,
        boundary: Array,
        activation: Activator,
        rand_key: int = 0,
    ) -> None:
        """Physics-Informed Neural Network

        Args:
            interior (np.ndarray): Points internal to the subdomain
            boundary (np.ndarray): Points along the boundary of the domain
            activation (Activator): Activation function for the layers
            rand_key (int): Initial key used to generate random values with jax
        """
        self.interior = interior
        self.boundary = boundary
        self.rand_key = rand_key
        self.model = neural_network(activation)
        self.v_model = jit(vmap(self.model, (None, 0)))

        self.residual = lambda params, points: 0
        self.v_residual = lambda params, points: 0

        self.interior_loss = lambda params, args: 0
        self.boundary_loss = lambda params, args: 0
        self.interface_loss = lambda params, args: 0

        self.loss = lambda params, args: 0
        self.grad_loss = lambda params, args: 0

    def init_params(self, sizes: Shape, optimizer: GradientTransformation) -> None:
        """Initialize the parameters of the network

        Args:
            sizes (Shape): Shape of the layers, [in_size, hidden1, ..., hiddenN, out_size]
            optimizer (GradientTransformation): Optax compatible optimizer
        """
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        key, self.rand_key = random.split(self.rand_key)
        self.params = init_network_params(sizes, key)

        self.optimizer = optimizer
        self.optstate = optimizer.init(self.params)

    @staticmethod
    @partial(jit, static_argnums=(0, 3))
    def update_iteration(
        grad_loss: Callable[[Params, dict], Updates],
        params: Params,
        args: dict[str, Array],
        optimizer: GradientTransformation,
        optstate: OptState,
    ) -> tuple[Params, OptState]:
        """One iteration of updating the parameters

        Args:
            grad_loss (Callable[[Params, dict], Updates]): Gradient of the loss function
            params (Params): Parameters of the neural network
            args (dict[str, Array]): Arguments needed in the loss functions
            optimizer (GradientTransformation): Chosen optax compatible optimizer
            optstate (OptState): Current state of the optimizer

        Returns:
            tuple[Params, OptState]: Updated parameters and state of the optimizer
        """
        grads = grad_loss(params, args)
        updates, optstate = optimizer.update(grads, optstate)
        params = optax.apply_updates(params, updates)

        return params, optstate

    def create_loss(self) -> Callable[[Params, dict[str, Array]], float]:
        """Creates the jitted loss function, after internal functions are set

        Returns:
            Callable[[Params, dict], float]: Loss function for the network
        """

        @jit
        def loss(params: Params, args: dict[str, Array]) -> float:
            """The loss function of the network

            Args:
                params (Params): Network parameters, i.e. weights and biases for the given layers
                args (dict): Relevant points, e.g.
                    {
                        "Internal": [[1, 2], ...]
                        "Boundary": [[0, 0], ...]
                        "12Interface_points": [[3, 3], ...]
                        "12Interface_values_here": [1, 2, ...]
                        "12Interface_values_there": [1.2, 1.8, ...]
                    }
                    such that the internal functions can be

                    def interface_loss_1(params, args):
                        I_p = args["12Interface_points"]
                        I_val_i = args["12Interface_values_here"]
                        I_val_j = args["12Interace_values_here"]
                        return l2_loss(I_val_i - I_val_j)

            Returns:
                float: Evaluated loss
            """
            return (
                self.interior_loss(params, args)
                + self.boundary_loss(params, args)
                + self.interface_loss(params, args)
            )

        self.loss = loss
        self.grad_loss = jit(grad(loss))

        return loss

    def predict(self, args: dict[str, Array]):
        b = args["boundary"]
        i = args["interior"]
        if b.size == 0:
            points = i
        else:
            points = np.vstack([b, i])

        prediction = self.v_model(self.params, points)
        return points, prediction

    def save_model(self, path: str) -> None:
        """Save the model parameters to a file

        Args:
            path (str): Path to save the model parameters
        """
        # Save optstate
        path.mkdir(parents=True, exist_ok=True)

        optstate_path = path / "optstate.pkl"
        with open(optstate_path, "wb") as f:
            pickle.dump(self.optstate, f)

        # Save params
        params_path = path / "params.npz"

        kwargs = {}
        for i, wb_tuple in enumerate(self.params):
            kwargs[f"weights_{i}"] = wb_tuple[0]
            kwargs[f"biases_{i}"] = wb_tuple[1]

        # Use np.savez to save the arrays in the file
        np.savez(params_path, **kwargs)

    def load_model(self, path: str) -> None:
        """Load the model parameters from a file

        Args:
            path (str): Path to load the model parameters
        """
        path = Path(path)
        # Load optstate
        optstate_path = path / "optstate.pkl"
        with open(optstate_path, "rb") as f:
            self.optstate = pickle.load(f)

        # Load params
        params_path = path / "params.npz"
        params = np.load(params_path)

        wb = []
        for i in range(len(params.items()) // 2):
            wb.append((params[f"weights_{i}"], params[f"biases_{i}"]))

        self.params = wb

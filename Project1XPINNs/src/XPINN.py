import jax.numpy as np
from jax import jit, random, grad  # noqa
import json
from pathlib import Path
from dataclasses import dataclass
from NN_model import init_network_params, neural_network
from type_util import (
    Array,
    Activator,
    Shape,
    Callable,
    GradientTransformation,
    Params,
    OptState,
    Updates,
)
import optax


@dataclass
class Interface:
    """Dataclass for storing interface information

    args:
        indices (list[int]): indicies of corresponding PINNs in the XPINN class
        points (np.ndarray): points along the interface
    """

    indices: list[int]
    points: Array


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
        """
        self.interior = interior
        self.boundary = boundary
        self.rand_key = rand_key
        self.model = neural_network(activation)

        self.interior_loss = lambda params, args: 0
        self.boundary_loss = lambda params, args: 0
        self.interface_loss = lambda params, args: 0

    def init_params(self, sizes: Shape) -> None:
        key, self.rand_key = random.split(self.rand_key)
        self.params = init_network_params(sizes, key)

    @staticmethod
    @jit
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

        return loss

    def set_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _feedforward(self, *args, **kwargs):
        raise NotImplementedError

    def _backpropogate(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @jit
    def compute_add(a: int, b: int):
        "Example method of jit-ing a method of a class"
        return a + b


class XPINN:
    def __init__(
        self,
        input_file: str | Path,
        activation: Activator,
        seed: int = 0,
    ) -> None:
        """Extended Physics-Informed Neural Network

        Args:
            input_file (str | Path): JSON-file containing the information about the domain/subdomain points
            activation (Activator): Activation function for the PINNs
            seed (int, optional): Seed to use for random values. Defaults to 0.

        Raises:
            FileExistsError: If the given file does not exist
        """
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileExistsError(f"The file {input_file} does not exist.")

        with open(input_file, "r") as infile:
            data = json.load(infile)

        key = random.PRNGKey(seed)

        # print(data)

        self.PINNs: list[PINN] = []
        self.Interfaces: list[Interface] = []

        for item in data["XPINNs"]:
            interior = np.asarray(item["Internal points"])
            boundary = np.asarray(item["Boundary points"])

            key, subkey = random.split(key)
            new_PINN = PINN(interior, boundary, activation, key)
            self.PINNs.append(new_PINN)

            key = subkey

        for item in data["Interfaces"]:
            indices = item["XPINNs"]
            points = np.asarray(item["Points"])
            new_Interface = Interface(indices, points)
            self.Interfaces.append(new_Interface)

        # print(self.PINNs)
        # print(self.Interfaces)

    def transfer_interface_values(self):
        raise NotImplementedError


if __name__ == "__main__":
    # base_dir = Path(__file__).parents[1]
    # data_dir = base_dir / "data"

    # network = XPINN(data_dir / "XPINN_template.json")

    for i in range(10):
        res = PINN.compute_add(i, i)
        print(res)

    a = np.arange(0, 10)
    b = np.arange(4, 14)

    print(PINN.compute_add(a, b))
    A = PINN(np.zeros(3), np.zeros(3), np.tanh)
    # loss = A.create_loss()
    # for i in range(10):
    #     print(loss(float(i)))

    # grad_loss = grad(loss)
    # for i in range(10):
    #     print(grad_loss(float(i)))

    x = np.arange(10)

    # @jit
    def foo():
        # global x
        return x * x

    print(foo())
    x = x.at[0].set(100)
    print(foo())

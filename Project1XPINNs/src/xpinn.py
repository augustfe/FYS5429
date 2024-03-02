import jax.numpy as np
from jax import jit, random, grad, vmap
import json
from pathlib import Path
from dataclasses import dataclass
from base_network import init_network_params, neural_network
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
from functools import partial


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

        # Set initial jax random key
        key = random.PRNGKey(seed)

        self.PINNs: list[PINN] = []
        self.Interfaces: list[Interface] = []
        self.main_args = {}

        # Store data for each PINN
        for i, item in enumerate(data["XPINNs"]):
            interior = np.asarray(item["Internal points"])
            boundary = np.asarray(item["Boundary points"])

            self.main_args[i] = {"boundary": boundary, "interior": interior}

            key, subkey = random.split(key)
            new_PINN = PINN(interior, boundary, activation, key)
            self.PINNs.append(new_PINN)

            key = subkey

        # Store data for each Interface
        for item in data["Interfaces"]:
            indices = item["XPINNs"]
            points = np.asarray(item["Points"])
            new_Interface = Interface(indices, points)
            self.Interfaces.append(new_Interface)

        for interface in self.Interfaces:
            a, b = sorted(interface.indices)
            for idx in interface.indices:
                args = self.main_args[idx]
                args[f"interface {a}{b}"] = interface.points

    def transfer_interface_values(self) -> None:
        """Communicate relevant values across each interface."""
        for interface in self.Interfaces:
            i, j = interface.indices
            self._interface_comm(i, j)
            self._interface_comm(j, i)

    def _interface_comm(self, i: int, j: int) -> None:
        """Compute values across boundary, and transfer.

        Args:
            i (int): Index of PINN to compute values for
            j (int): Index of PINN to save values for
        """
        a, b = sorted([i, j])
        pinn_i = self.PINNs[i]
        args_i = self.main_args[i]
        args_j = self.main_args[j]

        points = args_i[f"interface {a}{b}"]
        res_i = pinn_i.v_residual(pinn_i.params, points)
        args_j[f"interface res {i}"] = res_i

        val_i = pinn_i.v_model(pinn_i.params, points)
        args_j[f"interface val {i}"] = val_i

    def optimize_iter(self) -> list[float]:
        """One iteration of optimizing the networks.

        Returns:
            list[int]: Losses for the different networks
        """
        self.transfer_interface_values()
        losses: list[float] = []

        for i, pinn in enumerate(self.PINNs):
            args = self.main_args[i]

            iter_loss = pinn.loss(pinn.params, args)

            losses.append(iter_loss)

            params, optstate = pinn.update_iteration(
                pinn.grad_loss,
                pinn.params,
                args,
                pinn.optimizer,
                pinn.optstate,
            )
            # Typehinting
            params: Params
            optstate: OptState

            pinn.params, pinn.optstate = params, optstate

        return losses

    def run_iters(self, epoch: int) -> Array:
        losses = []
        for i in range(epoch):
            iter_loss = self.optimize_iter()
            losses.append(iter_loss)

            if i % 1000 == 0:
                print(
                    f"{i / epoch * 100:.2f}% iter = {i} of {epoch}: Total loss = {sum(iter_loss)}"
                )

        return np.asarray(losses)


@jit
def insert_elem(arr: Array, idx: int, jdx: int, elem) -> Array:
    arr = arr.at[idx, jdx].set(elem)
    return arr


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

import jax.numpy as np
from jax import jit, random
import json
from pathlib import Path
from dataclasses import dataclass
from NN_model import init_network_params, neural_network
from type_util import Array, Activator, Shape
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
        """Physcics-Informed Neural Network

        Args:
            interior (np.ndarray): Points internal to the subdomain
            boundary (np.ndarray): Points along the boundary of the domain
        """
        self.interior = interior
        self.boundary = boundary
        self.rand_key = rand_key
        self.model = neural_network(activation)

    def init_params(self, sizes: Shape) -> None:
        key, self.rand_key = random.split(self.rand_key)
        self.params = init_network_params(sizes, key)

    @staticmethod
    @jit
    def update_iteration(grad_loss, params, optimizer, optstate):
        grads = grad_loss(params)
        updates, optstate = optimizer.update(grads, optstate)
        params = optax.apply_updates(params, updates)

        return params, optstate

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
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileExistsError(f"The file {input_file} does not exist.")

        with open(input_file, "r") as infile:
            data = json.load(infile)

        key = random.PRNGKey(seed)

        print(data)

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

        print(self.PINNs)
        print(self.Interfaces)


if __name__ == "__main__":
    # base_dir = Path(__file__).parents[1]
    # data_dir = base_dir / "data"

    # network = XPINN(data_dir / "XPINN_template.json")

    for i in range(10):
        res = PINN.compute_add(i, i * 2)
        print(res)

    a = np.arange(0, 10)
    b = np.arange(4, 14)

    print(PINN.compute_add(a, b))

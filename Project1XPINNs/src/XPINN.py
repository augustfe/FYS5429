import jax.numpy as np
import numpy as onp
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Interface:
    """Dataclass for storing interface information

    args:
        indices (list[int]): indicies of corresponding PINNs in the XPINN class
        points (np.ndarray): points along the interface
    """

    indices: list[int]
    points: np.ndarray


class PINN:
    def __init__(self, interior: np.ndarray, boundary: np.ndarray) -> None:
        """Physcics-Informed Neural Network

        Args:
            interior (np.ndarray): Points internal to the subdomain
            boundary (np.ndarray): Points along the boundary of the domain
        """
        self.interior = interior
        self.boundary = boundary

    def set_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _feedforward(self, *args, **kwargs):
        raise NotImplementedError

    def _backpropogate(self, *args, **kwargs):
        raise NotImplementedError


class XPINN:
    def __init__(self, input_file: str | Path, seed: int = 0):
        input_file = Path(input_file)

        if not input_file.exists():
            raise FileExistsError(f"The file {input_file} does not exist.")

        with open(input_file, "r") as infile:
            data = json.load(infile)

        onp.random.seed(seed)

        print(data)

        self.PINNs: list[PINN] = []
        self.Interfaces: list[Interface] = []

        for item in data["XPINNs"]:
            interior = np.asarray(item["Internal points"])
            boundary = np.asarray(item["Boundary points"])
            new_PINN = PINN(interior, boundary)
            self.PINNs.append(new_PINN)

        for item in data["Interfaces"]:
            indices = item["XPINNs"]
            points = np.asarray(item["Points"])
            new_Interface = Interface(indices, points)
            self.Interfaces.append(new_Interface)

        print(self.PINNs)
        print(self.Interfaces)


if __name__ == "__main__":
    base_dir = Path(__file__).parents[1]
    data_dir = base_dir / "data"

    network = XPINN(data_dir / "XPINN_template.json")

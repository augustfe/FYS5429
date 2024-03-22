import jax.numpy as np
import json

from jax import random
from pathlib import Path
from dataclasses import dataclass
from type_util import Array, Activator, Params, OptState, Shape
from pinn import PINN
import numpy as onp


@dataclass
class Interface:
    """Dataclass for storing interface information

    args:
        indices (list[int]): indicies of corresponding PINNs in the XPINN class
        points (np.ndarray): points along the interface
    """

    indices: list[int]
    points: Array


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
            interior = np.asarray(item["Internal points"], dtype=np.float32)
            boundary = np.asarray(item["Boundary points"], dtype=np.float32)

            self.main_args[i] = {}

            for dkey in item:
                if dkey != "Internal points" and dkey != "Boundary points":
                    self.main_args[i][dkey] = np.asarray(
                        item[dkey], dtype=np.float32)

            self.main_args[i]['boundary'] = boundary
            self.main_args[i]['interior'] = interior

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

    def set_loss(self) -> None:
        """Initialize the loss functions for each PINN."""
        for pinn in self.PINNs:
            pinn.create_loss()

    def initialize_params(self, sizes: list[Shape], optimizer) -> None:
        """Initialize the parameters for each PINN.

        Args:
            sizes (list[int]): Sizes of the layers for each PINN
            optimizer ([type]): Optax compatible optimizer
        """
        self.input_size = sizes[0][0]
        self.output_size = sizes[0][-1]

        for pinn, size in zip(self.PINNs, sizes):
            pinn.init_params(size, optimizer)

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

    def optimize_iter(self, epoch: int) -> None:
        """One iteration of optimizing the networks.

        Returns:
            list[int]: Losses for the different networks
        """
        self.transfer_interface_values()

        for i, pinn in enumerate(self.PINNs):
            args = self.main_args[i]

            iter_loss = pinn.loss(pinn.params, args)

            self.losses[i, epoch] = iter_loss

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

    def run_iters(self, num_epoch: int) -> Array:
        self.losses = onp.zeros((len(self.PINNs), num_epoch))

        print_num = max(num_epoch // 10, 1)

        self.set_loss()

        for epoch in range(num_epoch):
            self.optimize_iter(epoch)

            iter_loss = sum(self.losses[:, epoch])
            if epoch % print_num == 0:
                print(
                    f"{epoch / num_epoch * 100:.2f}% iter = {epoch} of {num_epoch}: Total loss = {iter_loss}"
                )

        print(
            f"{(epoch+1) / num_epoch * 100:.2f}% iter = {epoch + 1} of {num_epoch}: Total loss = {iter_loss}"
        )

        return np.asarray(self.losses)

    def predict(self, input_file: str | Path = None):
        if input_file:
            main_args = {}
            with open(input_file) as infile:
                data = json.load(infile)

            for i, item in enumerate(data["XPINNs"]):
                interior = np.asarray(item["Internal points"])
                boundary = np.asarray(item["Boundary points"])

                main_args[i] = {"boundary": boundary, "interior": interior}

        else:
            main_args = self.main_args

        total_points = []
        predictions = []

        for i, pinn in enumerate(self.PINNs):
            points, prediction = pinn.predict(main_args[i])

            total_points.append(points)
            predictions.append(prediction)

        return total_points, predictions

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for i, pinn in enumerate(self.PINNs):
            pinn.save_model(path / f"pinn_{i}")
    
    def load_model(self, path: str | Path) -> None:
        path = Path(path)

        for i, pinn in enumerate(self.PINNs):
            pinn.load_model(path / f"pinn_{i}")
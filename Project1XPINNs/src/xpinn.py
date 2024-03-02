import jax.numpy as np
import json

from jax import random
from pathlib import Path
from dataclasses import dataclass
from type_util import (
    Array,
    Activator,
    Params,
    OptState,
)
from pinn import PINN


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

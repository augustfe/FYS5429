from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from type_util import Array
from xpinn import XPINN
from jax import vmap, lax

# Set the default DPI to specific value (e.g., 300)
plt.rcParams["figure.dpi"] = 300


# Set up for LaTeX rendering
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["figure.titlesize"] = 20
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14


def plot_poisson(
    points: Array,
    val: Array,
    title: str,
    savepath: Path,
    save_name: str,
    clim: tuple = None,
) -> None:
    """Plot the values of the Poisson equation.

    Args:
        points (Array): Points to plot
        val (Array): Values to plot
        title (str): Title of the plot
        savepath (Path): Path to save the figure
        save_name (str): Name of the file to save the figure as
        clim (tuple): Color limits
    """
    plt.scatter(points[:, 0], points[:, 1], c=val, cmap="turbo", s=1)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    if clim:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    plt.title(title)
    plt.savefig(savepath / f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_navier_stokes(
    points: Array,
    val: Array,
    title: str,
    savepath: Path,
    save_name: str,
    clim: tuple = None,
) -> None:
    """Plot the values of the Navier-Stokes equation.

    Args:
        points (Array): Points to plot
        val (Array): Values to plot
        title (str): Title of the plot
        savepath (Path): Path to save the figure
        save_name (str): Name of the file to save the figure as
        clim (tuple): Color limits
    """
    plt.scatter(points[:, 0], points[:, 1], c=val, cmap="turbo", s=1)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.xlim(0, 2.2)
    plt.ylim(0, 0.41)
    plt.gca().set_aspect("equal", adjustable="box")
    if clim:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath / f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_domain(
    xpinn: XPINN,
    savepath: Path,
    title: str,
    save_name: str,
    axis_labels: tuple[str, str] = ("x", "y"),
) -> None:
    """Plot the domain of the PINNs.

    Args:
        xpinn (XPINN): Instance of the XPINN class
        savepath (Path): Path to save the figure
        title (str): Title of the figure
        save_name (str): Name of the file to save the figure as
        axis_labels (tuple[str, str], optional): Labels for the x and y axes. Defaults to ("x", "y").
    """
    if len(xpinn.PINNs) == 1:
        pinn = xpinn.PINNs[0]
        plt.scatter(pinn.interior[:, 0], pinn.interior[:, 1], label="Interior")
        plt.scatter(pinn.boundary[:, 0], pinn.boundary[:, 1], label="Boundary")

    else:
        for i, pinn in enumerate(xpinn.PINNs):
            plt.scatter(
                pinn.interior[:, 0],
                pinn.interior[:, 1],
                label=f"Interior {i}",
            )
            if pinn.boundary.size != 0:
                plt.scatter(
                    pinn.boundary[:, 0],
                    pinn.boundary[:, 1],
                    label=f"Boundary {i}",
                )

        for interface in xpinn.Interfaces:
            plt.scatter(
                interface.points[:, 0],
                interface.points[:, 1],
                label=f"Interface {interface.indices}",
            )

    plt.legend()
    plt.xlabel(f"${axis_labels[0]}$")
    plt.ylabel(f"${axis_labels[1]}$")
    plt.title(title)
    plt.savefig(savepath / f"{save_name}.pdf", bbox_inches="tight")
    plt.show()


def plot_losses(
    a_losses: Array,
    n_iter: int,
    title: str,
    savepath: Path,
    save_name: str,
    t_0: int = 0,
) -> None:
    """Plot the losses of the PINNs.

    Args:
        a_losses (Array): Array of losses
        n_iter (int): Number of iterations
        title (str): Title of the figure
        savepath (Path): Path to save the figure
        save_name (str): Name of the file to save the figure as
        t_0 (int, optional): Starting point for the plot. Defaults to 0.
    """
    t = np.arange(t_0, n_iter)
    for i, loss in enumerate(a_losses):
        plt.plot(t, loss[t_0:], label=f"PINN {i}")
    plt.plot(t, np.sum(a_losses, axis=0)[t_0:], "--", label="Total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.title(title)
    plt.savefig(savepath / f"{save_name}.pdf", bbox_inches="tight")
    plt.show()


def compare_analytical_advection(
    xpinn: XPINN,
    file_test: Path,
    savepath: Path,
    savename: str,
    alpha: float = 0.5,
    prefix: str = "",
) -> None:
    """Compare the analytical solution with the predictions for the advection equation.

    Args:
        xpinn (XPINN): Instance of the XPINN class
        file_test (Path): Path to the test file
        savepath (Path): Path to save the figures
        savename (str): Name of the file to save the figures as
        alpha (float, optional): Advection velocity. Defaults to 0.5.
        prefix (str, optional): Prefix for the figures. Defaults to "".
    """

    def analytical_solution(point: Array, alpha: float = 0.5) -> float:
        x = point[0]
        t = point[1]
        condition = (x - alpha * t > -0.2) & (x - alpha * t < 0.2)
        return lax.select(condition, 1.0, 0.0)

    points, predictions = xpinn.predict(file_test)
    total_points = np.concatenate(points)
    total_pred = np.concatenate(predictions).flatten()

    # Compute all errors and predictions first
    analytical_values = vmap(lambda point: analytical_solution(point, alpha))(
        total_points
    )
    analytical_values: Array

    errors = np.abs(analytical_values - total_pred)

    # Scatter plot for errors
    scatter1 = plt.scatter(
        total_points[:, 0],
        total_points[:, 1],
        c=errors,
        cmap="turbo",
        s=1,
    )
    func_name = f"$u_t {'+' if alpha > 0 else '-'} {abs(alpha)}u_x = 0$"
    plt.title(f"Absolute error for {func_name}")
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    scatter1.set_clim(0, errors.max())
    plt.colorbar(scatter1)

    plt.savefig(savepath / f"{savename}_error.png", bbox_inches="tight")
    plt.show()

    # Scatter plot for predictions
    scatter2 = plt.scatter(
        total_points[:, 0],
        total_points[:, 1],
        c=total_pred,
        cmap="coolwarm",
        s=1,
    )
    plt.title(f"Predictions for {func_name}")
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    scatter2.set_clim(total_pred.min(), total_pred.max())
    plt.colorbar(scatter2)
    plt.savefig(savepath / f"{savename}_predictions.png", bbox_inches="tight")
    plt.show()

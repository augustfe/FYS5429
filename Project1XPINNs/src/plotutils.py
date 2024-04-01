from pathlib import Path
import numpy as np
from matplotlib import colormaps, pyplot as plt, cm
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
# xtick.labelsize : 16
# ytick.labelsize : 16


def setColors(
    variable_arr: np.ndarray,
    cmap_name: str = "viridis",
    norm_type: str = "log",
) -> tuple[mpl.colors.Colormap, mpl.colors.LogNorm, mpl.cm.ScalarMappable]:
    """
    Returns a colormap, a normalization instance, and a scalar mappable instance.

    Args:
        variable_arr (np.ndaray):
            Array of values to be plotted.
        cmap_name (str, optional):
            Name of the colormap. Defaults to "viridis".
        norm_type (str, optional):
            Type of normalization to use. Defaults to "log".

    Returns:
        tuple[mpl.colors.Colormap, mpl.colors.LogNorm, mpl.cm.ScalarMappable]:
            A tuple containing the colormap, normalization instance, and scalar mappable instance.
    """
    cmap = colormaps.get_cmap(cmap_name)
    if norm_type == "log":
        norm = mpl.colors.LogNorm(vmin=np.min(variable_arr), vmax=np.max(variable_arr))
    elif norm_type == "linear":
        norm = mpl.colors.Normalize(
            vmin=np.min(variable_arr), vmax=np.max(variable_arr)
        )
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return cmap, norm, sm


def plot_poisson(
    points: Array,
    val: Array,
    title: str,
    savepath: Path,
    save_name: str,
    clim: tuple = None,
):
    """Plot the values of the Poisson equation.

    Args:
        points (Array): Points to plot
        val (Array): Values to plot
        title (str): Title of the plot
        savepath (Path): Path to save the figure
        save_name (str): Name of the file to save the figure as
        clim (tuple): Color limits
    """
    # fig, ax = plt.figure()
    # ax.set_aspect("equal")
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
):
    """Plot the values of the Navier-Stokes equation.

    Args:
        points (Array): Points to plot
        val (Array): Values to plot
        title (str): Title of the plot
        savepath (Path): Path to save the figure
        save_name (str): Name of the file to save the figure as
        clim (tuple): Color limits
    """
    # fig, ax = plt.figure()
    # ax.set_aspect("equal")
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
        xpinn (XPINN): The XPINN object containing the PINNs.
        savepath (Path): Path to save the figure.
        title (str): Title of the figure.
        cmap_name (str, optional): Name of the colormap. Defaults to "viridis".
        norm_type (str, optional): Type of normalization to use. Defaults to "log".
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
    # alpha: float = 0.5,
) -> None:
    """Plot the losses of the PINNs over the training iterations.

    Args:
        a_losses (Array): Array of losses for each PINN
        t_0 (int): Starting index for the plot
        n_iter (int): Number of iterations
        title (str): Title of the plot
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
    # plt.title(f"Loss per Pinn over {n_iter} epochs")


def compare_analytical_advection(
    xpinn: XPINN,
    file_test: Path,
    savepath: Path,
    savename: str,
    alpha: float = 0.5,
    prefix: str = "",
):
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
        total_points[:, 0], total_points[:, 1], c=errors, cmap="turbo", s=1
    )
    func_name = f"$u_t {'+' if alpha > 0 else '-'} {abs(alpha)}u_x = 0$"
    plt.title(f"Absolute error for {func_name}")
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    scatter1.set_clim(0, errors.max())
    plt.colorbar(scatter1)  # Add one colorbar based on the errors

    plt.savefig(savepath / f"{savename}_error.png", bbox_inches="tight")
    plt.show()

    # Scatter plot for predictions
    scatter2 = plt.scatter(
        total_points[:, 0], total_points[:, 1], c=total_pred, cmap="coolwarm", s=1
    )
    plt.title(f"Predictions for {func_name}")
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    scatter2.set_clim(total_pred.min(), total_pred.max())
    plt.colorbar(scatter2)  # Add one colorbar based on the predictions
    plt.savefig(savepath / f"{savename}_predictions.png", bbox_inches="tight")
    plt.show()


def setup_axis(xlim: tuple[int], ylim: tuple[int]) -> plt.Axes:
    """Set up the axis for a function plot.

    Args:
        xlim (tuple[int]): The limits of the x-axis.
        ylim (tuple[int]): The limits of the y-axis.

    Returns:
        plt.Axes: The axis for the plot.
    """
    _, ax = plt.subplots()

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    for s in ax.spines.values():
        s.set_zorder(0)

    return ax


def set_plot_limits(ax, xlim=None, ylim=None, zlim=None):
    """
    Set the limits for the x, y, and z axes of a Matplotlib plot.

    Parameters:
    - ax: The axis object of the plot.
    - xlim: Tuple containing the lower and upper limits for the x-axis, e.g., (xmin, xmax).
    - ylim: Tuple containing the lower and upper limits for the y-axis, e.g., (ymin, ymax).
    - zlim: Tuple containing the lower and upper limits for the z-axis, e.g., (zmin, zmax).
    """

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)


def plot_at_timestep(
    x: np.ndarray,
    res_dnn: np.ndarray,
    res_analytic: np.ndarray,
    t: float,
    func_name: str,
    save: bool,
    savePath: Path = None,
    saveName: str = None,
) -> None:
    """Plot the computed solution at a given time step against the analytic solution.

    Args:
        x (np.ndarray): Spatial coordinates computed at
        res_dnn (np.ndarray): Predicted from the DNN
        res_analytic (np.ndarray): Analytic solution
        t (float): Time step
        func_name (str): Name of the activation function used
        save (bool): Whether to save the plot
        savePath (Path, optional): Path of the directory to save figures. Defaults to None.
        saveName (str, optional): Name of the file to save figure as. Defaults to None.
    """
    plt.figure(figsize=plt.figaspect(0.5))
    plt.title(f"{func_name}: Computed solutions at time = {t:g}")
    plt.plot(x, res_dnn, label="Deep neural network")
    plt.plot(x, res_analytic, label="Analytical")
    plt.xlabel("Position $x$")
    plt.legend()
    if save:
        plt.savefig(savePath / f"{saveName}_timestep_{t}.pdf", bbox_inches="tight")
    plt.close()


def plot_at_timestepEuler(
    x_numeric: np.ndarray,
    res_numeric: np.ndarray,
    x_anal: np.ndarray,
    res_analytic: np.ndarray,
    t: float,
    func_name: str,
    save: bool,
    savePath: Path = None,
    saveName: str = None,
) -> None:
    """Plot the computed solution at a given time step agains the analytic.

    Args:
        x_numeric (np.ndarray): x-coordinates for the numerical solution
        res_numeric (np.ndarray): Computed solution
        x_anal (np.ndarray): x-coordinates for the analytic solution
        res_analytic (np.ndarray): Analytic solution for comparison
        t (float): Time step computed at
        func_name (str): Name of the method used
        save (bool): Whether to save the figure
        savePath (Path, optional): Path to the directory to save the figure to. Defaults to None.
        saveName (str, optional): What to save the figure as. Defaults to None.
    """
    plt.figure(figsize=plt.figaspect(0.5))
    plt.title(f"{func_name}: Computed solutions at time = {t:g}")
    plt.plot(x_anal, res_analytic, label="Analytical")
    plt.plot(x_numeric, res_numeric, label="Forward Euler")
    plt.xlabel("Position $x$")
    plt.legend()
    if save:
        plt.savefig(savePath / f"{saveName}_timestep_{t}.pdf", bbox_inches="tight")
    plt.close()


def plot_surface(
    T: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    title: str,
    save: bool,
    savePath: Path = None,
    saveName: str = None,
) -> None:
    """Plot the predicted solution as a surface plot.

    Args:
        T (np.ndarray): Time coordinates
        X (np.ndarray): Spatial coordinates
        Z (np.ndarray): Predicted / Computed solution
        title (str): Title of the figure
        save (bool): Whether to save the figure
        savePath (Path, optional): Where to save the figure to. Defaults to None.
        saveName (str, optional): What to save the figure as. Defaults to None.
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)
    ax.plot_surface(T, X, Z, linewidth=0, antialiased=False, cmap=cm.viridis)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Position $x$")
    if save:
        plt.savefig(savePath / f"{saveName}.pdf", bbox_inches="tight")
    plt.close()

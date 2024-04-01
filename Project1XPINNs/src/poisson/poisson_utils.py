from typing import Callable

import jax.numpy as np
from jax import hessian, jit, vmap

from pinn import PINN
from region_utils.region import Domain, Subdomain
from region_utils.shapes import ConvexPolygon
from type_util import Array, Params
from utils import data_path
from xpinn import XPINN

LFunc = Callable[[Params, dict[str, Array]], Array]


def boundary_loss_factory(
    pinn_model: PINN, target: float | Array, weight: float = 1.0
) -> LFunc:
    """
    Factory for generating the boundary loss for the Poisson problem for Dirichlet boundary conditions.

    Args:
        pinn_model (PINN): Your PINN that you want to design the boundary loss for.
        target (float | Array): A singular value or an array of values over the

    Returns:
        LFunc: Your boundary loss function
    """

    def boundary_loss(params: Params, points: dict[str, Array]) -> Array:
        pts = points["boundary"]
        eval = pinn_model.v_model(params, pts)
        return weight * np.mean((eval - target) ** 2)

    return boundary_loss


def interior_loss_factory(
    pinn_model: PINN, rhs: Callable[[Array], Array] | Array, weight: float = 1.0
) -> LFunc:
    """
    Factory for generating the interior loss for the Poisson problem and the residual in the problem.
    Args:
        pinn_model (PINN): The relevant PINN
        rhs (Callable[[Array], Array]): The rhs of your Poisson problem -Δu = f

    Returns:
        LFunc: The interior loss function
        Sets the v_residual in your PINN
    """

    def hess(params):
        return hessian(lambda x: pinn_model.model(params, x))

    if callable(rhs):

        def residual(params, x):
            return np.trace(hess(params)(x)[0]) + rhs(x)

    else:

        def residual(params, x):
            return np.trace(hess(params)(x)[0]) + rhs

    v_residual = jit(vmap(residual, (None, 0)))
    pinn_model.v_residual = v_residual

    def interior_loss(params: Params, points: dict[str, Array]) -> Array:
        pts = points["interior"]
        return weight * np.mean(v_residual(params, pts) ** 2)

    return interior_loss


def interface_loss_factory(
    xpinn_model: XPINN, i: int, j: int, weight: float = 1.0
) -> LFunc:
    a, b = sorted([i, j])
    pi = xpinn_model.PINNs[i]

    def interface_loss(params: Params, args: dict[str, Array]) -> Array:
        inter_points = args[f"interface {a}{b}"]
        res_j = args[f"interface res {j}"]
        res_ij = np.mean((pi.v_residual(params, inter_points) - res_j) ** 2)
        # res_ij = 0

        val_j = args[f"interface val {j}"]
        avg_ij = np.mean(((pi.v_model(params, inter_points) - val_j) / 2.0) ** 2)
        # avg_ij = 0

        return weight * (res_ij + avg_ij)

    return interface_loss


def create_PINN_domain(internal_points: int, boundary_points: int):
    vertecies = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape(4, -1)
    square = ConvexPolygon(
        vertecies,
        [0, 1, 2, 3],
    )

    subdom = Subdomain([square])
    domain = Domain([subdom])

    ### Create domain after paper specifications https://epubs.siam.org/doi/epdf/10.1137/21M1447039
    domain.create_interior(internal_points, [0, 0], [1, 1])
    domain.create_boundary(boundary_points)

    domain.write_to_file(data_path / "poisson_single_pinn_train.json")


def create_PINN_test_domain(n: int = 1000):
    vertecies = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape(4, -1)
    square = ConvexPolygon(
        vertecies,
        [0, 1, 2, 3],
    )
    subdom = Subdomain([square])
    domain = Domain([subdom])
    domain.create_testing_data(n, [0, 0], [1, 1])
    domain.write_to_file(data_path / "poisson_single_pinn_test.json", False)


def crude_rel_L2(prediction: Array, true_value: Array, true_integral: float):
    return np.sqrt(np.sum((true_value - prediction) ** 2)) / true_integral


def create_XPINN_domain(
    internal_points: int, boundary_points: int, interface_points: int
):
    total_vertices = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.75, 0.25],
        ]
    )
    region_idxs = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
    boundary_idxs = [[0, 1, 2, 3], []]

    outer_rectangle = ConvexPolygon(
        total_vertices[region_idxs[0]],
        boundary_idxs[0],
    )
    inner_rectangle = ConvexPolygon(
        total_vertices[region_idxs[1]],
        boundary_idxs[1],
    )

    subdomain1 = Subdomain([outer_rectangle], subtraction=[inner_rectangle])
    subdomain2 = Subdomain([inner_rectangle])

    domain = Domain([subdomain1, subdomain2])

    ### Create domain after paper specifications https://epubs.siam.org/doi/epdf/10.1137/21M1447039
    domain.create_interior(internal_points, [0, 0], [1, 1])
    domain.create_boundary(boundary_points)
    idx = [4, 5, 6, 7, 4]
    points_per_side = int(interface_points / 4)
    for i in range(4):
        domain.create_interface(
            points_per_side,
            (0, 1),
            (total_vertices[idx[i]], total_vertices[idx[i + 1]]),
        )

    domain.write_to_file(data_path / "poisson_train.json")


def create_XPINN_test_domain(n: int = 1000):
    total_vertices = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.75, 0.25],
        ]
    )
    region_idxs = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
    boundary_idxs = [[0, 1, 2, 3], []]

    outer_rectangle = ConvexPolygon(
        total_vertices[region_idxs[0]],
        boundary_idxs[0],
    )
    inner_rectangle = ConvexPolygon(
        total_vertices[region_idxs[1]],
        boundary_idxs[1],
    )

    subdomain1 = Subdomain([outer_rectangle], subtraction=[inner_rectangle])
    subdomain2 = Subdomain([inner_rectangle])

    domain = Domain([subdomain1, subdomain2])
    domain.create_testing_data(n, [0, 0], [1, 1])
    domain.write_to_file(data_path / "poisson_test.json", False)

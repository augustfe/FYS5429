# fmt: off


import jax.numpy as np
from jax import jit, vmap
from type_util import Params

from navier_stokes_funcs import model, hess_psi

# fmt: on


def drag_lift_force(params: Params, nu: float, n_points: int = 100000):
    points = []
    norm_vecs = []

    r = 0.05
    c_x = 0.2
    c_y = 0.2
    for i in range(0, n_points):
        points.append(
            np.array(
                [
                    c_x + r * np.cos(2 * np.pi * (i / n_points)),
                    c_y + r * np.sin(2 * np.pi * (i / n_points)),
                ]
            )
        )
        norm_vecs.append(
            np.array(
                [np.cos(2 * np.pi * (i / n_points)),
                 np.sin(2 * np.pi * (i / n_points))]
            )
        )

    points = np.array(points)
    norm_vecs = np.array(norm_vecs)
    interval = 2 * r * 3.14 / n_points

    def sigma(params, xy):
        hess = hess_psi(params, xy)
        # hess = np.array([[1, 0], [1, 1]])

        press = model(params, xy)[1]

        return np.array(
            [
                [nu * hess[1, 0] - press, nu * (-hess[0, 0])],
                [nu * hess[1, 1], nu * (-hess[0, 1]) - press],
            ]
        )

    sigma = jit(vmap(sigma, (None, 0)))

    sigma_matrices = sigma(params, points)

    inside_int = np.array(
        [mat @ norm for (mat, norm) in zip(sigma_matrices, norm_vecs)]
    )

    drag_force = np.sum(inside_int[:, 0], axis=0) * interval
    lift_force = np.sum(inside_int[:, 1], axis=0) * interval

    return drag_force.item(), lift_force.item()


def drag_lift_coefficients(params: Params, nu: float, n_points: int = 100000):
    drag_force, lift_force = drag_lift_force(params, nu, n_points)
    U_mean = 0.2
    L = 0.1

    C_D = 2 * drag_force / (U_mean ** 2 * L)
    C_L = 2 * lift_force / (U_mean ** 2 * L)

    return C_D, C_L

# Example usage
# params = xpinn.PINNs[0].params
# C_D, C_L = drag_lift_coefficients(params, nu = 0.001)


def pressure_diff(params: Params):
    p1 = model(params, np.array((0.15, 0.2)))[1]
    p2 = model(params, np.array((0.25, 0.2)))[1]

    return p1 - p2


# Example usage
# params = xpinn.PINNs[0].params
# pressure_diff = pressure_diff(params)
if __name__ == "__main__":
    h = hess_psi([np.array([1, 2, 3, 4]).reshape(2, -1),
                  np.array([1, 2])], np.array([0.2, 0.2]))
    print(h)

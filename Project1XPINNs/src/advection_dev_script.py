"""Script version of advection_dev.ipynb.

Should only currently be used for profiling XPINN.py, might not be updated
"""

from pathlib import Path
from XPINN import XPINN
import optax
from type_util import Array
from jax import jacobian, jit, vmap
import jax.numpy as np

file = Path("../data/advection_constraints.json")
xpinn = XPINN(file, np.tanh)


def create_boundary_loss(index: int, target: float | Array):
    def boundary_loss(params, args):
        points = args["boundary"]
        eval = xpinn.PINNs[index].v_model(params, points)
        return np.mean((eval - target) ** 2)
        # return optax.l2_loss(eval, target)

    return boundary_loss


# fmt: off
def create_interior_loss(index: int):
    model = xpinn.PINNs[index].model
    jacob = lambda params: jacobian(lambda x: model(params, x))                  # noqa: E731
    N_dt = lambda params, x: jacob(params)(x)[1]                                 # noqa: E731
    N_dx = lambda params, x: jacob(params)(x)[0]                                 # noqa: E731
    residual = lambda params, x: (N_dt(params, x) + 0.5 * N_dx(params, x)) ** 2  # noqa: E731
    v_residual = jit(vmap(residual, (None, 0)))
    xpinn.PINNs[index].v_residual = v_residual

    def interior_loss(params, args):
        points = args["interior"]
        return np.mean(v_residual(params, points))

    return interior_loss
# fmt: on


def create_inter(i: int, j: int):

    a, b = sorted([i, j])
    pi = xpinn.PINNs[i]

    def interface_loss(params, args):
        inter_points = args[f"interface {a}{b}"]
        res_j = args[f"interface res {j}"]
        res_ij = np.mean((pi.v_residual(params, inter_points) - res_j) ** 2)
        # res_ij = 0

        val_j = args[f"interface val {j}"]
        avg_ij = np.mean((pi.v_model(params, inter_points) - val_j) ** 2)
        # avg_ij = 0

        return res_ij + avg_ij

    return interface_loss


# in_size = xpinn.PINNs[0].input_size

p0, p1, p2 = xpinn.PINNs

p0.boundary_loss = create_boundary_loss(0, 1.0)
p1.boundary_loss = create_boundary_loss(1, 0.0)
p2.boundary_loss = create_boundary_loss(2, 0.0)

p0.interface_loss = lambda params, args: create_inter(0, 1)(
    params,
    args,
) + create_inter(0, 2)(
    params,
    args,
)
p1.interface_loss = lambda params, args: create_inter(1, 0)(
    params,
    args,
)
p2.interface_loss = lambda params, args: create_inter(2, 0)(
    params,
    args,
)

for p in xpinn.PINNs:
    p.create_loss()

# losses = xpinn.run_iters(10000)
# print(losses)

import matplotlib.pyplot as plt  # noqa

for epoch in [100, 1000, 10000]:
    shapes = [[2, 10, 10, 1], [2, 50, 50, 1], [2, 100, 1]]
    for pinn, shape in zip(xpinn.PINNs, shapes):
        optimizer = optax.adam(learning_rate=0.0003)

        pinn.init_params(shape, optimizer)

    losses = xpinn.run_iters(epoch).T
    t_0 = 0
    t = np.arange(t_0, epoch)

    for i in range(3):
        plt.plot(t, losses[i, t_0:], label=f"PINN {i}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title(f"Loss per Pinn over {epoch} epochs")
    plt.show()

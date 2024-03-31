from xpinn import XPINN
import jax.numpy as np
from utils import data_path, model_path
from typing import Callable
from type_util import Params, Array
from base_network import neural_network
from jax import hessian, jit, vmap, grad, jacobian
from typing import Tuple
import optax
import jax

jax.config.update("jax_enable_x64", True)

data_path = data_path / "NavierStokes"

file_train = data_path / "single_pinn_train_400_5000.json"
file_test = data_path / "test.json"
activation = np.tanh
xpinn = XPINN(file_train, activation)

LFunc = Callable[[Params, dict[str, Array]], Array]

model = neural_network(activation)
v_model = vmap(model, (None, 0))


def psi(params, xy): return model(params, xy)[0]


hess_psi = hessian(psi, argnums=1)
d_psi_dxy = grad(psi, argnums=1)


def advection_term(params: Params, xy: Array) -> Array:
    hess = hess_psi(params, xy)
    processed_hess = np.array(
        [[-hess[1, 1], hess[0, 1]], [hess[1, 0], -hess[0, 0]]])
    return processed_hess @ d_psi_dxy(params, xy)


jacobi_hess_psi = jacobian(hess_psi, argnums=1)


def diffusion_term(params: Params, xy: Array):
    jachessi = jacobi_hess_psi(params, xy)
    u_diffusion = jachessi[0, 1, 0] + jachessi[1, 1, 1]  # psi_yxx + psi_yyy
    v_diffusion = - jachessi[1, 0, 1] - jachessi[0, 0, 0]  # psi_xyy + psi_xxx
    return np.array([u_diffusion, v_diffusion])


def p(params, xy): return model(params, xy)[1]


d_p = grad(p, argnums=1)


def navier_stokes_residual_factory(index: int, nu: float, weight: int = 1) -> LFunc:

    def residual(params, xy):
        return advection_term(params, xy) - nu * diffusion_term(params, xy) + d_p(params, xy)

    v_residual = jit(vmap(residual, (None, 0)))
    xpinn.PINNs[index].v_residual = v_residual

    def interior_loss(params: Params, points: dict[str, Array]) -> Array:
        pts = points["interior"]
        return weight * 2 * np.mean(v_residual(params, pts) ** 2)
    return interior_loss
# Good until here


U = 0.3


def inflow_func(xy): return np.array(
    (4 * U * xy[1] * (0.41 - xy[1])/(0.41**2), 0.0))


def uv(params: Params, xy: Array) -> Array:
    d_psi = d_psi_dxy(params, xy)
    u = d_psi[1]
    v = -d_psi[0]
    return np.array([u, v])


def boundary_loss_factory2(inflow_func: Callable[[Array], Array], nu: float, weights: Tuple[int, int, int, int] = (1, 1, 1, 1)) -> LFunc:

    def left_boundary_loss(params, xy):
        # (u - inflow)**2 + v**2
        return np.sum(np.square(uv(params, xy) - inflow_func(xy)))

    def wall_boundary_loss(params, xy):
        return np.sum(np.square(uv(params, xy)))  # u**2 + v**2

    def right_boundary_loss(params, xy):
        u_ = hess_psi(params, xy)[:, 1]
        u_x = u_[0]
        u_y = u_[1]
        return (nu * u_x - p(params, xy)) ** 2 + (nu * u_y) ** 2

    v_left_boundary_loss = vmap(left_boundary_loss, (None, 0))
    v_wall_boundary_loss = vmap(wall_boundary_loss, (None, 0))
    v_right_boundary_loss = vmap(right_boundary_loss, (None, 0))

    def boundary_loss(params: Params, points: dict[str, Array]) -> Array:

        left_pts = points['left boundary']
        right_pts = points['right boundary']
        wall_pts = points['wall boundary']
        cylinder_pts = points['cylinder boundary']

        left = np.mean(v_left_boundary_loss(
            params, left_pts) ** 2) * weights[0]
        right = np.mean(v_right_boundary_loss(
            params, right_pts) ** 2) * weights[1]
        wall = np.mean(v_wall_boundary_loss(
            params, wall_pts) ** 2) * weights[2]
        cylinder = np.mean(v_wall_boundary_loss(
            params, cylinder_pts) ** 2) * weights[3]

        return left + right + wall + cylinder

    return boundary_loss


p0 = xpinn.PINNs[0]
p0.boundary_loss = boundary_loss_factory2(
    inflow_func, nu=0.001, weights=(20, 1,  1, 1))
p0.interior_loss = navier_stokes_residual_factory(0, nu=0.001, weight=20)
p0.create_loss()

shape = [2] + 25*[30] + [2]

exponential_decay = optax.exponential_decay(
    init_value=0.001,
    transition_steps=1000,
    transition_begin=2000,
    decay_rate=0.1,
    end_value=0.00001,
)
optimizer = optax.adam(learning_rate=exponential_decay)

xpinn.PINNs[0].init_params(shape, optimizer)
# load_model_iter = 5000
# our_model_path = model_path / "NavierStokes"/ "single_pinn"/ "laminar" / f"ADAM_6000_JUNMAIO_x64_2100"
# xpinn.load_model(our_model_path)

n_iter = 5000
losses = xpinn.run_iters(n_iter)

# ab[ #inetional syntax error. DON'T RUN THIS CELL unless you want to save/overwrite the model
our_model_path = model_path / "NavierStokes" / \
    "single_pinn" / "laminar" / f"ADAM_5000_JUNMAIO_x64_5000"
xpinn.save_model(our_model_path)

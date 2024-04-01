import sys

import jax.numpy as np
import numpy as onp
import optax
from jax import vmap, lax
from tqdm import tqdm

from base_network import neural_network
from poisson.poisson_utils import (
    boundary_loss_factory,
    interior_loss_factory,
    crude_rel_L2,
    interface_loss_factory,
)
from utils import data_path
from xpinn import XPINN
from type_util import Array


def rhs(point: Array) -> Array:
    """The right-hand side of the Poisson equation.

    Args:
        point (Array): A point in the domain.

    Returns:
        Array: The value of the right-hand side at the point.
    """
    x = point[0]
    y = point[1]
    x_cond = (0.25 <= x) & (x <= 0.75)
    y_cond = (0.25 <= y) & (y <= 0.75)
    condition = x_cond & y_cond

    return lax.select(condition, -1.0, 0.0)


if __name__ == "__main__":
    ### Set files
    name = str(sys.argv[1])
    file = data_path / "poisson_train.json"
    file_test = data_path / "poisson_test.json"
    activation = np.tanh

    ### Set model (FFNN)
    xpinn = XPINN(file, activation)
    model = neural_network(activation)
    v_model = vmap(model, (None, 0))

    # Set losses
    p0, p1 = xpinn.PINNs
    p0.boundary_loss = boundary_loss_factory(p0, 0.0, weight=20)
    p0.interior_loss = interior_loss_factory(p0, rhs)  # implementing rhs
    p1.interior_loss = interior_loss_factory(p1, rhs)
    p0.interface_loss = interface_loss_factory(xpinn, 0, 1, weight=20)
    p1.interface_loss = interface_loss_factory(xpinn, 1, 0, weight=20)

    ### Initializing optimizer
    shapes = [[2, 20, 20, 20, 20, 20, 20, 20, 1], [2, 20, 20, 20, 20, 20, 20, 20, 1]]
    exponential_decay = optax.exponential_decay(
        init_value=0.001,
        transition_steps=10000,
        transition_begin=15000,
        decay_rate=0.1,
        end_value=0.00001,
    )
    optimizer = optax.adam(learning_rate=exponential_decay)
    xpinn.initialize_params(shapes, optimizer)

    ### Setting iterations
    n_iter = 200000
    xpinn.set_loss()
    nr_saved_points = 100
    ### prepping relative errors
    ##Getting initial predictions
    points, predictions = xpinn.predict(file_test)
    total_points = np.concatenate(points)
    total_pred = np.concatenate(predictions)
    sorted_indices = np.lexsort((total_points[:, 1], total_points[:, 0]))
    sorted_pred = total_pred[sorted_indices]

    # Getting true value
    file = f"../true_solution.npz"
    with onp.load(file) as true_file:
        true_sol = true_file["arr_0"]

    true_integral = np.sqrt(np.sum(true_sol**2))
    # Setting the L2 relative errors
    l2_errors = onp.zeros(nr_saved_points + 1)
    print("calculating crude_rel_L2")
    l2_errors[0] = crude_rel_L2(sorted_pred, true_sol, true_integral)
    print(l2_errors[0])
    for i in tqdm(range(nr_saved_points)):
        ### Run iterations
        losses = xpinn.run_iters(int(n_iter / nr_saved_points))

        ### Calculate rel l2 error
        _, predictions = xpinn.predict(file_test)
        sorted_pred = np.concatenate(predictions)[sorted_indices]
        l2_errors[i + 1] = crude_rel_L2(sorted_pred, true_sol, true_integral)
    print(l2_errors)
    onp.savez(f"results/L2_errors_{name}", l2_errors)

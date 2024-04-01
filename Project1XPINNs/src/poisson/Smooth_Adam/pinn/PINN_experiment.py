import sys

import jax.numpy as np
import numpy as onp
import optax
from jax import jit, vmap
from tqdm import tqdm

from base_network import neural_network
from poisson.poisson_utils import (
    boundary_loss_factory,
    interior_loss_factory,
    crude_rel_L2,
)
from utils import data_path
from xpinn import XPINN

if __name__ == "__main__":
    ### Set files
    name = str(sys.argv[1])
    file = data_path / "poisson_single_pinn_train.json"
    file_test = data_path / "poisson_single_pinn_test.json"
    activation = np.tanh

    ### Set model (FFNN)
    xpinn = XPINN(file, activation)
    model = neural_network(activation)
    v_model = vmap(model, (None, 0))

    ### Set true solution and RHS
    @jit
    def u_star(x):
        return np.prod(np.sin(np.pi * x))

    # rhs
    @jit
    def f(x):
        return 2.0 * np.pi**2 * u_star(x)

    f_v = f
    f_final = jit(f_v)
    # Set losses
    p0 = xpinn.PINNs[0]
    p0.boundary_loss = boundary_loss_factory(p0, 0)
    p0.interior_loss = interior_loss_factory(p0, f_final)
    p0.create_loss()
    xpinn.set_loss()

    ### Initializing optimizer
    shapes = [[2, 64, 1]]
    for pinn, shape in zip(xpinn.PINNs, shapes):
        exponential_decay = optax.exponential_decay(
            init_value=0.001,
            transition_steps=10000,
            transition_begin=15000,
            decay_rate=0.1,
            end_value=0.0000001,
        )
        optimizer = optax.adam(learning_rate=exponential_decay)

        pinn.init_params(shape, optimizer)

    ### Setting iterations
    n_iter = 200000
    nr_saved_points = 200
    ### prepping relative errors
    ##Getting initial predictions
    points, predictions = xpinn.predict(file_test)
    total_points = np.concatenate(points)
    total_pred = np.concatenate(predictions)

    # Getting true value
    u_vmap = vmap(u_star, (0))
    true_value = u_vmap(total_points).reshape(-1, 1)
    true_integral = np.sqrt(np.sum(true_value**2))

    # Setting the L2 relative errors
    l2_errors = onp.zeros(nr_saved_points + 1)
    l2_errors[0] = crude_rel_L2(total_pred, true_value, true_integral)

    for i in tqdm(range(nr_saved_points)):
        ### Run iterations
        losses = xpinn.run_iters(int(n_iter / nr_saved_points))

        ### Calculate rel l2 error
        _, predictions = xpinn.predict(file_test)
        total_pred = np.concatenate(predictions)
        l2_errors[i + 1] = crude_rel_L2(total_pred, true_value)

    print(l2_errors)
    onp.savez(f"results/L2_errors_{name}", l2_errors)

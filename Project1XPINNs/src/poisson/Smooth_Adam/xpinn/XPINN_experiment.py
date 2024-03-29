from xpinn import XPINN
import optax
from type_util import Array
from jax import hessian, jacobian, jit, vmap
import jax.numpy as np
import numpy as onp
import sys
from tqdm import tqdm
from utils import data_path
from base_network import neural_network
from poisson.poisson_utils import boundary_loss_factory, interior_loss_factory, crude_rel_L2, interface_loss_factory

if __name__ == '__main__':
    ### Set files
    name = str(sys.argv[1])
    file = data_path / "poisson_train.json"
    file_test = data_path / "poisson_test.json"
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
        return 2. * np.pi**2 * u_star(x)

    f_v = f
    f_final = jit(f_v)

    # Set losses
    p0, p1 = xpinn.PINNs
    p0.boundary_loss = boundary_loss_factory(p0,0.0)
    p0.interior_loss = interior_loss_factory(p0, f)  # implementing rhs
    p1.interior_loss = interior_loss_factory(p1, f)
    p0.interface_loss = interface_loss_factory(xpinn,0, 1)
    p1.interface_loss = interface_loss_factory(xpinn,1, 0)


    ### Initializing optimizer
    shapes = [[2] + [64] + [1],[2] + [64] + [64] + [1]]
    exponential_decay = optax.exponential_decay(
            init_value=0.001,
            transition_steps=10000,
            transition_begin=15000,
            decay_rate=0.1,
            end_value=0.0000001,
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

    # Getting true value
    u_vmap = vmap(u_star, (0))
    true_value = u_vmap(total_points).reshape(-1, 1)
    true_integral = np.sqrt(np.sum(true_value**2))

    #Setting the L2 relative errors
    l2_errors = onp.zeros(nr_saved_points + 1)
    l2_errors[0] = crude_rel_L2(total_pred, true_value, true_integral)
    
    for i in tqdm(range(nr_saved_points)):
        ### Run iterations
        losses = xpinn.run_iters(int(n_iter/nr_saved_points))
        
        ### Calculate rel l2 error
        _, predictions = xpinn.predict(file_test)
        total_pred = np.concatenate(predictions)
        l2_errors[i+1] = crude_rel_L2(total_pred, true_value)

    print(l2_errors)
    onp.savez(f"results/L2_errors_{name}", l2_errors)
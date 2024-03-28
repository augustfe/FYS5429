from pathlib import Path
from xpinn import XPINN
import optax
from type_util import Array
from jax import hessian, jacobian, jit, vmap
import jax.numpy as np
import numpy as onp

from tqdm import tqdm
import sys
from utils import data_path, fig_path

from type_util import Params
from base_network import neural_network

from poisson.poisson_utils import interface_loss_factory, interior_loss_factory, boundary_loss_factory


### Read file name to save data
file_name = str(sys.argv[1])

### Read file 
file = data_path / "poisson_train.json"
file_test = data_path / "poisson_test.json"
### Initializing xpinn
activation = np.tanh
xpinn = XPINN(file, activation)
### Initializing feedforward NN model
model = neural_network(activation)
v_model = vmap(model, (None, 0))

### True value for our problem
@jit
def u_star(x):
    return np.prod(np.sin(np.pi * x))

# rhs
@jit
def f(x):
    return 2. * np.pi**2 * u_star(x)

f_v = f
f_final = jit(f_v)

p0, p1 = xpinn.PINNs

p0.boundary_loss = boundary_loss_factory(p0,0.0)

p0.interior_loss = interior_loss_factory(p0, f)  # implementing rhs
p1.interior_loss = interior_loss_factory(p1, f)

p0.interface_loss = interface_loss_factory(xpinn,0, 1)
p1.interface_loss = interface_loss_factory(xpinn,1, 0)


shapes = [[2] + [64] + [1],[2] + [64] + [1]]
exponential_decay = optax.exponential_decay(
        init_value=0.001,
        transition_steps=10000,
        transition_begin=15000,
        decay_rate=0.1,
        end_value=0.0000001,
    )
optimizer = optax.adam(learning_rate=exponential_decay)
xpinn.initialize_params(shapes, optimizer)

### Setting points
n_iter = 200000
nr_saved_points = 200
l2_errors = onp.zeros(nr_saved_points+1)

points, predictions = xpinn.predict(file_test)
total_pred = np.concatenate(predictions)
total_points = np.concatenate(points)

# Setting true value
u_vmap = vmap(u_star, (0))
true_value = np.sin(np.pi * total_points[:, 0]) * np.sin(np.pi * total_points[:, 1])

n = int(np.sqrt(total_points.shape[0]))
X = total_points[:, 0]
Y = total_points[:, 1]
X = X.reshape(n, n)
Y = Y.reshape(n, n)
total_pred=total_pred.reshape(n,n)
true_value= true_value.reshape(n,n)

inner = onp.trapz((total_pred-true_value)**2, total_points[0:n,0], axis=0)
outer = onp.trapz(inner, total_points[::n,1], axis=0)
l2_err = np.sqrt(outer)
normalizer = np.sqrt(onp.trapz(onp.trapz((true_value)**2, total_points[0:n,0], axis=0),total_points[::n,1], axis=0))

xpinn.set_loss()
for i in tqdm(range(nr_saved_points)):
    ### Calculate rel l2 error
    _, predictions = xpinn.predict(file_test)
    total_pred = np.concatenate(predictions)

    total_pred=total_pred.reshape(n,n)
    inner = onp.trapz((total_pred-true_value)**2, total_points[0:n,0], axis=0)
    outer = onp.trapz(inner, total_points[::n,1], axis=0)
    l2_err = np.sqrt(outer)
    l2_errors[i] = l2_err/normalizer

    ### Run iterations
    losses = xpinn.run_iters(int(n_iter/nr_saved_points))

### Calculate rel l2 error
_, predictions = xpinn.predict(file_test)
total_pred = np.concatenate(predictions)
total_pred=total_pred.reshape(n,n)
inner = onp.trapz((total_pred-true_value)**2, total_points[0:n,0], axis=0)
outer = onp.trapz(inner, total_points[::n,1], axis=0)
l2_err = np.sqrt(outer)
l2_errors[-1] = l2_err/normalizer


print(l2_errors)
onp.savez(f"xpinn/L2_errors_{file_name}", l2_errors)
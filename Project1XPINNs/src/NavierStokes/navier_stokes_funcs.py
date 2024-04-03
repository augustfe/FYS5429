from jax import vmap, jit, grad
import jax.numpy as np
from base_network import neural_network

activation = np.tanh
model = neural_network(activation)
v_model = vmap(model, (None, 0))

# streamfunction and pressure


def psi(params, xy): return model(params, xy)[0]
def p(params, xy): return model(params, xy)[1]


d_psi_dxy = grad(psi, argnums=1)


def uv(params, xy):
    """Calculates the flow field from the streamfunction."""
    v_negated, u = d_psi_dxy(params, xy)
    return np.array([u, -v_negated])


jv_uv = jit(vmap(uv, (None, 0)))
jv_p = jit(vmap(p, (None, 0)))
jv_model = jit(v_model)

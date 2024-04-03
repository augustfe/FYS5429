from jax import vmap, jit, grad
import jax.numpy as np
from base_network import neural_network
from type_util import Array
from pathlib import Path
import json


activation = np.tanh
model = neural_network(activation)
v_model = vmap(model, (None, 0))

#streamfunction and pressure
psi = lambda params, xy: model(params, xy)[0]
p = lambda params, xy: model(params, xy)[1]


d_psi_dxy = grad(psi, argnums=1)

def uv(params, xy):
    """Calculates the flow field from the streamfunction."""
    v_negated, u = d_psi_dxy(params, xy)
    return np.array([u, -v_negated])

uv = jit(vmap(uv, (None, 0)))
p = jit(vmap(p, (None, 0)))
j_model = jit(v_model)

def pinn_predict(params, args: dict[str, Array]):
    """Predicts the streamfunction, pressure, flow field and flow magnitude using the trained PINN model.

    Args:
        params: Trained PINN model parameters
        args: Dictionary with boundary and interior points
    """
    b = args["boundary"]
    i = args["interior"]
    if b.size == 0:
        points = i
    else:
        points = np.vstack([b, i])

    net_output = j_model(params, points)
    pressure = net_output[:, 1]
    #pressure = p(params, points)
    streamfunction = net_output[:, 0]
    flow = uv(params, points)
    flow_magnitude = np.sqrt(np.sum(flow**2, axis=1))

    return points, flow, flow_magnitude, pressure, streamfunction


def NSpredict(xpinn, input_file: str | Path = None):
    """" Predicts streamfunction, pressure, flow and flow magnitude with inputed XPINN model params.

    Args:
        xpinn: XPINN model
        input_file: JSON file with input data
    """
    if input_file:
        main_args = {}
        with open(input_file) as infile:
            data = json.load(infile)

        for i, item in enumerate(data["XPINNs"]):
            interior = np.asarray(item["Internal points"])
            boundary = np.asarray(item["Boundary points"])

            main_args[i] = {"boundary": boundary, "interior": interior}

    else:
        main_args = xpinn.main_args

    total_streamfunction = []
    total_points = []
    total_flow = []
    total_p = []
    total_flow_magnitude = []


    for i, pinn in enumerate(xpinn.PINNs):
        params = pinn.params
        points, flow, flow_magnitude, p, streamfunction = pinn_predict(params, main_args[i])
        
        total_flow_magnitude.append(flow_magnitude)
        total_streamfunction.append(streamfunction)
        total_points.append(points)
        total_flow.append(flow)
        total_p.append(p)

    return total_points, total_flow, total_flow_magnitude, total_p, total_streamfunction



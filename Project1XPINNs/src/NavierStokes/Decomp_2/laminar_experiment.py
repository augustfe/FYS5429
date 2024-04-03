
from xpinn import XPINN
import jax.numpy as np
from utils import data_path
from typing import Tuple
import sys
import matplotlib.pyplot as plt
from typing import Callable
from type_util import Params, Array
from base_network import neural_network
from jax import hessian, jit, vmap, grad, jacobian
import jax

LFunc = Callable[[Params, dict[str, Array]], Array]

### Reading files

if len(sys.argv) != 5:  # The first argument is the script name itself
    print("Usage: python script.py <string> <float1> <float2> <float3> <float4>")
    sys.exit(1)

# First argument is the script name itself, so the string starts from index 1
file_train = sys.argv[1]

# Extracting floats from command-line arguments
try:
    w_boundary = float(sys.argv[2])
    w_interface = float(sys.argv[3])
    w_residual = float(sys.argv[4])
except ValueError:
    print("Error: Invalid float value provided.")
    sys.exit(1)

# Print the input string and list of floats
print("Input String:", file_train)
print("List of Floats:", [w_boundary, w_interface, w_residual])

### Activating
activation = np.tanh
xpinn = XPINN(file_train, activation)

model = neural_network(activation)
v_model = vmap(model, (None, 0))

# #### Advection term

psi = lambda params, xy: model(params, xy)[0]


hess_psi = hessian(psi, argnums=1)
d_psi_dxy = grad(psi, argnums=1)
v_d_psi_dxy = vmap(d_psi_dxy, (None, 0))

def advection_term(params: Params, xy: Array) -> Array:
    hess = hess_psi(params, xy)
    processed_hess = np.array([[-hess[1, 1], hess[0, 1]], [hess[1, 0], -hess[0, 0]]])
    return processed_hess @ d_psi_dxy(params, xy)


# #### Diffusion term

jacobi_hess_psi = jacobian(hess_psi, argnums=1)

def diffusion_term(params: Params, xy: Array):
    jachessi = jacobi_hess_psi(params, xy)
    u_diffusion = jachessi[0,1,0] + jachessi[1,1,1] #psi_yxx + psi_yyy
    v_diffusion = jachessi[1,0,1] + jachessi[0,0,0] #psi_xyy + psi_xxx
    return np.array([u_diffusion, -v_diffusion])

# #### Pressure

p = lambda params, xy: model(params, xy)[1]

d_p = grad(p, argnums=1)

# #### Navier stokes residual

def navier_stokes_residual_factory(index: int, nu:float, weight : int = 1) -> LFunc:

    def residual(params, xy):
        return advection_term(params, xy) - nu* diffusion_term(params, xy) + d_p(params, xy)

    v_residual = jit(vmap(residual, (None, 0)))
    xpinn.PINNs[index].v_residual = v_residual
    
    def interior_loss(params: Params, points: dict[str, Array]) -> Array:
        pts = points["interior"]
        return weight * 2 * np.mean(v_residual(params, pts) ** 2)
    
    return interior_loss

# #### Inflow

U = 0.3
inflow_func = lambda xy: np.array((4 * U * xy[1] * (0.41 - xy[1])/(0.41**2), 0.0))

def uv(params: Params, xy: Array) -> Array:
    d_psi = d_psi_dxy(params, xy)
    u = d_psi[1]
    v = -d_psi[0]
    return np.array([u, v])

# #### Boundary Losses

def boundary_loss_factory_1(inflow_func: Callable[[Array], Array], nu:float, weights:Tuple[int, int, int]= (1,1,1) ) -> LFunc:
    
    def left_boundary_loss(params, xy):
        return np.sum(np.square(uv(params, xy) - inflow_func(xy))) #(u - inflow)**2 + v**2
        
    def wall_boundary_loss(params, xy):
        return np.sum(np.square(uv(params, xy))) #return u**2 + v**2
    
    v_wall_boundary_loss = vmap(wall_boundary_loss, (None, 0))
    v_left_boundary_loss = vmap(left_boundary_loss, (None, 0))
    
    def boundary_loss(params: Params, points: dict[str, Array]) -> Array:

        left_pts = points['left boundary']
        wall_pts = points['wall boundary']
        cylinder_pts = points['cylinder boundary']
    
        left = np.mean(v_left_boundary_loss(params, left_pts)) * weights[0]
        wall = np.mean(v_wall_boundary_loss(params, wall_pts)) * weights[1]
        cylinder = np.mean(v_wall_boundary_loss(params, cylinder_pts))* weights[2]
    
        return left + wall + cylinder
    
    return boundary_loss

def boundary_loss_factory_2(nu:float, weights:Tuple[int, int]= (1,1) ) -> LFunc:
        
    def right_boundary_loss(params, xy):
        u_ = hess_psi(params, xy)[:,1]
        u_x = u_[0]
        u_y = u_[1]
        return (nu * u_x - p(params, xy)) ** 2 + (nu * u_y) ** 2
        
    def wall_boundary_loss(params, xy):
        return np.sum(np.square(uv(params, xy))) #return u**2 + v**2
    
    v_wall_boundary_loss = vmap(wall_boundary_loss, (None, 0))
    v_right_boundary_loss = vmap(right_boundary_loss, (None, 0))
    
    def boundary_loss(params: Params, points: dict[str, Array]) -> Array:

        right_pts = points['right boundary']
        wall_pts = points['wall boundary']
    
        right = np.mean(v_right_boundary_loss(params, right_pts)) * weights[0]
        wall = np.mean(v_wall_boundary_loss(params, wall_pts)) * weights[1]
    
        return right + wall 
    
    return boundary_loss

# #### Interface loss

def interface_loss_factory(i: int, j: int, weight = 1) -> LFunc:
    a, b = sorted([i, j])
    pi = xpinn.PINNs[i]

    def interface_loss(params: Params, args: dict[str, Array]) -> Array:
        inter_points = args[f"interface {a}{b}"]
        res_j = args[f"interface res {j}"] ### 2D as well
        res_ij = 2 * np.mean((pi.v_residual(params, inter_points) - res_j) ** 2) # Multiply by 2 for interface
        # res_ij = 0

        val_j = args[f"interface val {j}"]

        avg_ij = 2 * np.mean(((v_model(params, inter_points) - val_j) / 2) ** 2) #Multiply by 2 because meaning over 2 dimensions
        # avg_ij = 0

        ### Neumann interface
        params_j = args[f"params {j}"]
        d_psi_j = v_d_psi_dxy(params_j, inter_points)
        d_psi_i = v_d_psi_dxy(params, inter_points)

        neumann =  2* np.mean(np.square((d_psi_j - d_psi_i)))


        return weight * (res_ij + avg_ij + neumann)

    return interface_loss

p0 = xpinn.PINNs[0]
p0.boundary_loss = boundary_loss_factory_1(inflow_func, 0.001, (w_boundary, w_boundary, w_boundary))
p0.interior_loss = navier_stokes_residual_factory(0, 0.001, w_residual)


p1 = xpinn.PINNs[1]
p1.interior_loss = navier_stokes_residual_factory(1, 0.001, w_residual)
p1.boundary_loss = boundary_loss_factory_2(0.001, (w_boundary, w_boundary))

p0.interface_loss = interface_loss_factory(0, 1, weight = w_interface)
p1.interface_loss = interface_loss_factory(1, 0, weight = w_interface)

p0.create_loss()
p1.create_loss()

import optax

shapes = [[2] + [30] * 25 + [2],[2] + 10 * [20] + [2]]

exponential_decay = optax.exponential_decay(
    init_value=0.001,
    transition_steps=3000,
    transition_begin=3000,
    decay_rate=0.1,
    end_value=0.000001
)

optimizer = optax.adam(learning_rate=exponential_decay)

xpinn.initialize_params(shapes, optimizer)

print("Starting iterations")
n_iter = 10000
losses = xpinn.run_iters(n_iter)

from utils import model_path\

our_model_path = model_path / "NavierStokes"/ "Decomp_2"/ f"{file_train[:-5]}_{w_boundary}_{w_interface}_{w_residual}_iterations={n_iter}_Neumann_finaltest"
print(f"Writing to {our_model_path}")
xpinn.save_model(our_model_path)

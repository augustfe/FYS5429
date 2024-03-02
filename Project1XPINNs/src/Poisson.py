"""
Adam optimization. 
Two dimensional homogenous Poisson equation for a single PINN on the unit square domain. 

-u_xx - u_yy = 0
u(0,y) = u(1,y) = u(x,0) = 0
u(x,1) = sin(pi*x)

The equation has analytical solution (MAT3360 SIUUU)
u(x,y) = sin(pi*x)*sinh(k*pi*y)/sinh(pi)
"""
import jax
import jax.numpy as jnp
from jax import random, jit, vmap,grad
import optax
import numpy as np


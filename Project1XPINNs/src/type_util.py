from typing import Callable
from jax._src.typing import ArrayLike, Array, Shape
from optax import Params

Activator = Callable[[ArrayLike], Array]

from typing import Callable, Union
from jax._src.typing import (  # noqa
    ArrayLike,
    Array,
    Shape,
)
from optax import (  # noqa
    Params,
    GradientTransformation,
    OptState,
    Updates,
)

Activator = Callable[[ArrayLike], Array]
Scalar = Union[int, float]

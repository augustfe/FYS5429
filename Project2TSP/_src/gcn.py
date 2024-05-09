from typing import Sequence, Callable
import flax.linen as nn
import jraph
from flax.typing import Array, PRNGKey
from flax.training import train_state


class TrainState(train_state.TrainState):
    key: PRNGKey


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    activation: Callable[[Array], Array] = nn.leaky_relu

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=nn.initializers.glorot_normal(),
            )(x)
            x = self.activation(x)

        return x


class GCN(nn.Module):
    hidden_size: int
    number_classes: int
    activation: Callable[[Array], Array] = nn.leaky_relu
    dropout_rate: float = 0.1
    num_layers: int = 2

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, training: bool) -> jraph.GraphsTuple:
        feature_sizes = [self.hidden_size] * self.num_layers
        gn = jraph.GraphConvolution(
            MLP(feature_sizes, activation=self.activation),
            symmetric_normalization=True,
        )
        graph = gn(graph)

        dropout = nn.Dropout(rate=self.dropout_rate, deterministic=not training)
        graph = graph._replace(nodes=dropout(graph.nodes))

        gn = jraph.GraphConvolution(
            nn.Dense(self.number_classes, kernel_init=nn.initializers.glorot_normal()),
            symmetric_normalization=True,
        )
        graph = gn(graph)
        graph = graph._replace(nodes=nn.softmax(graph.nodes))

        return graph

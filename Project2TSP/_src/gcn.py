from typing import Sequence, Callable
import flax.linen as nn
import jraph
from flax.typing import Array, PRNGKey
from flax.training import train_state


class TrainState(train_state.TrainState):
    """TrainState with added PRNGKEY.

    Args:
        step (int): Counter starts at 0 and is incremented by every call to
        ``.apply_gradients()``.
        apply_fn (Callable): Usually set to ``model.apply()``. Kept in this dataclass for
        convenience to have a shorter params list for the ``train_step()`` function
        in your training loop.
        params (FrozenDict): The parameters to be updated by ``tx`` and used by ``apply_fn``.
        tx (optax.GradientTransformation): An Optax gradient transformation.
        opt_state (OptState): The state for ``tx``.
        key (PRNGKey): The key for the random number generator.
    """

    key: PRNGKey


class MLP(nn.Module):
    """A simple multi-layer perceptron.

    Args:
        feature_sizes (Sequence[int]): The sizes of the hidden layers.
        activation (Callable): The activation function to use. Note that there is no
        added activation for the outputlayer. Default to leaky_relu.
    """

    feature_sizes: Sequence[int]
    activation: Callable[[Array], Array] = nn.leaky_relu

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Apply the MLP to the inputs.

        Args:
            inputs (Array): Input array for the MLP.

        Returns:
            Array: Result of applying the MLP to the inputs.
        """
        outputs = inputs

        for size in self.feature_sizes:
            outputs = nn.Dense(
                features=size,
                kernel_init=nn.initializers.he_normal(),
                # kernel_init=nn.initializers.glorot_normal(),
            )(inputs)
            inputs = self.activation(outputs)

        return outputs


class GCN(nn.Module):
    """Graph Convolutional Network.

    Args:
        hidden_size (int): The size of the hidden layers of the MLPs.
        number_classes (int): The number of classes in the output.
        activation (Callable): The activation function to use in the hidden layers.
        Default to leaky_relu.
        dropout_rate (float): The dropout rate between the intial convolutions
        and the output layer. Default to 0.5.
        num_layers (int): The number of layers in each MLP in the initial convolutions.
        Default to 2.
        output_activation (Callable): The activation function for the output layer.
        Default to softmax.
        num_convolutions (int): The number of graph convolutions to apply before decoding.
        Default to 1.
        add_self_edges (bool): Whether to add self edges to the graph. Note that the initial
        article adds self edges. Default to False.
    """

    hidden_size: int
    number_classes: int
    activation: Callable[[Array], Array] = nn.leaky_relu
    dropout_rate: float = 0.5
    num_layers: int = 2
    output_activation: Callable[[Array], Array] = nn.softmax
    num_convolutions: int = 1
    add_self_edges: bool = False

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, training: bool) -> jraph.GraphsTuple:
        """Apply the GCN to the graph.

        Args:
            graph (jraph.GraphsTuple): The input graph.
            training (bool): Whether the model is in training mode.

        Returns:
            jraph.GraphsTuple: The output graph.
        """
        feature_sizes = [self.hidden_size] * self.num_layers

        for _ in range(self.num_convolutions):
            gn = jraph.GraphConvolution(
                MLP(feature_sizes, activation=self.activation),
                symmetric_normalization=True,
                add_self_edges=self.add_self_edges,
            )
            graph = gn(graph)

            # GraphConvolution does not add an activation after communication
            graph = graph._replace(nodes=self.activation(graph.nodes))

        dropout = nn.Dropout(rate=self.dropout_rate, deterministic=not training)
        graph = graph._replace(nodes=dropout(graph.nodes))

        gn = jraph.GraphConvolution(
            nn.Dense(self.number_classes, kernel_init=nn.initializers.he_normal()),
            symmetric_normalization=True,
            add_self_edges=self.add_self_edges,
        )
        graph = gn(graph)
        graph = graph._replace(nodes=self.output_activation(graph.nodes))

        return graph

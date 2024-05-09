import jraph
from jax import numpy as np, random, jit, value_and_grad  # , custom_jvp, lax
import optax
import flax.linen as nn
from flax.typing import Array, Optional, FrozenDict, Any
from flax.training.train_state import TrainState
import networkx as nx
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from matrix_helper import calculate_distances
from typing import Sequence

# import jax
# from jax._src.numpy import util as numpy_util


# @jax.jit
# def leaky_zero_one(x: Array, negative_slope: Array = 1e-2) -> Array:
#     r"""Leaky rectified linear unit activation function.

#   Computes the element-wise function:

#   .. math::
#     \mathrm{leaky\_relu}(x) = \begin{cases}
#       x, & x \ge 0\\
#       \alpha x, & x < 0
#     \end{cases}

#   where :math:`\alpha` = :code:`negative_slope`.

#   Args:
#     x : input array
#     negative_slope : array or scalar specifying the negative slope (default: 0.01)

#   Returns:
#     An array.

#   See also:
#     :func:`relu`
#   """
#     numpy_util.check_arraylike("leaky_zero_one", x)
#     x_arr = np.asarray(x)
#     return np.where((x_arr >= 0) & (x_arr <= 1), x_arr, negative_slope * x_arr)


# @custom_jvp
# @jax.jit
# def zero_one(x: Array) -> Array:
#     r"""Rectified linear unit activation function.

#     Computes the element-wise function:

#     .. math::
#       \mathrm{relu}(x) = \max(x, 0)

#     except under differentiation, we take:

#     .. math::
#       \nabla \mathrm{relu}(0) = 0

#     For more information see
#     `Numerical influence of ReLU’(0) on backpropagation
#     <https://openreview.net/forum?id=urrcVI-_jRm>`_.

#     Args:
#       x : input array

#     Returns:
#       An array.

#     Example:
#       >>> jax.nn.relu(jax.numpy.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
#       Array([0. , 0. , 0. , 0. , 0.5, 1. , 2. ], dtype=float32)

#     See also:
#       :func:`relu6`

#     """
#     return np.minimum(np.maximum(x, 0), 1)


# # For behavior at 0, see https://openreview.net/forum?id=urrcVI-_jRm
# zero_one.defjvps(
#     lambda g, ans, x: lax.select((1 > x) & (x > 0), g, lax.full_like(g, 0))
# )


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.leaky_relu(x)
            # x = leaky_zero_one(x)
            # x = zero_one(x)

            # x = nn.gelu(x)
        return x


class GCN_dev(nn.Module):
    input_size: int
    embedding_size: int
    hidden_size: int
    number_classes: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        embedding_fn = nn.Embed(self.input_size, self.embedding_size)

        graph = jraph.GraphMapFeatures(embed_node_fn=embedding_fn)(graph)

        # gn = jraph.GraphConvolution(
        #     nn.Dense(self.hidden_size),
        #     add_self_edges=True,
        #     symmetric_normalization=True,
        # )
        # graph = gn(graph)

        # graph = graph._replace(nodes=nn.relu(graph.nodes))
        feature_sizes = [self.hidden_size] * 2
        gn = jraph.GraphConvolution(
            MLP(feature_sizes),
            symmetric_normalization=True,
            add_self_edges=True,
        )
        graph = gn(graph)

        gn = jraph.GraphConvolution(
            nn.Dense(self.number_classes),
            symmetric_normalization=True,
            add_self_edges=True,
        )
        graph = gn(graph)

        # graph = graph._replace(nodes=nn.sigmoid(graph.nodes))
        nodes = graph.nodes
        nodes = nn.sigmoid(nodes)

        # Set the first node as the start and end node
        nodes = nodes.at[0, :].set(0.0)
        nodes = nodes.at[:, 0].set(0.0)
        nodes = nodes.at[0, 0].set(1.0)
        graph = graph._replace(nodes=nodes)

        return graph


def generate_graph(
    n: int,
    degree: Optional[int] = None,
    prob: Optional[float] = None,
    graph_type: str = "reg",
    random_seed: int = 1,
) -> nx.Graph:
    """Generate a random NetworkX graph with n nodes.

    Args:
        n (int): The number of nodes in the graph.
        degree (int, optional): The degree of the nodes in the graph. Defaults to None.
        prob (float, optional): The probability of the edges in the graph. Defaults to None.
        graph_type (str, optional): The type of the graph. Defaults to "reg".
        random_seed (int, optional): The random seed. Defaults to 0.

    Returns:
        nx.Graph: The generated graph.
    """
    pos = None
    if graph_type == "reg":
        # Generates a d-regular graph
        nx_graph = nx.random_regular_graph(degree, n, seed=random_seed)
    elif graph_type == "erdos":
        # Generates an Erdos-Renyi (or binomial) graph, with probability prob for each edge
        nx_graph = nx.erdos_renyi_graph(n, prob, seed=random_seed)
    elif graph_type == "prob":
        # Generates a graph with probability prob for each edge
        nx_graph = nx.fast_gnp_random_graph(n, prob, seed=random_seed)
    elif graph_type == "grid":
        n = int(np.sqrt(n))

        nx_graph = nx.grid_2d_graph(n, n)
        new_graph = nx.Graph()
        pos = {}

        for i, j in nx_graph.nodes:
            new_graph.add_node(i * n + j)
            pos[i * n + j] = np.asarray([i, j]) / n
        for (i, j), (k, l) in nx_graph.edges:
            new_graph.add_edge(i * n + j, k * n + l)
        nx_graph = new_graph
    elif graph_type == "chess":
        n = int(np.sqrt(n))

        nx_graph = nx.grid_2d_graph(n, n)
        new_graph = nx.Graph()
        pos = {}
        for i, j in nx_graph.nodes:
            if i in [0, n - 1] or j in [0, n - 1]:
                continue
            nx_graph.add_edge((i, j), (i + 1, j + 1))
            nx_graph.add_edge((i, j), (i - 1, j - 1))
            nx_graph.add_edge((i, j), (i + 1, j - 1))
            nx_graph.add_edge((i, j), (i - 1, j + 1))

        for i, j in nx_graph.nodes:
            new_graph.add_node(i * n + j)
            pos[i * n + j] = np.asarray([i, j]) / n
        for (i, j), (k, l) in nx_graph.edges:
            new_graph.add_edge(i * n + j, k * n + l)
        nx_graph = new_graph
    else:
        raise ValueError(
            f"Invalid graph type {graph_type}. Please choose 'reg', 'erdos', 'prob', 'grid', or 'chess'."
        )

    if pos is None:
        pos = nx.kamada_kawai_layout(nx_graph)

    return nx_graph, pos


def graph_to_jraph(nx_graph: nx.Graph, pos_arr: Array) -> jraph.GraphsTuple:
    """Convert a NetworkX graph to a jraph.GraphsTuple.

    Args:
        nx_graph (nx.Graph): The NetworkX graph to convert.

    Returns:
        jraph.GraphsTuple: The converted jraph.GraphsTuple.
    """
    # nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    n = nx_graph.number_of_nodes()
    e = nx_graph.number_of_edges()

    print(f"Number of edges: {e}")

    nodes = np.arange(n)
    senders, receivers = np.asarray(nx_graph.edges).T
    senders, receivers = np.concatenate([senders, receivers]), np.concatenate(
        [receivers, senders]
    )

    VALID = np.zeros((n, n)).at[senders, receivers].set(1)
    A = calculate_distances(pos_arr) * VALID

    print(A)

    # A = np.asarray(nx.adjacency_matrix(nx_graph, nodelist=range(n)).toarray())

    # Make sure the graph is undirected
    assert np.all(A == A.T)

    graph = jraph.GraphsTuple(
        n_node=np.array([n]),
        n_edge=np.array([e * 2]),
        nodes=nodes,
        edges=None,
        globals=A,
        senders=senders,
        receivers=receivers,
    )
    return graph


def TSP_loss_func(probs: Array, A: Array) -> Array:
    """Compute the vertex cover loss function.

    Args:
        probs (Array): The probabilities of the nodes.
        A (Array): The adjacency matrix of the graph.

    Returns:
        Array: The loss value.
    """
    H_1 = np.sum((1 - np.sum(probs, axis=1)) ** 2)
    H_2 = np.sum((1 - np.sum(probs, axis=0)) ** 2)

    FROM = probs
    TO = np.roll(probs.T, -1, axis=0)
    X = FROM @ TO
    # X = X + X.T
    H_3 = np.sum(np.where(A == 0, X, 0))

    H_A = 2 * np.max(A) * (H_1 + H_2 + H_3)
    H_A = H_1 + H_2 + H_3

    # X = X + X.T

    # H_4 = np.sum((1 - np.sum(X, axis=1)) ** 2)
    # H_5 = np.sum((1 - np.sum(X, axis=0)) ** 2)

    # H_C = H_4 + H_5

    print(H_1, H_2, H_3, H_A)
    # print(H_4, H_5, H_C)

    H_B = np.sum(A * X)
    # H_B = 0

    print(H_B)
    print(X)

    return H_A + H_B


@jit
def train_step(state: TrainState, graph: jraph.GraphsTuple) -> TrainState:
    """Perform a single training step.

    Args:
        state (TrainState): The training state.
        graph (jraph.GraphsTuple): The graph to train on.

    Returns:
        TrainState: The updated training state.
    """

    def loss_fn(params: FrozenDict[str, Any], graph: jraph.GraphsTuple) -> Array:
        predictions = state.apply_fn(params, graph)
        loss = TSP_loss_func(predictions.nodes, graph.globals)
        return loss, predictions.nodes

    # loss = loss_fn(state.params, graph)

    (loss, probs), grads = value_and_grad(loss_fn, has_aux=True)(state.params, graph)
    state = state.apply_gradients(grads=grads)

    return state, loss, probs


def train(
    nx_graph: nx.Graph,
    net: nn.Module,
    optimizer: optax.GradientTransformation,
    pos_arr: Array,
    num_epochs: int = 100,
    random_seed: int = 0,
    tol: float = 0.01,
    patience: int = 100000,
    threshold: float = 0.5,
) -> tuple[TrainState, Array]:
    """Train a neural network on a graph.

    Args:
        nx_graph (nx.Graph): The graph to train on.
        net (nn.Module): The neural network model.
        optimizer (optax.GradientTransformation): The optimizer.
        num_epochs (int): The number of epochs to train for.
        random_seed (int): The random seed.

    Returns:
        TrainState: The final training state.
    """
    graph = graph_to_jraph(nx_graph, pos_arr)
    rng = random.PRNGKey(random_seed)
    rng, init_rng = random.split(rng)

    params = net.init(init_rng, graph)
    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=optimizer,
    )

    prev_loss = 1.0
    count = 0

    n = graph.n_node[0]
    best_bitstring = np.zeros((n, n))
    bitstring_loss = jit(lambda x: TSP_loss_func(x, graph.globals))
    best_loss = bitstring_loss(best_bitstring)
    # best_loss = TSP_loss_func(best_bitstring, graph.globals)

    gnn_start = time()

    # print every 10% of the epochs

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    def loop_inner(state, prev_loss, count, best_loss, best_bitstring):
        state, loss, bitstring_ = train_step(state, graph)

        bitstring = bitstring_ > threshold

        # loss_ = TSP_loss(bitstring, graph.globals)
        # loss_ = bitstring_loss(bitstring)

        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring_

        if prev_loss - loss <= tol:
            count += 1
        else:
            count = 0

        return state, loss, count, best_loss, best_bitstring

    try:
        vals = (state, prev_loss, count, best_loss, best_bitstring)
        for i in pbar:
            vals = loop_inner(*vals)
            pbar.set_postfix(
                {"loss": f"{vals[1]:.4f}", "count": f"{vals[2] / patience:.2f}"}
            )
            # if vals[2] >= patience:
            #     print(f"Converged after {i} epochs.")
            #     break
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        state, loss, count, best_loss, best_bitstring = vals

    """
    for i in pbar:
        state, loss, bitstring_ = train_step(state, graph)

        pbar.set_postfix({"loss": f"{loss:.4f}", "count": f"{count / patience:.2f}"})

        bitstring = bitstring_ > threshold

        if i % (num_epochs // 10) == 0:
            print(
                f"Epoch {i}: Loss: {loss} VC: {np.sum(bitstring)} ({np.sum(bitstring_)})"
            )
            print(bitstring_)
            # print(bitstring_.flatten())

        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring
            best_bitstring_ = bitstring_

        if abs(prev_loss - loss) <= tol:
            count += 1
        else:
            count = 0

        # if count >= patience:
        #     print(f"Converged after {i} epochs.")
        #     break

        prev_loss = loss
    """

    print(f"Training time: {time() - gnn_start}")
    print(f"Final loss: {loss}")
    print(f"Best loss:  {best_loss}")

    best_bitstring_ = best_bitstring

    return state, best_bitstring, best_bitstring_


def evaluate_TSP(nx_graph: nx.Graph, bitstring: Array, pos, draw: bool = True) -> None:
    """Evaluate the vertex cover of a graph.

    Args:
        nx_graph (nx.Graph): The graph to evaluate.
        bitstring (Array): The bitstring representing the vertex cover.
    """
    n = nx_graph.number_of_nodes()
    plt.scatter(pos[:, 0], pos[:, 1])
    for i in range(n):
        for j in range(n):
            if not nx_graph.has_edge(i, j):
                continue
            color = "red" if bitstring[i, j] else "blue"
            a = pos[i]
            b = pos[j]

            plt.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                # alpha=float(bitstring[i, j]),
            )
    plt.show()

    graph = graph_to_jraph(nx_graph, pos)

    TSP_loss_func(bitstring, graph.globals)


def draw_TSP(nx_graph, pos, bitstring):

    # bitstring = bitstring > 0.5

    FROM = bitstring
    TO = np.roll(bitstring.T, -1, axis=0)

    adj = FROM @ TO
    # adj = adj > 0.5

    n = nx_graph.number_of_nodes()
    plt.scatter(pos[:, 0], pos[:, 1])
    for i, j in nx_graph.edges:
        a = pos[i]
        b = pos[j]

        plt.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="red",
            alpha=float(adj[i, j]),
        )
        plt.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="red",
            alpha=float(adj[j, i]),
        )
    for i in range(n):
        for j in range(n):
            if nx_graph.has_edge(i, j):
                continue
            # if not nx_graph.has_edge(i, j):
            #     continue
            color, style = ("red", "-") if nx_graph.has_edge(i, j) else ("blue", "--")
            a = pos[i]
            b = pos[j]

            plt.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                alpha=float(adj[i, j]),
                linestyle=style,
            )
    plt.show()


if __name__ == "__main__":
    n = 8 * 8

    # n = 100
    d = 3
    prob = 0.03
    nx_graph, pos = generate_graph(n, degree=d, graph_type="reg")
    nx_graph, pos = generate_graph(n, graph_type="grid")
    nx_graph, pos = generate_graph(n, graph_type="chess")
    # nx_graph, pos = generate_graph(25, graph_type="reg", degree=24)
    # nx_graph = nx.grid_2d_graph(8, 8)

    # nx.draw_spring(nx_graph)
    # nx.draw_networkx(nx_graph, pos=pos, with_labels=True)
    # plt.show()
    # quit()
    # nx_graph = generate_graph(n, prob=prob, graph_type="erdos")
    n = nx_graph.number_of_nodes()

    n_epochs = 300000
    lr = 1e-3
    optimizer = optax.adam(lr)

    embedding_size = int(np.sqrt(n))
    hidden_size = max(int(embedding_size / 2), 2)

    pos_arr = np.asarray([pos[i] for i in range(n)])
    jgraph = graph_to_jraph(nx_graph, pos_arr)

    A = jgraph.globals
    print(A)

    X = np.ones((n, n))
    print(np.where(A == 0, X, 0))

    net = GCN_dev(n, embedding_size, hidden_size, n)

    state, best_bitstring, best_bitstring_ = train(
        nx_graph, net, optimizer, pos_arr, n_epochs, tol=1e-4, patience=1e4
    )

    TSP_loss_func(best_bitstring, A)

    # evaluate_TSP(nx_graph, best_bitstring, pos_arr, draw=True)

    adj = best_bitstring @ np.roll(best_bitstring.T, -1, axis=0)
    adj = adj + adj.T

    with open("best_bitstring.csv", "w") as f:
        n_square = int(np.sqrt(n))
        f.write(
            ",".join([f"({i} {j})" for i in range(n_square) for j in range(n_square)])
            + "\n"
        )
        for i in range(n):
            for j in range(n):
                f.write(f"{adj[i, j]},")
            f.write("\n")

    # print((best_bitstring_ > 0.5) * 1)
    best_bitstring_ = (best_bitstring_ > 0.5) * 1
    TSP_loss_func(best_bitstring_, A)
    draw_TSP(nx_graph, pos_arr, best_bitstring_)

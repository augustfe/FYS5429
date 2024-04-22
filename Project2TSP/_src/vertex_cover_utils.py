import jraph
from jax import numpy as np, random, jit, value_and_grad
import optax
import flax.linen as nn
from flax.typing import Array, Optional, FrozenDict, Any
from flax.training.train_state import TrainState
import networkx as nx
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm


class GCN_dev(nn.Module):
    input_size: int
    embedding_size: int
    hidden_size: int
    number_classes: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        embedding_fn = nn.Embed(self.input_size, self.embedding_size)

        graph = jraph.GraphMapFeatures(embed_node_fn=embedding_fn)(graph)
        gn = jraph.GraphConvolution(nn.Dense(self.hidden_size, use_bias=False))
        graph = gn(graph)

        graph = graph._replace(nodes=nn.leaky_relu(graph.nodes))

        gn = jraph.GraphConvolution(
            nn.Dense(self.number_classes), symmetric_normalization=False
        )
        graph = gn(graph)

        graph = graph._replace(nodes=nn.sigmoid(graph.nodes))

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
    if graph_type == "reg":
        # Generates a d-regular graph
        nx_graph = nx.random_regular_graph(degree, n, seed=random_seed)
    elif graph_type == "erdos":
        # Generates an Erdos-Renyi (or binomial) graph, with probability prob for each edge
        nx_graph = nx.erdos_renyi_graph(n, prob, seed=random_seed)
    elif graph_type == "prob":
        # Generates a graph with probability prob for each edge
        nx_graph = nx.fast_gnp_random_graph(n, prob, seed=random_seed)
    else:
        raise ValueError(
            f"Invalid graph type {graph_type}. Please choose 'reg', 'erdos', or 'prob'."
        )

    return nx_graph


def graph_to_jraph(nx_graph: nx.Graph) -> jraph.GraphsTuple:
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

    # print(len(senders))

    # A = np.asarray(nx.adjacency_matrix(nx_graph, nodelist=range(n)).toarray())
    A = np.zeros((n, n)).at[senders, receivers].set(1)
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


def vertex_loss_func(probs: Array, A: Array) -> Array:
    """Compute the vertex cover loss function.

    Args:
        probs (Array): The probabilities of the nodes.
        A (Array): The adjacency matrix of the graph.

    Returns:
        Array: The loss value.
    """
    neg_probs = 1 - probs
    Q = neg_probs @ neg_probs.T
    # Penalty for not covering an edge
    H_A = np.sum(A * Q)
    # Penalty for covering a node
    H_B = np.sum(probs)

    return 2 * H_A + H_B


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
        loss = vertex_loss_func(predictions.nodes, graph.globals)
        return loss, predictions.nodes

    loss = loss_fn(state.params, graph)

    (loss, probs), grads = value_and_grad(loss_fn, has_aux=True)(state.params, graph)
    state = state.apply_gradients(grads=grads)

    return state, loss, probs


def train(
    nx_graph: nx.Graph,
    net: nn.Module,
    optimizer: optax.GradientTransformation,
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
    graph = graph_to_jraph(nx_graph)
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

    best_bitstring = np.zeros(graph.n_node[0])
    best_loss = vertex_loss_func(best_bitstring, graph.globals)

    gnn_start = time()

    # print every 10% of the epochs

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for i in pbar:
        state, loss, bitstring_ = train_step(state, graph)

        pbar.set_postfix({"loss": loss, "count": count})

        # bitstring_ = state.apply_fn(state.params, graph).nodes
        bitstring = bitstring_ > threshold

        if i % (num_epochs // 10) == 0:
            print(
                f"Epoch {i}: Loss: {loss} VC: {np.sum(bitstring)} ({np.sum(bitstring_)})"
            )
            # print(bitstring_.flatten())

        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if abs(prev_loss - loss) <= tol:
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f"Converged after {i} epochs.")
            break

        prev_loss = loss

    print(f"Training time: {time() - gnn_start}")
    print(f"Final loss: {loss}")
    print(f"Best loss:  {best_loss}")

    return state, best_bitstring


def evaluate_vertex_cover(
    nx_graph: nx.Graph, bitstring: Array, draw: bool = True
) -> None:
    """Evaluate the vertex cover of a graph.

    Args:
        nx_graph (nx.Graph): The graph to evaluate.
        bitstring (Array): The bitstring representing the vertex cover.
    """
    color_map = ["lightblue" if bitstring[i] else "orange" for i in nx_graph.nodes]

    size_vc = np.sum(bitstring)
    violations = 0

    for u, v in nx_graph.edges:
        if not bitstring[u] and not bitstring[v]:
            violations += 1

    print(f"Vertex cover size: {size_vc}")
    print(f"Number of violations: {violations}")
    print(f"Bitstring: {bitstring.astype(int).T}")

    redundant = 0

    # Count number of covered nodes only connected to covered nodes
    for u in nx_graph.nodes:
        if bitstring[u]:
            for v in nx_graph.neighbors(u):
                if not bitstring[v]:
                    break
            else:
                redundant += 1

    print(f"Number of redundant nodes: {redundant}")

    from networkx.algorithms.approximation import min_weighted_vertex_cover

    vc = min_weighted_vertex_cover(nx_graph)
    print(len(vc))
    print(vc)

    redunant = 0

    for u in vc:
        for v in nx_graph.neighbors(u):
            if v not in vc:
                break
        else:
            redunant += 1

    print(f"Number of redundant nodes from networkx: {redunant}")

    # pos = nx.kamada_kawai_layout(nx_graph)
    # nx.draw(nx_graph, pos, node_color=color_map, with_labels=True)
    if draw:
        nx.draw_spring(nx_graph, node_color=color_map)
        plt.show()


if __name__ == "__main__":
    n = 400
    d = 2
    prob = 0.03
    nx_graph = generate_graph(n, degree=d, graph_type="reg")
    # nx_graph = generate_graph(n, prob=prob, graph_type="erdos")
    n = nx_graph.number_of_nodes()

    n_epochs = 300000
    lr = 1e-4
    optimizer = optax.adam(lr)

    embedding_size = int(np.sqrt(n))
    hidden_size = int(embedding_size / 2)

    net = GCN_dev(n, embedding_size, hidden_size, 1)

    state, best_bitstring = train(
        nx_graph, net, optimizer, n_epochs, tol=1e-5, patience=1e4
    )
    evaluate_vertex_cover(nx_graph, best_bitstring, draw=True)

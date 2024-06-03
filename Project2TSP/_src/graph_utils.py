from typing import Optional
import networkx as nx
import jax.numpy as np
from flax.typing import Array
import jraph
from matrix_helper import calculate_distances


def generate_graph(
    n: int,
    degree: Optional[int] = None,
    prob: Optional[float] = None,
    graph_type: str = "reg",
    random_seed: int = 1,
) -> tuple[nx.Graph, dict[int, Array]]:
    """Generate a random NetworkX graph with n nodes.

    Args:
        n (int): The number of nodes in the graph.
        degree (int, optional): The degree of the nodes in the graph. Defaults to None.
        prob (float, optional): The probability of the edges in the graph. Defaults to None.
        graph_type (str, optional):
            The type of the graph, either "reg", "erdos", "prob", "grid" or "chess".
            Defaults to "reg".
        random_seed (int, optional): The random seed. Defaults to 0.

    Returns:
        nx.Graph: The generated graph.
        dict[int, Array]: The positions of the nodes in the graph.
    """
    pos = None
    graph_types = ("reg", "erdos", "prob", "grid", "chess")

    if graph_type not in graph_types:
        raise ValueError(
            f"Invalid graph type {graph_type}. Please choose from {graph_types}."
        )

    if graph_type == "reg":
        nx_graph = nx.random_regular_graph(degree, n, seed=random_seed)
    elif graph_type == "erdos":
        nx_graph = nx.erdos_renyi_graph(n, prob, seed=random_seed)
    elif graph_type == "prob":
        nx_graph = nx.fast_gnp_random_graph(n, prob, seed=random_seed)
    else:
        n = int(np.sqrt(n))
        nx_graph = nx.grid_2d_graph(n, n)
        new_graph = nx.Graph()
        pos = {}

        if graph_type == "chess":
            for i, j in nx_graph.nodes:
                if not 0 < i < n - 1 or not 0 < j < n - 1:
                    continue

                nx_graph.add_edge((i, j), (i + 1, j + 1))
                nx_graph.add_edge((i, j), (i - 1, j - 1))
                nx_graph.add_edge((i, j), (i + 1, j - 1))
                nx_graph.add_edge((i, j), (i - 1, j + 1))

            nx_graph.add_edge((0, 1), (1, 0))
            nx_graph.add_edge((0, n - 2), (1, n - 1))
            nx_graph.add_edge((n - 1, 1), (n - 2, 0))
            nx_graph.add_edge((n - 1, n - 2), (n - 2, n - 1))

        for i, j in nx_graph.nodes:
            new_graph.add_node(i * n + j)
            pos[i * n + j] = np.asarray([i, j]) / (n - 1)
        for (i, j), (k, l) in nx_graph.edges:
            new_graph.add_edge(i * n + j, k * n + l)
        nx_graph = new_graph

    if pos is None:
        pos = nx.kamada_kawai_layout(nx_graph)

    return nx_graph, pos


def graph_to_jraph(
    nx_graph: nx.Graph, pos: Optional[dict[int, Array]] = None
) -> jraph.GraphsTuple:
    """Convert a NetworkX graph to a jraph.GraphsTuple.

    Args:
        nx_graph (nx.Graph): The NetworkX graph to convert.
        pos (dict[int, Array]): The positions of the nodes in the graph.

    Returns:
        jraph.GraphsTuple: The converted jraph.GraphsTuple.
    """
    n = nx_graph.number_of_nodes()
    e = nx_graph.number_of_edges()

    senders, receivers = np.asarray(nx_graph.edges).T
    senders, receivers = np.concatenate([senders, receivers]), np.concatenate(
        [receivers, senders]
    )
    A = np.zeros((n, n)).at[senders, receivers].set(1.0)
    edges = None

    if pos is not None:
        pos_arr = np.stack([pos[i] for i in range(n)])
        distances = calculate_distances(pos_arr)
        distances: Array
        edges = distances.at[senders, receivers].get()

    graph = jraph.GraphsTuple(
        n_node=np.array([n]),
        n_edge=np.array([e]),
        nodes=A,
        edges=edges,
        globals=None,
        senders=senders,
        receivers=receivers,
    )

    return graph


def graph_to_jraph_2(nx_graph: nx.Graph, pos: dict[int, Array]) -> jraph.GraphsTuple:
    n = nx_graph.number_of_nodes()
    e = nx_graph.number_of_edges()

    senders, receivers = np.asarray(nx_graph.edges).T
    senders, receivers = np.concatenate([senders, receivers]), np.concatenate(
        [receivers, senders]
    )

    pos_arr = np.stack([pos[i] for i in range(n)])
    distances = calculate_distances(pos_arr)
    distances: Array
    edges = distances.at[senders, receivers].get()

    graph = jraph.GraphsTuple(
        n_node=np.array([n]),
        n_edge=np.array([e]),
        nodes=pos_arr,
        edges=edges,
        globals=None,
        senders=senders,
        receivers=receivers,
    )

    return graph

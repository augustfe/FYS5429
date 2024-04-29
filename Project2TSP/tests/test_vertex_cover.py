import pytest
import numpy as np
import networkx as nx

from _src.vertex_cover_utils import vertex_loss_func, generate_graph, graph_to_jraph


def test_generate_graph():
    n = 4
    nx_graph = generate_graph(n, 3)
    assert nx_graph.number_of_nodes() == n
    assert nx_graph.number_of_edges() == 4 * 3 / 2
    assert nx_graph.is_directed() is False

    expected_nodes = (0, 1, 2, 3)
    for node in nx_graph.nodes:
        assert node in expected_nodes

    expected_edges = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    for edge in nx_graph.edges:
        assert edge in expected_edges


def test_graph_to_jraph():
    n = 4
    nx_graph = generate_graph(n, 3)

    jraph_graph = graph_to_jraph(nx_graph)
    assert jraph_graph.n_node == np.array([n])
    assert jraph_graph.n_edge == np.array([nx_graph.number_of_edges() * 2])
    assert jraph_graph.nodes.shape == (n,)
    assert jraph_graph.edges is None

    adjacency = jraph_graph.globals
    assert adjacency.shape == (n, n)
    assert adjacency.sum() == nx_graph.number_of_edges() * 2
    np.testing.assert_array_equal(adjacency, adjacency.T)

    senders = jraph_graph.senders
    receivers = jraph_graph.receivers

    np.testing.assert_array_equal(senders[:6], receivers[6:])


@pytest.mark.parametrize(
    "probs, A",
    [
        (np.ones(4), np.eye(4)),
        (np.zeros(4), np.eye(4)),
        (np.array([0, 1, 0, 1]), np.ones((4, 4)) - np.eye(4)),
        (np.array([0.5, 0.5, 0.5, 0.5]), np.ones((4, 4)) - np.eye(4)),
    ],
)
def test_vertex_loss(probs: np.ndarray, A: np.ndarray):
    probs = probs.reshape(-1, 1)
    computed_loss = vertex_loss_func(probs, A)

    n = probs.shape[0]

    H_A = 0
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                H_A += 2 * (1 - probs[i]) * (1 - probs[j])

    H_B = np.sum(probs)
    expected_loss = H_A + H_B
    np.testing.assert_allclose(computed_loss, expected_loss)


@pytest.mark.parametrize(
    "n",
    [4, 6, 8, 10, 12, 14, 16, 18, 20],
)
def test_adjacency_match(n: int):
    nx_graph = generate_graph(n, 3)
    jraph_graph = graph_to_jraph(nx_graph)

    A = nx.adjacency_matrix(nx_graph, nodelist=range(n)).toarray()
    np.testing.assert_array_equal(A, jraph_graph.globals)

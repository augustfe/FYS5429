import networkx as nx
import jraph
from pathlib import Path
import jax.numpy as np


def BHOSLIB_to_nx_graph(filename: str | Path) -> nx.Graph:
    file = Path(filename)
    with open(file) as infile:
        header = infile.readline()
        N, M, E = map(int, infile.readline().split())

        assert N == M

        G = nx.Graph()
        for _ in range(E):
            u, v = map(int, infile.readline().split())
            # BHOSLIB is 1-indexed
            G.add_edge(u - 1, v - 1)

    return G


def BHOSLIB_to_jraph(filename: str | Path) -> nx.Graph:
    file = Path(filename)
    with open(file) as infile:
        header = infile.readline()
        N, M, E = map(int, infile.readline().split())

        assert N == M

        senders = [0] * E
        receivers = [0] * E
        for i in range(E):
            u, v = map(int, infile.readline().split())
            # BHOSLIB is 1-indexed
            senders[i] = u - 1
            receivers[i] = v - 1

    adjacency = np.zeros((N, N)).at[senders, receivers].set(1)

    senders, receivers = (senders + receivers), (receivers + senders)
    G = jraph.GraphsTuple(
        n_node=np.array([N]),
        n_edge=np.array([E]),
        senders=np.array(senders),
        receivers=np.array(receivers),
        nodes=np.arange(N),
        edges=None,
        globals=adjacency,
    )
    return G


if __name__ == "__main__":
    G = BHOSLIB_to_nx_graph("data/frb30-15-1/frb30-15-1.mtx")
    print(G)
    G = BHOSLIB_to_jraph("data/frb30-15-1/frb30-15-1.mtx")
    print(G)

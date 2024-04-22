import flax.linen as nn
from flax.typing import Array
from flax.training import train_state
from jax import numpy as np
from functools import partial
import jax
import jraph
import optax

from matrix_helper import calculate_distances


def create_dummy_graph(n: int) -> jraph.GraphsTuple:
    indices = np.arange(n)

    senders, receivers = np.meshgrid(indices, indices)
    senders, receivers = senders.flatten(), receivers.flatten()

    key = jax.random.PRNGKey(30)
    points = jax.random.uniform(key, shape=(n, 2))
    distances = calculate_distances(points)
    # edges = distances.flatten().reshape(-1, 1)

    mask = senders != receivers
    senders, receivers = senders[mask], receivers[mask]
    # edges = edges[mask]
    # distances = distances[mask]

    graph = jraph.GraphsTuple(
        n_node=np.array([n]),
        n_edge=np.array([receivers.shape[0]]),
        nodes=points,
        # nodes=distances,
        edges=None,
        globals=None,
        senders=senders,
        receivers=receivers,
    )

    return graph, points


class GCN(nn.Module):
    latent_size: int
    number_of_classes: int
    dropout_rate: float = 0.1
    deterministic: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Implements the GCN from Kipf et al https://arxiv.org/pdf/1609.02907.pdf.

        A' = D^{-0.5} A D^{-0.5}
        Z = f(X, A') = A' relu(A' X W_0) W_1

        Args:
          graphs: GraphsTuple the network processes.

        Returns:
          processed nodes.
        """
        # aggregate edges for nodes
        # gn = jraph.GraphMapFeatures(
        #     embed_node_fn=jraph.concatenated_args(nn.Dense(self.latent_size))
        # )
        # graphs = gn(graphs)
        # graphs = graphs._replace(nodes=nn.leaky_relu(graphs.nodes))

        gn = jraph.GraphConvolution(
            update_node_fn=nn.Dense(self.latent_size),
            # add_self_edges=True,
            # symmetric_normalization=False,
        )
        graphs = gn(graphs)
        # graphs = graphs._replace(nodes=nn.leaky_relu(graphs.nodes))
        # gn = jraph.GraphConvolution(
        #     update_node_fn=nn.Dense(self.latent_size),
        # )
        # graphs = gn(graphs)
        graphs = graphs._replace(nodes=nn.leaky_relu(graphs.nodes))
        # graphs = graphs._replace(nodes=nn.leaky_relu(graphs.nodes))

        # gn = jraph.GraphMapFeatures()

        # update_edge_fn = jraph.concatenated_args(
        #             MLP(
        #                 self.latent_size,
        #                 dropout_rate=self.dropout_rate,
        #                 deterministic=self.deterministic,
        #             )
        #         )

        # graphs = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(
        #     graphs
        # )

        def decoder_fn(x):
            x = nn.Dense(self.number_of_classes)(x)
            # x = nn.tanh(x)
            # x = x > 0
            x = nn.sigmoid(x)
            # x = 1 / 2 + 1 / 2 * nn.tanh(10 * (x - 1 / 2))
            return x

        gn = jraph.GraphConvolution(
            # update_node_fn=nn.Dense(self.number_of_classes),
            update_node_fn=decoder_fn,
            symmetric_normalization=False,
            # symmetric_normalization=False,
        )
        graphs = gn(graphs)
        # graphs = graphs._replace(nodes=nn.sigmoid(graphs.nodes))

        # def round_fn(x):
        #     x = nn.sigmoid(x)
        #     x = nn.tanh(10 * (x - 1 / 2))
        #     return 1 / 2 + 1 / 2 * x

        # def decoder_fn(x):
        #     x = nn.Dense(self.number_of_classes)(x)
        #     # x = nn.sigmoid(x)
        #     x = nn.softmax(x)
        #     x = round_fn(x)
        #     return x

        # graphs = graphs._replace(nodes=round_fn(graphs.nodes))
        # graphs = graphs._replace(nodes=nn.softmax(graphs.nodes, axis=1))
        # graphs = graphs._replace(nodes=nn.relu(graphs.nodes))
        return graphs.nodes


@partial(jax.jit, static_argnums=(1,))
def adjacency_graphs(graph: jraph.GraphsTuple, n: int) -> tuple[Array, Array]:
    in_graph = np.zeros((n, n))
    not_in_graph = np.ones((n, n))

    senders = graph.senders
    receivers = graph.receivers
    # distances = graph.nodes
    distances = calculate_distances(graph.nodes)
    in_graph = distances

    # mask = distances > 0
    # not_in_graph = not_in_graph.at[mask].set(0)
    # edges = graph.edges.flatten()

    # in_graph = in_graph.at[senders, receivers].add(edges)
    not_in_graph = not_in_graph.at[senders, receivers].set(0)

    # for i, j, e in zip(senders, receivers, edges):
    #     in_graph = in_graph.at[i, j].add(e)
    #     not_in_graph = not_in_graph.at[i, j].set(0)

    return in_graph, not_in_graph


@partial(jax.jit, static_argnums=(2,))
def train_step(state: train_state.TrainState, graph: jraph.GraphsTuple, n: int):

    in_graph, not_in_graph = adjacency_graphs(graph, n)
    A = 1
    # B = 1 / np.max(graph.nodes)
    B = 1

    def hamiltonian(X: Array) -> Array:
        """x = 1 - np.sum(X, axis=0)
        a = np.sum(x**2)

        x = 1 - np.sum(X, axis=1)
        b = np.sum(x**2)"""  # Old hamiltonian

        # X = X.T
        # X = X > 0.5

        cycle_const = 1 - np.sum(X, axis=1)
        a = np.sum(cycle_const**2)

        vert_const = 1 - np.sum(X, axis=0)
        b = np.sum(vert_const**2)
        # a = 0

        adjacents = X.T @ np.roll(X, 1, axis=0)
        # adjacents = adjacents.T
        not_int_const = np.sum(adjacents * not_in_graph)

        # adjacents = adjacents > 0.5

        receiver_const = 1 - np.sum(adjacents, axis=0)
        c = np.sum(receiver_const**2)
        c = 0

        sender_const = 1 - np.sum(adjacents, axis=1)
        d = np.sum(sender_const**2)
        d = 0

        return A * (a + b + c + d + not_int_const)

    def traveling_salesman(X: Array) -> Array:
        # X = X.T
        adjacents = X.T @ np.roll(X, 1, axis=0)
        # adjacents = adjacents.T
        a = np.sum(adjacents * in_graph)
        return B * a

    def soft_round(x):
        return 1 / 2 + 1 / 2 * np.tanh(25 * (x - 1 / 2))

    def loss_fn(params):
        out = model.apply(params, graph)

        # scaled = soft_round(out)

        # row = nn.softmax(out, axis=1) > 0.5
        # col = nn.softmax(out, axis=0) > 0.5

        # predicted = soft_round(out)
        # predicted = nn.softmax(out, axis=1)
        # predicted = np.round(predicted)
        # best = predicted > 0.5
        # best = soft_round(predicted)
        predicted = out > 0.5
        # out = soft_round(out)

        # additional = np.sqrt(np.sum((row - col) ** 2)) * 10 + np.sum(
        #     (scaled - out) ** 2
        # )

        # print(predicted)

        # probs = nn.softmax(out, axis=1)
        prob_loss = hamiltonian(out) + traveling_salesman(out)
        pred_loss = hamiltonian(predicted) + traveling_salesman(predicted)
        # pred_loss = 0
        # prob_loss = 0
        # pred_loss = hamiltonian(row) + traveling_salesman(row)
        # pred_loss = 0

        # best_loss = hamiltonian(col) + traveling_salesman(col)
        # best_loss = 0
        # pred_loss = 0

        # h = hamiltonian(out)
        # t = traveling_salesman(out)
        losses = (
            prob_loss,
            pred_loss,
            hamiltonian(predicted),
            traveling_salesman(predicted),
            # best_loss,
        )
        losses = (
            prob_loss,
            pred_loss,
            hamiltonian(predicted),
            traveling_salesman(predicted),
        )
        loss = prob_loss + pred_loss  # + best_loss + additional
        return loss, losses

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (losses)), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)

    return state, loss, losses


if __name__ == "__main__":
    key = jax.random.PRNGKey(30)
    key, init_key = jax.random.split(key)
    num_cities = 10

    print("Creating graph")
    graph, points = create_dummy_graph(num_cities)

    print(graph)

    print("Creating model")
    model = GCN(latent_size=2, number_of_classes=num_cities)
    print("Initizaling params")
    initial_params = model.init(init_key, graph)

    print("Creating adjacency graph")
    in_graph, not_in_graph = adjacency_graphs(graph, num_cities)

    n_epochs = 400000

    # exponential_decay = optax.exponential_decay(1e-2, 10000, 0.9, end_value=1e-5)

    optimizer = optax.adam(1e-4)

    print("Creating state")
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=optimizer
    )

    for i in range(n_epochs):
        state, loss, losses = train_step(state, graph, num_cities)
        if i % 10000 == 0:
            print(loss, *losses)

    out = model.apply(state.params, graph)
    X = nn.softmax(out, axis=1)

    print(out)

    # X = out.reshape(num_cities, num_cities)
    adjacency = X.T @ np.roll(X, 1, axis=1)

    # best_tour = adjacency > 0.5
    best_tour = np.argmax(adjacency, axis=0)
    adjacency = np.clip(adjacency, 0, 1)
    # adjacency = adjacency > 0.5

    print(adjacency)

    import matplotlib.pyplot as plt

    plt.scatter(points[:, 0], points[:, 1])

    for i in range(num_cities):
        for j in range(num_cities):
            a = points[i]
            b = points[j]

            plt.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="red",
                alpha=float(adjacency[i, j]),
            )

    plt.show()

    for i in range(num_cities):
        next = best_tour[i]
        a = points[i]
        b = points[next]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="green")

    plt.show()

    # print(hamiltonian(out))
    # print(traveling_salesman(out))

import jraph
import optax
import jax
import networkx as nx
import jax.numpy as np
import flax.linen as nn
from flax.typing import Array, FrozenDict, Any, PRNGKey

# from flax.training.train_state import TrainState
from jax import jit, value_and_grad

import matplotlib.pyplot as plt
from typing import Callable
from time import time
from tqdm import tqdm
from pathlib import Path

from gcn import GCN, TrainState
from graph_utils import generate_graph, graph_to_jraph
from matrix_helper import adjacency


@jit
def post_process(probs: Array) -> Array:
    "We can arbitrarily set the first node as the start and end node"
    probs = probs.at[0, :].set(0.0)
    probs = probs.at[:, 0].set(0.0)
    probs = probs.at[0, 0].set(1.0)
    return probs


def hamiltonian_cycle_loss(graph: jraph.GraphsTuple) -> Callable[[Array], Array]:

    senders = graph.senders
    receivers = graph.receivers
    edges = graph.edges

    def loss_function(probs: Array) -> Array:
        H_1 = np.sum((1.0 - np.sum(probs, axis=1)) ** 2)
        H_2 = np.sum((1.0 - np.sum(probs, axis=0)) ** 2)

        A_hat = adjacency(probs)
        invalid = A_hat.at[senders, receivers].set(0.0)
        H_3 = np.sum(invalid)

        H_A = H_1 + H_2 + H_3

        weighted = A_hat.at[senders, receivers].get()

        H_B = np.dot(weighted, edges)
        # H_B = 0

        return H_A + H_B

    return loss_function


@jit
def train_step(
    state: TrainState, graph: jraph.GraphsTuple, dropout_key: PRNGKey
) -> tuple[TrainState, Array, Array]:
    """Perform a single training step.

    Args:
        state (TrainState): The training state.
        graph (jraph.GraphsTuple): The graph to train on.

    Returns:
        TrainState: The updated training state.
    """
    loss_function = hamiltonian_cycle_loss(graph)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params: FrozenDict[str, Any]) -> tuple[Array, Array]:
        predictions = state.apply_fn(
            params,
            graph,
            training=True,
            rngs={"dropout": dropout_train_key},
        )

        predictions: jraph.GraphsTuple
        nodes = post_process(predictions.nodes)
        loss = loss_function(nodes)
        return loss, nodes

    (loss, probs), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, probs


def train(
    nx_graph: nx.Graph,
    net: GCN,
    optimizer: optax.GradientTransformation,
    num_epochs: int = 100,
    random_seed: int = 0,
    tol: float = 0.01,
    patience: int = 100,
) -> tuple[TrainState, Array]:
    graph = graph_to_jraph(nx_graph, pos)
    main_key = jax.random.PRNGKey(random_seed)
    main_key, init_rng, dropout_key = jax.random.split(main_key, num=3)

    params = net.init(
        # {"params": init_rng, "dropout": dropout_key},
        init_rng,
        graph,
        training=True,
    )
    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        key=dropout_key,
        tx=optimizer,
    )

    prev_loss = 1.0
    count = 0

    n = graph.n_node[0]
    best_bitstring = np.zeros((n, n))
    best_loss = np.inf

    gnn_start = time()

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        state, loss, probs = train_step(state, graph, dropout_key)

        if epoch % (num_epochs // 10) == 0:
            draw_cycle(nx_graph, pos, probs, rounding=False, save=True, step=epoch)

        if loss < best_loss:
            best_loss = loss
            best_bitstring = probs

        if prev_loss - loss < tol:
            count += 1
        else:
            count = 0

        if count > patience:
            print(f"Early stopping at epoch {epoch}.")
            break

        prev_loss = loss
        pbar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "patience": f"{count / patience * 100:.1f}%",
            }
        )

    gnn_end = time()
    print(f"Training took {gnn_end - gnn_start:.2f} seconds.")
    print(f"Final loss: {loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    return state, best_bitstring


def draw_cycle(
    nx_graph: nx.Graph,
    pos: dict[int, Array],
    bitstring: Array,
    rounding: bool = True,
    save: bool = False,
    step: int = 0,
):
    n = nx_graph.number_of_nodes()

    pos_arr = np.stack([pos[i] for i in range(n)])
    plt.scatter(pos_arr[:, 0], pos_arr[:, 1])

    A_hat = adjacency(bitstring)
    if rounding:
        A_hat = (A_hat > 0.5) * 1.0
    for i in range(n):
        for j in range(n):
            a = pos_arr[i]
            b = pos_arr[j]

            if nx_graph.has_edge(i, j):
                color, style = "black", "-"
            else:
                color, style = "red", "--"

            plt.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                linestyle=style,
                alpha=A_hat[i, j].item(),
            )

    if save:
        path = Path(__file__).parent / "cycle_imgs"
        path.mkdir(exist_ok=True, parents=True)
        plt.savefig(path / f"cycle_{step}.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    n = 8 * 8
    nx_graph, pos = generate_graph(n, graph_type="chess")
    # nx_graph, pos = generate_graph(n, graph_type="grid")
    n = nx_graph.number_of_nodes()

    n_epochs = 100000
    lr = 0.001
    optimizer = optax.adam(learning_rate=lr)

    hidden_size = 64
    net = GCN(hidden_size, n, nn.leaky_relu)

    state, best_bitstring = train(
        nx_graph,
        net,
        optimizer,
        n_epochs,
        tol=1e-4,
        patience=100000,
    )

    graph = graph_to_jraph(nx_graph, pos)
    final_graph = state.apply_fn(state.params, graph, training=False)
    final_bitstring = post_process(final_graph.nodes)

    draw_cycle(nx_graph, pos, best_bitstring)

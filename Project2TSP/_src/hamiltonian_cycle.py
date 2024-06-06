import jraph
import optax
import jax
import networkx as nx
import jax.numpy as np
import flax.linen as nn
from flax.typing import Array, FrozenDict, Any, PRNGKey
from jax import jit, value_and_grad, lax

import matplotlib.pyplot as plt
from typing import Callable
from time import time
from tqdm import tqdm
from pathlib import Path

from gcn import GCN, TrainState
from graph_utils import generate_graph, graph_to_jraph as graph_to_jraph
from matrix_helper import adjacency


@jit
def post_process(probs: Array) -> Array:
    "We can arbitrarily set the first node as the start and end node"
    probs = probs.at[0, :].set(0.0)
    probs = probs.at[:, 0].set(0.0)
    probs = probs.at[0, 0].set(1.0)
    return probs


def hamiltonian_cycle_loss(
    graph: jraph.GraphsTuple, use_TSP: bool
) -> Callable[[Array], Array]:

    senders = graph.senders
    receivers = graph.receivers
    edges = graph.edges

    def loss_function(probs: Array) -> Array:
        A_hat = adjacency(probs)
        invalid = A_hat.at[senders, receivers].set(0.0)

        H_1 = np.sum((1.0 - np.sum(probs, axis=1)) ** 2)
        H_2 = np.sum((1.0 - np.sum(probs, axis=0)) ** 2)
        H_3 = np.sum(invalid)

        H_A = H_1 + H_2 + H_3

        weighted = A_hat.at[senders, receivers].get()

        H_B = lax.select(use_TSP, np.dot(weighted, edges), 0.0)

        return H_A + H_B

    return loss_function


@jit
def train_step(
    state: TrainState,
    graph: jraph.GraphsTuple,
    dropout_key: PRNGKey,
    use_TSP: bool = True,
) -> tuple[TrainState, Array, Array]:
    """Perform a single training step.

    Args:
        state (TrainState): The training state.
        graph (jraph.GraphsTuple): The graph to train on.

    Returns:
        TrainState: The updated training state.
    """
    loss_function = hamiltonian_cycle_loss(graph, use_TSP=use_TSP)
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
    graph: jraph.GraphsTuple,
    net: GCN,
    optimizer: optax.GradientTransformation,
    num_epochs: int = 100,
    random_seed: int = 0,
    tol: float = 0.01,
    patience: int = 1000,
    warm_up: int = 2500,
    show_progress: bool = True,
    use_TSP: bool = True,
) -> tuple[TrainState, Array]:
    main_key = jax.random.PRNGKey(random_seed)
    main_key, init_rng, dropout_key = jax.random.split(main_key, num=3)

    params = net.init(
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

    best_loss = np.inf
    count = 0

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        state, epoch_loss, probs = train_step(state, graph, dropout_key, use_TSP)

        pbar.set_postfix(
            {
                "loss": f"{epoch_loss:.4f}",
                "patience": f"{count / patience * 100:.1f}%",
            }
        )

        if epoch > warm_up:
            if best_loss - epoch_loss > tol or np.isinf(best_loss):
                count = 0
            else:
                count += 1
                if count > patience:
                    if show_progress:
                        print(f"Early stopping at epoch {epoch}")
                    pbar.close()
                    break

        if epoch_loss < best_loss:
            best_bitstring = probs
            best_loss = epoch_loss

    print(f"Final loss: {epoch_loss:.4f}")
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
        loss_func = hamiltonian_cycle_loss(graph_to_jraph(nx_graph, pos))
        A_hat = (A_hat > 0.5) * 1.0

        det_loss = loss_func(bitstring)
        order_violations = (1 - np.sum(bitstring, axis=0)) ** 2
        node_violations = (1 - np.sum(bitstring, axis=1)) ** 2
        degree_violations = np.abs(np.sum(A_hat, axis=0) - 1)

        print(f"Deterministic loss: {det_loss:.4f}")
        print(f"Order violations: {np.sum(order_violations):.4f}")
        print(f"Node violations: {np.sum(node_violations):.4f}")
        print(f"Degree violations: {np.sum(degree_violations):.4f}")

    A_hat = np.clip(A_hat, 0, 1)
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
    # nx_graph, pos = generate_graph(100, degree=5, graph_type="reg")

    n = nx_graph.number_of_nodes()

    n_epochs = 100_000
    lr = 0.0001
    optimizer = optax.adam(learning_rate=lr)
    # lr = 0.01
    # optimizer = optax.noisy_sgd(learning_rate=lr, eta=0.4, gamma=0.0)
    # optimizer = optax.lamb(learning_rate=lr)

    hidden_size = 64
    net = GCN(
        hidden_size,
        n,
        nn.leaky_relu,
        output_activation=nn.softmax,
        num_layers=2,
        dropout_rate=0.3,
        num_convolutions=2,
    )

    state, best_bitstring = train(
        nx_graph,
        net,
        optimizer,
        n_epochs,
        tol=1e-4,
        patience=10000,
    )

    graph = graph_to_jraph(nx_graph, pos)
    final_graph = state.apply_fn(state.params, graph, training=False)
    final_bitstring = post_process(final_graph.nodes)
    # final_bitstring = final_graph.nodes

    # print(best_bitstring.shape)
    print(
        net.tabulate(
            jax.random.key(0),
            graph,
            training=False,
            compute_flops=True,
            compute_vjp_flops=True,
        )
    )

    draw_cycle(nx_graph, pos, final_bitstring, savename="no_TSP")

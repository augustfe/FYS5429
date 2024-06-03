import jraph
import optax
import jax
import networkx as nx
import jax.numpy as np
import flax.linen as nn
from flax.typing import Array, FrozenDict, Any, PRNGKey
from jax import jit, value_and_grad

import matplotlib.pyplot as plt
from typing import Callable
from time import time
from tqdm import tqdm
from pathlib import Path

from gcn import GCN, TrainState
from graph_utils import generate_graph, graph_to_jraph


def vertex_cover_loss(graph: jraph.GraphsTuple) -> Callable[[Array], Array]:

    senders = graph.senders
    receivers = graph.receivers

    def loss_function(probs: Array) -> Array:
        neg_probs = 1.0 - probs
        H_A = np.dot(neg_probs[senders].T, neg_probs[receivers]).sum()
        # Q = neg_probs @ neg_probs.T

        # H_A = np.sum(Q[senders, receivers])
        H_B = np.sum(probs)

        return 2 * H_A + H_B

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
    loss_function = vertex_cover_loss(graph)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params: FrozenDict[str, Any]) -> tuple[Array, Array]:
        predictions = state.apply_fn(
            params,
            graph,
            training=True,
            rngs={"dropout": dropout_train_key},
        )

        predictions: jraph.GraphsTuple
        loss = loss_function(predictions.nodes)
        return loss, predictions.nodes

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

    best_loss = np.inf

    gnn_start = time()

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        state, loss, probs = train_step(state, graph, dropout_key)

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
    # print(f"Training took {gnn_end - gnn_start:.2f} seconds.")
    # print(f"Final loss: {loss:.4f}")
    # print(f"Best loss: {best_loss:.4f}")

    return state, best_bitstring


def draw_vertex_cover(
    nx_graph: nx.Graph,
    pos: dict[int, Array],
    bitstring: Array,
    rounding: bool = True,
    save: bool = False,
    step: int = 0,
):
    n = nx_graph.number_of_nodes()

    if rounding:
        bitstring = (bitstring > 0.5) * 1.0

    pos_arr = np.stack([pos[i] for i in range(n)])
    plt.scatter(pos_arr[:, 0], pos_arr[:, 1], c=bitstring, cmap="coolwarm", zorder=2)

    size_vc = np.sum(bitstring)
    violations = 0

    for i, j in nx_graph.edges:
        a = pos_arr[i]
        b = pos_arr[j]

        if bitstring[i] > 0.5 or bitstring[j] > 0.5:
            color = "black"
        else:
            violations += 1
            color = "red"

        plt.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            zorder=1,
        )

    plt.title(f"Vertex Cover: {size_vc} nodes, {violations} violations")
    plt.axis("off")

    if save:
        path = Path(__file__).parent / "vertex_imgs"
        path.mkdir(exist_ok=True, parents=True)
        plt.savefig(path / f"vc_{step}.png")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    n = 8 * 8
    p = 0.05
    d = 3
    # nx_graph, pos = generate_graph(n, prob=p, graph_type="erdos")
    nx_graph, pos = generate_graph(n, graph_type="grid")
    n = nx_graph.number_of_nodes()

    n_epochs = 10000
    lr = 0.01
    optimizer = optax.adam(learning_rate=lr)
    # optimizer = optax.noisy_sgd(learning_rate=lr)

    hidden_size = 5
    net = GCN(
        hidden_size, 1, nn.leaky_relu, output_activation=nn.sigmoid, dropout_rate=0.0
    )

    sizes = []

    for i in tqdm(range(100)):
        state, best_bitstring = train(
            nx_graph,
            net,
            optimizer,
            n_epochs,
            tol=0.1,
            patience=10000,
            random_seed=i,
        )

        graph = graph_to_jraph(nx_graph, pos)
        final_graph = state.apply_fn(state.params, graph, training=False)
        final_bitstring = final_graph.nodes
        final_bitstring = (final_bitstring > 0.5) * 1.0
        sizes.append(np.sum(final_bitstring))

    # print(best_bitstring)

    # draw_vertex_cover(nx_graph, pos, final_bitstring, rounding=False)
    sizes = np.array(sizes)

    print(np.mean(sizes))
    print(np.std(sizes))
    print(np.min(sizes))
    print(np.max(sizes))
    print(np.median(sizes))
    print(np.percentile(sizes, 25))
    print(np.percentile(sizes, 75))

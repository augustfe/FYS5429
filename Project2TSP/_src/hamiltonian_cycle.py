import jraph
import optax
import jax
import jax.numpy as np
from flax.typing import Array, FrozenDict, Any, PRNGKey
from jax import jit, value_and_grad, lax

from typing import Callable
from tqdm import tqdm

from gcn import GCN, TrainState
from graph_utils import adjacency


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

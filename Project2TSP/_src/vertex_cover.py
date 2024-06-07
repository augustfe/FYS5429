import jraph
import jax
import jax.numpy as np
from jax import jit, value_and_grad, Array
from tqdm.auto import tqdm
from flax.typing import PRNGKey, FrozenDict, Any, Callable
from gcn import GCN, TrainState
import optax


def vertex_cover_loss(graph: jraph.GraphsTuple) -> Callable[[Array], Array]:
    """Compute the loss function for the vertex cover problem.

    Args:
        graph (jraph.GraphsTuple): The graph to compute the loss on.

    Returns:
        Callable[[Array], Array]: The loss function.
    """

    senders = graph.senders
    receivers = graph.receivers

    def loss_function(probs: Array) -> Array:
        neg_probs = 1.0 - probs

        H_A = np.dot(neg_probs[senders].T, neg_probs[receivers]).sum()
        H_B = np.sum(probs)

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
        predictions = predictions.nodes.flatten()
        loss = loss_function(predictions)
        return loss, predictions

    (loss, probs), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, probs


def train(
    graphs: jraph.GraphsTuple,
    net: GCN,
    optimizer: optax.GradientTransformation,
    num_epochs: int = 100,
    random_seed: int = 0,
    tol: float = 0.01,
    patience: int = 100,
    warm_up: int = 1000,
    show_progress: bool = True,
) -> tuple[TrainState, Array]:
    """Train the GCN on the vertex cover problem.

    Args:
        graphs (jraph.GraphsTuple): The graphs to train on.
        net (GCN): The GCN model to train.
        optimizer (optax.GradientTransformation): The optimizer to use.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 100.
        random_seed (int, optional): The random seed to use. Defaults to 0.
        tol (float, optional): The tolerance for early stopping. Defaults to 0.01.
        patience (int, optional): The patience for early stopping. Defaults to 100.
        warm_up (int, optional): The number of epochs before starting early stopping.
        show_progress (bool, optional): Whether to show the progress bar. Defaults to True.

    Returns:
        tuple[TrainState, Array]: The trained state and the losses.
    """
    graph = graphs
    num_graphs = graph.n_node.shape[0]

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

    @jit
    def eval_fn(probs: Array, graph: jraph.GraphsTuple) -> Array:
        """Evaluate the predicted vertex cover.

        Args:
            probs (Array): Predicted probabilities of the nodes being in the vertex cover.
            graph (jraph.GraphsTuple): The graph to evaluate the vertex cover on.

        Returns:
            Array: Evaluation of the vertex cover.
        """
        eval_probs = (probs > 0.5) * 1.0
        eval_loss = vertex_cover_loss(graph)(eval_probs)

        return eval_loss

    count = 0
    best_loss = np.inf
    pbar = tqdm(
        range(num_epochs), desc="Training", unit="epoch", disable=not show_progress
    )

    losses = []
    evals = []

    for epoch in pbar:
        epoch_loss = 0.0
        epoch_eval = 0.0

        state, epoch_loss, probs = train_step(state, graph, dropout_key)

        epoch_eval = eval_fn(probs, graph)

        epoch_loss /= num_graphs
        epoch_eval /= num_graphs
        losses.append(epoch_loss)
        evals.append(epoch_eval)

        pbar.set_postfix(
            {
                "loss": f"{epoch_loss:.4f}",
                "eval": f"{epoch_eval:.4f}",
                "diff": f"{best_loss - epoch_loss:.4f}",
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
            best_loss = epoch_loss

    return state, losses, evals

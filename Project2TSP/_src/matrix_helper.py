from jax import numpy as np, Array, random, vmap, jit, lax
from typing import Callable
from functools import partial
import matplotlib.pyplot as plt


def adjacency(X: Array) -> Array:
    "Generate the predicted adjacency matrix"

    FROM = X
    TO = np.roll(X.T, -1, axis=0)
    return FROM @ TO


def create_C0(n: int) -> Array:
    A = np.eye(n, dtype=np.int8)
    B = np.ones(n, dtype=np.int8)
    return np.kron(A, B)


def create_C1(n: int) -> Array:
    A = np.eye(n, dtype=np.int8)
    B = np.ones(n, dtype=np.int8)
    return np.kron(B, A)


@jit
def distance_between(x: Array, y: Array) -> float:
    return np.linalg.norm(x - y)


@jit
def calculate_distances(x: Array) -> Array:
    """The matrix of pairwise distances between the rows of x."""
    dist_a_to_B = vmap(distance_between, in_axes=(0, None))
    dist_a_to_B = vmap(dist_a_to_B, in_axes=(None, 0))

    return dist_a_to_B(x, x)


@jit
def evaluate_term(x: Array, C: Array, size: int) -> float:
    b = np.sum(x)
    x_tmp = C @ x
    c = x_tmp.T @ x_tmp
    return size - 2 * b + c


def total_energy_factory(C0: Array, C1: Array, M: Array) -> Callable[[Array], float]:
    size = C0.shape[0]
    A = 100.0
    B = A / (2 * np.max(M))
    # B = 1.0

    @jit
    def total_energy(x: Array) -> float:
        H_A = evaluate_term(x, C0, size) + evaluate_term(x, C1, size)
        H_B = distance_of_tour(x, M, size)
        return A * H_A + B * H_B

    return total_energy


@partial(jit, static_argnums=(2,))
def distance_of_tour(x: Array, M: Array, size: int) -> float:
    adj = adjacency_(x, size)

    return np.sum(adj * M)


def adjacency_(x: Array, size: int) -> Array:
    X = x.reshape(size, size)

    TO = np.roll(X, 1, axis=0)
    FROM = X.T

    return FROM @ TO


@partial(jit, static_argnums=(1,))
def cycle_step(counter_period, _):
    counter, period = counter_period
    counter = lax.cond(counter >= period - 1, lambda: -period, lambda: counter + 1)
    bit = lax.cond(
        counter >= 0,
        lambda: np.array(0, dtype=np.int8),
        lambda: np.array(1, dtype=np.int8),
    )
    return (counter, period), bit


def build_column(index, size):
    counter_period = (-1, 2**index)
    return lax.scan(cycle_step, counter_period, None, 2**size)[1]


def all_bitstrings_jax(size):
    return vmap(lambda i: build_column(i, size), out_axes=1)(
        np.arange(size - 1, -1, -1)
    )


def plot_tour(x: Array, points: Array):
    n = points.shape[0]
    dim = points.shape[1]
    adjacency_matrix = np.clip(adjacency_(x, n), 0, 1)
    # distance_matrix = distances(points) * adjacency_matrix
    # weighted = distance_matrix / np.max(distance_matrix)

    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        for i in range(n):
            for j in range(n):
                a = points[i]
                b = points[j]

                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    [a[2], b[2]],
                    color="red",
                    alpha=float(adjacency_matrix[i, j]),
                )
        plt.show()
        return

    plt.scatter(points[:, 0], points[:, 1])

    for i in range(n):
        for j in range(n):
            a = points[i]
            b = points[j]

            plt.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="red",
                alpha=float(adjacency_matrix[i, j]),
            )
    plt.show()


def tryout(points: Array = None):
    if points is None:
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    n = points.shape[0]

    C0 = create_C0(n)
    C1 = create_C1(n)

    M = calculate_distances(points)

    # print(distance_of_tour(x, M, C0.shape[0]))
    total_energy = total_energy_factory(C0, C1, M)
    X = all_bitstrings_jax(n * n)
    energies = vmap(total_energy, in_axes=0)(X)
    idxs = np.argmin(energies)
    print(X[idxs])
    print(energies[idxs])
    print(total_energy(X[idxs]))

    plot_tour(X[idxs], points)


if __name__ == "__main__":
    key = random.PRNGKey(30)

    points = random.uniform(key, (5, 3))
    # x = random.uniform(key, (25,))
    # plot_tour(x, points)
    tryout(points)
    # import matplotlib.pyplot as plt

    # # Create 64 evenly spaced points in [0, 1]^2
    # x = np.linspace(0, 1, 2)
    # x = np.stack(np.meshgrid(x, x), axis=-1).reshape(-1, 2)

    # # plt.scatter(x[:, 0], x[:, 1])
    # # plt.show()

    # print(x)

    # print(distances(x))

    # C0 = create_C0(3)
    # C1 = create_C1(3)

    # print(C0.T @ C0)
    # print(C1.T @ C1)

from region_utils.shapes import Shape
import numpy as onp
import jax.numpy as np
from type_util import Array, Callable
from jax import vmap, jit  # noqa
from typing import Optional
from collections import defaultdict
import json


class Subdomain:
    def __init__(
        self,
        composition: list[Shape],
        subtraction: Optional[list[Shape]] = None,
    ) -> None:
        if subtraction is None:
            subtraction = []

        self.boundary_len = 0
        self.boundaries: list[Callable[[float], Array]] = []
        for shape in composition + subtraction:
            self.boundaries += shape.boundary
            self.boundary_len += shape.boundary_length

        self.composition = composition
        self.subtraction = subtraction

        self.args: dict[str, Array] = {}

    def are_inside(self, points: Array) -> Array:
        are_valid = np.zeros(points.shape[0], dtype=bool)

        for shape in self.composition:
            additive = shape.are_inside(points)
            are_valid = bool_or(are_valid, additive)

        for shape in self.subtraction:
            subtractive = shape.are_inside(points)
            are_valid = bool_sub(are_valid, subtractive)

        return points[are_valid]

    def create_boundary(self, num_points: int) -> Array:
        n_bound = len(self.boundaries)
        boundary_counts = onp.random.multinomial(num_points, [1 / n_bound] * n_bound)
        boundary_points = np.zeros((num_points, 2))
        so_far = 0
        for boundary, boundary_counts in zip(self.boundaries, boundary_counts):
            t_vals = onp.random.uniform(0, 1, boundary_counts)
            x_vals = vmap(boundary)(t_vals)
            boundary_points = boundary_points.at[
                so_far : so_far + boundary_counts  # noqa: E203
            ].set(x_vals)
            so_far += boundary_counts

        return boundary_points


@jit
def bool_or(vec1: Array, vec2: Array) -> Array:
    vec1 = np.logical_or(vec1, vec2)
    return vec1


@jit
def bool_sub(vec1: Array, vec2: Array) -> Array:
    vec1 = np.logical_and(vec1, np.invert(vec2))
    return vec1


class Domain:
    def __init__(self, subdomains: list[Subdomain]) -> None:
        self.subdomains = subdomains
        self.pinn_points: dict[int, dict[str, Array]] = {
            i: {} for i in range(len(subdomains))
        }
        self.interfaces: defaultdict[list[int], list[Array]] = defaultdict(list)

    def create_interior(self, n: int, limits: list[list[int]]) -> None:
        lower, upper = limits

        points = []
        for low, high in zip(lower, upper):
            points.append(onp.random.uniform(low, high, n))
        points = np.column_stack(points)

        for i, subdomain in enumerate(self.subdomains):
            args = self.pinn_points[i]
            valid_points = subdomain.are_inside(points)
            args["interior"] = valid_points

    def create_boundary(self, n: int) -> None:
        total_length = sum(subdomain.boundary_len for subdomain in self.subdomains)

        for i, subdom in enumerate(self.subdomains):
            args = self.pinn_points[i]
            num_points = int(subdom.boundary_len / total_length * n)
            args["boundary"] = subdom.create_boundary(num_points)

    def create_interface(
        self, n: int, indexes: tuple[int, int], points: tuple[Array, Array]
    ) -> None:
        start, end = points
        i, j = sorted(indexes)

        inter_points = np.linspace(start, end, n)
        self.interfaces[(i, j)].append(inter_points)

    def write_to_file(self, filename: str) -> None:
        with open(filename, "w") as outfile:
            data = {"XPINNs": [], "Interfaces": []}
            for i, subdomain in enumerate(self.subdomains):
                args = self.pinn_points[i]
                subdomain_data = {
                    "Internal points": args["interior"].tolist(),
                    "Boundary points": args["boundary"].tolist(),
                }
                data["XPINNs"].append(subdomain_data)

            for key, val in self.interfaces.items():
                data["Interfaces"].append(
                    {
                        "XPINNs": key,
                        "Points": sum([point.tolist() for point in val], []),
                    }
                )
            json.dump(data, outfile)

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        for i, args in enumerate(self.pinn_points.values()):
            plt.scatter(*args["interior"].T, label=f"Interior {i}")
            plt.scatter(*args["boundary"].T, label=f"Boundary {i}")

        for key, val in self.interfaces.items():
            points = sum([point.tolist() for point in val], [])
            plt.scatter(*np.array(points).T, label=f"Interface {key}")

        plt.legend()
        plt.title("Domain Points")
        plt.show()

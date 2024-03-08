from _region.shapes import Shape, ConvexPolygon, Circle
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
from type_util import Array, Callable
from jax import vmap, jit  # noqa
from typing import Optional


class Subdomain:
    def __init__(
        self,
        composition: list[Shape],
        index: int,
        subtraction: Optional[list[Shape]] = None,
    ) -> None:
        if subtraction is None:
            subtraction = []

        self.boundary_len = 0
        # for shape in composition + subtraction:
        #     self.boundary_len += shape.boundary_length()

        self.boundaries: list[Callable[[float], Array]] = []
        for shape in composition + subtraction:
            self.boundaries += shape.boundary

        self.composition = composition
        self.subtraction = subtraction

        self.index = index
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
            boundary_points = boundary_points.at[so_far : so_far + boundary_counts].set(
                x_vals
            )
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
        self.main_args: dict[int, dict[str, Array]] = {}

    def create_interior(self, n: int, limits: list[list[int]]) -> None:
        lower, upper = limits

        points = []
        for low, high in zip(lower, upper):
            points.append(onp.random.uniform(low, high, n))
        points = np.column_stack(points)

        for i, subdomain in enumerate(self.subdomains):
            args = self.main_args[i]
            args["interior"] = subdomain.are_inside(points)

    def create_boundary(self, n: int) -> None:
        total_length = sum(subdomain.boundary_len for subdomain in self.subdomains)

        for i, subdom in enumerate(self.subdomains):
            args = self.main_args[i]
            num_points = int(subdom.boundary_len / total_length * n)
            args["boundary"] = subdom.create_boundary(num_points)

    def create_interface(self, n: int) -> None:
        pass


class _Domain:
    def __init__(self, *regions: tuple[Shape, ...]) -> None:
        # reverse regions such that we add domain region last
        self.regions = regions[::-1]
        self.all_points: dict[str, dict[str, Array]] = dict()

    def add_shape(self, shape: Shape) -> None:
        self.regions.insert(0, shape)

    def generate_data(self, num_points: list[tuple[int, int]]):
        for i, region in enumerate(self.regions):
            other_regions = self.regions[:i]
            region_num_points = num_points[i]
            if isinstance(region, ConvexPolygon):
                self.all_points[f"region{i}"] = self._generate_data_ConvexPoly(
                    region,
                    region_num_points,
                    other_regions,
                )
            else:
                self.all_points[f"region{i}"] = self._generate_data_Circle(
                    region,
                    region_num_points,
                    other_regions,
                )

    def _generate_data_ConvexPoly(
        self,
        region: ConvexPolygon,
        region_num_points: tuple[int, int],
        other_regions: list[Shape],
    ) -> dict[str, Array]:
        n_interior, n_boundary = region_num_points

        boundary_points = []

        vertecies = region.vertices
        x_min, t_min = np.min(vertecies, axis=0)
        x_max, t_max = np.max(vertecies, axis=0)

        # Calculate the number of points to sample on each edge based on their length
        num_vertices = len(region.vertices)
        edge_lengths = np.linalg.norm(region.vectors, axis=1)

        total_perimeter = np.sum(edge_lengths)
        points_per_edge = np.round(
            (edge_lengths / total_perimeter) * n_boundary
        ).astype(int)

        # Generate points for each edge
        for j in range(num_vertices):
            vertex1 = region.vertices[j]
            vertex2 = region.vertices[(j + 1) % num_vertices]
            for j in range(points_per_edge[j]):
                t = j / points_per_edge[j]
                point = (1 - t) * vertex1 + t * vertex2
                boundary_points.append(point)

        interior_points = _generate_interior_points(
            x_min, x_max, t_min, t_max, n_interior, region, other_regions
        )

        return {"interior": interior_points, "boundary": np.array(boundary_points)}

    def _generate_data_Circle(
        self,
        region: ConvexPolygon,
        region_num_points: tuple[int, int],
        other_regions: list[Shape],
    ) -> dict[str, Array]:
        n_interior, n_boundary = region_num_points
        n_interior = int(n_interior * 4 / np.pi)

        center, radius = region.center, region.radius
        x_center, t_center = center

        x_min, t_min = center - radius
        x_max, t_max = center + radius

        # Generate uniformly distributed points around the circle
        angles = onp.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
        boundary_points = region.center + region.radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )

        interior_points = _generate_interior_points(
            x_min, x_max, t_min, t_max, n_interior, region, other_regions
        )

        return {"interior": interior_points, "boundary": np.array(boundary_points)}

    def plot_points(self) -> None:
        for region_key, region_data in self.all_points.items():
            plt.scatter(
                region_data["interior"][:, 0],
                region_data["interior"][:, 1],
                label=region_key,
                s=1,
            )
            plt.scatter(
                region_data["boundary"][:, 0],
                region_data["boundary"][:, 1],
                label=f"boundary {region_key}",
                s=1,
            )
            plt.legend()
        plt.show()


def _generate_interior_points(
    x_min,
    x_max,
    t_min,
    t_max,
    n_interior: int,
    region: Shape,
    other_regions: list[Shape],
):
    x = onp.random.uniform(x_min, x_max, size=n_interior)
    t = onp.random.uniform(t_min, t_max, size=n_interior)

    points = np.column_stack([x, t])

    is_inside_mask = region.are_inside(points)

    for other_region in other_regions:
        not_inside_region_mask = np.logical_not(other_region.are_inside(points))
        is_inside_mask = np.logical_and(is_inside_mask, not_inside_region_mask)

    return points[is_inside_mask]


if __name__ == "__main__":
    # Define a domain, which in this case will just be a larger square
    domain_vertices = np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    domain = ConvexPolygon(domain_vertices)

    # Define a circle and a square inside this domain
    circle = Circle(center=np.array([1.5, 1.5]), radius=0.5)
    square_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    square = ConvexPolygon(square_vertices)

    # Create a region with these shapes
    thingy = _Domain(domain, square, circle)
    thingy.generate_data([(20, 20), (20, 20), (20, 20)])

    # Generate 100 interior points for each shape in the region
    for shape_key, shape_data in thingy.all_points.items():
        interior_points = thingy.all_points[shape_key]["interior"]
        print(f"{shape_key} has {len(interior_points)} interior points")

    thingy.plot_points()

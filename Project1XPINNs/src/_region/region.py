from _region.shapes import Shape, ConvexPolygon, Circle
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
from type_util import Array


class Domain:
    def __init__(self, *regions: tuple[Shape, ...]) -> None:
        # reverse regions such that we add domain region last
        self.regions = regions[::-1]
        print(self.regions, type(self.regions))
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
        edge_lengths = np.linalg.norm(
            np.roll(region.vertices, -1, axis=0) - region.vertices, axis=1
        )
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
    thingy = Domain(domain, square, circle)
    thingy.generate_data([(20, 20), (20, 20), (20, 20)])

    # Generate 100 interior points for each shape in the region
    for shape_key, shape_data in thingy.all_points.items():
        interior_points = thingy.all_points[shape_key]["interior"]
        print(f"{shape_key} has {len(interior_points)} interior points")

    thingy.plot_points()

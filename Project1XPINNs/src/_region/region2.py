from shapes import Shape, ConvexPolygon, Circle
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class Domain:
    def __init__(self, *regions: Shape):
        # reverse regions such that we add domain region last
        self.regions = regions[::-1]
        self.all_points = dict()

    def add_shape(self, shape):
        self.regions.append(shape)

    def generate_data(self, num_points: List[Tuple[int]]):
        self._generate_data(self.regions, num_points)

    @staticmethod
    def _generate_data(regions, num_points: [int, int]):
        all_points = {}
        for count, region in enumerate(regions):
            region_key = f'region{count}'
            n_interior, n_boundary = num_points[count]
            if isinstance(region, ConvexPolygon):
                vertecies = region.vertices
                x_min, t_min = np.min(vertecies, axis=0)
                x_max, t_max = np.max(vertecies, axis=0)

            elif isinstance(region, Circle):
                n_interior = int(n_interior*4/np.pi)
                center, radius = region.center, region.radius
                x_center, t_center = center
                x_min, t_min = center - radius
                x_max, t_max = center + radius

            x = onp.random.uniform(x_min, x_max, size=n_interior)
            t = onp.random.uniform(t_min, t_max, size=n_interior)

            points = np.column_stack([x, t])

            is_inside_mask = region.are_inside(points)
            for other_count, other_region in enumerate(regions):
                if count == other_count:
                    continue
                not_inside_region_mask = np.logical_not(
                    other_region.are_inside(points))
                is_inside_mask = np.logical_and(
                    is_inside_mask, not_inside_region_mask)

            interior_points = points[is_inside_mask]
            boundary_points = self._generate_boundary_data(region, n_boundary)

            all_points[region_key] = {
                'interior': interior_points, 'boundary': boundary_points}

        return all_points

    @staticmethod
    def _generate_boundary_data(region: Shape, n: int) -> list:
        """Generates uniformly distributed boundary data given a shape.

        Args:
            region (Shape): The shape to generate boundary data from.
            n (int): The approximate number of points to generate.

        Returns:
            list: A list of boundary points.
        """
        boundary_points = []

        if isinstance(region, ConvexPolygon):
            # Calculate the number of points to sample on each edge based on their length
            num_vertices = len(region.vertices)
            edge_lengths = np.linalg.norm(
                np.roll(region.vertices, -1, axis=0) - region.vertices, axis=1)
            total_perimeter = np.sum(edge_lengths)
            points_per_edge = np.round(
                (edge_lengths / total_perimeter) * n).astype(int)

            # Generate points for each edge
            for i in range(num_vertices):
                vertex1 = region.vertices[i]
                vertex2 = region.vertices[(i + 1) % num_vertices]
                for j in range(points_per_edge[i]):
                    t = j / points_per_edge[i]
                    point = (1 - t) * vertex1 + t * vertex2
                    boundary_points.append(point)

        elif isinstance(region, Circle):
            # Generate uniformly distributed points around the circle
            angles = onp.linspace(0, 2 * np.pi, n, endpoint=False)
            boundary_points = region.center + region.radius * \
                np.stack([np.cos(angles), np.sin(angles)], axis=1)

        return np.array(boundary_points)

    def plot_points(self):
        for region_key, region_data in self.all_points.items():
            plt.scatter(
                region_data['interior'][:, 0], region_data['interior'][:, 1], label=region_key, s=1)
            plt.scatter(region_data['boundary'][:, 0],
                        region_data['boundary'][:, 1], label=f"boundary {region_key}", s=1)
            plt.legend()
        plt.show()


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
    thingy.generate_data([[100, 100], [100, 100], [100, 100]])

    # Generate 100 interior points for each shape in the region
    for shape_key, shape_data in thingy.all_points.items():
        interior_points = thingy.all_points[shape_key]['interior']
        print(f'{shape_key} has {len(interior_points)} interior points')

    thingy.plot_points()

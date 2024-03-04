from type_util import Array, Scalar
import jax.numpy as np
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def is_inside(self, point: Array) -> bool:
        """Check Whether a given point is inside the region

        Args:
            point (Array): Point to check

        Returns:
            bool: Whether the point is inside
        """
        pass

    def are_inside(self, points: Array) -> Array:
        """Check whether multiple points are inside a region

        Args:
            points (Array): Array of points to check

        Returns:
            Array: Array of booleans, corresponing to whether each point is inside
        """
        return vmap(self.is_inside)(points)


class ConvexPolygon(Shape):
    def __init__(self, vertices: Array) -> None:
        """A convex polygon, defined by a sequence of points in the plane

        Args:
            vertices (Array): The vertices of the polygon, in the order they appear
        """
        self.vertices = vertices
        self.vectors = create_vectors(vertices)

    def is_inside(self, point: Array) -> bool:
        affine_points = point - self.vertices
        return point_inside_poly(self.vectors, affine_points)


class Circle(Shape):
    def __init__(self, center: Array, radius: Scalar):
        self.center = center
        self.radius = radius

    def is_inside(self, point: Array) -> bool:
        return point_inside_circ(self.center, point, self.radius)


@jit
def create_vectors(points: Array) -> Array:
    shifted_points = np.roll(points, -1, axis=0)
    return shifted_points - points


@jit
def get_side(affine_segment: Array, affine_point: Array) -> bool:
    return lax.select(np.cross(affine_segment, affine_point) < 0, 1, -1)


@jit
def point_inside_poly(affine_segments: Array, affine_points: Array) -> bool:
    v_get_side = vmap(get_side)
    side = v_get_side(affine_segments, affine_points)
    side_C = np.roll(side, 1)
    return np.array_equal(side, side_C)


@jit
def point_inside_circ(center: Array, point: Array, radius: Scalar) -> bool:
    dist = np.sum((center - point) ** 2)
    return dist < radius


if __name__ == "__main__":
    vertices = np.asarray([[1, 1], [2, 1], [2, 2], [1, 2]])
    square = ConvexPolygon(vertices)
    circ = Circle(np.asarray([1.5, 1.5]), 1)

    for shape in [square, circ]:
        x = np.linspace(0, 3, 50)
        X, Y = np.meshgrid(x, x)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        # inside_idx = square.are_inside(points)
        inside_idx = circ.are_inside(points)

        def plot_points(coords: Array) -> None:
            plt.scatter(coords[:, 0], coords[:, 1])

        plot_points(points[inside_idx])
        plot_points(points[np.bitwise_not(inside_idx)])
        plt.axis("equal")
        plt.show()

from type_util import Array, Scalar, ArrayLike
import jax.numpy as np
from jax import jit, vmap, lax
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

    def is_on_boundary(self, point: Array) -> bool:
        """Check Whether a given point is on boundary of the region

        Args:
            point (Array): Point to check

        Returns:
            bool: Whether the point is inside
        """
        pass

    def are_on_boundary(self, points: Array) -> Array:
        """Check whether multiple points are on the boundary of a region

        Args:
            points (Array): Array of points to check

        Returns:
            Array: Array of booleans, corresponing to whether each point is on boundary
        """
        return vmap(self.is_on_boundary)(points)


class ConvexPolygon(Shape):
    def __init__(self, vertices: ArrayLike) -> None:
        """A convex polygon, defined by a sequence of points in the plane

        Args:
            vertices (Array): The vertices of the polygon, in the order they appear
        """
        self.vertices = np.asarray(vertices)
        self.vectors = _create_vectors(self.vertices)

    def is_inside(self, point: Array) -> bool:
        return _point_inside_poly(point, self.vectors, self.vertices)

    def is_on_boundary(self, point: Array) -> bool:
        return _point_on_edge_poly(point, self.vectors, self.vertices)


class Circle(Shape):
    def __init__(self, center: Array, radius: Scalar):
        self.center = center
        self.radius = radius

    def is_inside(self, point: Array) -> bool:
        return _point_inside_circ(point, self.center, self.radius)

    def is_on_boundary(self, point: Array) -> bool:
        return _point_on_circumfrance(point, self.center, self.radius)


# We place the methods outside the classes, such
# that they behave properly when jitted
# ----------------- POLYGON METHODS ----------------- #
@jit
def _create_vectors(points: Array) -> Array:
    """Calculates the vector between $x_i$ and $x_{i+1}$

    Args:
        points (Array): Points to turn into vectors

    Returns:
        Array: Vectors, where index i is the vector between i and i+1
    """
    shifted_points = np.roll(points, -1, axis=0)
    return shifted_points - points


@jit
def _point_inside_poly(point: Array, affine_segments: Array, vertices: Array) -> bool:
    """Check whether a point is inside a polygon.

    Args:
        point (Array): Point to check
        affine_segments (Array): Vectors between polygon vertices
        vertices (Array): Vertices of the polygon

    Returns:
        bool: True, if the point lies inside the shape
    """
    affine_points = point - vertices

    # Vectorize to check each segment-point pair
    v_get_side = vmap(_get_side)
    side = v_get_side(affine_segments, affine_points)

    # Roll and array_equal to check all values are the same (all left or all right)
    side_C = np.roll(side, 1)
    return np.array_equal(side, side_C)


@jit
def _point_on_edge_poly(point: Array, affine_segments: Array, vertices: Array):
    """Check whether a point is on edge of a polygon.

    Args:
        point (Array): Point to check
        affine_segments (Array): Vectors between polygon vertices
        vertices (Array): Vertices of the polygon

    Returns:
        bool: True, if the point lies on the edge of a polygon
    """
    v_point_on_segment = vmap(_point_on_segment)
    on_segment = v_point_on_segment(point, vertices)

    # roll and array_equal to check all values are the same(all left or all right)
    on_segment


@jit
def _point_on_segment(point: Array, vertecies: Array):
    """Check wheter point is on line segment

    Args:
        point (Array): Point to check
        affine_segment (Array): vector between polygon vertecies

    Returns:
        _type_: _description_
    """
    pass


@jit
def _get_side(affine_segment: Array, affine_point: Array) -> bool:
    """Calculate which side of the line a point lies.

    We use the cross product in order to check which side of the line we are on

    lax.select(pred, on_true, on_false)
    is equivalent to
    if pred:
        return on_true
    else:
        return on_false

    Args:
        affine_segment (Array): Vector segment
        affine_point (Array): Point - origin

    Returns:
        bool: 1, if the point lies to the left of the line, -1 else.
    """
    return lax.select(np.cross(affine_segment, affine_point) < 0, 1, -1)


# ----------------- CIRCLE METHODS ----------------- #
@jit
def _point_inside_circ(point: Array, center: Array, radius: Scalar) -> bool:
    """Check if the point is inside the circle

    Args:
        point (Array): Point to check
        center (Array): Center of the circle
        radius (Scalar): Radius of the circle

    Returns:
        bool: True, if the point is contained in the circle
    """
    dist = np.sum((center - point) ** 2)
    return dist < radius**2


@jit
def _point_on_circumfrance(point: Array, center: Array, radius: Scalar) -> bool:
    """Check if the point is on the circumfrance of the circle

    Args:
        point (Array): Point to check
        center (Array): Center of the circle
        radius (Scalar): Radius of the circle

    Returns:
        bool: True, if the point is contained in circumfrance circle
    """
    dist = np.sum((center - point) ** 2)
    return np.isclose(dist, radius**2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vertices = np.asarray([[1, 1], [2, 1], [2, 2], [1, 2]])
    square = ConvexPolygon(vertices)
    circ = Circle(np.asarray([1.5, 1.5]), 1)

    shapes: list[Shape] = [square, circ]

    for shape in shapes:
        x = np.linspace(0, 3, 50)
        X, Y = np.meshgrid(x, x)
        points = np.vstack([X.ravel(), Y.ravel()]).T

        inside_idx = shape.are_inside(points)

        def plot_points(coords: Array) -> None:
            plt.scatter(coords[:, 0], coords[:, 1])

        plot_points(points[inside_idx])
        plot_points(points[np.bitwise_not(inside_idx)])
        plt.axis("equal")
        plt.show()

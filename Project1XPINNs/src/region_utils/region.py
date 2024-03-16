from region_utils.shapes import Shape
import numpy as onp
import jax.numpy as np
from type_util import Array, Callable
from jax import vmap, jit
from collections import defaultdict
import json
from pathlib import Path
import matplotlib.pyplot as plt


class Subdomain:
    def __init__(
        self,
        composition: list[Shape],
        subtraction: list[Shape] = None,
    ) -> None:
        """
        Initializes a Subdomain, based on Shapes.

        Args:
            composition (list[Shape]):
                A list of Shape objects representing the composition of the subdomain.
            subtraction (list[Shape], optional):
                A list of Shape objects representing the subtraction from the subdomain. Defaults to None.
        """
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
        """Get the points that are inside the subdomain

        Args:
            points (Array): Total points of the domain

        Returns:
            Array: The valid points within the subdomain
        """
        are_valid = np.zeros(points.shape[0], dtype=bool)

        for shape in self.composition:
            additive = shape.are_inside(points)
            are_valid = bool_or(are_valid, additive)

        for shape in self.subtraction:
            subtractive = shape.are_inside(points)
            are_valid = bool_sub(are_valid, subtractive)

        return points[are_valid]

    def create_boundary(self, num_points: int) -> Array:
        """Create the boundary points for the subdomain

        Args:
            num_points (int): The number of boundary points to create

        Returns:
            Array: Boundary points for the subdomain
        """
        n_bound = len(self.boundaries)
        if n_bound == 0:
            return np.array([[],[]]).T

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
    """Bitwise or for boolean arrays

    Args:
        vec1 (Array): First boolean array
        vec2 (Array): Second boolean array

    Returns:
        Array: Bitwise or of the two arrays
    """
    vec1 = np.logical_or(vec1, vec2)
    return vec1


@jit
def bool_sub(vec1: Array, vec2: Array) -> Array:
    """Bitwise subtraction for boolean arrays

    Args:
        vec1 (Array): First boolean array
        vec2 (Array): Second boolean array

    Returns:
        Array: Bitwise result of vec1 and not vec2
    """
    vec1 = np.logical_and(vec1, np.invert(vec2))
    return vec1


class Domain:
    def __init__(self, subdomains: list[Subdomain]) -> None:
        """Domain object, containing subdomains. Used to create the domain for the XPINN

        Args:
            subdomains (list[Subdomain]): Subdomains that make up the domain
        """
        self.subdomains = subdomains
        self.pinn_points: dict[int, dict[str, Array]] = {
            i: {} for i in range(len(subdomains))
        }
        self.interfaces: defaultdict[tuple[int, int], list[Array]] = defaultdict(list)

    def create_interior(self, n: int, lower: list[int], upper: list[int]) -> None:
        """Create the interior points of the domain, dividing them among the subdomains

        Args:
            n (int): Number of points to create
            lower (list[int]): Lower bounds of the domain
            upper (list[int]): Upper bounds of the domain
        """
        points = []
        for low, high in zip(lower, upper):
            points.append(onp.random.uniform(low, high, n))
        points = np.column_stack(points)

        for i, subdomain in enumerate(self.subdomains):
            args = self.pinn_points[i]
            valid_points = subdomain.are_inside(points)
            args["interior"] = valid_points
            print(valid_points.shape)

    def create_boundary(self, n: int) -> None:
        """Generate the boundary points for the subdomains

        Args:
            n (int): Number of boundary points to create
        """
        # Divide the number of points evenly among the subdomains
        total_length = sum(subdomain.boundary_len for subdomain in self.subdomains)

        for i, subdom in enumerate(self.subdomains):
            args = self.pinn_points[i]
            num_points = int(subdom.boundary_len / total_length * n)
            #print(num_points)
            args["boundary"] = subdom.create_boundary(num_points)

    def create_interface(
        self, n: int, indexes: tuple[int, int], points: tuple[Array, Array], add_noise: bool = False
    ) -> None:
        """Generate an interface between two subdomains

        Args:
            n (int): Number of points to create
            indexes (tuple[int, int]): Indexes of the subdomain
            points (tuple[Array, Array]): Start and end points of the interface
        """
        start, end = points
        i, j = sorted(indexes)

        inter_points = np.linspace(start, end, n)
        #if add_noise:
            #### TODO: 
        self.interfaces[(i, j)].append(inter_points)

    def find_overlap(
            self, indexes: tuple[int,int]
    ) -> None:
        """Pick out overlapping points to use for interface
        """
        i, j = sorted(indexes)
        subdomain_i_points = onp.concatenate((self.pinn_points[i]["interior"] , self.pinn_points[i]["boundary"]))
        subdomain_j_points = onp.concatenate((self.pinn_points[j]["interior"] , self.pinn_points[j]["boundary"]))
        nrows,ncols = subdomain_i_points.shape

        dtype= {'names':[f'f{i}' for i in range(ncols)],
                'formats': ncols*[subdomain_i_points.dtype]}
        overlap =onp.intersect1d(subdomain_i_points.view(dtype),subdomain_j_points.view(dtype))
        overlap = overlap.view(subdomain_i_points.dtype).reshape(-1,ncols)

        self.interfaces[(i,j)].append(overlap)

    def create_testing_data(self, n: int, lower: list[int], upper: list[int]) -> None:
        """Create the testing data for the XPINN

        Args:
            n (int): Number of points per axes
            lower (list[int]): Lower bounds of the domain
            upper (list[int]): Upper bounds of the domain
        """
        points = []
        for low, high in zip(lower, upper):
            points.append(np.linspace(low, high, n))
        points = np.column_stack([x.ravel() for x in np.meshgrid(*points)])

        self.testing_points = defaultdict(dict)
        for i, subdomain in enumerate(self.subdomains):
            args = self.testing_points[i]
            valid_points = subdomain.are_inside(points)
            args["interior"] = valid_points
            args["boundary"] = np.array([])

    def write_to_file(self, filename: str | Path, train: bool = True) -> None:
        """Write the domain data to a JSON file

        Args:
            filename (str | Path): File to write the data to
            train (bool, optional): Whether to write the training data. Defaults to True.
        """
        filename = Path(filename)
        if not filename.suffix == ".json":
            raise ValueError("Filename must have a .json extension")

        if train:
            main_data = self.pinn_points
            interfaces = self.interfaces
        else:
            main_data = self.testing_points
            interfaces = {}

        with open(filename, "w") as outfile:
            data = {"XPINNs": [], "Interfaces": []}
            for i, subdomain in enumerate(self.subdomains):
                args = main_data[i]
                subdomain_data = {
                    "Internal points": args["interior"].tolist(),
                    "Boundary points": args["boundary"].tolist(),
                }
                data["XPINNs"].append(subdomain_data)

            for key, val in interfaces.items():
                data["Interfaces"].append(
                    {
                        "XPINNs": key,
                        "Points": sum([point.tolist() for point in val], []),
                    }
                )
            json.dump(data, outfile)

    def plot(self, train: bool = True) -> None:
        """Plot the domain points.

        Args:
            train (bool, optional): Whether to use the training or testing data. Defaults to True.
        """
        if train:
            data = self.pinn_points
            interfaces = self.interfaces
        else:
            data = self.testing_points
            interfaces = {}

        for i, args in enumerate(data.values()):
            self._plot_array(args["interior"], f"Interior {i}")
            self._plot_array(args["boundary"], f"Boundary {i}")

        for key, val in interfaces.items():
            points = sum([point.tolist() for point in val], [])
            self._plot_array(np.array(points), f"Interface {key}")

        plt.legend()
        plt.title("Domain Points")
        plt.show()

    def _plot_array(self, arr: Array, label: str) -> None:
        """Plot an array of points

        Args:
            arr (Array): Array of points
            label (str): Label for the plot
        """
        if arr.size == 0:
            return

        plt.scatter(*arr.T, label=label)

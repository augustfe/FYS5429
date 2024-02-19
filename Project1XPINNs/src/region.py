from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Region:
    def __init__(
        self, constraint_file: str | Path, seed: int = 0, use_seed: bool = True
    ):
        self.constraint_file = Path(constraint_file)
        self.areas: list[Area] = []
        self.interfaces: list[Interface] = []
        self.limits = []
        self.num_boundaries: int = 0
        self.num_interfaces: int = 0

        if use_seed:
            np.random.seed(seed)

        self.read_constraint_file()

    def read_constraint_file(self):
        """Read constraints from file and generate corresponding regions.

        File must be of the format
        # of arguments
            low high
        # of areas
            # of constraints for area
            constraits of the form
            C0 C1 .. Cn K
            meaning
            x0 * C0 + x1 * C1 + ... + xn * Cn <= K
        # of boundaries
            # of boundaries for region
                x00 x10 ... xn0 x01 x11 ... xn1
                translating to the start point
                [x00, x10, ..., xn0]
                and the end point
                [x01, x11, ..., xn1]
        """
        with open(self.constraint_file, "r") as infile:
            # Number of arguments
            n_args = int(infile.readline())
            for _ in range(n_args):
                # low, high constraint for each arg
                lower, higher = map(float, infile.readline().split())
                self.limits.append((lower, higher))

            # Number of areas
            N = int(infile.readline())
            for _ in range(N):
                # Number of constraints for area
                M = int(infile.readline())
                curr_area = Area()
                for j in range(M):
                    x_C, t_C, const = map(float, infile.readline().split())
                    curr_area.add_ineq(x_C, t_C, const)
                self.areas.append(curr_area)

            # Number of boundaries
            N = int(infile.readline())
            for i in range(N):
                # Number of boundaries per region
                m = int(infile.readline())
                self.num_boundaries += 1 if m != 0 else 0
                for _ in range(m):
                    # Start and end points of boundary
                    values = list(map(float, infile.readline().split()))
                    start, end = self.two_points(values, n_args)
                    self.areas[i].add_boundary(start, end)

            # Number of interfaces
            N = int(infile.readline())
            for i in range(N):
                a, b = map(int, infile.readline().split())
                values = list(map(float, infile.readline().split()))
                start, end = self.two_points(values, n_args)
                new_Interface = Interface(
                    self.areas[a], self.areas[b], start, end, a, b
                )
                self.interfaces.append(new_Interface)
                self.num_interfaces += 1

    def two_points(self, values: list, n_args: int) -> tuple[np.ndarray]:
        """Convert list of values into corresponding start and end point

        Args:
            values (list): List of variable values
            n_args (int): Number of arguments

        Returns:
            tuple[list,list]: start, end
        """
        start = np.asarray(values[n_args:])
        end = np.asarray(values[:n_args])

        return start, end

    def add_points(self, X: np.ndarray):
        """Add array of points to the corresponding areas

        Args:
            X (np.ndarray): Array of all points
        """
        for point in X:
            for area in self.areas:
                if not area.is_in_area(point):
                    continue
                area.add_point(point)
                break
            else:
                print(f"WARNING: Can't place ({point}) in an area!!")

    def test_points(self, N: int = 2000) -> np.ndarray:
        """Populate region with test-points.

        Args:
            N (int, optional): Number of points to generate. Defaults to 2000.

        Returns:
            np.ndarray: Array of generated points.
        """
        arrs = []
        for lower, higher in self.limits:
            arrs.append(np.random.uniform(lower, higher, (N, 1)))

        X = np.hstack(arrs)
        self.add_points(X)

        return X

    def test_boundary_and_interface(self, N: int = 200) -> np.ndarray:
        points_per = N // (self.num_boundaries + self.num_interfaces)
        total_points = []
        for area in self.areas:
            bounds = area.boundary_constraints
            if not len(bounds):
                continue

            bounds = np.asarray(bounds)
            num_b = len(bounds)
            idx = np.random.choice(num_b, points_per)
            scales = np.random.uniform(0, 1, size=(points_per, 2))

            chosen_points = bounds[idx]
            starts = chosen_points[:, 0, :]
            ends = chosen_points[:, 1, :]

            print(starts)
            print(ends)
            print(ends - starts)
            print(starts * scales)

            new_points = starts + (ends - starts) * scales
            area.bound_points = new_points
            total_points.append(new_points)

        for interface in self.interfaces:
            new_points = interface.add_points(points_per)
            total_points.append(new_points)

        return total_points

    def test_boundary_points(self, N: int = 200) -> np.ndarray:
        placed = 0
        num_areas = len(self.areas)
        new_points = []
        while placed <= N:
            # Choose area
            chosen_area = self.areas[np.random.choice(num_areas)]
            n_bound = len(chosen_area.boundary_constraints)
            # Check area has boundary constraints
            if n_bound == 0:
                continue

            # Choose boundary
            chosen_bound = np.random.choice(n_bound)
            start, end = chosen_area.boundary_constraints[chosen_bound]

            # Generate new point through interpolation
            dist = np.random.uniform(0, 1)
            new_point = start + (end - start) * dist
            chosen_area.bound_points.append(new_point)

            new_points.append(new_point)
            placed += 1

        for area in self.areas:
            area.bound_points = np.asarray(area.bound_points)

        return np.asarray(new_points)

    def plot_points(self):
        """Plot the points in the corresponding areas of the region."""
        for i, area in enumerate(self.areas):
            area.points_to_df()
            plt.scatter(
                area.df_points["x"], area.df_points["t"], s=2.5, label=f"Region {i+1}"
            )
            bounds = area.bound_points
            plt.scatter(
                bounds[:, 0],
                bounds[:, 1],
                s=2.5,
                # marker="x",
                # alpha=0.4,
                label=f"Boundary {i + 1}",
            )

        for interface in self.interfaces:
            plt.scatter(
                interface.points[:, 0],
                interface.points[:, 1],
                s=2.5,
                label=f"Interface {interface.idx_left}{interface.idx_right}",
            )
        plt.legend()
        plt.show()


class Area:
    def __init__(self):
        self.constraints: list[list[float, float, float]] = []
        self.points: list[list[float, float]] = []
        self.df_points: pd.DataFrame = None
        self.coeff_matrix: np.ndarray = None
        self.const_matrix: np.ndarray = None
        self.boundary_constraints: list[tuple[list]] = []
        self.bound_points = []

    def add_ineq(self, x_C: float, t_C: float, const: float) -> None:
        """Add inequality for 2D

        Must be of the form
            x * x_C + t * t_C <= const

        Args:
            x (float): x coefficient
            t (float): t coefficient
            const (float): right side of inequality
        """
        vals = (x_C, t_C, const)
        for val in vals:
            if not isinstance(val, float):
                raise ValueError(f"{val} must be float.")
        self.constraints.append([x_C, t_C, const])

    def add_boundary(self, start_point: np.ndarray, end_point: np.ndarray) -> None:
        """Add pair of points defining a boundary.

        e.g.:
            [1, 1] to [1, 0]

        Args:
            start_point (np.ndarray): Start point for boundary
            end_point (np.ndarray): End point for boundary
        """
        self.boundary_constraints.append((start_point, end_point))

    def read_ineq_from_file(self, filename: str | Path) -> None:
        """Read the constraints of the region from file.

        File must be of the form:
            x_C t_C const
            x_C t_C const
            ...

        Args:
            filename (str | Path): filename containing the constraints

        Raises:
            FileNotFoundError: if the file given is not a file or does not exist
        """
        file = Path(filename)

        if not file.is_file():
            raise FileNotFoundError(f"Object pointed to by {file} is not a file.")

        with open(file, "r") as infile:
            for line in infile:
                x_C, t_C, const = map(float, line.split())
                self.add_ineq(x_C, t_C, const)

    def read_ineq_from_list(self, values: list[list[float, float, float]]) -> None:
        """Add inequalities from list of values

        Tuples must be of the form
            (x_C, t_C, const)

        Args:
            values (list[tuple[float, float, float]]): List of constraints

        Raises:
            ValueError: if wrong number of arguments in a constraint
        """
        for value in values:
            value = list(value)

            if not len(value) == 3:
                raise ValueError(
                    f"Wrong number of values given, {len(value)} instead of 3."
                )

            if value not in self.constraints:
                self.constraints.append(value)

    def is_in_area(self, point: np.ndarray) -> bool:
        """Check if the point is in the region

        Args:
            x (float): x value
            t (float): t value

        Returns:
            bool: Whether the point is within the given area
        """
        if self.coeff_matrix is None:
            tmp = np.asarray(self.constraints)
            self.coeff_matrix = tmp[:, :-1]
            self.const_matrix = tmp[:, -1]

        return np.all(self.coeff_matrix @ point <= self.const_matrix)

    def add_point(self, point: np.ndarray) -> None:
        """Add a point to the area

        Args:
            x (float): x value of point
            t (float): t value of point
        """
        self.points.append(point)

    def points_to_df(self) -> None:
        """Convert points from list to dataframe."""
        if self.df_points is not None:
            print("WARNING: Overwriting previous points.")
        self.df_points = pd.DataFrame(self.points, columns=["x", "t"], dtype=float)

    def write_points_to_file(self, filename: str | Path) -> None:
        """Write dataframe of points to file

        Args:
            filename (str | Path): filename of where to save points.
        """
        file = Path(filename)

        if self.df_points is None:
            self.points_to_df()

        self.df_points.to_csv(file)


class Interface:
    def __init__(
        self,
        left_area: Area,
        right_area: Area,
        start_point: np.ndarray,
        end_point: np.ndarray,
        idx_left: int,
        idx_right: int,
    ):
        self.left: Area = left_area
        self.right: Area = right_area
        self.start = start_point
        self.end = end_point
        self.points = None
        self.idx_left = idx_left
        self.idx_right = idx_right

    def add_points(self, n: int = 100) -> np.ndarray:
        """Interpolates n points between endpoints

        Args:
            n (int): Number of points to place
        """
        self.points = np.linspace(self.start, self.end, n)
        return self.points

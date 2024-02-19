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

        if use_seed:
            np.random.seed(seed)

        self.read_constraint_file()

    def read_constraint_file(self):
        """Read constraints from file and generate corresponding regions."""
        with open(self.constraint_file, "r") as infile:
            N = int(infile.readline())
            for i in range(N):
                M = int(infile.readline())
                curr_area = Area()
                for j in range(M):
                    x_C, t_C, const = map(float, infile.readline().split())
                    curr_area.add_ineq(x_C, t_C, const)
                self.areas.append(curr_area)

    def add_points(self, X: np.ndarray):
        """Add array of points to the corresponding areas

        Args:
            X (np.ndarray): Array of all points
        """
        for x, t in X:
            for area in self.areas:
                if not area.is_in_area(x, t):
                    continue
                area.add_point(x, t)
                break
            else:
                print(f"WARNING: Can't place ({x}, {t}) in an area!!")

    def test_points(
        self, xlim: tuple[float] = (-1, 1), tlim: tuple[float] = (0, 1), N: int = 2000
    ) -> np.ndarray:
        """Populate region with test-points.

        Args:
            xlim (tuple[float], optional): (min, max) for x values. Defaults to (-1, 1).
            tlim (tuple[float], optional): (min, max) for t values. Defaults to (0, 1).
            N (int, optional): Number of points to generate. Defaults to 2000.

        Returns:
            np.ndarray: Array of generated points.
        """
        x_vals = np.random.uniform(xlim[0], xlim[1], (N, 1))
        t_vals = np.random.uniform(tlim[0], tlim[1], (N, 1))
        X = np.hstack([x_vals, t_vals])

        self.add_points(X)

        return X

    def plot_points(self):
        """Plot the points in the corresponding areas of the region."""
        for i, area in enumerate(self.areas):
            area.points_to_df()
            plt.scatter(
                area.df_points["x"], area.df_points["t"], s=2.5, label=f"Region {i+1}"
            )
        plt.legend()
        plt.show()


class Area:
    def __init__(self):
        self.constraints: list[tuple[float, float, float]] = []
        self.points: list[list[float, float]] = []
        self.df_points: pd.DataFrame = None

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
        self.constraints.append((x_C, t_C, const))

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

    def read_ineq_from_list(self, values: list[tuple[float, float, float]]) -> None:
        """Add inequalities from list of values

        Tuples must be of the form
            (x_C, t_C, const)

        Args:
            values (list[tuple[float, float, float]]): List of constraints

        Raises:
            ValueError: if wrong number of arguments in a constraint
        """
        for value in values:
            value = tuple(value)

            if not len(value) == 3:
                raise ValueError(
                    f"Wrong number of values given, {len(value)} instead of 3."
                )

            if value not in self.constraints:
                self.constraints.append(value)

    def is_in_area(self, x: float, t: float) -> bool:
        """Check if the point is in the region

        Args:
            x (float): x value
            t (float): t value

        Returns:
            bool: Whether the point is within the given area
        """
        for x_C, t_C, const in self.constraints:
            if not x * x_C + t * t_C <= const:
                return False

        return True

    def add_point(self, x: float, t: float) -> None:
        """Add a point to the area

        Args:
            x (float): x value of point
            t (float): t value of point
        """
        self.points.append([x, t])

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

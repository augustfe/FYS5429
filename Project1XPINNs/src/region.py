from pathlib import Path
import pandas as pd


class Area:
    def __init__(self):
        self.constraints: list[tuple[float, float, float]] = []
        self.points: list[list[float, float]] = []

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

    def write_points_to_file(self, filename: str | Path):
        file = Path(filename)

        df = pd.DataFrame(self.points, columns=["x", "t"], dtype=float)
        df.to_csv(file)

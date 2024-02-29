from pathlib import Path
from region import Region


def create_JSON(input_file: str | Path):
    input_file = Path(input_file)

    total_region = Region(input_file)
    total_region.test_points()
    total_region.test_boundary_and_interface()
    print(total_region.areas)


if __name__ == "__main__":
    create_JSON(Path(__file__).parent / "advection_constraints.txt")

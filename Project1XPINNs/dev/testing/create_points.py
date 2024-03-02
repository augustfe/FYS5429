from pathlib import Path
from region import Region
import json


def create_JSON(input_file: str | Path):
    input_file = Path(input_file)

    total_region = Region(input_file)
    total_region.test_points()
    total_region.test_boundary_and_interface()

    vals = {"XPINNs": [], "Interfaces": []}

    for area in total_region.areas:
        pinn_dict = {"Internal points": [], "Boundary points": []}
        pinn_dict["Internal points"] = area.points
        pinn_dict["Boundary points"] = area.bound_points

        vals["XPINNs"].append(pinn_dict)

    for interface in total_region.interfaces:
        interface_dict = {"XPINNs": [], "Points": []}
        interface_dict["XPINNs"] = interface.indices
        interface_dict["Points"] = interface.points.tolist()

        vals["Interfaces"].append(interface_dict)

    with open(input_file.with_suffix(".json"), "w") as outfile:
        outfile.write(json.dumps(vals, indent=1))


if __name__ == "__main__":
    create_JSON(Path(__file__).parent / "advection_constraints.txt")

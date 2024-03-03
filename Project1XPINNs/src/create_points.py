from pathlib import Path
from region import Region
import json


def create_JSON(
    input_file: str | Path, interior: int = 2000, edge: int = 200, test: bool = False
):
    input_file = Path(input_file)

    total_region = Region(input_file)
    total_region.test_points(interior)
    total_region.test_boundary_and_interface(edge)

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

    out = Path(input_file.stem + "_test" * test).with_suffix(".json")
    out = input_file.parent / out
    with open(out, "w") as outfile:
        outfile.write(json.dumps(vals))


if __name__ == "__main__":
    infile = Path(__file__).parents[1] / "data/advection_constraints.txt"
    create_JSON(infile)
    create_JSON(infile, 20000, 2000, True)

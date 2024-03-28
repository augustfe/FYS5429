from region_utils.region import Subdomain, Domain
from region_utils.shapes import ConvexPolygon
import jax.numpy as np

total_vertices = np.asarray(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0.25, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
        [0.75, 0.25],
    ]
)

region_idxs = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]])
boundary_idxs = [[0, 1, 2, 3], []]

outer_rectangle = ConvexPolygon(
    total_vertices[region_idxs[0]],
    boundary_idxs[0],
)
inner_rectangle = ConvexPolygon(
    total_vertices[region_idxs[1]],
    boundary_idxs[1],
)

subdomain1 = Subdomain([outer_rectangle], subtraction=[inner_rectangle])
subdomain2 = Subdomain([inner_rectangle])

domain = Domain([subdomain1, subdomain2])


domain.create_interior(900, [0, 0], [1, 1])
domain.create_boundary(120)

idx = [4, 5, 6, 7, 4]
for i in range(4):
    domain.create_interface(
        20,
        (0, 1),
        (total_vertices[idx[i]], total_vertices[idx[i + 1]]),
    )

from utils import data_path

domain.write_to_file(data_path / "poisson_train.json")

# adaptive.py
import numpy as np
from scipy.linalg import solve

def node_moving_1d(coordinate, estimated_error, boundary_nodes_id, ratio_nodal_distance=5, relaxation=1.0):
    b = np.log(ratio_nodal_distance) / np.log(np.max(estimated_error) / np.min(estimated_error))
    adjusted_error = estimated_error ** b
    ke = 0.5 * (adjusted_error[:-1] + adjusted_error[1:])

    k = np.diag(np.hstack([0, ke])) + np.diag(np.hstack([ke, 0]))
    k -= np.diag(ke, -1)
    k -= np.diag(ke, 1)
    k += np.eye(k.shape[0]) * 1e-6  # regularization

    f = np.zeros(len(coordinate))
    for i in boundary_nodes_id:
        f -= k[:, i] * coordinate[i]
    k = np.delete(k, boundary_nodes_id, axis=0)
    k = np.delete(k, boundary_nodes_id, axis=1)
    f = np.delete(f, boundary_nodes_id)

    new_coords = solve(k, f)
    coords = coordinate.copy()
    interior_ids = [i for i in range(len(coords)) if i not in boundary_nodes_id]
    coords[interior_ids] = new_coords

    return coordinate + relaxation * (coords - coordinate)

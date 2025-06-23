import numpy as np

def random_resampling_1d(
    coordinate: np.ndarray,
    boundary_nodes_id: np.ndarray,
    **kwargs  # absorb other arguments like ratio_nodal_distance, stifness
) -> np.ndarray:
    """
    Randomly resamples all non-boundary nodes within [x_min, x_max].

    Parameters
    ----------
    coordinate : np.ndarray
        Current 1D node coordinates.

    boundary_nodes_id : np.ndarray
        Indices of fixed boundary nodes.

    Returns
    -------
    np.ndarray
        New coordinates array with random interior resampling.
    """
    num_nodes = len(coordinate)
    x_min, x_max = coordinate.min(), coordinate.max()

    interior_ids = [i for i in range(num_nodes) if i not in boundary_nodes_id]
    new_coords = coordinate.copy()

    # Random uniform resampling for interior nodes
    resampled = np.random.uniform(x_min, x_max, size=len(interior_ids))

    new_coords[interior_ids] = resampled
    return new_coords
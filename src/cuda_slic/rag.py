import numpy as np

from ._rag import _extract_neighbours_2d, _extract_neighbours_3d


def create_rag(
    rlabels,
    connectivity=1,
    min_boundary=None,
    norm_counts="unit",
    margin=0,
    return_counts=False,
):
    """Creates a Region Adjacency Graph between superpixels.

    Parameters
    ----------
    rlabels : 2D or 3D ndarray
        Input superpixel labels for a 2D or 3D image.
    connectivity : int, optional
        Order of the neighbours, which must have a value of ${1, 2}$.
        For a 2D image `1` corresponds to 4-connected pixels while `2`
        corresponds to 8-connected pixels. For a 3D image `1` and `2`
        correspond to 6-connected and 18-connected voxels respectively.
    min_boundary : int, optional
        Minimum number of boundary pixels that a pair of superpixels
        need to share in order to consider them neighbours.
    norm_counts : string, optional
        Normalize the number of boundary pixels that any pair of
        superpixels share. Possible values are 'unit' which performs
        unit normalization as `counts /= counts.max()` and
        `margin` which performs `counts = np.max(counts, margin) / margin`.
    margin : int, optional
        An heuristic number of boundary pixels required to consider two
        neighbouring superpixels *perfect neighbours*. Used only if
        `norm_counts='margin'`.

    Returns
    -------
    graph : RAG (subclass of networkx.Graph)
        An undirected graph containing all the superpixels as nodes
        and the connections between them as edges. The number of
        boundary pixels that a pair of superpixels share is returned
        as an edge property: `boundary`.

    """
    n_labels = rlabels.max() + 1

    if (rlabels.ndim == 2 and connectivity not in (4, 8)) or (
        rlabels.ndim == 3 and connectivity not in (6, 18, 26)
    ):
        raise Exception("Only {1, 2} values are supported for `connectivity`")

    if rlabels.ndim == 2:
        nodes, neighbors = _extract_neighbours_2d(rlabels, connectivity)
    else:
        nodes, neighbors = _extract_neighbours_3d(rlabels, connectivity)

    nodes = np.tile(nodes, connectivity // 2)
    neighbors = neighbors.flatten("f")

    idx = (neighbors != -1) & (neighbors != nodes)
    nodes = nodes[idx]
    neighbors = neighbors[idx]

    idx = nodes > neighbors
    nodes[idx], neighbors[idx] = neighbors[idx], nodes[idx]

    n_nodes = np.int64(rlabels.max() + 1)
    crossing_hash = nodes + neighbors.astype(np.int64) * n_nodes
    if min_boundary is not None or return_counts:
        unique_hash, counts = np.unique(crossing_hash, return_counts=True)
    else:
        unique_hash = np.unique(crossing_hash)

    neighbors = np.c_[unique_hash % n_nodes, unique_hash // n_nodes]
    neighbors = neighbors.astype(np.int32)

    if min_boundary is not None:
        idx = counts >= min_boundary
        neighbors = neighbors[idx]
        counts = counts[idx]

    if return_counts:
        if norm_counts == "unit":
            counts = counts / float(counts.max())
        elif norm_counts == "margin":
            counts = np.minimum(counts, margin) / float(margin)
        return neighbors, counts.astype(np.float32)

    return neighbors

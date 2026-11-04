"""Strict loading and aligned cropping of optional monomer geometry features."""

from pathlib import Path

import torch


def load_geometry(root, uid, sequence, required, loader):
    if root is None:
        if required:
            raise ValueError("Required geometry feature root is missing")
        return None
    path = Path(root) / f"{uid}.pt"
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    data = loader(path)
    if data["sequence"] != sequence:
        raise ValueError(f"{uid}: geometry sequence mismatch")
    n = len(sequence)
    edge = torch.as_tensor(data["geometry_edge_index"], dtype=torch.long)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError(f"{uid}: invalid geometry index")
    e = edge.shape[1]
    shapes = {"geometry_vectors": (n, 8, 3), "geometry_edge_features": (e, 30),
              "geometry_edge_vectors": (e, 1, 3), "geometry_edge_scores": (e,)}
    result = {"geometry_edge_index": edge}
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= n):
        raise ValueError(f"{uid}: out-of-range geometry edges")
    for key, shape in shapes.items():
        value = torch.as_tensor(data[key], dtype=torch.float32)
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"{uid}: invalid {key}")
        result[key] = value
    return result


def crop_geometry(data, crop, n_res):
    if data is None or crop == slice(None):
        return data
    start, stop, step = crop.indices(n_res)
    if step != 1:
        raise ValueError("Geometry cropping requires contiguous residues")
    edge = data["geometry_edge_index"]
    keep = ((edge >= start) & (edge < stop)).all(dim=0)
    return {"geometry_vectors": data["geometry_vectors"][start:stop],
            "geometry_edge_index": edge[:, keep] - start,
            **{key: data[key][keep] for key in ("geometry_edge_features", "geometry_edge_vectors", "geometry_edge_scores")}}

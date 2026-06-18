"""Dataset for PPC complete protein features."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.is_floating_point():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def _norm_chain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _feature_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


def _discover_feature_paths(features_root: Path, ids: list[str] | None = None) -> list[Path]:
    if ids is None:
        return sorted(features_root.glob("*/*_protein.pt"))
    paths: list[Path] = []
    for pdb_id in ids:
        pdb_id = pdb_id.lower()
        path = features_root / pdb_id / f"{pdb_id}_protein.pt"
        if path.exists():
            paths.append(path)
    return sorted(paths)


def _chain_ids_to_int(chain_ids: list[Any]) -> torch.Tensor:
    mapping: OrderedDict[str, int] = OrderedDict()
    out: list[int] = []
    for chain_id in chain_ids:
        key = _norm_chain(chain_id)
        if key not in mapping:
            mapping[key] = len(mapping)
        out.append(mapping[key])
    return torch.tensor(out, dtype=torch.long)


def _compute_backbone_vectors(all_atom_coords: list[Any], ca_coords: torch.Tensor) -> torch.Tensor:
    n_res = ca_coords.shape[0]
    vectors = torch.zeros(n_res, 4, 3, dtype=torch.float32)
    for idx, atoms in enumerate(all_atom_coords):
        atom_tensor = _as_tensor(atoms, dtype=torch.float32)
        if atom_tensor.ndim != 2 or atom_tensor.shape[0] == 0:
            continue
        ca = ca_coords[idx]
        vectors[idx, 0] = atom_tensor.mean(dim=0) - ca
        take = min(3, atom_tensor.shape[0])
        vectors[idx, 1 : 1 + take] = atom_tensor[:take] - ca
    return vectors


def _locate_esm(esm_root: Path, pdb_id: str) -> Path | None:
    candidates = (
        esm_root / pdb_id / f"{pdb_id}_esm2.pt",
        esm_root / pdb_id / f"{pdb_id}_protein.pt",
        esm_root / f"{pdb_id}_esm2.pt",
        esm_root / f"{pdb_id}_protein.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_labels(label_root: Path | None, pdb_id: str, n_res: int) -> torch.Tensor:
    if label_root is None:
        return torch.full((n_res,), -100, dtype=torch.long)
    candidates = (
        label_root / pdb_id / f"{pdb_id}_labels.pt",
        label_root / pdb_id / f"{pdb_id}_contact.pt",
        label_root / f"{pdb_id}_labels.pt",
        label_root / f"{pdb_id}.pt",
    )
    for path in candidates:
        if not path.exists():
            continue
        data = _torch_load(path)
        if isinstance(data, dict):
            value = data.get("labels", data.get("is_contact", data.get("contact_labels")))
        else:
            value = data
        labels = torch.as_tensor(value, dtype=torch.long)
        if labels.shape[0] != n_res:
            raise ValueError(f"{pdb_id}: label length {labels.shape[0]} != n_res {n_res}")
        return labels
    return torch.full((n_res,), -100, dtype=torch.long)


def _slice_first_dim(value: Any, residue_slice: slice) -> Any:
    if isinstance(value, torch.Tensor):
        return value[residue_slice]
    try:
        return value[residue_slice]
    except TypeError:
        return list(value)[residue_slice]


class ProteinFeatureDataset(Dataset):
    """Load PPC protein_v4 features, optional ESM features, and optional labels."""

    def __init__(
        self,
        features_root: str | Path,
        esm_root: str | Path | None = None,
        label_root: str | Path | None = None,
        ids: list[str] | None = None,
        max_residues: int | None = None,
        crop_mode: str = "none",
        positive_crop_prob: float = 0.7,
        seed: int = 0,
    ) -> None:
        self.features_root = Path(features_root)
        self.esm_root = Path(esm_root) if esm_root else None
        self.label_root = Path(label_root) if label_root else None
        self.max_residues = int(max_residues) if max_residues else None
        self.crop_mode = crop_mode
        self.positive_crop_prob = float(positive_crop_prob)
        self.rng = random.Random(seed)
        self.feature_paths = _discover_feature_paths(self.features_root, ids)
        if not self.feature_paths:
            raise FileNotFoundError(f"No *_protein.pt files under {self.features_root}")

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.feature_paths[index]
        pdb_id = _feature_id(path)
        data = _torch_load(path)

        ca_raw = data["ca_coords"]
        n_res = int(ca_raw.shape[0])
        labels = _load_labels(self.label_root, pdb_id, n_res)
        crop_start: int | None = None
        crop_end: int | None = None
        if self.max_residues is not None and n_res > self.max_residues and self.crop_mode != "none":
            crop_start = self._select_crop_start(labels, n_res, self.max_residues)
            crop_end = crop_start + self.max_residues
            residue_slice = slice(crop_start, crop_end)
        else:
            residue_slice = slice(None)

        ca_coords = _as_tensor(_slice_first_dim(ca_raw, residue_slice), dtype=torch.float32)
        chain_ids = list(_slice_first_dim(list(data["chain_ids"]), residue_slice))
        all_atom_coords = list(_slice_first_dim(list(data["all_atom_coords"]), residue_slice))
        labels = labels[residue_slice]
        item: dict[str, Any] = {
            "pdb_id": pdb_id,
            "protein_physchem": _as_tensor(_slice_first_dim(data["physchem_features"], residue_slice), dtype=torch.float32),
            "protein_spatial_scalar": _as_tensor(
                _slice_first_dim(data["spatial_scalar_features"], residue_slice),
                dtype=torch.float32,
            ),
            "protein_spatial_vector": _as_tensor(
                _slice_first_dim(data["spatial_vector_features"], residue_slice),
                dtype=torch.float32,
            ),
            "protein_coords": ca_coords,
            "protein_backbone_vector": _compute_backbone_vectors(all_atom_coords, ca_coords),
            "chain_ids": _chain_ids_to_int(chain_ids),
            "labels": labels,
        }
        if crop_start is not None and crop_end is not None:
            item["crop_start"] = crop_start
            item["crop_end"] = crop_end

        if self.esm_root is not None:
            esm_path = _locate_esm(self.esm_root, pdb_id)
            if esm_path is not None:
                esm_data = _torch_load(esm_path)
                esm_raw = esm_data["embeddings"]
                if int(esm_raw.shape[0]) != n_res:
                    raise ValueError(f"{pdb_id}: ESM length {esm_raw.shape[0]} != n_res {n_res}")
                esm_embeddings = _as_tensor(_slice_first_dim(esm_raw, residue_slice), dtype=torch.float32)
                item["esm_embeddings"] = esm_embeddings
        return item

    def _select_crop_start(self, labels: torch.Tensor, n_res: int, max_residues: int) -> int:
        if n_res <= max_residues or self.crop_mode == "none":
            return 0
        if self.crop_mode == "first":
            return 0
        use_positive = (
            self.crop_mode == "positive_window"
            and self.rng.random() < self.positive_crop_prob
            and bool((labels == 1).any())
        )
        if use_positive:
            positives = torch.nonzero(labels == 1, as_tuple=False).flatten().tolist()
            center = self.rng.choice(positives)
        else:
            center = self.rng.randrange(n_res)
        return max(0, min(center - max_residues // 2, n_res - max_residues))

    def _crop_item(self, item: dict[str, Any], max_residues: int) -> dict[str, Any]:
        n_res = int(item["protein_coords"].shape[0])
        if n_res <= max_residues or self.crop_mode == "none":
            return item

        if self.crop_mode == "first":
            start = 0
        else:
            use_positive = (
                self.crop_mode == "positive_window"
                and self.rng.random() < self.positive_crop_prob
                and bool((item["labels"] == 1).any())
            )
            if use_positive:
                positives = torch.nonzero(item["labels"] == 1, as_tuple=False).flatten().tolist()
                center = self.rng.choice(positives)
            else:
                center = self.rng.randrange(n_res)
            start = max(0, min(center - max_residues // 2, n_res - max_residues))

        idx = torch.arange(start, start + max_residues, dtype=torch.long)
        cropped: dict[str, Any] = {"pdb_id": item["pdb_id"], "crop_start": start, "crop_end": start + max_residues}
        for key, value in item.items():
            if key in {"pdb_id", "crop_start", "crop_end"}:
                continue
            if isinstance(value, torch.Tensor) and value.shape[0] == n_res:
                cropped[key] = value.index_select(0, idx)
            else:
                cropped[key] = value
        return cropped


def _pad_tensor(value: torch.Tensor, max_len: int, pad_value: float | int = 0) -> torch.Tensor:
    shape = (max_len,) + tuple(value.shape[1:])
    out = value.new_full(shape, pad_value)
    out[: value.shape[0]] = value
    return out


def collate_protein_features(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(item["protein_coords"].shape[0] for item in items)
    batch: dict[str, Any] = {"pdb_id": [item["pdb_id"] for item in items]}
    if "crop_start" in items[0]:
        batch["crop_start"] = [item.get("crop_start") for item in items]
        batch["crop_end"] = [item.get("crop_end") for item in items]
    float_keys = (
        "protein_physchem",
        "protein_spatial_scalar",
        "protein_spatial_vector",
        "protein_backbone_vector",
        "protein_coords",
    )
    for key in float_keys:
        batch[key] = torch.stack([_pad_tensor(item[key], max_len, 0.0) for item in items], dim=0)
    batch["chain_ids"] = torch.stack([_pad_tensor(item["chain_ids"], max_len, 0) for item in items], dim=0)
    batch["labels"] = torch.stack([_pad_tensor(item["labels"], max_len, -100) for item in items], dim=0)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        mask[idx, : item["protein_coords"].shape[0]] = True
    batch["protein_mask"] = mask

    if all("esm_embeddings" in item for item in items):
        batch["esm_embeddings"] = torch.stack(
            [_pad_tensor(item["esm_embeddings"], max_len, 0.0) for item in items],
            dim=0,
        )
    return batch

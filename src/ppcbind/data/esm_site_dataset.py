"""ESM-only dataset for residue-level protein contact-site prediction."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
AA_GROUPS = {
    "positive": set("KRH"),
    "negative": set("DE"),
    "polar": set("STNQCY"),
    "hydrophobic": set("AILMFWV"),
    "aromatic": set("FWYH"),
    "small": set("AGSTCV"),
}
AA_MASS = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}
AA_HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _discover_esm_paths(esm_root: Path, ids: list[str] | None = None) -> list[Path]:
    if ids is None:
        return sorted(esm_root.glob("*/*_esm2.pt"))
    paths: list[Path] = []
    for pdb_id in ids:
        pdb_id = pdb_id.lower()
        candidates = (
            esm_root / pdb_id / f"{pdb_id}_esm2.pt",
            esm_root / pdb_id / f"{pdb_id}_protein.pt",
            esm_root / f"{pdb_id}_esm2.pt",
            esm_root / f"{pdb_id}_protein.pt",
        )
        for path in candidates:
            if path.exists():
                paths.append(path)
                break
    return sorted(paths)


def _esm_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_esm2"):
        return stem[: -len("_esm2")]
    if stem.endswith("_protein"):
        return stem[: -len("_protein")]
    return path.parent.name


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
        value = data.get("labels", data.get("is_contact", data.get("contact_labels"))) if isinstance(data, dict) else data
        labels = torch.as_tensor(value, dtype=torch.long)
        if labels.shape[0] != n_res:
            raise ValueError(f"{pdb_id}: label length {labels.shape[0]} != ESM length {n_res}")
        return labels
    return torch.full((n_res,), -100, dtype=torch.long)


def _local_chain_ids(chain_ids: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mapping: OrderedDict[str, int] = OrderedDict()
    chain_counts: dict[int, int] = {}
    chain_ints: list[int] = []
    chain_pos: list[int] = []
    for chain_id in chain_ids:
        key = _norm_text(chain_id)
        if key not in mapping:
            mapping[key] = len(mapping)
        cid = mapping[key]
        pos = chain_counts.get(cid, 0)
        chain_counts[cid] = pos + 1
        chain_ints.append(cid)
        chain_pos.append(pos)

    lengths = torch.tensor([chain_counts[cid] for cid in chain_ints], dtype=torch.float32)
    pos = torch.tensor(chain_pos, dtype=torch.float32)
    rel_pos = pos / torch.clamp(lengths - 1.0, min=1.0)
    return torch.tensor(chain_ints, dtype=torch.long), pos.long(), rel_pos


def _sequence_features(residue_names: list[Any]) -> torch.Tensor:
    features = torch.zeros((len(residue_names), len(AA_ORDER) + len(AA_GROUPS) + 2), dtype=torch.float32)
    for row, value in enumerate(residue_names):
        aa = _norm_text(value).upper()
        if len(aa) != 1 or aa not in AA_TO_INDEX:
            continue
        features[row, AA_TO_INDEX[aa]] = 1.0
        offset = len(AA_ORDER)
        for col, group in enumerate(AA_GROUPS.values()):
            features[row, offset + col] = 1.0 if aa in group else 0.0
        features[row, offset + len(AA_GROUPS)] = (AA_MASS[aa] - 75.0) / (205.0 - 75.0)
        features[row, offset + len(AA_GROUPS) + 1] = (AA_HYDROPATHY[aa] + 4.5) / 9.0
    return features


def _compare_text_list(name: str, expected: list[Any], observed: list[Any], pdb_id: str) -> None:
    if len(expected) != len(observed):
        raise ValueError(f"{pdb_id}: {name} length {len(observed)} != ESM length {len(expected)}")
    for idx, (a_value, b_value) in enumerate(zip(expected, observed)):
        a = _norm_text(a_value)
        b = _norm_text(b_value)
        if a != b:
            raise ValueError(f"{pdb_id}: {name} mismatch at residue {idx}: ESM={a!r} seq_feature={b!r}")


def _load_external_sequence_features(
    sequence_feature_root: Path | None,
    pdb_id: str,
    n_res: int,
    residue_names: list[Any],
    chain_ids: list[Any],
    required: bool,
) -> torch.Tensor | None:
    if sequence_feature_root is None:
        return None
    candidates = (
        sequence_feature_root / pdb_id / f"{pdb_id}_seq.pt",
        sequence_feature_root / pdb_id / f"{pdb_id}_sequence.pt",
        sequence_feature_root / f"{pdb_id}_seq.pt",
        sequence_feature_root / f"{pdb_id}.pt",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        if required:
            raise FileNotFoundError(f"{pdb_id}: sequence feature file not found under {sequence_feature_root}")
        return None

    data = _torch_load(path)
    if isinstance(data, dict):
        value = data.get("seq_features", data.get("features"))
        if "residue_names_1" in data:
            _compare_text_list("residue_names_1", residue_names, list(data["residue_names_1"]), pdb_id)
        elif "sequence" in data:
            _compare_text_list("sequence", residue_names, list(str(data["sequence"])), pdb_id)
        if "chain_ids" in data:
            _compare_text_list("chain_ids", chain_ids, list(data["chain_ids"]), pdb_id)
    else:
        value = data
    if value is None:
        raise ValueError(f"{pdb_id}: sequence feature file has no seq_features/features tensor")

    features = torch.as_tensor(value, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError(f"{pdb_id}: sequence features must be 2D, got shape {tuple(features.shape)}")
    if int(features.shape[0]) != n_res:
        raise ValueError(f"{pdb_id}: sequence feature length {features.shape[0]} != ESM length {n_res}")
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _slice_value(value: torch.Tensor, residue_slice: slice) -> torch.Tensor:
    return value[residue_slice]


def _load_contact_graph(data: dict[str, Any], pdb_id: str, n_res: int) -> tuple[torch.Tensor, torch.Tensor]:
    edge_value = data.get("contact_edge_index", data.get("contact_edges"))
    if edge_value is None:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

    edge_index = torch.as_tensor(edge_value, dtype=torch.long)
    if edge_index.ndim != 2:
        raise ValueError(f"{pdb_id}: contact edges must be 2D, got shape {tuple(edge_index.shape)}")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.shape[0] != 2:
        raise ValueError(f"{pdb_id}: contact edges must have shape (2, E), got {tuple(edge_index.shape)}")

    score_value = data.get("contact_edge_scores", data.get("contact_scores"))
    if score_value is None:
        scores = torch.ones(edge_index.shape[1], dtype=torch.float32)
    else:
        scores = torch.as_tensor(score_value, dtype=torch.float32).flatten()
    if scores.shape[0] != edge_index.shape[1]:
        raise ValueError(f"{pdb_id}: contact score length {scores.shape[0]} != edge count {edge_index.shape[1]}")

    valid = (
        torch.isfinite(scores)
        & (edge_index[0] >= 0)
        & (edge_index[1] >= 0)
        & (edge_index[0] < n_res)
        & (edge_index[1] < n_res)
        & (edge_index[0] != edge_index[1])
    )
    return edge_index[:, valid].contiguous(), torch.nan_to_num(scores[valid], nan=0.0, posinf=1.0, neginf=0.0)


def _slice_contact_graph(
    edge_index: torch.Tensor,
    scores: torch.Tensor,
    residue_slice: slice,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0 or residue_slice == slice(None):
        return edge_index, scores
    start = int(residue_slice.start or 0)
    stop = residue_slice.stop
    if stop is None:
        return edge_index, scores
    keep = (
        (edge_index[0] >= start)
        & (edge_index[0] < stop)
        & (edge_index[1] >= start)
        & (edge_index[1] < stop)
    )
    sliced = edge_index[:, keep] - start
    return sliced.contiguous(), scores[keep].contiguous()


class ESMProteinSiteDataset(Dataset):
    """Load per-residue ESM embeddings and offline contact labels.

    This dataset intentionally does not read complete structure features such
    as coordinates, DSSP-from-PDB, spatial neighbors, or backbone vectors.
    """

    def __init__(
        self,
        esm_root: str | Path,
        label_root: str | Path | None = None,
        ids: list[str] | None = None,
        sequence_feature_root: str | Path | None = None,
        require_sequence_features: bool = False,
        max_residues: int | None = None,
        crop_mode: str = "none",
        seed: int = 0,
        preload: bool = False,
    ) -> None:
        self.esm_root = Path(esm_root)
        self.label_root = Path(label_root) if label_root else None
        self.sequence_feature_root = Path(sequence_feature_root) if sequence_feature_root else None
        self.require_sequence_features = bool(require_sequence_features)
        self.max_residues = int(max_residues) if max_residues else None
        self.crop_mode = crop_mode
        self.rng = random.Random(seed)
        self.esm_paths = _discover_esm_paths(self.esm_root, ids)
        if not self.esm_paths:
            raise FileNotFoundError(f"No *_esm2.pt files under {self.esm_root}")
        self._preloaded_items: list[dict[str, Any]] | None = None
        if preload:
            self._preloaded_items = [self._load_item(i) for i in range(len(self.esm_paths))]

    def __len__(self) -> int:
        return len(self.esm_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._preloaded_items is not None:
            return self._preloaded_items[index]
        return self._load_item(index)

    def _load_item(self, index: int) -> dict[str, Any]:
        path = self.esm_paths[index]
        pdb_id = _esm_id(path)
        data = _torch_load(path)
        embeddings = torch.as_tensor(data["embeddings"], dtype=torch.float32)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        n_res = int(embeddings.shape[0])
        labels = _load_labels(self.label_root, pdb_id, n_res)
        chain_ids_raw = list(data.get("chain_ids", data.get("chain_id", [""] * n_res)))
        if len(chain_ids_raw) != n_res:
            raise ValueError(f"{pdb_id}: chain_ids length {len(chain_ids_raw)} != ESM length {n_res}")
        residue_names = list(data.get("residue_names_1", data.get("residue_name_1", ["X"] * n_res)))
        if len(residue_names) != n_res:
            raise ValueError(f"{pdb_id}: residue_names_1 length {len(residue_names)} != ESM length {n_res}")
        chain_ids, chain_pos, chain_rel_pos = _local_chain_ids(chain_ids_raw)
        protein_rel_pos = torch.arange(n_res, dtype=torch.float32) / max(1, n_res - 1)
        seq_features = _load_external_sequence_features(
            self.sequence_feature_root,
            pdb_id,
            n_res,
            residue_names,
            chain_ids_raw,
            self.require_sequence_features,
        )
        if seq_features is None:
            seq_features = _sequence_features(residue_names)
        contact_edge_index, contact_edge_scores = _load_contact_graph(data, pdb_id, n_res)

        residue_slice = self._select_slice(n_res)
        contact_edge_index, contact_edge_scores = _slice_contact_graph(contact_edge_index, contact_edge_scores, residue_slice)
        item: dict[str, Any] = {
            "pdb_id": pdb_id,
            "esm_embeddings": _slice_value(embeddings, residue_slice),
            "seq_features": _slice_value(seq_features, residue_slice),
            "chain_ids": _slice_value(chain_ids, residue_slice),
            "chain_pos": _slice_value(chain_pos, residue_slice),
            "chain_rel_pos": _slice_value(chain_rel_pos, residue_slice),
            "protein_rel_pos": _slice_value(protein_rel_pos, residue_slice),
            "labels": _slice_value(labels, residue_slice),
            "contact_edge_index": contact_edge_index,
            "contact_edge_scores": contact_edge_scores,
        }
        if residue_slice != slice(None):
            item["crop_start"] = residue_slice.start
            item["crop_end"] = residue_slice.stop
        return item

    def _select_slice(self, n_res: int) -> slice:
        if self.max_residues is None or n_res <= self.max_residues or self.crop_mode == "none":
            return slice(None)
        if self.crop_mode == "first":
            start = 0
        elif self.crop_mode == "random":
            start = self.rng.randrange(0, n_res - self.max_residues + 1)
        else:
            raise ValueError(f"Unsupported crop_mode={self.crop_mode!r}; use none, first, or random")
        return slice(start, start + self.max_residues)


def _pad_tensor(value: torch.Tensor, max_len: int, pad_value: float | int = 0) -> torch.Tensor:
    shape = (max_len,) + tuple(value.shape[1:])
    out = value.new_full(shape, pad_value)
    out[: value.shape[0]] = value
    return out


def collate_esm_site_features(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["esm_embeddings"].shape[0]) for item in items)
    batch: dict[str, Any] = {"pdb_id": [item["pdb_id"] for item in items]}
    for key in ("crop_start", "crop_end"):
        if key in items[0]:
            batch[key] = [item.get(key) for item in items]

    batch["esm_embeddings"] = torch.stack([_pad_tensor(item["esm_embeddings"], max_len, 0.0) for item in items])
    batch["seq_features"] = torch.stack([_pad_tensor(item["seq_features"], max_len, 0.0) for item in items])
    batch["chain_ids"] = torch.stack([_pad_tensor(item["chain_ids"], max_len, 0) for item in items])
    batch["chain_pos"] = torch.stack([_pad_tensor(item["chain_pos"], max_len, 0) for item in items])
    batch["chain_rel_pos"] = torch.stack([_pad_tensor(item["chain_rel_pos"], max_len, 0.0) for item in items])
    batch["protein_rel_pos"] = torch.stack([_pad_tensor(item["protein_rel_pos"], max_len, 0.0) for item in items])
    batch["labels"] = torch.stack([_pad_tensor(item["labels"], max_len, -100) for item in items])
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        mask[idx, : int(item["esm_embeddings"].shape[0])] = True
    batch["protein_mask"] = mask

    edge_indices: list[torch.Tensor] = []
    edge_scores: list[torch.Tensor] = []
    for batch_idx, item in enumerate(items):
        edge_index = item.get("contact_edge_index")
        if edge_index is None or edge_index.numel() == 0:
            continue
        edge_indices.append(edge_index.long() + batch_idx * max_len)
        edge_scores.append(item["contact_edge_scores"].float())
    if edge_indices:
        batch["contact_edge_index"] = torch.cat(edge_indices, dim=1)
        batch["contact_edge_scores"] = torch.cat(edge_scores, dim=0)
    else:
        batch["contact_edge_index"] = torch.empty((2, 0), dtype=torch.long)
        batch["contact_edge_scores"] = torch.empty((0,), dtype=torch.float32)
    return batch

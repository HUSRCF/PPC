"""ESM-only dataset for residue-level protein contact-site prediction."""

from __future__ import annotations

import csv
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

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
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
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


def _load_chain_filter_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id = _norm_text(row.get("seq_id"))
            pdb_id = _norm_text(row.get("pdb_id")).lower()
            chain_id = _norm_text(row.get("chain_id"))
            if not seq_id or not pdb_id or not chain_id:
                continue
            rows[seq_id] = {
                **row,
                "seq_id": seq_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "n_residues": int(row.get("n_residues") or row.get("len_seq") or 0),
            }
    if not rows:
        raise ValueError(f"{path}: no chain rows with seq_id/pdb_id/chain_id")
    return rows


def _chain_slice(chain_ids: list[Any], chain_id: str, sample_id: str) -> slice:
    positions = [idx for idx, value in enumerate(chain_ids) if _norm_text(value) == chain_id]
    if not positions:
        raise ValueError(f"{sample_id}: chain {chain_id!r} not found in ESM metadata")
    start = positions[0]
    stop = positions[-1] + 1
    if positions != list(range(start, stop)):
        raise ValueError(f"{sample_id}: chain {chain_id!r} residues are not contiguous")
    return slice(start, stop)


def _load_labels(
    label_root: Path | None,
    pdb_id: str,
    n_res: int,
    required: bool = False,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if label_root is None:
        if required:
            raise ValueError(f"{pdb_id}: label_root is required but None")
        return torch.full((n_res,), -100, dtype=torch.long), None
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
        return labels, data if isinstance(data, dict) else None
    if required:
        raise FileNotFoundError(f"{pdb_id}: label file not found under {label_root}")
    return torch.full((n_res,), -100, dtype=torch.long), None


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


def _get_metadata_list(data: dict[str, Any], *keys: str) -> list[Any] | None:
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, str):
                return list(value)
            return list(value)
    return None


def _norm_residue_name(value: Any, target: str) -> str:
    text = _norm_text(value).upper()
    if target == "one":
        return AA3_TO_1.get(text, text)
    return text


def _compare_label_metadata(
    data: dict[str, Any],
    label_data: dict[str, Any] | None,
    pdb_id: str,
    n_res: int,
) -> None:
    """Verify residue/chain alignment between the ESM payload and its label file.

    This invariant can also be checked once offline with
    scripts/analysis/validate_esm_site_metadata.py after regenerating labels or
    features, instead of re-checking every __getitem__ call during training.
    """
    if label_data is None:
        raise ValueError(f"{pdb_id}: strict label metadata requested but label payload has no metadata")

    checks = [
        ("chain_ids", _get_metadata_list(data, "chain_ids", "chain_id"), label_data.get("chain_ids")),
        ("residue_indices", _get_metadata_list(data, "residue_indices", "residue_index"), label_data.get("residue_indices")),
        ("insertion_codes", _get_metadata_list(data, "insertion_codes", "insertion_code"), label_data.get("insertion_codes")),
    ]
    for name, esm_values, label_values in checks:
        if esm_values is None or label_values is None:
            raise ValueError(f"{pdb_id}: missing metadata field for strict label check: {name}")
        _compare_text_list(name, list(esm_values), list(label_values), pdb_id)

    label_res = label_data.get("residue_names")
    if label_res is None:
        raise ValueError(f"{pdb_id}: missing label residue_names for strict label check")
    esm_res3 = _get_metadata_list(data, "residue_names_3", "residue_name_3")
    esm_res1 = _get_metadata_list(data, "residue_names_1", "residue_name_1")
    if esm_res3 is not None:
        observed = [_norm_residue_name(x, "three") for x in label_res]
        expected = [_norm_residue_name(x, "three") for x in esm_res3]
        _compare_text_list("residue_names_3", expected, observed, pdb_id)
    elif esm_res1 is not None:
        observed = [_norm_residue_name(x, "one") for x in label_res]
        expected = [_norm_residue_name(x, "one") for x in esm_res1]
        _compare_text_list("residue_names_1", expected, observed, pdb_id)
    else:
        raise ValueError(f"{pdb_id}: missing ESM residue names for strict label check")
    if len(label_res) != n_res:
        raise ValueError(f"{pdb_id}: label residue metadata length {len(label_res)} != ESM length {n_res}")


def _compare_sequence_feature_metadata(
    data: dict[str, Any],
    residue_names: list[Any],
    chain_ids: list[Any],
    pdb_id: str,
) -> None:
    """Verify residue/chain alignment between ESM and external sequence features.

    This is the same invariant checked by scripts/analysis/validate_esm_site_metadata.py.
    Use strict_sequence_feature_metadata=False only after that offline check passes.
    """
    if "residue_names_1" in data:
        _compare_text_list("residue_names_1", residue_names, list(data["residue_names_1"]), pdb_id)
    elif "sequence" in data:
        _compare_text_list("sequence", residue_names, list(str(data["sequence"])), pdb_id)
    if "chain_ids" in data:
        _compare_text_list("chain_ids", chain_ids, list(data["chain_ids"]), pdb_id)


def _load_external_sequence_features(
    sequence_feature_root: Path | None,
    pdb_id: str,
    n_res: int,
    residue_names: list[Any],
    chain_ids: list[Any],
    required: bool,
    strict_metadata: bool = True,
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
        if strict_metadata:
            _compare_sequence_feature_metadata(data, residue_names, chain_ids, pdb_id)
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


def _load_external_prottrans_embeddings(
    prottrans_embedding_root: Path | None,
    pdb_id: str,
    n_res: int,
    residue_names: list[Any],
    chain_ids: list[Any],
    required: bool,
    load_fn: Callable[[Path], Any] = _torch_load,
) -> torch.Tensor | None:
    """Load frozen ProtTrans/ProtBert-style per-residue embeddings.

    Expected shape is [L, D].  Payloads may be a raw tensor or a dict with one
    of several conventional keys, so the same loader can handle ProtBert,
    ProtBert-BFD, ProtT5, or future frozen PLM embeddings.
    """

    if prottrans_embedding_root is None:
        return None
    candidates = (
        prottrans_embedding_root / pdb_id / f"{pdb_id}_prottrans.pt",
        prottrans_embedding_root / pdb_id / f"{pdb_id}_protbert.pt",
        prottrans_embedding_root / pdb_id / f"{pdb_id}_protbert_bfd.pt",
        prottrans_embedding_root / pdb_id / f"{pdb_id}_embeddings.pt",
        prottrans_embedding_root / pdb_id / f"{pdb_id}.pt",
        prottrans_embedding_root / f"{pdb_id}_prottrans.pt",
        prottrans_embedding_root / f"{pdb_id}_protbert.pt",
        prottrans_embedding_root / f"{pdb_id}_protbert_bfd.pt",
        prottrans_embedding_root / f"{pdb_id}_embeddings.pt",
        prottrans_embedding_root / f"{pdb_id}.pt",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        if required:
            raise FileNotFoundError(f"{pdb_id}: ProtTrans embedding file not found under {prottrans_embedding_root}")
        return None

    data = load_fn(path)
    if isinstance(data, dict):
        for key in ("prottrans_embeddings", "protbert_embeddings", "embeddings", "features", "residue_embeddings"):
            if key in data:
                value = data[key]
                break
        else:
            raise ValueError(f"{pdb_id}: ProtTrans payload has no embedding tensor: {path}")
        if "residue_names_1" in data:
            _compare_text_list("prottrans residue_names_1", residue_names, list(data["residue_names_1"]), pdb_id)
        elif "sequence" in data:
            _compare_text_list("prottrans sequence", residue_names, list(str(data["sequence"])), pdb_id)
        if "chain_ids" in data:
            _compare_text_list("prottrans chain_ids", chain_ids, list(data["chain_ids"]), pdb_id)
    else:
        value = data

    embeddings = torch.as_tensor(value, dtype=torch.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"{pdb_id}: ProtTrans embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if int(embeddings.shape[0]) != n_res:
        raise ValueError(f"{pdb_id}: ProtTrans embedding length {embeddings.shape[0]} != ESM length {n_res}")
    return torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)


def _slice_value(value: torch.Tensor, residue_slice: slice) -> torch.Tensor:
    return value[residue_slice]


def _load_contact_graph(
    data: dict[str, Any],
    pdb_id: str,
    n_res: int,
    required: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_value = data.get("contact_edge_index", data.get("contact_edges"))
    if edge_value is None:
        if required:
            raise ValueError(f"{pdb_id}: contact graph required but contact_edge_index/contact_edges is missing")
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
    edge_index = edge_index[:, valid].contiguous()
    scores = torch.nan_to_num(scores[valid], nan=0.0, posinf=1.0, neginf=0.0)
    if required and edge_index.shape[1] == 0:
        raise ValueError(f"{pdb_id}: contact graph required but no valid edges remain after filtering")
    return edge_index, scores


def _load_external_contact_graph(
    contact_graph_root: Path | None,
    pdb_id: str,
    n_res: int,
    required: bool,
    load_fn: Callable[[Path], Any] = _torch_load,
) -> tuple[torch.Tensor, torch.Tensor]:
    if contact_graph_root is None:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    candidates = (
        contact_graph_root / pdb_id / f"{pdb_id}_contact_graph.pt",
        contact_graph_root / pdb_id / f"{pdb_id}_predcontact.pt",
        contact_graph_root / pdb_id / f"{pdb_id}_pred_struct_contact.pt",
        contact_graph_root / f"{pdb_id}_contact_graph.pt",
        contact_graph_root / f"{pdb_id}_predcontact.pt",
        contact_graph_root / f"{pdb_id}.pt",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        if required:
            raise FileNotFoundError(f"{pdb_id}: external contact graph file not found under {contact_graph_root}")
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    data = load_fn(path)
    if isinstance(data, dict) and "n_residues" in data and int(data["n_residues"]) != int(n_res):
        raise ValueError(f"{pdb_id}: external contact graph n_residues {data['n_residues']} != ESM length {n_res}")
    if isinstance(data, dict):
        return _load_contact_graph(data, pdb_id, n_res, required=required)
    raise ValueError(f"{pdb_id}: external contact graph payload must be a dict: {path}")


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
        primary_embedding_root: str | Path | None = None,
        prottrans_embedding_root: str | Path | None = None,
        contact_graph_root: str | Path | None = None,
        aux_contact_graph_root: str | Path | None = None,
        require_sequence_features: bool = False,
        require_primary_embeddings: bool = False,
        require_prottrans_embeddings: bool = False,
        max_residues: int | None = None,
        crop_mode: str = "none",
        seed: int = 0,
        preload: bool = False,
        strict_ids: bool = False,
        require_labels: bool = False,
        strict_label_metadata: bool = False,
        strict_sequence_feature_metadata: bool = True,
        require_contact_graph: bool = False,
        require_aux_contact_graph: bool = False,
        chain_filter_manifest: str | Path | None = None,
        payload_cache_size: int = 0,
    ) -> None:
        self.esm_root = Path(esm_root)
        self.label_root = Path(label_root) if label_root else None
        self.sequence_feature_root = Path(sequence_feature_root) if sequence_feature_root else None
        self.primary_embedding_root = Path(primary_embedding_root) if primary_embedding_root else None
        self.prottrans_embedding_root = Path(prottrans_embedding_root) if prottrans_embedding_root else None
        self.contact_graph_root = Path(contact_graph_root) if contact_graph_root else None
        self.aux_contact_graph_root = Path(aux_contact_graph_root) if aux_contact_graph_root else None
        self.require_sequence_features = bool(require_sequence_features)
        self.require_primary_embeddings = bool(require_primary_embeddings)
        self.require_prottrans_embeddings = bool(require_prottrans_embeddings)
        self.max_residues = int(max_residues) if max_residues else None
        self.crop_mode = crop_mode
        self.rng = random.Random(seed)
        self.requested_ids = [str(pdb_id).lower() for pdb_id in ids] if ids is not None else None
        self.require_labels = bool(require_labels)
        self.strict_label_metadata = bool(strict_label_metadata)
        self.strict_sequence_feature_metadata = bool(strict_sequence_feature_metadata)
        self.require_contact_graph = bool(require_contact_graph)
        self.require_aux_contact_graph = bool(require_aux_contact_graph)
        self.chain_filter_manifest = Path(chain_filter_manifest) if chain_filter_manifest else None
        self.payload_cache_size = max(0, int(payload_cache_size))
        self._payload_cache: OrderedDict[Path, Any] = OrderedDict()
        self.chain_filter_rows = (
            _load_chain_filter_manifest(self.chain_filter_manifest) if self.chain_filter_manifest is not None else None
        )
        self.sample_entries: list[dict[str, Any]] | None = None
        if self.chain_filter_rows is not None:
            sample_ids = [str(seq_id).strip() for seq_id in ids] if ids is not None else sorted(self.chain_filter_rows)
            self.sample_entries = []
            missing: list[str] = []
            for sample_id in sample_ids:
                row = self.chain_filter_rows.get(sample_id)
                if row is None:
                    missing.append(sample_id)
                    continue
                pdb_id = str(row["pdb_id"]).lower()
                candidates = (
                    self.esm_root / pdb_id / f"{pdb_id}_esm2.pt",
                    self.esm_root / pdb_id / f"{pdb_id}_protein.pt",
                    self.esm_root / f"{pdb_id}_esm2.pt",
                    self.esm_root / f"{pdb_id}_protein.pt",
                )
                path = next((candidate for candidate in candidates if candidate.exists()), None)
                if path is None:
                    missing.append(sample_id)
                    continue
                self.sample_entries.append({"sample_id": sample_id, "path": path, **row})
            if strict_ids and missing:
                raise FileNotFoundError(
                    f"Missing chain-filtered samples/files for {len(missing)} requested ids; examples={missing[:20]}"
                )
            self.esm_paths = [entry["path"] for entry in self.sample_entries]
            self.sample_ids = [entry["sample_id"] for entry in self.sample_entries]
        else:
            self.esm_paths = _discover_esm_paths(self.esm_root, ids)
            self.sample_ids = [_esm_id(path) for path in self.esm_paths]
        if not self.esm_paths:
            raise FileNotFoundError(f"No *_esm2.pt files under {self.esm_root}")
        if self.chain_filter_rows is None and strict_ids and self.requested_ids is not None:
            requested = set(self.requested_ids)
            discovered = {_esm_id(path).lower() for path in self.esm_paths}
            missing = sorted(requested - discovered)
            extra = sorted(discovered - requested)
            if missing or extra or len(self.esm_paths) != len(self.requested_ids):
                raise FileNotFoundError(
                    f"ESM files under {self.esm_root} do not match split ids: "
                    f"requested={len(self.requested_ids)} discovered={len(discovered)} "
                    f"missing={missing[:20]} extra={extra[:20]}"
                )
        self._preloaded_items: list[dict[str, Any]] | None = None
        if preload:
            self._preloaded_items = [self._load_item(i) for i in range(len(self.esm_paths))]

    def __len__(self) -> int:
        return len(self.esm_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._preloaded_items is not None:
            return self._preloaded_items[index]
        return self._load_item(index)

    def _load_payload(self, path: Path) -> Any:
        if self.payload_cache_size <= 0:
            return _torch_load(path)
        key = path.resolve()
        cached = self._payload_cache.pop(key, None)
        if cached is not None:
            self._payload_cache[key] = cached
            return cached
        payload = _torch_load(key)
        self._payload_cache[key] = payload
        while len(self._payload_cache) > self.payload_cache_size:
            self._payload_cache.popitem(last=False)
        return payload

    def _load_item(self, index: int) -> dict[str, Any]:
        path = self.esm_paths[index]
        entry = self.sample_entries[index] if self.sample_entries is not None else None
        pdb_id = str(entry["pdb_id"]) if entry is not None else _esm_id(path)
        sample_id = str(entry["sample_id"]) if entry is not None else pdb_id
        data = self._load_payload(path)
        stored_embeddings = data.get("embeddings")
        if stored_embeddings is not None:
            full_n_res = int(stored_embeddings.shape[0])
        else:
            full_n_res = int(data.get("feature_n_residues") or len(data.get("chain_ids", ())))
        if full_n_res <= 0:
            raise ValueError(f"{pdb_id}: source payload does not define a positive residue count")
        if stored_embeddings is None and self.primary_embedding_root is None:
            raise ValueError(f"{pdb_id}: source payload has no embeddings and no primary_embedding_root was provided")
        labels, label_data = _load_labels(self.label_root, pdb_id, full_n_res, self.require_labels)
        if self.strict_label_metadata:
            _compare_label_metadata(data, label_data, pdb_id, full_n_res)
        chain_ids_raw = list(data.get("chain_ids", data.get("chain_id", [""] * full_n_res)))
        if len(chain_ids_raw) != full_n_res:
            raise ValueError(f"{pdb_id}: chain_ids length {len(chain_ids_raw)} != ESM length {full_n_res}")
        residue_names = list(data.get("residue_names_1", data.get("residue_name_1", ["X"] * full_n_res)))
        if len(residue_names) != full_n_res:
            raise ValueError(f"{pdb_id}: residue_names_1 length {len(residue_names)} != ESM length {full_n_res}")
        if entry is not None:
            first_slice = _chain_slice(chain_ids_raw, str(entry["chain_id"]), sample_id)
            labels = _slice_value(labels, first_slice)
            chain_ids_raw = chain_ids_raw[first_slice]
            residue_names = residue_names[first_slice]
        else:
            first_slice = slice(None)
        n_res = len(chain_ids_raw)
        expected_len = int(entry.get("n_residues") or 0) if entry is not None else 0
        if expected_len and n_res != expected_len:
            raise ValueError(f"{sample_id}: chain length {n_res} != manifest length {expected_len}")
        if self.primary_embedding_root is not None:
            embeddings = _load_external_prottrans_embeddings(
                self.primary_embedding_root,
                sample_id,
                n_res,
                residue_names,
                chain_ids_raw,
                self.require_primary_embeddings,
                load_fn=self._load_payload,
            )
            if embeddings is None:
                raise FileNotFoundError(f"{sample_id}: primary embedding is unavailable")
        else:
            embeddings = torch.as_tensor(_slice_value(stored_embeddings, first_slice), dtype=torch.float32)
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        chain_ids, chain_pos, chain_rel_pos = _local_chain_ids(chain_ids_raw)
        protein_rel_pos = torch.arange(n_res, dtype=torch.float32) / max(1, n_res - 1)
        seq_features = _load_external_sequence_features(
            self.sequence_feature_root,
            sample_id,
            n_res,
            residue_names,
            chain_ids_raw,
            self.require_sequence_features,
            self.strict_sequence_feature_metadata,
        )
        if seq_features is None:
            seq_features = _sequence_features(residue_names)
        prottrans_embeddings = _load_external_prottrans_embeddings(
            self.prottrans_embedding_root,
            sample_id,
            n_res,
            residue_names,
            chain_ids_raw,
            self.require_prottrans_embeddings,
            load_fn=self._load_payload,
        )
        if self.contact_graph_root is not None:
            contact_edge_index, contact_edge_scores = _load_external_contact_graph(
                self.contact_graph_root,
                sample_id,
                n_res,
                self.require_contact_graph,
                load_fn=self._load_payload,
            )
        else:
            contact_edge_index, contact_edge_scores = _load_contact_graph(data, pdb_id, full_n_res, self.require_contact_graph)
            contact_edge_index, contact_edge_scores = _slice_contact_graph(contact_edge_index, contact_edge_scores, first_slice)
        aux_contact_edge_index, aux_contact_edge_scores = _load_external_contact_graph(
            self.aux_contact_graph_root,
            sample_id,
            n_res,
            self.require_aux_contact_graph,
            load_fn=self._load_payload,
        )

        residue_slice = self._select_slice(n_res)
        contact_edge_index, contact_edge_scores = _slice_contact_graph(contact_edge_index, contact_edge_scores, residue_slice)
        aux_contact_edge_index, aux_contact_edge_scores = _slice_contact_graph(
            aux_contact_edge_index,
            aux_contact_edge_scores,
            residue_slice,
        )
        item: dict[str, Any] = {
            "pdb_id": sample_id,
            "source_pdb_id": pdb_id,
            "esm_embeddings": _slice_value(embeddings, residue_slice),
            "seq_features": _slice_value(seq_features, residue_slice),
            "chain_ids": _slice_value(chain_ids, residue_slice),
            "chain_ids_raw": [_norm_text(value) for value in chain_ids_raw[residue_slice]],
            "chain_pos": _slice_value(chain_pos, residue_slice),
            "chain_rel_pos": _slice_value(chain_rel_pos, residue_slice),
            "protein_rel_pos": _slice_value(protein_rel_pos, residue_slice),
            "labels": _slice_value(labels, residue_slice),
            "contact_edge_index": contact_edge_index,
            "contact_edge_scores": contact_edge_scores,
            "aux_contact_edge_index": aux_contact_edge_index,
            "aux_contact_edge_scores": aux_contact_edge_scores,
        }
        if entry is not None:
            item["source_chain_id"] = str(entry["chain_id"])
        if prottrans_embeddings is not None:
            item["prottrans_embeddings"] = _slice_value(prottrans_embeddings, residue_slice)
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


def _stack_padded(
    values: list[torch.Tensor],
    max_len: int,
    pad_value: float | int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pad and stack variable-length tensors in one allocation."""
    tail_shape = tuple(values[0].shape[1:])
    out = values[0].new_full((len(values), max_len) + tail_shape, pad_value, dtype=dtype)
    for idx, value in enumerate(values):
        out[idx, : value.shape[0]] = value
    return out


def _collate_contact_edges(
    items: list[dict[str, Any]],
    max_len: int,
    index_key: str,
    score_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_indices: list[torch.Tensor] = []
    edge_scores: list[torch.Tensor] = []
    for batch_idx, item in enumerate(items):
        edge_index = item.get(index_key)
        if edge_index is None or edge_index.numel() == 0:
            continue
        edge_indices.append(edge_index.long() + batch_idx * max_len)
        edge_scores.append(item[score_key].float())
    if edge_indices:
        return torch.cat(edge_indices, dim=1), torch.cat(edge_scores, dim=0)
    return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)


def collate_esm_site_features(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["esm_embeddings"].shape[0]) for item in items)
    batch: dict[str, Any] = {"pdb_id": [item["pdb_id"] for item in items]}
    if "chain_ids_raw" in items[0]:
        batch["chain_ids_raw"] = [item["chain_ids_raw"] for item in items]
    for key in ("crop_start", "crop_end"):
        if key in items[0]:
            batch[key] = [item.get(key) for item in items]

    batch["esm_embeddings"] = _stack_padded([item["esm_embeddings"] for item in items], max_len, 0.0, torch.float32)
    if any("prottrans_embeddings" in item for item in items):
        missing = [item["pdb_id"] for item in items if "prottrans_embeddings" not in item]
        if missing:
            raise ValueError(f"Batch mixes missing ProtTrans embeddings for: {missing[:8]}")
        batch["prottrans_embeddings"] = _stack_padded(
            [item["prottrans_embeddings"] for item in items], max_len, 0.0, torch.float32
        )
    batch["seq_features"] = _stack_padded([item["seq_features"] for item in items], max_len, 0.0, torch.float32)
    batch["chain_ids"] = _stack_padded([item["chain_ids"] for item in items], max_len, 0, torch.long)
    batch["chain_pos"] = _stack_padded([item["chain_pos"] for item in items], max_len, 0, torch.long)
    batch["chain_rel_pos"] = _stack_padded([item["chain_rel_pos"] for item in items], max_len, 0.0, torch.float32)
    batch["protein_rel_pos"] = _stack_padded([item["protein_rel_pos"] for item in items], max_len, 0.0, torch.float32)
    batch["labels"] = _stack_padded([item["labels"] for item in items], max_len, -100, torch.long)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        mask[idx, : int(item["esm_embeddings"].shape[0])] = True
    batch["protein_mask"] = mask

    batch["contact_edge_index"], batch["contact_edge_scores"] = _collate_contact_edges(
        items,
        max_len,
        "contact_edge_index",
        "contact_edge_scores",
    )
    batch["aux_contact_edge_index"], batch["aux_contact_edge_scores"] = _collate_contact_edges(
        items,
        max_len,
        "aux_contact_edge_index",
        "aux_contact_edge_scores",
    )
    return batch

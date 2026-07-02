#!/usr/bin/env python3
"""Run official PIPENN-EMB dnet weights on prepared chain-level embeddings.

The official PIPENN-EMB dnet model uses 1170-residue slices and 1025 input
channels: normalized_length plus a 1024D ProtT5/ProtBert-style embedding.
This runner keeps the official architecture and slicing convention, but writes
the output directly as residue-level TSV for the PPC benchmark harness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SLICE_LEN = 1170
INPUT_DIM = 1025
FEATURE_PAD_CONST = np.float32(0.11111111)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-csv", required=True)
    parser.add_argument("--embeddings-npz", required=True)
    parser.add_argument("--model-hdf5", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--method", default="pipenn_emb_dnet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_dnet_model():
    from tensorflow.keras.initializers import he_uniform
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        Input,
        PReLU,
        TimeDistributed,
    )
    from tensorflow.keras.models import Model

    prot = Input(shape=(SLICE_LEN, INPUT_DIM))
    x = prot
    dilation = 1
    for channels in (64, 128, 128, 64, 32):
        x = Conv1D(
            channels,
            7,
            dilation_rate=dilation,
            padding="same",
            use_bias=False,
            kernel_initializer=he_uniform(),
        )(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        dilation *= 2
    out = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    return Model(inputs=prot, outputs=out)


def comma_values(value: object, dtype=np.float32) -> np.ndarray:
    return np.fromstring(str(value), sep=",", dtype=dtype)


def comma_sequence(value: object) -> str:
    text = str(value)
    if "," in text:
        return "".join(part.strip() for part in text.split(",") if part.strip())
    return text.strip()


def make_slices(norm_len: np.ndarray, emb: np.ndarray) -> np.ndarray:
    if norm_len.ndim != 1:
        raise ValueError(f"normalized_length must be 1D, got {norm_len.shape}")
    if emb.ndim != 2 or emb.shape[1] != 1024:
        raise ValueError(f"embedding must be [L,1024], got {emb.shape}")
    if emb.shape[0] != norm_len.shape[0]:
        raise ValueError(f"length mismatch: norm={norm_len.shape[0]} emb={emb.shape[0]}")
    features = np.concatenate([norm_len[:, None].astype(np.float32), emb.astype(np.float32)], axis=1)
    length = features.shape[0]
    n_slices = int(math.ceil(length / SLICE_LEN))
    padded_len = n_slices * SLICE_LEN
    if padded_len != length:
        pad = np.full((padded_len - length, INPUT_DIM), FEATURE_PAD_CONST, dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    return features.reshape(n_slices, SLICE_LEN, INPUT_DIM)


def main() -> None:
    args = parse_args()
    out_tsv = Path(args.output_tsv)
    summary_json = Path(args.summary_json)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.prepared_csv)
    if args.limit > 0:
        df = df.iloc[: args.limit].copy()

    emb_dict = np.load(args.embeddings_npz)
    model = build_dnet_model()
    model.load_weights(args.model_hdf5)

    total_chains = 0
    total_residues = 0
    skipped = []

    with out_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["seq_id", "position", "residue", "score", "method"])
        for row in df.itertuples(index=False):
            seq_id = getattr(row, "uniprot_id")
            seq = comma_sequence(getattr(row, "sequence"))
            rlength = int(getattr(row, "Rlength"))
            if seq_id not in emb_dict:
                skipped.append({"seq_id": seq_id, "reason": "missing_embedding"})
                continue
            emb = emb_dict[seq_id]
            norm_len = comma_values(getattr(row, "normalized_length"))
            if len(seq) != rlength or emb.shape[0] != rlength or norm_len.shape[0] != rlength:
                skipped.append(
                    {
                        "seq_id": seq_id,
                        "reason": "length_mismatch",
                        "sequence_len": len(seq),
                        "Rlength": rlength,
                        "embedding_len": int(emb.shape[0]),
                        "normalized_length_len": int(norm_len.shape[0]),
                    }
                )
                continue

            chunks = make_slices(norm_len, emb)
            pred = model.predict(chunks, batch_size=args.batch_size, verbose=0).reshape(-1)[:rlength]
            for pos, (aa, score) in enumerate(zip(seq, pred), start=1):
                writer.writerow([seq_id, pos, aa, float(score), args.method])
            total_chains += 1
            total_residues += rlength

    summary = {
        "method": args.method,
        "prepared_csv": args.prepared_csv,
        "embeddings_npz": args.embeddings_npz,
        "model_hdf5": args.model_hdf5,
        "output_tsv": str(out_tsv),
        "chains": total_chains,
        "residues": total_residues,
        "skipped": skipped,
        "skipped_count": len(skipped),
        "slice_len": SLICE_LEN,
        "input_dim": INPUT_DIM,
        "feature_pad_const": float(FEATURE_PAD_CONST),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run official PIPENN-EMB base architecture weights on prepared embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIM = 1025
FEATURE_PAD_CONST = np.float32(0.11111111)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-csv", required=True)
    p.add_argument("--embeddings-npz", required=True)
    p.add_argument("--model-hdf5", required=True)
    p.add_argument("--architecture", required=True, choices=["dnet", "rnn", "rnet", "unet"])
    p.add_argument("--output-tsv", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chain-batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def activation():
    from tensorflow.keras.layers import PReLU

    return PReLU()


def build_dnet_model(slice_len: int = 1170):
    from tensorflow.keras.initializers import he_uniform
    from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Input, TimeDistributed
    from tensorflow.keras.models import Model

    prot = Input(shape=(slice_len, INPUT_DIM))
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
        x = activation()(x)
        dilation *= 2
    out = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    return Model(inputs=prot, outputs=out)


def build_rnn_model(slice_len: int = 1024):
    from tensorflow.keras.layers import Dense, GRU, Input, TimeDistributed
    from tensorflow.keras.models import Model

    prot = Input(shape=(slice_len, INPUT_DIM))
    x = prot
    for _ in range(2):
        x = GRU(148, return_sequences=True, reset_after=True)(x)
    out = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    return Model(inputs=prot, outputs=out)


def build_rnet_model(slice_len: int = 1024):
    from tensorflow.keras.initializers import he_uniform
    from tensorflow.keras.layers import Add, BatchNormalization, Conv1D, Dense, Dropout, Input, TimeDistributed
    from tensorflow.keras.models import Model

    def act_norm(x):
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        return activation()(x)

    def conv(x, kernel_size):
        return Conv1D(
            filters=128,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=he_uniform(),
        )(x)

    prot = Input(shape=(slice_len, INPUT_DIM))
    x = conv(prot, 3)
    for _ in range(8):
        res = x
        x = act_norm(x)
        x = conv(x, 5)
        x = act_norm(x)
        x = conv(x, 3)
        x = Add()([res, x])
    x = act_norm(x)
    out = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    return Model(inputs=prot, outputs=out)


def build_unet_model(slice_len: int = 1024):
    from tensorflow.keras import backend as K
    from tensorflow.keras.initializers import he_uniform
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Conv1D,
        Conv2DTranspose,
        Dropout,
        Input,
        MaxPooling1D,
        Reshape,
        concatenate,
    )
    from tensorflow.keras.models import Model

    def conv1d_transpose(input_tensor, filters, kernel_size, strides=2, padding="same"):
        _, h, c = input_tensor.shape
        x = Reshape((h, 1, c))(input_tensor)
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding=padding,
            use_bias=False,
            kernel_initializer=he_uniform(),
        )(x)
        _, h2, _, c2 = x.shape
        return Reshape((h2, c2))(x)

    def block(x, is_conv, channels):
        if is_conv:
            x = Conv1D(
                channels,
                7,
                padding="same",
                use_bias=False,
                kernel_initializer=he_uniform(),
            )(x)
        else:
            x = conv1d_transpose(x, filters=channels, kernel_size=7)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        return activation()(x)

    prot = Input(shape=(slice_len, INPUT_DIM))
    x = prot
    channels = 64 // 2
    copies = []

    for _ in range(2):
        channels *= 2
        x = block(x, True, channels)
        copies.append(x)
        x = MaxPooling1D(pool_size=2)(x)

    for _ in range(2):
        channels *= 2
        x = block(x, True, channels)
        x = block(x, True, channels)
        copies.append(x)
        x = MaxPooling1D(pool_size=2)(x)

    x = block(x, True, channels)
    x = block(x, True, channels)
    copies.append(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = block(x, True, channels)

    channels //= 2
    for level in range(4, 2, -1):
        x = block(x, False, channels)
        x = concatenate([copies[level], x], axis=2)
        x = block(x, True, channels * 2)

    for level in range(2, -1, -1):
        channels //= 2
        x = block(x, False, channels)
        x = concatenate([copies[level], x], axis=2)
        if level != 0:
            x = block(x, True, channels * 2)

    x = Conv1D(1, 7, padding="same")(x)
    out = Activation("sigmoid")(x)
    return Model(inputs=prot, outputs=out)


def build_model(architecture: str):
    if architecture == "dnet":
        return build_dnet_model(), 1170
    if architecture == "rnn":
        return build_rnn_model(), 1024
    if architecture == "rnet":
        return build_rnet_model(), 1024
    if architecture == "unet":
        return build_unet_model(), 1024
    raise ValueError(architecture)


def comma_values(value: object, dtype=np.float32) -> np.ndarray:
    return np.fromstring(str(value), sep=",", dtype=dtype)


def comma_sequence(value: object) -> str:
    text = str(value)
    if "," in text:
        return "".join(part.strip() for part in text.split(",") if part.strip())
    return text.strip()


def make_slices(norm_len: np.ndarray, emb: np.ndarray, slice_len: int) -> np.ndarray:
    if norm_len.ndim != 1:
        raise ValueError(f"normalized_length must be 1D, got {norm_len.shape}")
    if emb.ndim != 2 or emb.shape[1] != 1024:
        raise ValueError(f"embedding must be [L,1024], got {emb.shape}")
    if emb.shape[0] != norm_len.shape[0]:
        raise ValueError(f"length mismatch: norm={norm_len.shape[0]} emb={emb.shape[0]}")
    features = np.concatenate([norm_len[:, None].astype(np.float32), emb.astype(np.float32)], axis=1)
    length = features.shape[0]
    n_slices = int(math.ceil(length / slice_len))
    padded_len = n_slices * slice_len
    if padded_len != length:
        pad = np.full((padded_len - length, INPUT_DIM), FEATURE_PAD_CONST, dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    return features.reshape(n_slices, slice_len, INPUT_DIM)


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
    model, slice_len = build_model(args.architecture)
    model.load_weights(args.model_hdf5)

    total_chains = 0
    total_residues = 0
    skipped = []
    pending_chunks = []
    pending_records = []

    def flush_pending(writer):
        nonlocal pending_chunks, pending_records, total_chains, total_residues
        if not pending_chunks:
            return
        batch = np.concatenate(pending_chunks, axis=0)
        pred_all = model.predict(batch, batch_size=args.batch_size, verbose=0).reshape(-1)
        offset = 0
        for seq_id, seq, rlength, n_slices in pending_records:
            pred = pred_all[offset : offset + n_slices * slice_len][:rlength]
            offset += n_slices * slice_len
            for pos, (aa, score) in enumerate(zip(seq, pred), start=1):
                writer.writerow([seq_id, pos, aa, float(score), args.method])
            total_chains += 1
            total_residues += rlength
        pending_chunks = []
        pending_records = []

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

            chunks = make_slices(norm_len, emb, slice_len)
            pending_chunks.append(chunks)
            pending_records.append((seq_id, seq, rlength, chunks.shape[0]))
            if len(pending_records) >= args.chain_batch_size:
                flush_pending(writer)
        flush_pending(writer)

    summary = {
        "method": args.method,
        "architecture": args.architecture,
        "prepared_csv": args.prepared_csv,
        "embeddings_npz": args.embeddings_npz,
        "model_hdf5": args.model_hdf5,
        "output_tsv": str(out_tsv),
        "chains": total_chains,
        "residues": total_residues,
        "skipped": skipped,
        "skipped_count": len(skipped),
        "slice_len": slice_len,
        "input_dim": INPUT_DIM,
        "feature_pad_const": float(FEATURE_PAD_CONST),
        "batch_size": args.batch_size,
        "chain_batch_size": args.chain_batch_size,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

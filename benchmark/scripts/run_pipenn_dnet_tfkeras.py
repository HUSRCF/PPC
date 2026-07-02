#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from tf_keras.models import load_model


# PIPENN imports shap through its explanation helper at module import time.
# The dnet inference path used here does not call explanation code, and stubbing
# shap avoids an otherwise unrelated numba/numpy version conflict in BIO.
sys.modules.setdefault("shap", types.ModuleType("shap"))


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PIPENN BioDL-P dnet model on a prepared PIPENN-1 CSV using tf_keras.")
    ap.add_argument("--pipenn-root", required=True)
    ap.add_argument("--prepared-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.pipenn_root).resolve()
    sys.path.insert(0, str(root / "utils"))
    from PPIDataset import PPIDatasetCls, DatasetParams
    from PPIParams import PPIParamsCls

    PPIDatasetCls.setLogger(DummyLogger())
    DatasetParams.USE_COMET = False
    DatasetParams.ONE_HOT_ENCODING = True
    DatasetParams.FLOAT_TYPE = "float32"
    PPIParamsCls.setInitParams("dnet-ppi", dsParam=DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN, dsLabelParam="Biolip_P")

    input_shape = (1170, DatasetParams.getFeaturesDim())
    label_shape = (1170, 1)
    _, x, _, y = PPIDatasetCls.makeDataset(str(Path(args.prepared_csv).resolve()), input_shape, label_shape, training=False)
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    model = load_model(str(Path(args.model).resolve()), compile=False)
    pred = model.predict(x, batch_size=args.batch_size, verbose=0)

    y_flat = y.reshape(-1)
    s_flat = pred.reshape(-1)
    mask = y_flat != DatasetParams.LABEL_PAD_CONST
    y_valid = y_flat[mask].astype(int)
    s_valid = s_flat[mask].astype(float)

    metrics = {
        "n_proteins": int(x.shape[0]),
        "n_residues": int(y_valid.size),
        "n_positive": int(y_valid.sum()),
        "positive_rate": float(y_valid.mean()) if y_valid.size else 0.0,
        "auprc": float(average_precision_score(y_valid, s_valid)),
        "auroc": float(roc_auc_score(y_valid, s_valid)),
    }
    best = None
    for thr in np.linspace(0.01, 0.99, 99):
        pred_label = (s_valid >= thr).astype(int)
        f1 = f1_score(y_valid, pred_label, zero_division=0)
        mcc = matthews_corrcoef(y_valid, pred_label)
        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "mcc": float(mcc),
                "precision": float(precision_score(y_valid, pred_label, zero_division=0)),
                "recall": float(recall_score(y_valid, pred_label, zero_division=0)),
            }
    pred_05 = (s_valid >= 0.5).astype(int)
    metrics.update(
        {
            "f1_at_0p5": float(f1_score(y_valid, pred_05, zero_division=0)),
            "mcc_at_0p5": float(matthews_corrcoef(y_valid, pred_05)),
            "precision_at_0p5": float(precision_score(y_valid, pred_05, zero_division=0)),
            "recall_at_0p5": float(recall_score(y_valid, pred_05, zero_division=0)),
            "best_threshold_by_f1": best,
        }
    )

    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

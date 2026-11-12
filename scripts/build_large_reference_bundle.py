#!/usr/bin/env python3
"""Build a CPU fp32 reference bundle of LARGE synthetic datasets for the TPU accuracy gate (CPU only, no TPU).

Why: the OpenML parity bundle (`ref_cpu_bundle.pt`) has only small datasets (<= 8,192 training rows), which all take
the fused (non-tiled) path of `tabicl_tpu_bucketed.py`. The tiled column/row path, where the
TABICL_TPU_ROW_PRECISION / TABICL_TPU_COL_PRECISION / bucket / tile knobs act, is used for
B x T x (H + 4) > 4.5M elements (about > 11k rows at 100 features). This bundle covers that path.

Datasets (deliberately not saturated, so precision changes are visible in the metric):
  L1  binary,   50,000 rows (45k train / 5k test) x 100 features (same shape as the headline benchmark)
  L2  10-class, 20,000 rows (18k / 2k) x 60 features
  L3  regression, 30,000 rows (27k / 3k) x 80 features
Reference: TabICLClassifier / TabICLRegressor on CPU, fp32, use_amp=False, kv_cache=False, 8 estimators, seed 42.
CPU probabilities / predictions are stored too, so the gate can also report max |p_tpu - p_cpu|.

Output format matches ref_cpu_bundle.pt: {"seed": 42, "datasets": [{name, task_type, X_train, y_train, X_test,
y_test, cpu_fp32_metrics: {roc_auc | rmse}, cpu_fp32_pred, cpu_seconds}]}

Usage (inside any pod, CPU only):
  python3 v5p_make_large_bundle.py --out /models/tabiclv2-v5p-infer/ref_cpu_bundle_large.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated subset of L1,L2,L3")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import torch

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    from sklearn.datasets import make_classification, make_regression
    from sklearn.metrics import roc_auc_score, root_mean_squared_error
    from tabicl import TabICLClassifier, TabICLRegressor

    seed = 42
    specs = [
        ("L1_binary_45k5k_f100", "classification",
         dict(n_samples=50000, n_features=100, n_informative=30, n_redundant=10, n_classes=2, flip_y=0.05,
              class_sep=0.8, random_state=1), 45000),
        ("L2_10class_18k2k_f60", "classification",
         dict(n_samples=20000, n_features=60, n_informative=30, n_redundant=5, n_classes=10, n_clusters_per_class=1,
              flip_y=0.02, class_sep=1.0, random_state=2), 18000),
        ("L3_regression_27k3k_f80", "regression",
         dict(n_samples=30000, n_features=80, n_informative=40, noise=25.0, random_state=3), 27000),
    ]
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    out = {"seed": seed, "datasets": [], "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch_threads": torch.get_num_threads()}
    for name, task, kw, n_tr in specs:
        if only and name.split("_")[0] not in only:
            continue
        if task == "classification":
            X, y = make_classification(**kw)
        else:
            X, y = make_regression(**kw)
            # Mild non-linearity so the task is not purely linear.
            y = y + 0.002 * (X[:, 0] * X[:, 1]) * np.std(y)
        X = X.astype(np.float32)
        Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
        t0 = time.perf_counter()
        if task == "classification":
            est = TabICLClassifier(n_estimators=8, random_state=seed, device="cpu", use_amp=False, kv_cache=False)
            est.fit(Xtr, ytr)
            pred = est.predict_proba(Xte)
            if len(est.classes_) == 2:
                metric = {"roc_auc": float(roc_auc_score((yte == est.classes_[1]).astype(int), pred[:, 1]))}
            else:
                metric = {"roc_auc": float(roc_auc_score(yte, pred, labels=est.classes_, multi_class="ovr",
                                                         average="macro"))}
        else:
            est = TabICLRegressor(n_estimators=8, random_state=seed, device="cpu", use_amp=False, kv_cache=False)
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
            metric = {"rmse": float(root_mean_squared_error(yte, pred))}
        secs = time.perf_counter() - t0
        out["datasets"].append({"name": name, "task_type": task, "X_train": Xtr, "y_train": ytr, "X_test": Xte,
                                "y_test": yte, "cpu_fp32_metrics": metric, "cpu_fp32_pred": np.asarray(pred),
                                "cpu_seconds": round(secs, 1)})
        print(json.dumps({"name": name, "task": task, "metric": metric, "cpu_seconds": round(secs, 1)}), flush=True)
        torch.save(out, args.out + ".partial")
    torch.save(out, args.out)
    if os.path.exists(args.out + ".partial"):
        os.remove(args.out + ".partial")
    print("WROTE", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

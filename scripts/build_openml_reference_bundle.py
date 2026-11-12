#!/usr/bin/env python3
"""TabICLv2 CPU FP32 Correctness Baseline (10 Classification + 5 Regression OpenML/TabArena Datasets).

Loads TabICLClassifier and TabICLRegressor on CPU in fp32 (use_amp=False),
evaluates 10 OpenML/TabArena classification datasets (accuracy, roc_auc, log_loss)
and 5 OpenML/TabArena regression datasets (rmse, r2) with fixed seed=42,
strictly splitting train/test BEFORE fitting any preprocessing.
Saves:
  - results/ref_cpu.csv (exact 22-column schema)
  - results/ref_cpu_predictions.json (summary metrics + missing-value & schema audit)
  - results/ref_cpu_bundle.pt (exact raw splits + CPU fp32 predictions for TPU accuracy verification)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import torch

from tabicl import TabICLClassifier, TabICLRegressor
from csv_schema import write_rows


# 10 canonical OpenML / TabArena-suite classification datasets (with backup OpenML IDs)
CLASSIFICATION_CANDIDATES: list[tuple[int, str]] = [
    (31, "credit-g"),
    (37, "diabetes"),
    (1464, "blood-transfusion"),
    (1489, "phoneme"),
    (1067, "kc1"),
    (1494, "qsar-biodeg"),
    (40701, "churn"),
    (40984, "segment"),
    (46, "splice"),
    (54, "vehicle"),
    (1462, "banknote-authentication"),
    (1510, "wdbc"),
    (23, "cmc"),
]

# 5 canonical OpenML / TabArena-suite regression datasets (with backup OpenML IDs)
REGRESSION_CANDIDATES: list[tuple[int, str]] = [
    (562, "cpu_small"),
    (189, "kin8nm"),
    (287, "wine_quality"),
    (183, "abalone"),
    (44959, "concrete_compressive_strength"),
    (549, "strikes"),
    (44963, "physiochemical_protein"),
]


def load_openml_split(
    data_id: int,
    task_type: str,
    seed: int = 42,
    max_rows: int = 4096,
) -> dict[str, Any]:
    ds = openml.datasets.get_dataset(
        data_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, categorical_indicator, attribute_names = ds.get_data(
        target=ds.default_target_attribute,
        dataset_format="dataframe",
    )
    assert isinstance(X, pd.DataFrame)
    y_ser = pd.Series(y)

    # Drop rows with missing target (never impute target labels)
    valid_target = ~y_ser.isna()
    dropped_target_rows = int((~valid_target).sum())
    X = X.loc[valid_target].reset_index(drop=True)
    y_ser = y_ser.loc[valid_target].reset_index(drop=True)

    # Inspect missing feature values before split (TabICLv2 natively handles NaNs inside train-fitted preprocessors)
    missing_cells = int(X.isna().sum().sum())
    total_cells = int(X.shape[0] * max(1, X.shape[1]))
    missing_frac = missing_cells / float(total_cells)

    if task_type == "classification":
        y_arr = y_ser.astype(str).to_numpy()
        # Drop ultrarare classes (<2 samples) if any so stratified split is well-defined
        vc = pd.Series(y_arr).value_counts()
        keep_classes = set(vc[vc >= 2].index)
        mask = np.isin(y_arr, list(keep_classes))
        X = X.loc[mask].reset_index(drop=True)
        y_arr = y_arr[mask]
        if len(X) > max_rows:
            X, _, y_arr, _ = train_test_split(
                X, y_arr, train_size=max_rows, random_state=seed, stratify=y_arr
            )
            X = X.reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
        )
        n_classes = int(len(np.unique(y_train)))
    else:
        y_arr = pd.to_numeric(y_ser, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(y_arr)
        X = X.loc[finite].reset_index(drop=True)
        y_arr = y_arr[finite]
        if len(X) > max_rows:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), size=max_rows, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y_arr = y_arr[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_arr, test_size=0.2, random_state=seed
        )
        n_classes = 0

    return {
        "data_id": data_id,
        "name": ds.name,
        "task_type": task_type,
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train,
        "y_test": y_test,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": n_classes,
        "dropped_missing_target_rows": dropped_target_rows,
        "missing_feature_cells": missing_cells,
        "missing_feature_frac": round(missing_frac, 6),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    classes: np.ndarray,
    proba: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, proba, labels=classes))
    if len(classes) == 2:
        pos_label = classes[1]
        y_bin = (y_true == pos_label).astype(np.int32)
        auc = float(roc_auc_score(y_bin, proba[:, 1]))
    else:
        auc = float(roc_auc_score(y_true, proba, labels=classes, multi_class="ovr", average="macro"))
    return {"accuracy": acc, "roc_auc": auc, "log_loss": ll}


def upload_to_gcs(bucket_name: str, prefix: str, local_path: Path, rel_dest: str) -> None:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix.strip('/')}/{rel_dest.lstrip('/')}")
    blob.upload_from_filename(str(local_path))
    print(f"[GCS] Uploaded {local_path} -> gs://{bucket_name}/{blob.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/tabicl_phase1_ref")
    ap.add_argument("--bucket", default="")
    ap.add_argument("--run-prefix", default="")
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warmup checkpoint download + 1 tiny synthetic CPU pass so download/import is never in timed p50_ms
    print("[1.1] Downloading/warming TabICLv2 Classifier & Regressor checkpoints on CPU (fp32)...", flush=True)
    rng = np.random.RandomState(0)
    Xw = rng.randn(64, 4).astype(np.float32)
    yw_cls = (Xw[:, 0] > 0).astype(int)
    clf_warm = TabICLClassifier(n_estimators=2, device="cpu", use_amp=False, random_state=args.seed)
    clf_warm.fit(Xw[:48], yw_cls[:48])
    _ = clf_warm.predict_proba(Xw[48:])

    yw_reg = (Xw[:, 0] * 2.0 + 0.1).astype(np.float32)
    reg_warm = TabICLRegressor(n_estimators=2, device="cpu", use_amp=False, random_state=args.seed)
    reg_warm.fit(Xw[:48], yw_reg[:48])
    _ = reg_warm.predict(Xw[48:])
    print("[1.1] CPU fp32 warmup complete.", flush=True)

    csv_rows: list[dict[str, Any]] = []
    bundle_datasets: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    # --- 10 Classification Datasets ---
    cls_done = 0
    for data_id, alias in CLASSIFICATION_CANDIDATES:
        if cls_done >= 10:
            break
        try:
            item = load_openml_split(data_id, "classification", seed=args.seed)
        except Exception as exc:
            print(f"[1.1][WARN] Skipping OpenML {data_id} ({alias}): {exc}", flush=True)
            continue

        clf = TabICLClassifier(
            n_estimators=args.n_estimators,
            batch_size=args.n_estimators,
            kv_cache=False,
            device="cpu",
            use_amp=False,
            offload_mode=False,
            random_state=args.seed,
        )
        t0 = time.perf_counter()
        clf.fit(item["X_train"], item["y_train"])
        proba = clf.predict_proba(item["X_test"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        y_pred = clf.classes_[np.argmax(proba, axis=1)]

        metrics = compute_classification_metrics(item["y_test"], clf.classes_, proba, y_pred)
        cls_done += 1
        print(
            f"[1.1][CLS {cls_done:02d}/10] id={data_id} ({item['name']}): "
            f"train={item['n_train']} test={item['n_test']} feat={item['n_features']} classes={item['n_classes']} "
            f"AUC={metrics['roc_auc']:.5f} ACC={metrics['accuracy']:.5f} LL={metrics['log_loss']:.5f} "
            f"time={elapsed_ms:.1f}ms",
            flush=True,
        )

        for m_name, m_val in metrics.items():
            csv_rows.append(
                {
                    "phase": "1.1_cpu_ref",
                    "path": "path_a_cpu_fp32",
                    "model": "tabicl-classifier-v2-20260212",
                    "dtype": "fp32",
                    "n_train": item["n_train"],
                    "n_test": item["n_test"],
                    "n_features": item["n_features"],
                    "n_classes": item["n_classes"],
                    "n_estimators": args.n_estimators,
                    "kv_cache": False,
                    "devices": "cpu:1",
                    "batch_datasets": 1,
                    "compile_s": 0.0,
                    "p50_ms": round(elapsed_ms, 3),
                    "p99_ms": round(elapsed_ms, 3),
                    "datasets_per_s": round(1000.0 / elapsed_ms, 4),
                    "test_rows_per_s": round(item["n_test"] * 1000.0 / elapsed_ms, 2),
                    "peak_hbm_gib": 0.0,
                    "metric_name": m_name,
                    "metric_value": round(m_val, 6),
                    "ref_metric_value": round(m_val, 6),
                    "notes": f"openml_id={data_id};name={item['name']};missing_frac={item['missing_feature_frac']};paper_h100_50kx100_ref=<10s",
                }
            )

        item_save = dict(item)
        item_save["classes"] = clf.classes_
        item_save["cpu_fp32_proba"] = proba
        item_save["cpu_fp32_pred"] = y_pred
        item_save["cpu_fp32_metrics"] = metrics
        item_save["cpu_fp32_ms"] = elapsed_ms
        bundle_datasets.append(item_save)
        summary_records.append(
            {
                "data_id": data_id,
                "name": item["name"],
                "task_type": "classification",
                "n_train": item["n_train"],
                "n_test": item["n_test"],
                "n_features": item["n_features"],
                "n_classes": item["n_classes"],
                "missing_feature_frac": item["missing_feature_frac"],
                "metrics": metrics,
                "cpu_fp32_ms": round(elapsed_ms, 2),
            }
        )

    assert cls_done == 10, f"Expected 10 classification datasets, got {cls_done}"

    # --- 5 Regression Datasets ---
    reg_done = 0
    for data_id, alias in REGRESSION_CANDIDATES:
        if reg_done >= 5:
            break
        try:
            item = load_openml_split(data_id, "regression", seed=args.seed)
        except Exception as exc:
            print(f"[1.1][WARN] Skipping OpenML {data_id} ({alias}): {exc}", flush=True)
            continue

        reg = TabICLRegressor(
            n_estimators=args.n_estimators,
            batch_size=args.n_estimators,
            kv_cache=False,
            device="cpu",
            use_amp=False,
            offload_mode=False,
            random_state=args.seed,
        )
        t0 = time.perf_counter()
        reg.fit(item["X_train"], item["y_train"])
        y_pred = reg.predict(item["X_test"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rmse = float(root_mean_squared_error(item["y_test"], y_pred))
        r2 = float(r2_score(item["y_test"], y_pred))
        metrics = {"rmse": rmse, "r2": r2}
        reg_done += 1
        print(
            f"[1.1][REG {reg_done:02d}/05] id={data_id} ({item['name']}): "
            f"train={item['n_train']} test={item['n_test']} feat={item['n_features']} "
            f"RMSE={rmse:.5f} R2={r2:.5f} time={elapsed_ms:.1f}ms",
            flush=True,
        )

        for m_name, m_val in metrics.items():
            csv_rows.append(
                {
                    "phase": "1.1_cpu_ref",
                    "path": "path_a_cpu_fp32",
                    "model": "tabicl-regressor-v2-20260212",
                    "dtype": "fp32",
                    "n_train": item["n_train"],
                    "n_test": item["n_test"],
                    "n_features": item["n_features"],
                    "n_classes": 0,
                    "n_estimators": args.n_estimators,
                    "kv_cache": False,
                    "devices": "cpu:1",
                    "batch_datasets": 1,
                    "compile_s": 0.0,
                    "p50_ms": round(elapsed_ms, 3),
                    "p99_ms": round(elapsed_ms, 3),
                    "datasets_per_s": round(1000.0 / elapsed_ms, 4),
                    "test_rows_per_s": round(item["n_test"] * 1000.0 / elapsed_ms, 2),
                    "peak_hbm_gib": 0.0,
                    "metric_name": m_name,
                    "metric_value": round(m_val, 6),
                    "ref_metric_value": round(m_val, 6),
                    "notes": f"openml_id={data_id};name={item['name']};missing_frac={item['missing_feature_frac']};paper_h100_50kx100_ref=<10s",
                }
            )

        item_save = dict(item)
        item_save["cpu_fp32_pred"] = y_pred
        item_save["cpu_fp32_metrics"] = metrics
        item_save["cpu_fp32_ms"] = elapsed_ms
        bundle_datasets.append(item_save)
        summary_records.append(
            {
                "data_id": data_id,
                "name": item["name"],
                "task_type": "regression",
                "n_train": item["n_train"],
                "n_test": item["n_test"],
                "n_features": item["n_features"],
                "n_classes": 0,
                "missing_feature_frac": item["missing_feature_frac"],
                "metrics": metrics,
                "cpu_fp32_ms": round(elapsed_ms, 2),
            }
        )

    assert reg_done == 5, f"Expected 5 regression datasets, got {reg_done}"

    csv_path = out_dir / "ref_cpu.csv"
    write_rows(csv_path, csv_rows, append=False)

    json_path = out_dir / "ref_cpu_summary.json"
    json_path.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")

    bundle_path = out_dir / "ref_cpu_bundle.pt"
    torch.save({"seed": args.seed, "n_estimators": args.n_estimators, "datasets": bundle_datasets}, bundle_path)
    print(f"[1.1] Saved {csv_path}, {json_path}, and {bundle_path}", flush=True)

    if args.bucket and args.run_prefix:
        upload_to_gcs(args.bucket, args.run_prefix, csv_path, "results/ref_cpu.csv")
        upload_to_gcs(args.bucket, args.run_prefix, json_path, "results/ref_cpu_summary.json")
        upload_to_gcs(args.bucket, args.run_prefix, bundle_path, "results/ref_cpu_bundle.pt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

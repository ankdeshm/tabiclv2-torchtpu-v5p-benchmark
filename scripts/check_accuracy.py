#!/usr/bin/env python3
"""Accuracy gate for the compiled TabICLv2 runner on TPU v5p, per tuning-knob variant.

Runs every dataset in the CPU-reference bundle (`ref_cpu_bundle.pt`: 10 OpenML classification + 5 regression
datasets with CPU fp32 metrics) through `TabICLClassifier` / `TabICLRegressor` with `attach_compiled_runner`
(bf16 AMP, no KV cache, 8 estimators) and compares with the CPU fp32 reference:
  classification: |AUC_tpu - AUC_cpu| <= 0.005 ;  regression: |RMSE_tpu / RMSE_cpu - 1| <= 1%
Knobs are taken from the environment (see tabicl_tpu_bucketed.py) and recorded.

Usage: python3 check_accuracy.py --tag <variant> --bundle <path> --out <jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
KNOBS = ("TABICL_TPU_COL_PRECISION", "TABICL_TPU_ROW_PRECISION", "TABICL_TPU_ROW_BUCKET_STEP", "TABICL_TPU_COL_GROUP_TILE",
         "TABICL_TPU_ICL_KV_TRIM", "TABICL_TPU_FEATURE_STEP", "TABICL_TPU_SPLASH_BQ", "TABICL_TPU_SPLASH_BKV",
         "TABICL_TPU_SPLASH_BKV_COMPUTE", "TABICL_TPU_SDPA_MATH", "TABICL_TPU_SDPA_MATH_MAX_ELEMS", "TABICL_TPU_FAST_INPROJ", "TABICL_TPU_ROW_TILE",
         "TABICL_TPU_COL_AUTOCAST", "TABICL_TPU_ROW_AUTOCAST", "TABICL_TPU_MHA", "TABICL_TPU_MHA_PROJ")


def run_inner(args: argparse.Namespace) -> int:
    import torch
    import torch_tpu  # noqa: F401
    from sklearn.metrics import roc_auc_score, root_mean_squared_error
    from tabicl import TabICLClassifier, TabICLRegressor
    from tabicl_tpu_bucketed import attach_compiled_runner

    dev = torch.device("tpu:0")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    worst_auc, worst_rmse, n_fail = 0.0, 0.0, 0
    worst_pred = 0.0
    # Resumable across Spot preemptions: datasets with a PASS/FAIL row for this variant from an interrupted earlier
    # pass (fresh pod, same knobs) are not re-run; their rows are folded into this pass's summary.
    prev: dict[str, dict] = {}
    if out.exists():
        for ln in out.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("config", r.get("variant")) == args.tag and not r.get("summary") and r.get("gate") in ("PASS", "FAIL"):
                prev[r["name"]] = r
    resumed: list[str] = []
    for ds in bundle["datasets"]:
        if args.only and ds["name"] not in args.only.split(","):
            continue
        if ds["name"] in prev:
            r = prev[ds["name"]]
            worst_auc = max(worst_auc, float(r.get("abs_diff", 0.0) or 0.0))
            worst_rmse = max(worst_rmse, float(r.get("rel_diff", 0.0) or 0.0))
            worst_pred = max(worst_pred, float(r.get("max_abs_pred_diff_vs_cpu", 0.0) or 0.0))
            n_fail += int(r["gate"] == "FAIL")
            resumed.append(ds["name"])
            print(f"[v5p-parity] reuse {ds['name']}: {r['gate']} (row from an earlier interrupted pass)", flush=True)
            continue
        rec = {"config": args.tag, "name": ds["name"], "task": ds["task_type"], "knobs": {k: os.environ.get(k, "") for k in KNOBS},
               "n_train": int(len(ds["X_train"])), "n_features": int(np.asarray(ds["X_train"]).shape[1]),
               "compile_cache": os.environ.get("TORCH_TPU_CACHE_DIR", "")}
        try:
            t0 = time.perf_counter()
            if ds["task_type"] == "classification":
                est = TabICLClassifier(n_estimators=8, random_state=int(bundle.get("seed", 42)), device=dev, use_amp=True, kv_cache=False)
                attach_compiled_runner(est)
                est.fit(ds["X_train"], ds["y_train"])
                proba = est.predict_proba(ds["X_test"])
                pred_out = proba
                if len(est.classes_) == 2:
                    auc = float(roc_auc_score((ds["y_test"] == est.classes_[1]).astype(int), proba[:, 1]))
                else:
                    auc = float(roc_auc_score(ds["y_test"], proba, labels=est.classes_, multi_class="ovr", average="macro"))
                ref = float(ds["cpu_fp32_metrics"]["roc_auc"])
                d = abs(auc - ref)
                rec.update(auc=round(auc, 6), auc_cpu=round(ref, 6), abs_diff=round(d, 6), gate="PASS" if d <= 0.005 else "FAIL")
                worst_auc = max(worst_auc, d)
            else:
                est = TabICLRegressor(n_estimators=8, random_state=int(bundle.get("seed", 42)), device=dev, use_amp=True, kv_cache=False)
                attach_compiled_runner(est)
                est.fit(ds["X_train"], ds["y_train"])
                pred = est.predict(ds["X_test"])
                pred_out = pred
                rmse = float(root_mean_squared_error(ds["y_test"], pred))
                ref = float(ds["cpu_fp32_metrics"]["rmse"])
                rel = abs(rmse / ref - 1.0)
                rec.update(rmse=round(rmse, 6), rmse_cpu=round(ref, 6), rel_diff=round(rel, 6), gate="PASS" if rel <= 0.01 else "FAIL")
                worst_rmse = max(worst_rmse, rel)
            rec["seconds_first_call"] = round(time.perf_counter() - t0, 2)
            if ds.get("cpu_fp32_pred") is not None:
                try:
                    cpu_pred = np.asarray(ds["cpu_fp32_pred"])
                    tpu_pred = np.asarray(pred_out)
                    if cpu_pred.shape == tpu_pred.shape and np.issubdtype(cpu_pred.dtype, np.floating):
                        cpu_pred = cpu_pred.astype(np.float64)
                        diff = np.abs(tpu_pred.astype(np.float64) - cpu_pred)
                        if ds["task_type"] != "classification":
                            diff = diff / (np.std(cpu_pred) + 1e-12)  # regression: in units of the CPU prediction std
                        rec["max_abs_pred_diff_vs_cpu"] = round(float(diff.max()), 6)
                        rec["mean_abs_pred_diff_vs_cpu"] = round(float(diff.mean()), 6)
                        worst_pred = max(worst_pred, float(diff.max()))
                except Exception:  # noqa: BLE001  (diagnostic only; never affects the gate)
                    pass
            if args.save_preds_dir:
                try:
                    pdir = Path(args.save_preds_dir) / args.tag
                    pdir.mkdir(parents=True, exist_ok=True)
                    np.save(pdir / f"{ds['name']}.npy", np.asarray(pred_out))
                except Exception as _e:  # noqa: BLE001  (diagnostic only)
                    rec["save_preds_error"] = str(_e)[:120]
            n_fail += int(rec["gate"] == "FAIL")
        except Exception as exc:  # noqa: BLE001
            rec.update(gate="ERROR", error=str(exc)[:300])
            n_fail += 1
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[v5p-parity] " + json.dumps(rec), flush=True)
    summ = {"config": args.tag, "summary": True, "bundle": os.path.basename(args.bundle),
            "max_abs_auc_diff": round(worst_auc, 6), "max_rmse_rel_diff": round(worst_rmse, 6),
            "max_abs_pred_diff_vs_cpu": round(worst_pred, 6),
            "n_fail": n_fail, "all_pass": n_fail == 0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reused_rows_from_interrupted_pass": resumed,
            "knobs": {k: os.environ.get(k, "") for k in KNOBS}, "compile_cache": os.environ.get("TORCH_TPU_CACHE_DIR", "")}
    with out.open("a") as f:
        f.write(json.dumps(summ) + "\n")
    print("[v5p-parity] " + json.dumps(summ), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--preset", choices=["default", "best"], default="best", help="Use 'best' for the Pallas kernel + 4k/f4 bucketing configuration")
    ap.add_argument("--tag", default="")
    ap.add_argument("--bundle", required=True, help="Path to CPU fp32 reference bundle (.pt) built by build_openml_reference_bundle.py or build_large_reference_bundle.py")
    ap.add_argument("--out", default="/tmp/v5p_parity.jsonl")
    ap.add_argument("--only", default="", help="comma-separated dataset names (default: all)")
    # The TorchTPU persistent compile cache key does not include torch.set_float32_matmul_precision, so variants
    # that differ only in precision knobs MUST NOT share a cache directory (they would silently reuse executables).
    ap.add_argument("--cache-dir", default="", help="default: /tmp/torchtpu_cc_<tag> (fresh per variant)")
    ap.add_argument("--save-preds-dir", default="", help="optional: save per-dataset predictions to <dir>/<tag>/<name>.npy")
    args = ap.parse_args()
    if not args.tag:
        args.tag = "best" if args.preset == "best" else "baseline"
    if args.inner:
        return run_inner(args)
    from tpu_single_host_setup import build_single_host_env

    env = build_single_host_env(port=18080, cache_dir=args.cache_dir or f"/tmp/torchtpu_cc_{args.tag}")
    if args.preset == "best":
        for _k, _v in {'TABICL_TPU_COL_PRECISION': 'high', 'TABICL_TPU_ROW_BUCKET_STEP': '4096', 'TABICL_TPU_COL_GROUP_TILE': 'auto', 'TABICL_TPU_ICL_KV_TRIM': '1', 'TABICL_TPU_FEATURE_STEP': '4', 'TABICL_TPU_SPLASH_BKV': '4096', 'TABICL_TPU_SPLASH_BKV_COMPUTE': '256', 'TABICL_TPU_MHA': 'col1=pk,col2=pk,row=pk,cls=pk', 'TABICL_TPU_MHA_ROPE': 'mm', 'TABICL_TPU_MHA_PROJ': 'views', 'TABICL_TPU_MHA_HXQ': '1', 'TABICL_TPU_MHA_PACKED': '1', 'TABICL_TPU_FAST_GROUP': '1'}.items():
            env.setdefault(_k, _v)
    # Isolate every compile-cache layer per variant. The v2 gate (fresh TorchTPU cache dirs per variant, same pod)
    # still produced bit-identical outputs for variants that differ only in matmul precision, so another cache
    # (Inductor FX-graph / AOT-autograd cache, default /tmp/torchinductor_<user>) is not keyed on the precision.
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{args.tag}"
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"
    return subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "--inner", "--tag", args.tag,
                            "--bundle", args.bundle, "--out", args.out, "--only", args.only,
                            "--save-preds-dir", args.save_preds_dir], env=env)


if __name__ == "__main__":
    raise SystemExit(main())

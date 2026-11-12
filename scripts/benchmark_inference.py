#!/usr/bin/env python3
"""TabICLv2 inference benchmark on one TPU v5p chip (single-host TorchTPU, one device), with tuning knobs.

Measures, on `tpu:0` of one v5p host (run via the single-host env in `tpu_single_host_setup.py`):
  A. Model-forward timings (same method as the v7x Phase 1.5 harness: random tensors already in model layout,
     bucket padding + host<->device copies included, sklearn preprocessing excluded), p50 over timed iterations
     after warm-ups:
       - no-KV forward, KV-cache build (prefill, single timed run after a warm-up run) and KV query.
  B. End-to-end scikit-learn `fit` + `predict_proba` with `TabICLClassifier` (preprocessing + ensemble views +
     compiled forward), timed after one warm-up call, on a synthetic `make_classification` dataset.

Tuning knobs are read by `tabicl_tpu_bucketed.py` from the environment and recorded in every output row:
  TABICL_TPU_COL_PRECISION (highest|high|medium), TABICL_TPU_ROW_PRECISION, TABICL_TPU_ROW_BUCKET_STEP,
  TABICL_TPU_COL_GROUP_TILE.

Usage (inside a pod on a v5p host):
  python3 benchmark_inference.py --tag baseline --out /models/tabiclv2-v5p-infer/baseline.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

KNOBS = ("TABICL_TPU_COL_PRECISION", "TABICL_TPU_ROW_PRECISION", "TABICL_TPU_ROW_BUCKET_STEP", "TABICL_TPU_COL_GROUP_TILE",
         "TABICL_TPU_ICL_KV_TRIM", "TABICL_TPU_ICL_KV_STEP", "TABICL_TPU_SPLASH_BQ", "TABICL_TPU_SPLASH_BKV",
         "TABICL_TPU_SPLASH_BKV_COMPUTE", "TABICL_TPU_ROW_TILE", "TABICL_FAST_PREPROCESS", "TABICL_REUSE_MODEL",
         "TABICL_PREPROCESS_THREADS", "TABICL_TPU_FEATURE_STEP", "TABICL_TPU_SDPA_MATH", "TABICL_TPU_SDPA_MATH_MAX_ELEMS", "TABICL_TPU_FAST_INPROJ",
         "TABICL_TPU_COL_AUTOCAST", "TABICL_TPU_ROW_AUTOCAST", "TABICL_TPU_SPLASH_MIN_KV", "TABICL_TPU_MHA", "TABICL_TPU_MHA_PROJ")


def _emit(out: Path, row: dict[str, Any]) -> None:
    row = dict(row)
    row["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row["knobs"] = {k: os.environ.get(k, "") for k in KNOBS}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("[v5p-infer] " + json.dumps(row), flush=True)


def run_inner(args: argparse.Namespace) -> int:
    import torch
    import torch_tpu  # noqa: F401
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from tabicl import TabICLClassifier
    from tabicl_tpu_bucketed import BucketedCompiledTabICL, attach_compiled_runner

    out = Path(args.out)
    # Resumable across Spot preemptions: specs that already have an "ok" row for this variant in --out are skipped
    # (a resumed spec still compiles from scratch in its new pod, so first-call times stay comparable).
    done_tags: set[str] = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("config", r.get("variant")) == args.tag and r.get("status") == "ok":
                done_tags.add(str(r.get("tag")))
    dev = torch.device("tpu:0")
    rng = np.random.RandomState(42)
    X_init = rng.randn(64, 8).astype(np.float32)
    y_init = (X_init[:, 0] > 0).astype(np.int64)
    base = TabICLClassifier(n_estimators=8, random_state=42, device=dev)
    base.fit(X_init, y_init)
    runner = BucketedCompiledTabICL(base.model_, dev, use_amp=True, kv_cache_dtype=torch.bfloat16)

    def hbm_reset() -> None:
        try:
            torch.accelerator.reset_peak_memory_stats(dev)
        except Exception:  # noqa: BLE001
            pass

    def hbm_peak() -> float:
        try:
            return round(torch.accelerator.max_memory_allocated(dev) / 1024**3, 2)
        except Exception:  # noqa: BLE001
            return -1.0

    # ---------------- A. model-forward timings ----------------
    specs = [
        # tag, n_train, n_test, n_feat, n_classes, n_est, mode, iters, warmups
        ("headline_45k5k_nocache", 45000, 5000, 100, 2, 8, "nocache", 5, 2),
        ("headline_45k5k_kv", 45000, 5000, 100, 2, 8, "kv", 5, 2),
        ("50k1k_est1_nocache", 50000, 1000, 100, 2, 1, "nocache", 5, 2),
        ("10k1k_nocache", 10000, 1000, 100, 10, 8, "nocache", 10, 2),
        ("10k1k_kv", 10000, 1000, 100, 10, 8, "kv", 10, 2),
        ("1k1k_nocache", 1000, 1000, 100, 10, 8, "nocache", 20, 3),
    ]
    if args.quick:
        specs = [s for s in specs if s[0] in ("headline_45k5k_nocache", "10k1k_nocache")]
    if args.e2e_only:
        specs = []
    for tag, n_tr, n_te, n_f, n_c, n_est, mode, iters, warm in specs:
        if tag in done_tags:
            print(f"[v5p-infer] skip {tag}: already ok in {out}", flush=True)
            continue
        view_batches = [1] if n_est == 1 else [n_est // 2, n_est - n_est // 2]
        row: dict[str, Any] = {"suite": "model_forward", "tag": tag, "config": args.tag, "n_train": n_tr, "n_test": n_te,
                               "n_features": n_f, "n_classes": n_c, "n_estimators": n_est, "mode": mode,
                               "device": "tpu_v5p:1 (one chip)", "iters": iters, "warmups": warm}
        try:
            gc.collect()
            hbm_reset()
            if mode == "nocache":
                views = []
                for b in view_batches:
                    Xv = torch.randn(b, n_tr + n_te, n_f, dtype=torch.float32)
                    yv = torch.randint(0, max(n_c, 2), (b, n_tr), dtype=torch.float32)
                    views.append((Xv, yv))
                t0 = time.perf_counter()
                for Xv, yv in views:
                    runner.forward_nocache_bucketed(Xv, yv, num_classes=n_c, return_logits=True)
                row["first_call_s"] = round(time.perf_counter() - t0, 3)
                for _ in range(warm):
                    for Xv, yv in views:
                        runner.forward_nocache_bucketed(Xv, yv, num_classes=n_c, return_logits=True)
                hbm_reset()
                lat = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    for Xv, yv in views:
                        runner.forward_nocache_bucketed(Xv, yv, num_classes=n_c, return_logits=True)
                    lat.append((time.perf_counter() - t0) * 1000.0)
            else:
                trv, tev = [], []
                for b in view_batches:
                    trv.append((torch.randn(b, n_tr, n_f, dtype=torch.float32),
                                torch.randint(0, max(n_c, 2), (b, n_tr), dtype=torch.float32)))
                    tev.append(torch.randn(b, n_te, n_f, dtype=torch.float32))
                caches = [runner.prefill_kv_bucketed(X, y, num_classes=n_c, cache_dtype=torch.bfloat16)[0] for X, y in trv]
                del caches
                gc.collect()
                t0 = time.perf_counter()
                caches = [runner.prefill_kv_bucketed(X, y, num_classes=n_c, cache_dtype=torch.bfloat16)[0] for X, y in trv]
                row["kv_build_ms_single_run"] = round((time.perf_counter() - t0) * 1000.0, 2)
                for Xte, c in zip(tev, caches):
                    runner.query_kv_bucketed(Xte, cache=c, return_logits=True)
                for _ in range(warm):
                    for Xte, c in zip(tev, caches):
                        runner.query_kv_bucketed(Xte, cache=c, return_logits=True)
                hbm_reset()
                lat = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    for Xte, c in zip(tev, caches):
                        runner.query_kv_bucketed(Xte, cache=c, return_logits=True)
                    lat.append((time.perf_counter() - t0) * 1000.0)
                del caches
            row["p50_ms"] = round(float(np.median(lat)), 2)
            row["p99_ms"] = round(float(np.percentile(lat, 99)), 2)
            row["peak_hbm_gib"] = hbm_peak()
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)[:400]
        _emit(out, row)

    # ---------------- B. end-to-end sklearn fit + predict_proba ----------------
    e2e = [("e2e_45k5k_nocache", 45000, 5000, 100, False), ("e2e_45k5k_kv", 45000, 5000, 100, True),
           ("e2e_9k1k_nocache", 9000, 1000, 100, False)]
    if args.quick:
        e2e = e2e[:1]
    for tag, n_tr, n_te, n_f, kv in e2e:
        if tag in done_tags:
            print(f"[v5p-infer] skip {tag}: already ok in {out}", flush=True)
            continue
        row = {"suite": "e2e_sklearn", "tag": tag, "config": args.tag, "n_train": n_tr, "n_test": n_te, "n_features": n_f,
               "n_estimators": 8, "kv_cache": kv, "device": "tpu_v5p:1 (one chip)"}
        try:
            X, y = make_classification(n_samples=n_tr + n_te, n_features=n_f, n_informative=min(20, n_f), n_redundant=0,
                                       n_classes=2, random_state=0)
            X = X.astype(np.float32)
            Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
            clf = TabICLClassifier(n_estimators=8, random_state=0, device=dev, use_amp=True, kv_cache=kv)
            attach_compiled_runner(clf)
            t0 = time.perf_counter()
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)
            row["first_call_s"] = round(time.perf_counter() - t0, 2)
            fits, preds = [], []
            for _ in range(args.e2e_iters):
                gc.collect()
                t0 = time.perf_counter()
                clf.fit(Xtr, ytr)
                t1 = time.perf_counter()
                p = clf.predict_proba(Xte)
                t2 = time.perf_counter()
                fits.append(t1 - t0)
                preds.append(t2 - t1)
            row["fit_s_p50"] = round(float(np.median(fits)), 3)
            row["predict_s_p50"] = round(float(np.median(preds)), 3)
            row["fit_plus_predict_s_p50"] = round(float(np.median(np.array(fits) + np.array(preds))), 3)
            row["iters"] = args.e2e_iters
            row["auc"] = round(float(roc_auc_score(yte, p[:, 1])), 5)
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)[:400]
        _emit(out, row)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--preset", choices=["default", "best"], default="best", help="Use 'best' for the Pallas kernel + 4k/f4 bucketing configuration, or 'default' for power-of-two baseline")
    ap.add_argument("--fast-fit-procs", type=int, default=0, help="Enable shared-memory Yeo-Johnson parallel fit with N worker processes (e.g. 64) and model reuse")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="/tmp/v5p_infer.jsonl")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--e2e-only", action="store_true", help="skip the model-forward suite (A); run only e2e (B)")
    ap.add_argument("--e2e-iters", type=int, default=3)
    # The TorchTPU persistent compile cache key does not include torch.set_float32_matmul_precision: variants that
    # differ only in precision knobs must not share a cache directory.
    ap.add_argument("--cache-dir", default="", help="default: /tmp/torchtpu_cc_<tag> (fresh per variant)")
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
    if args.fast_fit_procs > 0:
        env["TABICL_FAST_PREPROCESS"] = "1"
        env["TABICL_PREPROCESS_PROCS"] = str(args.fast_fit_procs)
        env["TABICL_REUSE_MODEL"] = "1" 
    # Isolate every compile-cache layer per variant (see v5p_parity_compiled.py).
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{args.tag}"
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
    env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--inner", "--tag", args.tag, "--out", args.out,
           "--e2e-iters", str(args.e2e_iters)] + (["--quick"] if args.quick else []) + (["--e2e-only"] if args.e2e_only else [])
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

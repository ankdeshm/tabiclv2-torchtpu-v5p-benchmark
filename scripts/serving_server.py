#!/usr/bin/env python3
"""TabICLv2 Multi-Process & Multi-Host FastAPI Throughput Server on TorchTPU.

Architecture per TPU v5p Host (`4` chips = `4` worker processes):
  - `8` Worker Processes (`torchrun --nnodes=1 --nproc_per_node=8`, `backend="tpu_dist"`), each bound
    exclusively to `1` TPU TensorCore (`local_rank = 0..7`) and serving FastAPI on `127.0.0.1:(8100 + local_rank)`:
      * Pre-compiled `BucketedCompiledTabICL` (`torch.compile(backend="tpu", dynamic=False, fullgraph=True)`)
      * Cross-dataset batched execution along `B` (`POST /worker/predict_batch_bin`)
      * Per-`dataset_id` LRU `BucketedKVCache` store in TPU HBM (`POST /worker/prefill_bin`, `POST /worker/query_bin`)
  - `1` Async FastAPI Dispatcher (`0.0.0.0:8000`, exposed via Kubernetes `Service/tabicl-serving-svc`):
      * Shape-bucket routing + size-tier isolation (reserving fast lanes for `<5K` requests vs `50K-250K` heavy tables)
      * Dynamic micro-batching (`2.5 ms` window) stacking multiple concurrent datasets along `B` (`B <= 32`)
      * Deterministic `crc32(dataset_id) % 8` KV-cache affinity routing
"""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
import datetime as dt
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from typing import Any
import zlib

import numpy as np

# FastAPI resolves the (string, because of `from __future__ import annotations`) endpoint annotations such as
# `req: Request` against this module's globals. They must be importable here; otherwise FastAPI treats `req` as a
# required query parameter and every request fails with 422 Unprocessable Entity.
try:
    from fastapi import Request, Response  # noqa: F401
except Exception:  # noqa: BLE001  (module still importable without FastAPI, e.g. for pack/unpack helpers)
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ==============================================================================
# Fast Binary Wire Helpers (`header_json_len (uint32) + header_json + raw float32 bytes`)
# ==============================================================================
def pack_arrays(meta: dict[str, Any], arrays: list[np.ndarray]) -> bytes:
    arr_specs = []
    raw_parts = []
    for a in arrays:
        c = np.ascontiguousarray(a)
        arr_specs.append({"dtype": str(c.dtype), "shape": list(c.shape), "nbytes": int(c.nbytes)})
        raw_parts.append(c.tobytes())
    hdr = json.dumps({"meta": meta, "arrays": arr_specs}).encode("utf-8")
    return struct.pack("<I", len(hdr)) + hdr + b"".join(raw_parts)


def unpack_arrays(buf: bytes) -> tuple[dict[str, Any], list[np.ndarray]]:
    (hdr_len,) = struct.unpack("<I", buf[:4])
    hdr = json.loads(buf[4 : 4 + hdr_len].decode("utf-8"))
    offset = 4 + hdr_len
    out_arrays = []
    for sp in hdr["arrays"]:
        nb = int(sp["nbytes"])
        arr = np.frombuffer(buf, dtype=np.dtype(sp["dtype"]), count=nb // np.dtype(sp["dtype"]).itemsize, offset=offset)
        out_arrays.append(arr.reshape(sp["shape"]).copy())
        offset += nb
    return hdr["meta"], out_arrays


# ==============================================================================
# Worker Process (`1` per TPU TensorCore, `local_rank = 0..7`)
# ==============================================================================
def run_tpu_workers(base_worker_port: int, prewarm_mode: str) -> int:
    import fastapi
    from fastapi import Request, Response
    import torch
    import torch_tpu  # noqa: F401
    import torch.distributed as dist
    import uvicorn
    from tabicl import TabICLClassifier
    from tabicl_tpu_bucketed import (
        BATCH_BUCKETS,
        FEATURE_BUCKETS,
        ROW_BUCKETS,
        BucketedCompiledTabICL,
        BucketedKVCache,
        pick_bucket,
        pick_feature_bucket,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.accelerator.set_device_index(0)
    dev = torch.device("tpu:0")
    dist.init_process_group(backend="tpu_dist", timeout=dt.timedelta(minutes=20))

    rng = np.random.RandomState(42 + local_rank)
    X_init = rng.randn(64, 8).astype(np.float32)
    y_init = (X_init[:, 0] > 0).astype(np.int64)
    base_clf = TabICLClassifier(n_estimators=8, random_state=42, device=dev)
    base_clf.fit(X_init, y_init)

    runner = BucketedCompiledTabICL(
        model=base_clf.model_,
        device=dev,
        use_amp=True,
        kv_cache_dtype=torch.bfloat16,
    )

    # Pre-warm common serving buckets from persistent Tier-3 compilation cache so load test measures steady-state
    if prewarm_mode != "none":
        warm_shapes = [
            (8, 800, 200, 16),
            (8, 800, 200, 32),
            (8, 3000, 500, 32),
            (8, 3000, 500, 64),
            (16, 800, 200, 32),
        ]
        if prewarm_mode == "full":
            warm_shapes.extend(
                [
                    (8, 12000, 1000, 64),
                    (8, 25000, 2000, 100),
                    (8, 60000, 2000, 100),
                ]
            )
        for B_w, n_tr_w, n_te_w, H_w in warm_shapes:
            Xw = torch.randn(B_w, n_tr_w + n_te_w, H_w, dtype=torch.float32)
            yw = torch.randint(0, 2, (B_w, n_tr_w), dtype=torch.float32)
            runner.forward_nocache_bucketed(Xw, yw, num_classes=2, return_logits=True)
            del Xw, yw

    # Synchronize all 8 workers after pre-warm before opening ports
    bar = torch.zeros(1, device=dev)
    dist.all_reduce(bar)
    torch.accelerator.synchronize(dev)

    kv_store: OrderedDict[str, dict[str, Any]] = OrderedDict()
    MAX_KV_ENTRIES = 48
    gpu_lock = asyncio.Lock()
    stats = {
        "worker_rank": local_rank,
        "hostname": socket.gethostname(),
        "requests_served": 0,
        "datasets_served": 0,
        "batched_calls": 0,
        "kv_prefills": 0,
        "kv_queries": 0,
    }

    app = fastapi.FastAPI(title=f"TabICLv2-TPU-Worker-{local_rank}")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        hbm_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
        return {
            "status": "ok",
            "worker_rank": local_rank,
            "world_size": world_size,
            "hostname": socket.gethostname(),
            "kv_entries": len(kv_store),
            "peak_hbm_gib": round(hbm_gib, 4),
            **stats,
        }

    def _prep_ensemble_views(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None,
        n_estimators: int,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
        """Fast vectorized 2-view ('none' + mean/std standardized) + feature/class shuffle ensemble builder."""
        from tabicl._sklearn.preprocessing import EnsembleGenerator, TransformToNumerical

        # Encode classes 0..C-1
        classes, y_enc = np.unique(y_train, return_inverse=True)
        n_classes = len(classes)
        X_tr_f = np.asarray(X_train, dtype=np.float32)

        eg = EnsembleGenerator(
            classification=True,
            n_estimators=n_estimators,
            norm_methods=["none", "power"] if X_tr_f.shape[0] <= 8192 else ["none"],
            feat_shuffle_method="latin",
            class_shuffle_method="shift",
            outlier_threshold=4.0,
            random_state=42,
        )
        eg.fit(X_tr_f, y_enc)
        class_shuffles: list[np.ndarray] = []
        for sh_list in eg.class_shuffles_.values():
            class_shuffles.extend(sh_list)

        if X_test is None:
            tr_data = eg.transform(X=None, mode="train")
            Xs_all = np.concatenate([v[0] for v in tr_data.values()], axis=0).astype(np.float32, copy=False)
            ys_all = np.concatenate([v[1] for v in tr_data.values()], axis=0).astype(np.float32, copy=False)
            return Xs_all, ys_all, class_shuffles, n_classes

        X_te_f = np.asarray(X_test, dtype=np.float32)
        both = eg.transform(X_te_f, mode="both")
        Xs_all = np.concatenate([v[0] for v in both.values()], axis=0).astype(np.float32, copy=False)
        ys_all = np.concatenate([v[1] for v in both.values()], axis=0).astype(np.float32, copy=False)
        return Xs_all, ys_all, class_shuffles, n_classes

    def _exec_batch_sync(items_meta: list[dict[str, Any]], arrays: list[np.ndarray]) -> bytes:
        """Execute 1 or K cross-dataset requests stacked along `B` on this TPU TensorCore."""
        t0 = time.perf_counter()
        prepared = []
        arr_idx = 0
        for im in items_meta:
            X_tr = arrays[arr_idx]
            y_tr = arrays[arr_idx + 1]
            X_te = arrays[arr_idx + 2]
            arr_idx += 3
            n_est = int(im.get("n_estimators", 8))
            Xs_v, ys_v, cls_shuf, n_cls = _prep_ensemble_views(X_tr, y_tr, X_te, n_est)
            prepared.append((Xs_v, ys_v, cls_shuf, n_cls, X_tr.shape[0], X_te.shape[0], X_tr.shape[1]))

        # Check if all items in the batch share `n_train` and fit in a single stacked `B` call (`B_total <= 32`)
        same_ntrain = len({p[4] for p in prepared}) == 1
        B_total = sum(p[0].shape[0] for p in prepared)
        max_T = max(p[0].shape[1] for p in prepared)
        max_H = max(p[0].shape[2] for p in prepared)

        out_probs: list[np.ndarray] = []
        if len(prepared) > 1 and same_ntrain and B_total <= 32 and max_T <= 16384:
            # True cross-dataset batching along `B` in ONE TPU call!
            T_bkt = pick_bucket(max_T, ROW_BUCKETS)
            H_bkt = pick_feature_bucket(max_H)
            B_bkt = pick_bucket(B_total, BATCH_BUCKETS)
            n_tr_shared = prepared[0][4]

            X_stack = torch.zeros((B_bkt, T_bkt, H_bkt), device=dev, dtype=torch.float32)
            y_stack = torch.zeros((B_bkt, T_bkt), device=dev, dtype=torch.float32)
            d_list = []

            cursor = 0
            slices = []
            for Xs_v, ys_v, cls_shuf, n_cls, n_tr_i, n_te_i, H_i in prepared:
                bv, tv, hv = Xs_v.shape
                X_stack[cursor : cursor + bv, :tv, :hv] = torch.from_numpy(Xs_v).to(dev)
                y_stack[cursor : cursor + bv, :n_tr_i] = torch.from_numpy(ys_v).to(dev)
                d_list.extend([hv] * bv)
                slices.append((cursor, bv, cls_shuf, n_cls, n_tr_i, n_te_i))
                cursor += bv

            while len(d_list) < B_bkt:
                X_stack[ len(d_list) ] = X_stack[0]
                y_stack[ len(d_list) ] = y_stack[0]
                d_list.append(d_list[0])

            d_vec = torch.tensor(d_list, device=dev, dtype=torch.long)
            train_sz = torch.tensor(n_tr_shared, device=dev, dtype=torch.long)

            with torch.no_grad():
                prev_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision("highest")
                try:
                    raw_full = runner._nocache_compiled(X_stack, y_stack, d_vec, train_sz)
                finally:
                    torch.set_float32_matmul_precision(prev_prec)
                torch.accelerator.synchronize(dev)
                raw_cpu = raw_full.float().cpu().numpy()

            for c0, bv, cls_shuf, n_cls, n_tr_i, n_te_i in slices:
                logits_i = raw_cpu[c0 : c0 + bv, n_tr_i : n_tr_i + n_te_i, :n_cls]
                avg = np.zeros((n_te_i, n_cls), dtype=np.float32)
                for idx_s, sh in enumerate(cls_shuf):
                    avg += logits_i[idx_s][..., sh]
                avg /= max(len(cls_shuf), 1)
                avg = avg / 0.9
                avg -= np.max(avg, axis=-1, keepdims=True)
                exp_a = np.exp(avg)
                probs = exp_a / np.sum(exp_a, axis=-1, keepdims=True)
                out_probs.append(probs.astype(np.float32))
        else:
            for Xs_v, ys_v, cls_shuf, n_cls, n_tr_i, n_te_i, H_i in prepared:
                logits_t, _ = runner.forward_nocache_bucketed(
                    torch.from_numpy(Xs_v),
                    torch.from_numpy(ys_v),
                    num_classes=n_cls,
                    return_logits=True,
                )
                logits_np = logits_t.cpu().numpy()
                avg = np.zeros((n_te_i, n_cls), dtype=np.float32)
                for idx_s, sh in enumerate(cls_shuf):
                    avg += logits_np[idx_s][..., sh]
                avg /= max(len(cls_shuf), 1)
                avg = avg / 0.9
                avg -= np.max(avg, axis=-1, keepdims=True)
                exp_a = np.exp(avg)
                probs = exp_a / np.sum(exp_a, axis=-1, keepdims=True)
                out_probs.append(probs.astype(np.float32))

        tpu_ms = (time.perf_counter() - t0) * 1000.0
        hbm_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
        stats["batched_calls"] += 1
        stats["requests_served"] += len(items_meta)
        stats["datasets_served"] += len(items_meta)
        return pack_arrays(
            {
                "worker_rank": local_rank,
                "batch_datasets": len(items_meta),
                "worker_ms": round(tpu_ms, 3),
                "peak_hbm_gib": round(hbm_gib, 4),
            },
            out_probs,
        )

    @app.post("/worker/predict_batch_bin")
    async def worker_predict_batch_bin(req: Request) -> Response:
        body = await req.body()
        meta, arrays = unpack_arrays(body)
        items_meta = meta["items"]
        async with gpu_lock:
            resp_bytes = await asyncio.to_thread(_exec_batch_sync, items_meta, arrays)
        return Response(content=resp_bytes, media_type="application/octet-stream")

    def _exec_prefill_sync(meta: dict[str, Any], arrays: list[np.ndarray]) -> dict[str, Any]:
        from tabicl._sklearn.preprocessing import EnsembleGenerator

        t0 = time.perf_counter()
        ds_id = str(meta["dataset_id"])
        n_est = int(meta.get("n_estimators", 8))
        X_tr = np.asarray(arrays[0], dtype=np.float32)
        y_tr = arrays[1]
        classes, y_enc = np.unique(y_tr, return_inverse=True)
        n_cls = len(classes)

        eg = EnsembleGenerator(
            classification=True,
            n_estimators=n_est,
            norm_methods=["none", "power"] if X_tr.shape[0] <= 8192 else ["none"],
            feat_shuffle_method="latin",
            class_shuffle_method="shift",
            outlier_threshold=4.0,
            random_state=42,
        )
        eg.fit(X_tr, y_enc)
        tr_data = eg.transform(X=None, mode="train")
        Xs_all = np.concatenate([v[0] for v in tr_data.values()], axis=0).astype(np.float32, copy=False)
        ys_all = np.concatenate([v[1] for v in tr_data.values()], axis=0).astype(np.float32, copy=False)

        cache_obj, comp_s = runner.prefill_kv_bucketed(
            torch.from_numpy(Xs_all),
            torch.from_numpy(ys_all),
            num_classes=n_cls,
            cache_dtype=torch.bfloat16,
        )
        class_shuffles: list[np.ndarray] = []
        for sh_list in eg.class_shuffles_.values():
            class_shuffles.extend(sh_list)

        while len(kv_store) >= MAX_KV_ENTRIES:
            kv_store.popitem(last=False)
        kv_store[ds_id] = {
            "eg": eg,
            "cache": cache_obj,
            "class_shuffles": class_shuffles,
            "n_classes": n_cls,
        }
        stats["kv_prefills"] += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "cached",
            "dataset_id": ds_id,
            "worker_rank": local_rank,
            "kv_nbytes": cache_obj.nbytes(),
            "compile_s": round(comp_s, 4),
            "prefill_ms": round(elapsed_ms, 3),
        }

    def _exec_query_sync(meta: dict[str, Any], arrays: list[np.ndarray]) -> bytes:
        t0 = time.perf_counter()
        ds_id = str(meta["dataset_id"])
        if ds_id not in kv_store:
            raise KeyError(f"dataset_id={ds_id} not found on worker {local_rank}")
        kv_store.move_to_end(ds_id)
        entry = kv_store[ds_id]
        eg = entry["eg"]
        cache_obj: BucketedKVCache = entry["cache"]
        cls_shuf = entry["class_shuffles"]
        n_cls = int(entry["n_classes"])

        X_te = np.asarray(arrays[0], dtype=np.float32)
        te_data = eg.transform(X_te, mode="test")
        Xs_te = np.concatenate([v[0] for v in te_data.values()], axis=0).astype(np.float32, copy=False)

        logits_t, _ = runner.query_kv_bucketed(
            torch.from_numpy(Xs_te),
            cache=cache_obj,
            return_logits=True,
        )
        logits_np = logits_t.cpu().numpy()
        avg = np.zeros((X_te.shape[0], n_cls), dtype=np.float32)
        for idx_s, sh in enumerate(cls_shuf):
            avg += logits_np[idx_s][..., sh]
        avg /= max(len(cls_shuf), 1)
        avg = avg / 0.9
        avg -= np.max(avg, axis=-1, keepdims=True)
        exp_a = np.exp(avg)
        probs = (exp_a / np.sum(exp_a, axis=-1, keepdims=True)).astype(np.float32)

        stats["kv_queries"] += 1
        stats["requests_served"] += 1
        stats["datasets_served"] += 1
        tpu_ms = (time.perf_counter() - t0) * 1000.0
        hbm_gib = torch.accelerator.max_memory_allocated(dev) / (1024.0**3)
        return pack_arrays(
            {
                "worker_rank": local_rank,
                "dataset_id": ds_id,
                "worker_ms": round(tpu_ms, 3),
                "peak_hbm_gib": round(hbm_gib, 4),
            },
            [probs],
        )

    @app.post("/worker/prefill_bin")
    async def worker_prefill_bin(req: Request) -> dict[str, Any]:
        body = await req.body()
        meta, arrays = unpack_arrays(body)
        async with gpu_lock:
            return await asyncio.to_thread(_exec_prefill_sync, meta, arrays)

    @app.post("/worker/query_bin")
    async def worker_query_bin(req: Request) -> Response:
        body = await req.body()
        meta, arrays = unpack_arrays(body)
        async with gpu_lock:
            try:
                resp_bytes = await asyncio.to_thread(_exec_query_sync, meta, arrays)
            except KeyError as exc:
                return Response(content=str(exc).encode(), status_code=404)
        return Response(content=resp_bytes, media_type="application/octet-stream")

    port = base_worker_port + local_rank
    print(f"[Worker {local_rank}/{world_size}] Listening on 127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


# ==============================================================================
# Frontend Dispatcher (`0.0.0.0:8000`) — Bucket Routing + Dynamic Batching along `B` + KV Affinity
# ==============================================================================
def run_dispatcher(host: str, port: int, base_worker_port: int, num_workers: int, batch_window_ms: float) -> int:
    import fastapi
    from fastapi import Request, Response
    import httpx
    import uvicorn
    from tabicl_tpu_bucketed import FEATURE_BUCKETS, ROW_BUCKETS, pick_bucket, pick_feature_bucket  # noqa: F401

    app = fastapi.FastAPI(title="TabICLv2-TPU-Dispatcher")
    worker_urls = [f"http://127.0.0.1:{base_worker_port + i}" for i in range(num_workers)]
    inflight = [0] * num_workers
    rr_cursor = 0

    # Dynamic batch queues keyed by `(T_bkt, H_bkt, n_train, n_estimators)`
    batch_queues: dict[tuple[int, int, int, int], list[tuple[dict[str, Any], list[np.ndarray], asyncio.Future[tuple[dict[str, Any], np.ndarray]]]]] = {}
    batch_timers: dict[tuple[int, int, int, int], asyncio.Task[None]] = {}

    client_holder: dict[str, httpx.AsyncClient] = {}

    @app.on_event("startup")
    async def _startup() -> None:
        limits = httpx.Limits(max_keepalive_connections=128, max_connections=256)
        client_holder["http"] = httpx.AsyncClient(
            timeout=httpx.Timeout(float(os.environ.get("TABICL_DISPATCH_TIMEOUT_S", "1800"))), limits=limits
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if "http" in client_holder:
            await client_holder["http"].aclose()

    def _pick_worker_for_bucket(T_bkt: int, H_bkt: int) -> int:
        nonlocal rr_cursor
        # Prefer the last quarter of workers for mammoth >=65K requests so <5K requests keep fast lanes
        # (v7x layout: workers 6..7 of 8; v5p host: worker 3 of 4).
        if T_bkt >= 65536 and num_workers >= 4:
            n_heavy = max(1, num_workers // 4)
            heavy = list(range(num_workers - n_heavy, num_workers))
            candidates = heavy[::-1] + [i for i in range(num_workers) if i not in heavy]
        else:
            preferred = zlib.crc32(f"{T_bkt}:{H_bkt}".encode()) % num_workers
            candidates = [(preferred + off) % num_workers for off in range(num_workers)]
        best = min(candidates, key=lambda idx: (inflight[idx], (idx - rr_cursor) % num_workers))
        rr_cursor = (best + 1) % num_workers
        return best

    async def _flush_key(bkey: tuple[int, int, int, int]) -> None:
        await asyncio.sleep(batch_window_ms / 1000.0)
        batch_timers.pop(bkey, None)
        q = batch_queues.pop(bkey, [])
        if not q:
            return

        T_bkt, H_bkt, _, n_est = bkey
        # Cap each cross-dataset TPU batch so total ensemble views <= 32
        views_per_ds = n_est
        max_ds_per_call = max(1, 32 // max(views_per_ds, 1))

        for i in range(0, len(q), max_ds_per_call):
            chunk = q[i : i + max_ds_per_call]
            asyncio.create_task(_dispatch_chunk(T_bkt, H_bkt, chunk))

    async def _dispatch_chunk(
        T_bkt: int,
        H_bkt: int,
        chunk: list[tuple[dict[str, Any], list[np.ndarray], asyncio.Future[tuple[dict[str, Any], np.ndarray]]]],
    ) -> None:
        w_idx = _pick_worker_for_bucket(T_bkt, H_bkt)
        inflight[w_idx] += 1
        try:
            items_meta = [c[0] for c in chunk]
            flat_arrays: list[np.ndarray] = []
            for c in chunk:
                flat_arrays.extend(c[1])
            payload = pack_arrays({"items": items_meta}, flat_arrays)
            resp = await client_holder["http"].post(
                f"{worker_urls[w_idx]}/worker/predict_batch_bin",
                content=payload,
                headers={"Content-Type": "application/octet-stream"},
            )
            resp.raise_for_status()
            w_meta, out_arrays = unpack_arrays(resp.content)
            for idx_item, (_, _, fut) in enumerate(chunk):
                if not fut.done():
                    fut.set_result((w_meta, out_arrays[idx_item]))
        except Exception as exc:
            for _, _, fut in chunk:
                if not fut.done():
                    fut.set_exception(exc)
        finally:
            inflight[w_idx] -= 1

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        cl = client_holder["http"]
        workers = []
        for u in worker_urls:
            try:
                r = await cl.get(f"{u}/healthz", timeout=5.0)
                workers.append(r.json())
            except Exception as exc:
                workers.append({"status": "down", "error": str(exc)})
        ok_cnt = sum(1 for w in workers if w.get("status") == "ok")
        return {
            "status": "ok" if ok_cnt == num_workers else "degraded",
            "hostname": socket.gethostname(),
            "ready_workers": ok_cnt,
            "total_workers": num_workers,
            "inflight": inflight,
            "workers": workers,
        }

    @app.post("/v1/predict_bin")
    async def v1_predict_bin(req: Request) -> Response:
        body = await req.body()
        meta, arrays = unpack_arrays(body)
        X_tr, y_tr, X_te = arrays[0], arrays[1], arrays[2]
        n_tr, H = X_tr.shape
        n_te = X_te.shape[0]
        n_est = int(meta.get("n_estimators", 8))

        T_bkt = pick_bucket(n_tr + n_te, ROW_BUCKETS)
        H_bkt = pick_feature_bucket(H)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[dict[str, Any], np.ndarray]] = loop.create_future()

        # Enable cross-dataset dynamic batching along `B` for small/medium tables (`T_bkt <= 8192`)
        if batch_window_ms > 0 and T_bkt <= 8192 and n_est <= 16:
            bkey = (T_bkt, H_bkt, n_tr, n_est)
            batch_queues.setdefault(bkey, []).append((meta, [X_tr, y_tr, X_te], fut))
            if bkey not in batch_timers:
                batch_timers[bkey] = asyncio.create_task(_flush_key(bkey))
            elif len(batch_queues[bkey]) >= max(1, 32 // max(n_est, 1)):
                t = batch_timers.pop(bkey)
                t.cancel()
                q_now = batch_queues.pop(bkey, [])
                asyncio.create_task(_dispatch_chunk(T_bkt, H_bkt, q_now))
        else:
            asyncio.create_task(_dispatch_chunk(T_bkt, H_bkt, [(meta, [X_tr, y_tr, X_te], fut)]))

        w_meta, probs = await fut
        out_hdr = {
            "hostname": socket.gethostname(),
            "bucket": [T_bkt, H_bkt],
            **w_meta,
        }
        return Response(content=pack_arrays(out_hdr, [probs]), media_type="application/octet-stream")

    @app.post("/v1/prefill_bin")
    async def v1_prefill_bin(req: Request) -> dict[str, Any]:
        body = await req.body()
        meta, _ = unpack_arrays(body)
        ds_id = str(meta["dataset_id"])
        w_idx = zlib.crc32(ds_id.encode()) % num_workers
        inflight[w_idx] += 1
        try:
            r = await client_holder["http"].post(
                f"{worker_urls[w_idx]}/worker/prefill_bin",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
            r.raise_for_status()
            out = r.json()
            out["hostname"] = socket.gethostname()
            return out
        finally:
            inflight[w_idx] -= 1

    @app.post("/v1/query_bin")
    async def v1_query_bin(req: Request) -> Response:
        body = await req.body()
        meta, _ = unpack_arrays(body)
        ds_id = str(meta["dataset_id"])
        w_idx = zlib.crc32(ds_id.encode()) % num_workers
        inflight[w_idx] += 1
        try:
            r = await client_holder["http"].post(
                f"{worker_urls[w_idx]}/worker/query_bin",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
            if r.status_code == 404:
                return Response(content=r.content, status_code=404)
            r.raise_for_status()
            w_meta, arrs = unpack_arrays(r.content)
            w_meta["hostname"] = socket.gethostname()
            return Response(content=pack_arrays(w_meta, arrs), media_type="application/octet-stream")
        finally:
            inflight[w_idx] -= 1

    print(f"[Dispatcher] Listening on {host}:{port} -> {num_workers} TPU TensorCore workers", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ==============================================================================
# Host Supervisor (`launch-host`): Launches 8 TPU Workers + Dispatcher on `:8000`
# ==============================================================================
def launch_host_server(
    port: int,
    base_worker_port: int,
    num_workers: int,
    cache_dir: str,
    prewarm_mode: str,
    batch_window_ms: float,
) -> int:
    import urllib.request
    from tpu_single_host_setup import build_single_host_env

    env_w = build_single_host_env(
        port=18080,
        cache_dir=cache_dir,
        slicebuilder_base_port=39600,
        nproc=num_workers,
    )
    worker_cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={num_workers}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=127.0.0.1:29601",
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--base-worker-port",
        str(base_worker_port),
        "--prewarm",
        prewarm_mode,
    ]
    print(f"[Supervisor] Spawning {num_workers} TorchTPU workers on {socket.gethostname()}...", flush=True)
    w_proc = subprocess.Popen(worker_cmd, env=env_w)

    # Wait until all 8 TPU workers respond 200 OK on /healthz
    deadline = time.time() + 600.0
    ready = set()
    while time.time() < deadline and len(ready) < num_workers:
        if w_proc.poll() is not None:
            print(f"[Supervisor] Worker torchrun exited prematurely with code {w_proc.returncode}", flush=True)
            return w_proc.returncode or 1
        for i in range(num_workers):
            if i in ready:
                continue
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{base_worker_port + i}/healthz", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready.add(i)
            except Exception:
                pass
        if len(ready) < num_workers:
            time.sleep(2.0)

    if len(ready) < num_workers:
        w_proc.terminate()
        raise RuntimeError(f"Timed out waiting for {num_workers} TPU workers (ready={sorted(ready)})")

    print(f"[Supervisor] All {num_workers} TPU TensorCore workers READY! Starting Dispatcher on 0.0.0.0:{port}...", flush=True)
    try:
        return run_dispatcher("0.0.0.0", port, base_worker_port, num_workers, batch_window_ms)
    finally:
        w_proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["launch-host", "worker", "dispatcher"], default="launch-host")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--base-worker-port", type=int, default=8100)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--cache-dir", type=str, default="/tmp/tabicl_serving_cache")
    ap.add_argument("--prewarm", choices=["none", "fast", "full"], default="fast")
    ap.add_argument("--preset", choices=["default", "best"], default="best", help="Apply recommended v5p Pallas kernel and bucketing settings")
    ap.add_argument("--batch-window-ms", type=float, default=2.5)
    args = ap.parse_args()
    if getattr(args, "preset", "default") == "best":
        for _k, _v in {'TABICL_TPU_COL_PRECISION': 'high', 'TABICL_TPU_ROW_BUCKET_STEP': '4096', 'TABICL_TPU_COL_GROUP_TILE': 'auto', 'TABICL_TPU_ICL_KV_TRIM': '1', 'TABICL_TPU_FEATURE_STEP': '4', 'TABICL_TPU_SPLASH_BKV': '4096', 'TABICL_TPU_SPLASH_BKV_COMPUTE': '256', 'TABICL_TPU_MHA': 'col1=pk,col2=pk,row=pk,cls=pk', 'TABICL_TPU_MHA_ROPE': 'mm', 'TABICL_TPU_MHA_PROJ': 'views', 'TABICL_TPU_MHA_HXQ': '1', 'TABICL_TPU_MHA_PACKED': '1', 'TABICL_TPU_FAST_GROUP': '1'}.items():
            os.environ.setdefault(_k, _v)

    if args.mode == "worker":
        return run_tpu_workers(args.base_worker_port, args.prewarm)
    if args.mode == "dispatcher":
        return run_dispatcher("0.0.0.0", args.port, args.base_worker_port, args.num_workers, args.batch_window_ms)
    return launch_host_server(
        args.port,
        args.base_worker_port,
        args.num_workers,
        args.cache_dir,
        args.prewarm,
        args.batch_window_ms,
    )


if __name__ == "__main__":
    raise SystemExit(main())

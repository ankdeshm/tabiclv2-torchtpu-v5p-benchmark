#!/usr/bin/env python3
"""Closed-Loop Multi-Process & Multi-Host Serving Load Generator.

Drives realistic mixed-size tabular workloads (`70% <5K`, `25% 5-50K`, `5% 50-250K` rows)
at concurrency `{1, 4, 16, 64}` against:
  - 1-Host FastAPI TPU Server (`8` TensorCore workers on `tabicl-host0`, `devices="tpu:8"`)
  - 2-Host Kubernetes Service (`http://tabicl-serving-svc.default.svc.cluster.local:8000` or pod endpoints,
    `16` TensorCore workers across `tabicl-host0` + `tabicl-host1`, `devices="tpu:16"`)

Benchmarks BOTH:
  - `kv_cache=False`: Stateless mixed-size requests with shape-bucket routing + cross-dataset dynamic batching along `B`
  - `kv_cache=True`: Per-`dataset_id` HBM KV-cache prefill + closed-loop concurrent query load at `{1, 4, 16, 64}`

Writes strict 22-column CSV rows to `results/bench_serving_multihost.csv` (+ JSON summary).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

import httpx
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from csv_schema import write_rows  # noqa: E402
from serving_server import pack_arrays, unpack_arrays  # noqa: E402


def _build_workload_pool(seed: int = 42) -> dict[str, dict[str, Any]]:
    """Pre-generate realistic synthetic datasets across the 3 mandatory size tiers:
    - Tier S (`70%` `<5K` rows): `(800+200)x20`, `(2500+500)x32`
    - Tier M (`25%` `5K-50K` rows): `(12000+1000)x50`, `(35000+2000)x64`
    - Tier L (`5%` `50K-250K` rows): `(60000+2000)x100`, `(120000+2000)x64`
    """
    rng = np.random.RandomState(seed)
    templates = {
        "small_1k": (800, 200, 20, 2),
        "small_3k": (2500, 500, 32, 4),
        "med_13k": (12000, 1000, 50, 2),
        "med_37k": (35000, 2000, 64, 2),
        "large_62k": (60000, 2000, 100, 2),
        "large_122k": (120000, 2000, 64, 2),
    }
    pool: dict[str, dict[str, Any]] = {}
    for name, (n_tr, n_te, H, n_cls) in templates.items():
        X_tr = rng.randn(n_tr, H).astype(np.float32)
        y_tr = rng.randint(0, n_cls, size=(n_tr,)).astype(np.int64)
        X_te = rng.randn(n_te, H).astype(np.float32)
        pool[name] = {
            "name": name,
            "n_train": n_tr,
            "n_test": n_te,
            "n_features": H,
            "n_classes": n_cls,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "predict_payload": pack_arrays({"n_estimators": 8, "template": name}, [X_tr, y_tr, X_te]),
            "prefill_payload": pack_arrays({"dataset_id": f"ds_{name}", "n_estimators": 8}, [X_tr, y_tr]),
            "query_payload": pack_arrays({"dataset_id": f"ds_{name}"}, [X_te]),
        }
    return pool


def _sample_schedule(num_requests: int, seed: int) -> list[str]:
    """Deterministic `70% <5K`, `25% 5-50K`, `5% 50-250K` request sequence."""
    rng = np.random.RandomState(seed)
    items: list[str] = []
    n_small = int(round(num_requests * 0.70))
    n_med = int(round(num_requests * 0.25))
    n_large = max(1, num_requests - n_small - n_med)
    # Adjust n_small so sum == num_requests
    n_small = num_requests - n_med - n_large

    for i in range(n_small):
        items.append("small_1k" if (i % 2 == 0) else "small_3k")
    for i in range(n_med):
        items.append("med_13k" if (i % 3 != 2) else "med_37k")
    for i in range(n_large):
        items.append("large_62k" if (i % 2 == 0) else "large_122k")

    rng.shuffle(items)
    return items


async def _warmup_endpoints(urls: list[str], pool: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Send requests of each template to every host endpoint so compiled kernels are warm on all workers.

    Fault tolerant: a failing template is recorded (with its error) and the warm-up continues. Returns per-template
    wall times (the first call of each shape includes compilation).
    """
    limits = httpx.Limits(max_keepalive_connections=64, max_connections=128)
    report: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0), limits=limits) as cl:
        for u in urls:
            print(f"[LoadTest {time.strftime('%H:%M:%S')}] Warming up endpoint {u} across all 6 size-tier templates...", flush=True)
            for tname, item in pool.items():
                reps = 4 if "small" in tname else (2 if "med" in tname else 1)
                rec: dict[str, Any] = {"endpoint": u, "template": tname, "reps": reps}
                t0 = time.perf_counter()
                try:
                    resps = await asyncio.gather(*[
                        cl.post(f"{u}/v1/predict_bin", content=item["predict_payload"],
                                headers={"Content-Type": "application/octet-stream"})
                        for _ in range(reps)
                    ])
                    for r in resps:
                        r.raise_for_status()
                    rec["predict_warm_s"] = round(time.perf_counter() - t0, 2)
                    t1 = time.perf_counter()
                    r_pf = await cl.post(f"{u}/v1/prefill_bin", content=item["prefill_payload"],
                                         headers={"Content-Type": "application/octet-stream"})
                    r_pf.raise_for_status()
                    rec["prefill_warm_s"] = round(time.perf_counter() - t1, 2)
                    t2 = time.perf_counter()
                    r_q = await cl.post(f"{u}/v1/query_bin", content=item["query_payload"],
                                        headers={"Content-Type": "application/octet-stream"})
                    r_q.raise_for_status()
                    rec["query_warm_s"] = round(time.perf_counter() - t2, 2)
                    rec["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    rec["ok"] = False
                    rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(f"[LoadTest {time.strftime('%H:%M:%S')}] warm-up {json.dumps(rec)}", flush=True)
                report.append(rec)
    return report


async def _run_closed_loop(
    target_urls: list[str],
    pool: dict[str, dict[str, Any]],
    concurrency: int,
    total_requests: int,
    kv_cache: bool,
    seed: int,
    max_config_wall_s: float = 0.0,
    req_log: Any = None,
) -> dict[str, Any]:
    schedule = _sample_schedule(total_requests, seed=seed + concurrency)
    q: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
    for idx, tname in enumerate(schedule):
        q.put_nowait((idx, tname))

    latencies_ms: list[float] = []
    tier_latencies: dict[str, list[float]] = {"small": [], "med": [], "large": []}
    total_test_rows = 0
    max_batch_seen = 1
    max_hbm_gib = 0.0
    hosts_hit: dict[str, int] = {}

    # Create independent HTTP clients (or multi-connection pool) so K8s Service / multi-host load balances across TCP connections
    clients = [
        httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
        for _ in range(concurrency)
    ]

    t_wall0 = time.perf_counter()

    errors: list[str] = []
    n_errors = 0
    truncated = False

    async def _worker_coro(cid: int) -> None:
        nonlocal total_test_rows, max_batch_seen, max_hbm_gib, n_errors, truncated
        cl = clients[cid]
        while True:
            if max_config_wall_s > 0 and (time.perf_counter() - t_wall0) > max_config_wall_s:
                truncated = True
                return
            try:
                req_idx, tname = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            item = pool[tname]
            base_url = target_urls[(req_idx + cid) % len(target_urls)]
            endpoint = f"{base_url}/v1/query_bin" if kv_cache else f"{base_url}/v1/predict_bin"
            payload = item["query_payload"] if kv_cache else item["predict_payload"]

            t0 = time.perf_counter()
            try:
                resp = await cl.post(endpoint, content=payload, headers={"Content-Type": "application/octet-stream"})
                resp.raise_for_status()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                meta, arrs = unpack_arrays(resp.content)
                assert arrs[0].shape == (item["n_test"], item["n_classes"])
            except Exception as exc:  # noqa: BLE001
                n_errors += 1
                if len(errors) < 5:
                    errors.append(f"{tname}: {type(exc).__name__}: {str(exc)[:200]}")
                if req_log is not None:
                    req_log.write(json.dumps({"conc": concurrency, "kv": kv_cache, "template": tname, "ok": False,
                                              "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                                              "error": str(exc)[:200]}) + "\n")
                    req_log.flush()
                continue
            latencies_ms.append(dt_ms)
            tier = tname.split("_")[0]
            tier_latencies[tier].append(dt_ms)
            total_test_rows += int(item["n_test"])
            max_batch_seen = max(max_batch_seen, int(meta.get("batch_datasets", 1)))
            max_hbm_gib = max(max_hbm_gib, float(meta.get("peak_hbm_gib", 0.0)))
            hname = str(meta.get("hostname", "unknown"))
            hosts_hit[hname] = hosts_hit.get(hname, 0) + 1
            if req_log is not None:
                req_log.write(json.dumps({"conc": concurrency, "kv": kv_cache, "template": tname, "ok": True,
                                          "ms": round(dt_ms, 1), "batch": int(meta.get("batch_datasets", 1))}) + "\n")
                req_log.flush()

    try:
        await asyncio.gather(*(_worker_coro(i) for i in range(concurrency)))
    finally:
        await asyncio.gather(*(c.aclose() for c in clients))

    wall_s = time.perf_counter() - t_wall0
    p50 = float(np.median(latencies_ms)) if latencies_ms else float("nan")
    p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else float("nan")
    ds_per_s = len(latencies_ms) / max(wall_s, 1e-6)
    rows_per_s = total_test_rows / max(wall_s, 1e-6)

    small_p50 = float(np.median(tier_latencies["small"])) if tier_latencies["small"] else 0.0
    med_p50 = float(np.median(tier_latencies["med"])) if tier_latencies["med"] else 0.0
    large_p50 = float(np.median(tier_latencies["large"])) if tier_latencies["large"] else 0.0

    return {
        "concurrency": concurrency,
        "kv_cache": kv_cache,
        "requests_planned": total_requests,
        "total_requests": len(latencies_ms),
        "errors": n_errors,
        "first_errors": errors,
        "truncated_by_wall_cap": truncated,
        "wall_s": round(wall_s, 3),
        "p50_ms": round(p50, 3),
        "p99_ms": round(p99, 3),
        "small_p50_ms": round(small_p50, 3),
        "med_p50_ms": round(med_p50, 3),
        "large_p50_ms": round(large_p50, 3),
        "datasets_per_s": round(ds_per_s, 4),
        "test_rows_per_s": round(rows_per_s, 2),
        "max_batch_datasets": max_batch_seen,
        "peak_hbm_gib": round(max_hbm_gib, 4),
        "hosts_hit": hosts_hit,
    }


async def async_main(args: argparse.Namespace) -> int:
    pool = _build_workload_pool(seed=42)
    direct_urls = [u.strip().rstrip("/") for u in args.direct_urls.split(",") if u.strip()]
    target_urls = [u.strip().rstrip("/") for u in args.target_urls.split(",") if u.strip()]
    req_log = open(args.request_log, "a", encoding="utf-8") if args.request_log else None  # noqa: SIM115

    # Always warm up all direct backend host URLs so every host has warm kernels + prefilled KV entries
    t_w = time.perf_counter()
    warmup_report = await _warmup_endpoints(direct_urls, pool)
    warmup_s = round(time.perf_counter() - t_w, 1)

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    modes = [m.strip().lower() == "kv" for m in args.modes.split(",") if m.strip()]
    csv_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    def _flush_results() -> None:
        # Written after every configuration so partial results survive a preemption or a hang.
        write_rows(Path(args.out_csv), csv_rows, append=False)
        Path(args.out_json).write_text(json.dumps({"warmup_s": warmup_s, "warmup": warmup_report, "configs": details,
                                                   "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                                  indent=2))

    # --resume (Spot preemptions): configurations that already finished cleanly in an earlier run of the same
    # serving step (zero errors, and either all planned requests done or stopped by the wall cap) are kept as-is
    # and not re-run; anything else (e.g. rows from a broken server: all errors) is measured again.
    reused: dict[tuple[bool, int], dict[str, Any]] = {}
    if args.resume and Path(args.out_json).exists():
        try:
            prev = json.loads(Path(args.out_json).read_text())
            for c in prev.get("configs", []):
                ok = int(c.get("errors", 1)) == 0 and int(c.get("total_requests", 0)) > 0 and (
                    int(c.get("total_requests", 0)) >= int(c.get("requests_planned", 1)) or bool(c.get("truncated_by_wall_cap")))
                if ok:
                    reused[(bool(c["kv_cache"]), int(c["concurrency"]))] = dict(c, reused_from_earlier_run=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[LoadTest] --resume: could not read {args.out_json}: {exc!r}", flush=True)
        if reused:
            print(f"[LoadTest] --resume: reusing {sorted(reused)} from {args.out_json}", flush=True)

    _flush_results()
    for kv_flag in modes:
        for conc in concurrencies:
            # Scale request count with concurrency so closed-loop steady state is well-sampled
            n_reqs = max(40, conc * 4) if not kv_flag else max(60, conc * 6)
            if (kv_flag, conc) in reused:
                res = reused[(kv_flag, conc)]
            else:
                res = await _run_closed_loop(
                    target_urls=target_urls,
                    pool=pool,
                    concurrency=conc,
                    total_requests=n_reqs,
                    kv_cache=kv_flag,
                    seed=1000,
                    max_config_wall_s=args.max_config_wall_s,
                    req_log=req_log,
                )
            details.append(res)
            print(
                f"[LoadTest {time.strftime('%H:%M:%S')} {args.devices_label} kv={str(kv_flag):5s} conc={conc:2d}] "
                f"reqs={res['total_requests']:3d}/{res['requests_planned']} err={res['errors']} "
                f"trunc={res['truncated_by_wall_cap']} wall={res['wall_s']:6.2f}s "
                f"ds/s={res['datasets_per_s']:7.2f} rows/s={res['test_rows_per_s']:9.1f} "
                f"p50={res['p50_ms']:7.2f}ms p99={res['p99_ms']:8.2f}ms "
                f"max_b={res['max_batch_datasets']} hosts={res['hosts_hit']}",
                flush=True,
            )
            csv_rows.append(
                {
                    "phase": f"1.6_serving_{args.tag}",
                    "path": f"path_a_fastapi_{'kv' if kv_flag else 'nocache'}",
                    "model": "tabicl-classifier-v2-20260212",
                    "dtype": "bf16",
                    "n_train": -1,
                    "n_test": -1,
                    "n_features": -1,
                    "n_classes": 2,
                    "n_estimators": 8,
                    "kv_cache": kv_flag,
                    "devices": args.devices_label,
                    "batch_datasets": res["max_batch_datasets"],
                    "compile_s": 0.0,
                    "p50_ms": res["p50_ms"],
                    "p99_ms": res["p99_ms"],
                    "datasets_per_s": res["datasets_per_s"],
                    "test_rows_per_s": res["test_rows_per_s"],
                    "peak_hbm_gib": res["peak_hbm_gib"],
                    "metric_name": "serving_datasets_per_s",
                    "metric_value": res["datasets_per_s"],
                    "ref_metric_value": "",
                    "notes": (
                        f"concurrency={conc};mix=70pct_lt5k_25pct_5to50k_5pct_50to250k;"
                        f"ok={res['total_requests']}/{res['requests_planned']};errors={res['errors']};"
                        f"truncated={res['truncated_by_wall_cap']};"
                        f"small_p50={res['small_p50_ms']}ms;med_p50={res['med_p50_ms']}ms;"
                        f"large_p50={res['large_p50_ms']}ms;hosts={json.dumps(res['hosts_hit'])}"
                    ),
                }
            )
            _flush_results()

    if req_log is not None:
        req_log.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. 1host or 2host")
    ap.add_argument("--devices-label", required=True, help="e.g. tpu:8 or tpu:16")
    ap.add_argument("--direct-urls", required=True, help="Comma-separated host URLs for warmup/KV prefill")
    ap.add_argument("--target-urls", required=True, help="Comma-separated target URL(s) (e.g. K8s Service URL)")
    ap.add_argument("--concurrencies", default="1,4,16,64")
    ap.add_argument("--modes", default="nocache,kv", help="comma-separated subset of nocache,kv (run order)")
    ap.add_argument("--max-config-wall-s", type=float, default=0.0,
                    help="stop issuing new requests in a config after this many seconds (0 = no cap)")
    ap.add_argument("--request-log", default="", help="optional per-request JSONL log path")
    ap.add_argument("--out-csv", default="/tmp/bench_serving_multihost.csv")
    ap.add_argument("--out-json", default="/tmp/bench_serving_summary.json")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep configurations that finished cleanly in an earlier run (from --out-json) and skip them")
    args = ap.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())

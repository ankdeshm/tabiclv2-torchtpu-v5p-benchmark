#!/usr/bin/env python3
"""Distributed PyTorch (TorchTPU 0.2.0 + `tpu_dist`) Pre-Training & Fine-Tuning Runner for TabICLv2 on Cloud TPU v5p.

Executes the official TabICLv2 (arXiv:2602.11139) multi-stage pre-training curriculum on TPU v5p (`2x2x4` = 16 chips
or `2x4x4` = 32 chips) using native PyTorch (`torch==2.15.0` + `torch_tpu==0.2.0` + `tpu_dist` ICI AllReduce):
  - Synthetic tabular prior: `tabicl.prior.PriorDataset(prior_type="graph_scm")` on CPU DataLoader workers
  - Optimizer & schedule: Upstream `Muon` (Newton-Schulz-5 orthogonalized momentum) + cosine schedule w/ warmup
  - Precision: `torch.amp.autocast("tpu", dtype=torch.bfloat16)` with master FP32 weights (zero FP16 / GradScaler)
  - Shape bucketing along `(seq_len, train_size, max_features)` so XLA compiles a tiny finite set of static graphs
    and executes steady-state training steps at cached TPU speed with zero per-step recompilation storms.

Stages executed sequentially (each in an isolated `torchrun` multi-host invocation, writing checkpoints + live
`SUMMARY.md` / `summary.json` to `/models/tabiclv2-v5p-pretrain/`):
  1. `clf_stage1_1k`   — Classifier Stage 1 (`seq_len=1024`, `features<=100`, `global_bs=64`, `lr=8e-4`, 200 steps)
  2. `clf_stage2_10k`  — Classifier Stage 2 (`seq_len=400..10240`, `79-81%` train, `global_bs=64`, `lr=1e-4`, 120 steps, resumed from Stage 1)
  3. `clf_stage3_32k`  — Classifier Stage 3 (`seq_len=400..32768`, `79-81%` train, `global_bs=64`, `lr=2e-5`, 80 steps, resumed from Stage 2)
  4. `reg_stage1_1k`   — Quantile Regressor Stage 1 (`999` quantiles pinball loss, `bias_free_ln=True`, `seq_len=1024`, `global_bs=64`, 200 steps)
  5. `clf_stage1_long` — Sustained Classifier Stage 1 Pre-Training (`50,000` steps, checkpointing every `500` steps, resumed from `clf_stage1_1k`)
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from typing import Any


SEQ_BUCKETS = (512, 1024, 2048, 4096, 8192, 10240, 16384, 32768, 60000)


def _bucket_seq_and_split(seq_len: int, train_size: int, fixed_1k: bool) -> tuple[int, int]:
    """Snap `(seq_len, train_size)` to a small deterministic grid so XLA caches every graph after warmup."""
    if fixed_1k:
        b_seq = 1024
        # 6 split buckets across [0.3, 0.9] * 1024 -> {320, 448, 576, 704, 832, 928}
        grid = (320, 448, 576, 704, 832, 928)
        b_tr = min(grid, key=lambda g: abs(g - train_size))
        return b_seq, min(max(b_tr, 64), b_seq - 32)
    b_seq = SEQ_BUCKETS[-1]
    for b in SEQ_BUCKETS:
        if seq_len <= b:
            b_seq = b
            break
    # Stage 2 & 3 use 79-81% train split -> deterministic 80% of bucketed seq_len!
    b_tr = int(round(0.80 * b_seq))
    return b_seq, min(max(b_tr, 64), b_seq - 32)


def run_worker_stage(args: argparse.Namespace) -> int:
    """Per-rank worker invoked under `torchrun`."""
    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    sys.path.insert(0, "/workspace/tabicl/src")
    from tabicl._model.attention import set_flash_attn3_enabled
    from tabicl._model.tabicl import TabICL
    from tabicl.prior._dataset import PriorDataset
    from tabicl.prior._genload import seed_worker
    from tabicl.prior.graph_lib._config import PriorConfig
    from tabicl.train._muon import Muon
    from tabicl.train._optim import get_scheduler
    from tabicl.train._train_config import build_parser

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.accelerator.set_device_index(0)
    dev = torch.device("tpu:0")
    dist.init_process_group(backend="tpu_dist", timeout=dt.timedelta(minutes=30))

    if rank == 0:
        print(
            f"[TabICLv2-TPU] Stage={args.stage_name} world_size={world_size} "
            f"(local_rank={local_rank}) initialized tpu_dist",
            flush=True,
        )

    t_parser = build_parser()
    cfg_argv = [
        "--wandb_log", "False",
        "--device", "cpu",
        "--dtype", "bfloat16",
        "--np_seed", str(args.seed),
        "--torch_seed", str(args.seed),
        "--max_steps", str(args.max_steps),
        "--batch_size", str(args.global_batch_size),
        "--micro_batch_size", str(args.micro_batch_size),
        "--lr", str(args.lr),
        "--muon", "True",
        "--beta1", "0.9",
        "--weight_decay", "0.01",
        "--use_cautious_wd", "False",
        "--scheduler", "cosine_with_restarts",
        "--warmup_proportion", str(args.warmup_proportion),
        "--cosine_num_cycles", "1",
        "--cosine_amplitude_decay", "1",
        "--cosine_lr_end", "1e-7",
        "--gradient_clipping", str(args.grad_clip),
        "--prior_type", "graph_scm",
        "--prior_device", "cpu",
        "--n_jobs", str(args.num_workers),
        "--batch_size_per_gp", str(args.batch_size_per_gp),
        "--min_features", "4",
        "--max_features", str(args.max_features),
        "--max_classes", "10",
        "--max_seq_len", str(args.max_seq_len),
        "--log_seq_len", str(args.log_seq_len),
        "--min_train_size", str(args.min_train_size),
        "--max_train_size", str(args.max_train_size),
        "--seq_len_per_gp", "True",
        "--graph_noise", "False",
        "--filter_unpredictable_graphs", "True",
        "--filter_unpredictable_datasets", "True",
        "--allow_act_warping", "False",
        "--min_n_nodes", "2",
        "--max_n_nodes", "32",
        "--cauchy_dag_offset", "0.0",
        "--embed_dim", "128",
        "--col_num_blocks", "3",
        "--col_nhead", "8",
        "--col_num_inds", "128",
        "--col_affine", "False",
        "--col_feature_group", "same",
        "--col_feature_group_size", "3",
        "--col_target_aware", "True",
        "--col_ssmax", "True",
        "--row_num_blocks", "3",
        "--row_nhead", "8",
        "--row_num_cls", "4",
        "--row_rope_base", "100000",
        "--row_rope_interleaved", "False",
        "--icl_num_blocks", "12",
        "--icl_nhead", "8",
        "--icl_ssmax", "True",
        "--ssmax_type", "qassmax-mlp-elementwise",
        "--ff_factor", "2",
        "--norm_first", "True",
        "--zero_init", "False",
        "--use_flash_attn3", "False",
        "--norm_type", "layernorm_nobias" if args.regression else "default",
    ]
    if args.min_seq_len < args.max_seq_len:
        cfg_argv += ["--min_seq_len", str(args.min_seq_len)]
    if args.regression:
        cfg_argv += ["--regression_method", "quantile", "--num_quantiles", "999"]
    cfg = t_parser.parse_args(cfg_argv)

    set_flash_attn3_enabled(False)
    np.random.seed(cfg.np_seed + rank)
    torch.manual_seed(cfg.torch_seed + rank)

    local_batch_size = math.ceil(cfg.batch_size / world_size)
    num_micro_batches = math.ceil(local_batch_size / cfg.micro_batch_size)

    model_config = {
        "max_classes": 0 if args.regression else cfg.max_classes,
        "num_quantiles": cfg.num_quantiles,
        "embed_dim": cfg.embed_dim,
        "col_num_blocks": cfg.col_num_blocks,
        "col_nhead": cfg.col_nhead,
        "col_num_inds": cfg.col_num_inds,
        "col_affine": cfg.col_affine,
        "col_feature_group": cfg.col_feature_group,
        "col_feature_group_size": cfg.col_feature_group_size,
        "col_target_aware": cfg.col_target_aware,
        "col_ssmax": cfg.ssmax_type if cfg.col_ssmax else False,
        "row_num_blocks": cfg.row_num_blocks,
        "row_nhead": cfg.row_nhead,
        "row_num_cls": cfg.row_num_cls,
        "row_rope_base": cfg.row_rope_base,
        "row_rope_interleaved": cfg.row_rope_interleaved,
        "icl_num_blocks": cfg.icl_num_blocks,
        "icl_nhead": cfg.icl_nhead,
        "icl_ssmax": cfg.ssmax_type if cfg.icl_ssmax else False,
        "ff_factor": cfg.ff_factor,
        "dropout": cfg.dropout,
        "activation": cfg.activation,
        "norm_first": cfg.norm_first,
        "bias_free_ln": bool(args.regression),
        "zero_init": cfg.zero_init,
        "recompute": args.recompute,
    }

    # Build model identically on CPU with fixed seed first so all ranks start from identical weights
    torch.manual_seed(cfg.torch_seed)
    model = TabICL(**model_config)
    start_step = 0

    ckpt_dir = Path(args.ckpt_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from latest checkpoint in ckpt_dir if present, else load init_ckpt weights if specified
    resume_ckpt = None
    if ckpt_dir.exists():
        ckpts = sorted(
            ckpt_dir.glob("step-*.ckpt"),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        if ckpts:
            resume_ckpt = ckpts[-1]

    loaded_state = None
    if resume_ckpt is not None:
        loaded_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded_state["state_dict"], strict=True)
        start_step = int(loaded_state.get("step", 0))
        if rank == 0:
            print(f"[TabICLv2-TPU] Resumed stage {args.stage_name} from {resume_ckpt} at step {start_step}", flush=True)
    elif args.init_ckpt and Path(args.init_ckpt).exists():
        init_obj = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = init_obj["state_dict"] if isinstance(init_obj, dict) and "state_dict" in init_obj else init_obj
        model.load_state_dict(sd, strict=True)
        if rank == 0:
            print(f"[TabICLv2-TPU] Initialized weights from {args.init_ckpt}", flush=True)

    model = model.to(dev)
    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(
            f"[TabICLv2-TPU] Trainable params={num_params:,} local_bs={local_batch_size} "
            f"micro_bs={cfg.micro_batch_size} ga={num_micro_batches} global_bs={local_batch_size * world_size}",
            flush=True,
        )

    optimizer = Muon(
        param_groups=[dict(params=[p for p in model.parameters() if p.requires_grad], use_muon=True)],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        matched_adamw_rms=0.2,
        momentum=cfg.beta1,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(cfg.beta1, cfg.beta2),
        adamw_eps=1e-8,
        use_cautious_wd=cfg.use_cautious_wd,
    )
    scheduler = get_scheduler(config=cfg, optimizer=optimizer)
    if loaded_state is not None and "optimizer_state" in loaded_state:
        try:
            optimizer.load_state_dict(loaded_state["optimizer_state"])
            scheduler.load_state_dict(loaded_state["scheduler_state"])
        except Exception as exc:
            if rank == 0:
                print(f"[TabICLv2-TPU] Note: fresh optimizer state ({exc})", flush=True)

    # Re-seed per rank before synthetic prior DataLoader creation so all 16 ranks sample independent SCM graphs!
    np.random.seed(cfg.np_seed + 1009 * rank + start_step)
    torch.manual_seed(cfg.torch_seed + 1009 * rank + start_step)

    dataset = PriorDataset(
        regression=bool(args.regression),
        batch_size=local_batch_size,
        batch_size_per_gp=min(cfg.batch_size_per_gp, local_batch_size),
        min_features=cfg.min_features,
        max_features=cfg.max_features,
        max_classes=cfg.max_classes,
        min_seq_len=cfg.min_seq_len,
        max_seq_len=cfg.max_seq_len,
        log_seq_len=cfg.log_seq_len,
        log_n_features=cfg.log_n_features,
        seq_len_per_gp=cfg.seq_len_per_gp,
        min_train_size=cfg.min_train_size,
        max_train_size=cfg.max_train_size,
        replay_small=cfg.replay_small,
        prior_type=cfg.prior_type,
        config=PriorConfig.from_args(cfg),
        device="cpu",
        n_jobs=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=False,
        worker_init_fn=seed_worker,
        persistent_workers=(args.num_workers > 0),
    )

    fixed_1k = (args.min_seq_len is None or args.min_seq_len == args.max_seq_len) and args.max_seq_len == 1024
    data_iter = iter(dataloader)

    bar = torch.zeros(1, device=dev)
    dist.all_reduce(bar)
    torch.accelerator.synchronize(dev)
    try:
        torch.accelerator.reset_peak_memory_stats(dev)
    except Exception:
        pass

    metrics_path = Path(args.metrics_jsonl)
    step_records: list[dict[str, Any]] = []

    for step in range(start_step, args.max_steps):
        t_p0 = time.perf_counter()
        batch = next(data_iter)
        prior_s = time.perf_counter() - t_p0

        t_s0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]
        micro_batches = list(zip(*[torch.split(t, cfg.micro_batch_size, dim=0) for t in batch]))

        total_loss_val = 0.0
        total_acc_val = 0.0
        total_rows_local = 0

        for mb_X, mb_y, _mb_d, mb_seq, mb_tr in micro_batches:
            raw_seq = int(mb_seq[0].item())
            raw_tr = int(mb_tr[0].item())
            b_seq, b_tr = _bucket_seq_and_split(raw_seq, raw_tr, fixed_1k=fixed_1k)

            # Pad or slice sequence dimension to deterministic bucket `b_seq`; keep features at constant `max_features`
            cur_T = mb_X.shape[1]
            if cur_T > b_seq:
                mb_X = mb_X[:, :b_seq, :]
                mb_y = mb_y[:, :b_seq]
            elif cur_T < b_seq:
                pad_T = b_seq - cur_T
                mb_X = F.pad(mb_X, (0, 0, 0, pad_T), value=0.0)
                mb_y = F.pad(mb_y, (0, pad_T), value=0)

            cur_H = mb_X.shape[2]
            if cur_H < cfg.max_features:
                mb_X = F.pad(mb_X, (0, cfg.max_features - cur_H), value=0.0)
            elif cur_H > cfg.max_features:
                mb_X = mb_X[:, :, : cfg.max_features]

            total_rows_local += int(mb_X.shape[0] * b_seq)

            mb_X_dev = mb_X.to(dev, dtype=torch.float32)
            mb_y_dev = mb_y.to(dev)
            y_train = mb_y_dev[:, :b_tr].to(torch.float32)
            y_test = mb_y_dev[:, b_tr:]

            with torch.amp.autocast("tpu", dtype=torch.bfloat16):
                pred = model(mb_X_dev, y_train, None)
                if args.regression:
                    alphas = torch.linspace(
                        0.0, 1.0, cfg.num_quantiles + 2, device=dev, dtype=torch.float32
                    )[1:-1].view(1, 1, -1)
                    errors = y_test.float().unsqueeze(-1) - pred.float()
                    loss = torch.maximum(alphas * errors, (alphas - 1.0) * errors).mean()
                else:
                    logits_2d = pred.float().flatten(end_dim=-2)
                    targets_1d = y_test.long().clamp(0, cfg.max_classes - 1).flatten()
                    loss = F.cross_entropy(logits_2d, targets_1d)

            scaled_loss = loss / num_micro_batches
            scaled_loss.backward()

            with torch.no_grad():
                total_loss_val += float(scaled_loss.detach().cpu().item())
                if not args.regression:
                    acc = (logits_2d.argmax(dim=1) == targets_1d).float().mean()
                    total_acc_val += float(acc.detach().cpu().item()) / num_micro_batches

        # AllReduce gradients across all 16 TPU v5p chips via ICI
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        if cfg.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)

        optimizer.step()
        scheduler.step()
        torch.accelerator.synchronize(dev)
        train_s = time.perf_counter() - t_s0
        step_wall_s = prior_s + train_s

        # Aggregate metrics across all ranks every step for exact cluster-wide loss/accuracy/throughput
        stat_vec = torch.tensor(
            [total_loss_val, total_acc_val, float(total_rows_local), step_wall_s, train_s, prior_s],
            device=dev,
            dtype=torch.float32,
        )
        dist.all_reduce(stat_vec, op=dist.ReduceOp.AVG)
        torch.accelerator.synchronize(dev)
        avg_loss, avg_acc, avg_rows_per_rank, avg_wall_s, avg_train_s, avg_prior_s = [
            float(x) for x in stat_vec.cpu().tolist()
        ]

        global_tables = local_batch_size * world_size
        global_rows = avg_rows_per_rank * world_size
        tables_per_s = global_tables / max(avg_wall_s, 1e-6)
        compute_tables_per_s = global_tables / max(avg_train_s, 1e-6)
        rows_per_s = global_rows / max(avg_wall_s, 1e-6)
        compute_rows_per_s = global_rows / max(avg_train_s, 1e-6)

        cur_step = step + 1
        if rank == 0:
            try:
                peak_hbm_gib = round(torch.accelerator.max_memory_allocated(dev) / float(1024**3), 3)
            except Exception:
                peak_hbm_gib = 0.0
            rec = {
                "stage": args.stage_name,
                "step": cur_step,
                "loss": round(avg_loss, 5),
                "accuracy": None if args.regression else round(avg_acc, 4),
                "lr": round(float(scheduler.get_last_lr()[0]), 8),
                "step_wall_ms": round(avg_wall_s * 1000.0, 2),
                "tpu_train_ms": round(avg_train_s * 1000.0, 2),
                "cpu_prior_ms": round(avg_prior_s * 1000.0, 2),
                "global_tables_per_s": round(tables_per_s, 2),
                "tpu_compute_tables_per_s": round(compute_tables_per_s, 2),
                "global_rows_per_s": round(rows_per_s, 1),
                "tpu_compute_rows_per_s": round(compute_rows_per_s, 1),
                "peak_hbm_gib": peak_hbm_gib,
                "world_size": world_size,
            }
            step_records.append(rec)
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec) + "\n")

            if cur_step <= 15 or cur_step % args.log_every == 0 or cur_step == args.max_steps:
                metric_str = (
                    f"pinball={avg_loss:.4f}" if args.regression else f"ce={avg_loss:.4f} acc={avg_acc*100:.1f}%"
                )
                print(
                    f"[{args.stage_name}] step {cur_step:5d}/{args.max_steps} | {metric_str} | "
                    f"wall={avg_wall_s*1000:6.1f}ms (tpu={avg_train_s*1000:6.1f}ms prior={avg_prior_s*1000:5.1f}ms) | "
                    f"tables/s={tables_per_s:6.2f} (compute={compute_tables_per_s:6.2f}) | "
                    f"rows/s={rows_per_s:9.1f} | hbm={peak_hbm_gib:.2f}GiB",
                    flush=True,
                )

            if cur_step % args.save_every == 0 or cur_step == args.max_steps:
                ckpt_file = ckpt_dir / f"step-{cur_step}.ckpt"
                torch.save(
                    {
                        "config": model_config,
                        "step": cur_step,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                    },
                    ckpt_file,
                )
                print(f"[{args.stage_name}] Saved checkpoint -> {ckpt_file}", flush=True)

    if rank == 0 and args.stage_summary_json:
        steady = step_records[min(15, len(step_records) // 3) :] if step_records else []
        summary_obj = {
            "stage": args.stage_name,
            "regression": bool(args.regression),
            "world_size": world_size,
            "global_batch_size": local_batch_size * world_size,
            "micro_batch_size": cfg.micro_batch_size,
            "grad_accum": num_micro_batches,
            "steps_completed": len(step_records),
            "start_loss": step_records[0]["loss"] if step_records else None,
            "end_loss": step_records[-1]["loss"] if step_records else None,
            "start_acc": step_records[0]["accuracy"] if step_records else None,
            "end_acc": step_records[-1]["accuracy"] if step_records else None,
            "steady_median_wall_ms": round(statistics.median([r["step_wall_ms"] for r in steady]), 2) if steady else 0.0,
            "steady_median_tpu_ms": round(statistics.median([r["tpu_train_ms"] for r in steady]), 2) if steady else 0.0,
            "steady_tables_per_s": round(statistics.mean([r["global_tables_per_s"] for r in steady]), 2) if steady else 0.0,
            "steady_compute_tables_per_s": round(statistics.mean([r["tpu_compute_tables_per_s"] for r in steady]), 2) if steady else 0.0,
            "steady_rows_per_s": round(statistics.mean([r["global_rows_per_s"] for r in steady]), 1) if steady else 0.0,
            "steady_compute_rows_per_s": round(statistics.mean([r["tpu_compute_rows_per_s"] for r in steady]), 1) if steady else 0.0,
            "peak_hbm_gib": max((r["peak_hbm_gib"] for r in step_records), default=0.0),
            "latest_ckpt": str(ckpt_dir / f"step-{args.max_steps}.ckpt"),
        }
        Path(args.stage_summary_json).write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    dist.all_reduce(bar)
    torch.accelerator.synchronize(dev)
    dist.destroy_process_group()
    return 0


def _wait_dns(host: str, timeout_s: float = 300.0) -> str:
    t0 = time.time()
    candidates = [host]
    if not host.endswith(".cluster.local") and not host.replace(".", "").isdigit():
        candidates.append(f"{host}.default.svc.cluster.local")
    while time.time() - t0 < timeout_s:
        for cand in candidates:
            try:
                return socket.gethostbyname(cand)
            except Exception:
                pass
        time.sleep(2.0)
    raise RuntimeError(f"Timed out resolving host {host}")


def _write_overall_summary(out_root: Path, slice_name: str, world_size: int, stages: list[dict[str, Any]]) -> None:
    payload = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "TabICLv2 (Classifier 27.55M & Quantile Regressor 28.54M)",
        "framework": "PyTorch 2.15.0 + TorchTPU 0.2.0 (`tpu_dist` ICI AllReduce) + Muon Optimizer + Graph-SCM Prior",
        "slice_name": slice_name,
        "world_size_chips": world_size,
        "stages": stages,
    }
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# TabICLv2 Distributed PyTorch (`TorchTPU 0.2.0`) Pre-Training on `{slice_name}` (`{world_size}` TPU v5p Chips)",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Runtime Stack:** `{payload['framework']}`",
        f"- **Precision:** `torch.amp.autocast('tpu', dtype=torch.bfloat16)` with FP32 master weights & `Muon` Newton-Schulz-5 (`ns_steps=5`)",
        "",
        "| Stage | Task | Global BS (`mb x ga x chips`) | Steps | Steady Step (`ms`) | TPU Compute (`ms`) | **Tables/s (E2E / Compute)** | **Rows/s (E2E / Compute)** | Loss (`start -> end`) | Acc (`start -> end`) | Peak HBM (`GiB`) |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|:---|:---|---:|",
    ]
    for s in stages:
        acc_s = (
            f"{s['start_acc']*100:.1f}% -> {s['end_acc']*100:.1f}%"
            if s.get("start_acc") is not None and s.get("end_acc") is not None
            else "N/A (999-Quantile Pinball)"
        )
        lines.append(
            f"| `{s['stage']}` | {'Regressor' if s['regression'] else 'Classifier'} | "
            f"`{s['global_batch_size']}` (`{s['micro_batch_size']}x{s['grad_accum']}x{s['world_size']}`) | "
            f"`{s['steps_completed']}` | `{s['steady_median_wall_ms']:.1f}` | `{s['steady_median_tpu_ms']:.1f}` | "
            f"**{s['steady_tables_per_s']:.2f} / {s['steady_compute_tables_per_s']:.2f}** | "
            f"**{s['steady_rows_per_s']:,.1f} / {s['steady_compute_rows_per_s']:,.1f}** | "
            f"`{s['start_loss']:.4f} -> {s['end_loss']:.4f}` | `{acc_s}` | `{s['peak_hbm_gib']:.2f}` |"
        )
    (out_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_orchestrator(args: argparse.Namespace) -> int:
    world_size = args.nnodes * args.nproc_per_node
    raw_tpu_hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    if raw_tpu_hosts:
        tpu_host_list = [h.strip() for h in raw_tpu_hosts.split(",") if h.strip()]
        node_rank = int(os.environ.get("TPU_WORKER_ID", os.environ.get("JOB_COMPLETION_INDEX", "0")))
    else:
        svc_suffix = ".".join(args.master_addr.split(".")[1:])
        job_prefix = args.master_addr.split(".")[0].rsplit("-", 1)[0]
        tpu_host_list = [f"{job_prefix}-{i}.{svc_suffix}" for i in range(args.nnodes)]
        node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))

    hosts_ips = [_wait_dns(h) for h in tpu_host_list]
    raw_topo = os.environ.get("TPU_TOPOLOGY", "2x2x4" if world_size == 16 else "2x4x4")
    # On TPU v5p (viperfish), TORCH_TPU_TOPOLOGY is 3-dimensional `X,Y,Z` (e.g. `2,2,4` or `2,4,4`)
    torch_tpu_topo = ",".join(raw_topo.split("x"))

    if node_rank == 0:
        print(
            f"[TabICLv2-TPU Orchestrator] Resolved {len(hosts_ips)} TPU hosts={hosts_ips} "
            f"TORCH_TPU_TOPOLOGY={torch_tpu_topo} world_size={world_size}",
            flush=True,
        )

    out_root = Path(args.output_dir)
    if node_rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)

    stage1_ckpt = str(out_root / "checkpoints" / "clf_stage1_1k" / "step-200.ckpt")
    stage2_ckpt = str(out_root / "checkpoints" / "clf_stage2_10k" / "step-120.ckpt")

    suite = [
        {
            "name": "clf_stage1_1k",
            "max_steps": 200,
            "save_every": 100,
            "log_every": 10,
            "global_bs": 64,
            "micro_bs": 4,
            "bs_per_gp": 4,
            "lr": 8e-4,
            "grad_clip": 10.0,
            "min_seq": 1024,
            "max_seq": 1024,
            "log_seq": False,
            "min_tr": 0.3,
            "max_tr": 0.9,
            "regression": False,
            "recompute": False,
            "init_ckpt": "",
            "workers": 8,
        },
        {
            "name": "clf_stage2_10k",
            "max_steps": 120,
            "save_every": 60,
            "log_every": 10,
            "global_bs": 64,
            "micro_bs": 1,
            "bs_per_gp": 1,
            "lr": 1e-4,
            "grad_clip": 10.0,
            "min_seq": 1024,
            "max_seq": 10240,
            "log_seq": True,
            "min_tr": 0.79,
            "max_tr": 0.81,
            "regression": False,
            "recompute": True,
            "init_ckpt": stage1_ckpt,
            "workers": 8,
        },
        {
            "name": "clf_stage3_32k",
            "max_steps": 80,
            "save_every": 40,
            "log_every": 10,
            "global_bs": 64,
            "micro_bs": 1,
            "bs_per_gp": 1,
            "lr": 2e-5,
            "grad_clip": 1.0,
            "min_seq": 2048,
            "max_seq": 32768,
            "log_seq": True,
            "min_tr": 0.79,
            "max_tr": 0.81,
            "regression": False,
            "recompute": True,
            "init_ckpt": stage2_ckpt,
            "workers": 6,
        },
        {
            "name": "reg_stage1_1k",
            "max_steps": 200,
            "save_every": 100,
            "log_every": 10,
            "global_bs": 64,
            "micro_bs": 4,
            "bs_per_gp": 4,
            "lr": 8e-4,
            "grad_clip": 10.0,
            "min_seq": 1024,
            "max_seq": 1024,
            "log_seq": False,
            "min_tr": 0.3,
            "max_tr": 0.9,
            "regression": True,
            "recompute": False,
            "init_ckpt": "",
            "workers": 8,
        },
        {
            "name": "clf_stage1_sustained_50k",
            "max_steps": 50000,
            "save_every": 500,
            "log_every": 25,
            "global_bs": 64,
            "micro_bs": 4,
            "bs_per_gp": 4,
            "lr": 8e-4,
            "grad_clip": 10.0,
            "min_seq": 1024,
            "max_seq": 1024,
            "log_seq": False,
            "min_tr": 0.3,
            "max_tr": 0.9,
            "regression": False,
            "recompute": False,
            "init_ckpt": stage1_ckpt,
            "workers": 8,
        },
    ]

    completed_summaries: list[dict[str, Any]] = []
    for idx, st in enumerate(suite):
        port = args.base_port + idx * 17
        sb_port = 18480 + idx * 32
        sb_addrs = ",".join(
            f"{ip}:{sb_port + d_i}" for ip in hosts_ips for d_i in range(args.nproc_per_node)
        )
        stage_json = out_root / f"{st['name']}_summary.json"
        metrics_jsonl = out_root / f"{st['name']}_metrics.jsonl"
        ckpt_dir = out_root / "checkpoints" / st["name"]

        stage_env = os.environ.copy()
        for k_drop in ("TPU_PROCESS_ADDRESSES", "TPU_PROCESS_PORT"):
            stage_env.pop(k_drop, None)
        stage_env.update(
            {
                "PJRT_DEVICE": "TPU",
                "TPU_WORKER_ID": str(node_rank),
                "TPU_WORKER_HOSTNAMES": ",".join(hosts_ips),
                "TORCH_TPU_TOPOLOGY": torch_tpu_topo,
                "TORCH_TPU_DEVICES_PER_HOST": str(args.nproc_per_node),
                "TORCH_TPU_BASE_PORT": str(sb_port),
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addrs,
            }
        )

        cmd = [
            "torchrun",
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={args.nproc_per_node}",
            "--rdzv_backend=c10d",
            f"--rdzv_id=tabiclv2_{st['name']}_{idx}",
            f"--rdzv_endpoint={hosts_ips[0]}:{port}",
            f"--node_rank={node_rank}",
            __file__,
            "--worker",
            f"--stage-name={st['name']}",
            f"--max-steps={st['max_steps']}",
            f"--save-every={st['save_every']}",
            f"--log-every={st['log_every']}",
            f"--global-batch-size={st['global_bs']}",
            f"--micro-batch-size={st['micro_bs']}",
            f"--batch-size-per-gp={st['bs_per_gp']}",
            f"--lr={st['lr']}",
            f"--grad-clip={st['grad_clip']}",
            f"--min-seq-len={st['min_seq']}",
            f"--max-seq-len={st['max_seq']}",
            f"--min-train-size={st['min_tr']}",
            f"--max-train-size={st['max_tr']}",
            f"--num-workers={st['workers']}",
            f"--ckpt-dir={ckpt_dir}",
            f"--metrics-jsonl={metrics_jsonl}",
            f"--stage-summary-json={stage_json}",
        ]
        if st["log_seq"]:
            cmd.append("--log-seq-len")
        if st["regression"]:
            cmd.append("--regression")
        if st["recompute"]:
            cmd.append("--recompute")
        if st["init_ckpt"]:
            cmd.append(f"--init-ckpt={st['init_ckpt']}")

        if node_rank == 0:
            print(
                f"\n=== [Stage {idx+1}/{len(suite)}] Launching {st['name']} on rdzv={hosts_ips[0]}:{port} "
                f"sb_base={sb_port} ===",
                flush=True,
            )

        rc = subprocess.call(cmd, env=stage_env)
        if rc != 0:
            print(f"[TabICLv2-TPU] Stage {st['name']} exited with rc={rc} on node {node_rank}", flush=True)
            if st["name"] != "clf_stage1_sustained_50k":
                return rc
        if node_rank == 0 and stage_json.exists():
            completed_summaries.append(json.loads(stage_json.read_text(encoding="utf-8")))
            _write_overall_summary(out_root, args.slice_name, world_size, completed_summaries)
        time.sleep(5)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    # Orchestrator args
    ap.add_argument("--nnodes", type=int, default=4)
    ap.add_argument("--nproc-per-node", type=int, default=4)
    ap.add_argument("--master-addr", type=str, default="localhost")
    ap.add_argument("--base-port", type=int, default=29610)
    ap.add_argument("--slice-name", type=str, default="v5p-16-4n-2x2x4")
    ap.add_argument("--output-dir", type=str, default="/models/tabiclv2-v5p-pretrain/v5p-16")
    # Worker stage args
    ap.add_argument("--stage-name", type=str, default="clf_stage1_1k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--global-batch-size", type=int, default=64)
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--batch-size-per-gp", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--warmup-proportion", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--max-features", type=int, default=100)
    ap.add_argument("--min-seq-len", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--log-seq-len", action="store_true")
    ap.add_argument("--min-train-size", type=float, default=0.3)
    ap.add_argument("--max-train-size", type=float, default=0.9)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--init-ckpt", type=str, default="")
    ap.add_argument("--ckpt-dir", type=str, default="/tmp/ckpt")
    ap.add_argument("--metrics-jsonl", type=str, default="/tmp/metrics.jsonl")
    ap.add_argument("--stage-summary-json", type=str, default="")
    args = ap.parse_args()

    if args.worker:
        return run_worker_stage(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Distributed PyTorch (TorchTPU 0.2.0 + `tpu_dist`) Pre-Training & Fine-Tuning Runner for TabICLv2 on Cloud TPU v5p.

Optimized Stage-1 distributed training runner on TPU v5p (v5p-16 / v5p-32) supporting compiled forward+loss per split bucket,
grouped compiled Muon updates, fused Pallas small-head attention (`--mha=pk`), exact matmul RoPE (`--mha-rope=mm`), and packed projections (`--mha-packed`).

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


def _install_sdpa_math_train(max_elems: int = 1 << 20) -> None:
    """Route short, unmasked attention (q_len * k_len <= max_elems, keys below the Splash threshold) to an explicit
    matmul + fp32 softmax + matmul instead of F.scaled_dot_product_attention (TorchTPU's SDPA kernel)."""
    import torch
    from tabicl._model import attention as _att

    if getattr(_att, "_v5p_sdpa_math_installed", False):
        return
    orig = _att.sdpa_with_flattened_batch

    def sdpa_with_flattened_batch(q, k, v, attn_mask=None, dropout_p=0.0, ssmax_layer=None,  # noqa: ANN001
                                  src_len_override=None, valid_kv_len=None):
        if (q.device.type == "tpu" and dropout_p == 0.0 and attn_mask is None and valid_kv_len is None
                and getattr(_att, "_tpu_flash_attention_fn", None) is None
                and k.shape[-2] < _att._tpu_splash_min_kv() and q.shape[-2] * k.shape[-2] <= max_elems):
            q_shape = q.shape
            q2 = q.reshape(-1, *q.shape[-3:])
            k2 = k.reshape(-1, *k.shape[-3:])
            v2 = v.reshape(-1, *v.shape[-3:])
            if ssmax_layer is not None:
                q2 = ssmax_layer(q2, src_len_override if src_len_override is not None else k2.size(-2))
            scores = torch.matmul(q2 * (q2.shape[-1] ** -0.5), k2.transpose(-2, -1)).float()
            probs = torch.softmax(scores, dim=-1).to(v2.dtype)
            return torch.matmul(probs, v2).view(q_shape)
        return orig(q, k, v, attn_mask, dropout_p, ssmax_layer=ssmax_layer, src_len_override=src_len_override,
                    valid_kv_len=valid_kv_len)

    _att.sdpa_with_flattened_batch = sdpa_with_flattened_batch
    _att._v5p_sdpa_math_installed = True


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

    if getattr(args, "mha", ""):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        if getattr(args, "mha_rope", ""):
            os.environ["TABICL_TPU_MHA_ROPE"] = args.mha_rope
        if getattr(args, "mha_packed", False):
            os.environ["TABICL_TPU_MHA_PACKED"] = "1"
        import tabicl_tpu_mha as _mha_mod
        from tabicl._model import attention as _att_m
        from tabicl._model import layers as _lay_m

        _mha_mod.install(_att_m, _lay_m, mode=args.mha)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[TabICLv2-TPU] --mha={args.mha} rope={os.environ.get('TABICL_TPU_MHA_ROPE', 'roll')} "
                  f"packed={os.environ.get('TABICL_TPU_MHA_PACKED', '0')}: small-head attention (tabicl_tpu_mha.py)", flush=True)
    if getattr(args, "sdpa_math", False):
        _install_sdpa_math_train()
        if int(os.environ.get("RANK", "0")) == 0:
            print("[TabICLv2-TPU] --sdpa-math: short unmasked attention via explicit matmul + fp32 softmax", flush=True)

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
    fwd_model = model
    if args.compile_model:
        torch._dynamo.config.recompile_limit = max(getattr(torch._dynamo.config, "recompile_limit", 8), 64)
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                getattr(torch._dynamo.config, "accumulated_recompile_limit", 64), 256
            )
        fwd_model = torch.compile(model, backend="tpu", dynamic=False)
        if rank == 0:
            print("[TabICLv2-TPU] torch.compile(model, backend='tpu', dynamic=False) enabled", flush=True)
    fwd_loss_c = None
    if args.compile_step_fn:
        torch._dynamo.config.recompile_limit = max(getattr(torch._dynamo.config, "recompile_limit", 8), 64)

        def _fwd_loss(X, y_full, b_tr: int):
            y_tr = y_full[:, :b_tr].to(torch.float32)
            y_te = y_full[:, b_tr:]
            with torch.amp.autocast("tpu", dtype=torch.bfloat16):
                pred_ = model(X, y_tr, None, num_classes=cfg.max_classes)
                logits_ = pred_.float().flatten(end_dim=-2)
                targets_ = y_te.long().clamp(0, cfg.max_classes - 1).flatten()
                loss_ = F.cross_entropy(logits_, targets_)
            acc_ = (logits_.argmax(dim=1) == targets_).float().mean().detach()
            return loss_, acc_

        fwd_loss_c = torch.compile(_fwd_loss, backend="tpu", dynamic=False)
        if rank == 0:
            print("[TabICLv2-TPU] torch.compile(forward+loss, scalar outputs) enabled", flush=True)
    bd_lo, bd_hi = 0, -1
    if args.breakdown_steps:
        _lo, _hi = args.breakdown_steps.split("-")
        bd_lo, bd_hi = int(_lo), int(_hi)
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
    if args.compiled_muon:
        from tabicl.train._muon import zeropower_via_newtonschulz5 as _ns5

        torch._dynamo.config.recompile_limit = max(getattr(torch._dynamo.config, "recompile_limit", 8), 1024)
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                getattr(torch._dynamo.config, "accumulated_recompile_limit", 64), 8192
            )

        def _muon_param_update(p, g, buf, lr_t, momentum: float, nesterov: bool, ns_steps: int,
                               weight_decay: float, adj_ratio: float):
            buf.mul_(momentum).add_(g)
            ns_in = g.add(buf, alpha=momentum) if nesterov else buf
            ns_in = ns_in.reshape(g.shape[0], -1)
            upd = _ns5(ns_in, steps=ns_steps).view(g.shape)
            p.mul_(1.0 - lr_t * weight_decay)
            p.sub_(upd * (lr_t * adj_ratio))

        _muon_upd_c = torch.compile(_muon_param_update, backend="tpu", dynamic=False, fullgraph=True)

        def _compiled_muon_step():
            for group in optimizer.param_groups:
                assert group.get("use_muon", False) and not group["use_cautious_wd"]
                lr_t = torch.tensor(float(group["lr"]), device=dev, dtype=torch.float32)
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    st = optimizer.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    rows = g.shape[0]
                    cols = g.numel() // rows
                    adj = math.sqrt(max(rows, cols)) * float(group["matched_adamw_rms"])
                    _muon_upd_c(p.data, g, st["momentum_buffer"], lr_t, float(group["momentum"]),
                                bool(group["nesterov"]), int(group["ns_steps"]), float(group["weight_decay"]), adj)

        if getattr(args, "grouped_muon", False):
            def _muon_all(params, grads, bufs, lr_t, adjs, momentum: float, nesterov: bool, ns_steps: int,
                          weight_decay: float):
                for p_, g_, b_, a_ in zip(params, grads, bufs, adjs):
                    b_.mul_(momentum).add_(g_)
                    ns_in_ = g_.add(b_, alpha=momentum) if nesterov else b_
                    ns_in_ = ns_in_.reshape(g_.shape[0], -1)
                    upd_ = _ns5(ns_in_, steps=ns_steps).view(g_.shape)
                    p_.mul_(1.0 - lr_t * weight_decay)
                    p_.sub_(upd_ * (lr_t * a_))

            _muon_all_c = torch.compile(_muon_all, backend="tpu", dynamic=False, fullgraph=True)

            def _grouped_muon_step():
                for group in optimizer.param_groups:
                    assert group.get("use_muon", False) and not group["use_cautious_wd"]
                    lr_t = torch.tensor(float(group["lr"]), device=dev, dtype=torch.float32)
                    ps, gs, bs, adjs = [], [], [], []
                    for p in group["params"]:
                        g = p.grad
                        if g is None:
                            continue
                        st = optimizer.state[p]
                        if "momentum_buffer" not in st:
                            st["momentum_buffer"] = torch.zeros_like(g)
                        rows = g.shape[0]
                        cols = g.numel() // rows
                        ps.append(p.data)
                        gs.append(g)
                        bs.append(st["momentum_buffer"])
                        adjs.append(math.sqrt(max(rows, cols)) * float(group["matched_adamw_rms"]))
                    if ps:
                        _muon_all_c(ps, gs, bs, lr_t, adjs, float(group["momentum"]), bool(group["nesterov"]),
                                    int(group["ns_steps"]), float(group["weight_decay"]))

            optimizer.step = _grouped_muon_step
        else:
            optimizer.step = _compiled_muon_step
        if rank == 0:
            print("[TabICLv2-TPU] compiled Muon step enabled (" + ("one grouped graph for all parameters" if getattr(args, "grouped_muon", False)
                  else "per-parameter torch.compile") + ", backend='tpu')", flush=True)
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
    if args.precompile_splits and fwd_loss_c is not None and fixed_1k and not args.regression:
        t_pc0 = time.perf_counter()
        # Use a REAL first micro-batch (exact dtype/stride/shape of the training inputs) so the guards match.
        _b0 = next(data_iter)
        _b0 = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in _b0]
        _mb0 = list(zip(*[torch.split(t, cfg.micro_batch_size, dim=0) for t in _b0]))[0]
        _mbX, _mby = _mb0[0], _mb0[1]
        if _mbX.shape[1] > 1024:
            _mbX, _mby = _mbX[:, :1024, :], _mby[:, :1024]
        elif _mbX.shape[1] < 1024:
            _padT = 1024 - _mbX.shape[1]
            _mbX = F.pad(_mbX, (0, 0, 0, _padT), value=0.0)
            _mby = F.pad(_mby, (0, _padT), value=0)
        if _mbX.shape[2] < cfg.max_features:
            _mbX = F.pad(_mbX, (0, cfg.max_features - _mbX.shape[2]), value=0.0)
        elif _mbX.shape[2] > cfg.max_features:
            _mbX = _mbX[:, :, : cfg.max_features]
        _X0 = _mbX.to(dev, dtype=torch.float32)
        _y0 = _mby.to(dev)
        if rank == 0:
            print(f"[TabICLv2-TPU] precompile inputs X{tuple(_X0.shape)} {_X0.dtype} y{tuple(_y0.shape)} {_y0.dtype}", flush=True)
        for _btr in (320, 448, 576, 704, 832, 928):
            _l0, _a0 = fwd_loss_c(_X0, _y0, _btr)
            _l0.backward()
            optimizer.zero_grad(set_to_none=True)
            torch.accelerator.synchronize(dev)
        if rank == 0:
            print(f"[TabICLv2-TPU] precompiled 6 split buckets in {time.perf_counter() - t_pc0:.1f}s", flush=True)

    bar = torch.zeros(1, device=dev)
    dist.all_reduce(bar)
    torch.accelerator.synchronize(dev)
    try:
        torch.accelerator.reset_peak_memory_stats(dev)
    except Exception:
        pass

    metrics_path = Path(args.metrics_jsonl)
    # --local-metrics: per-step rows go to local disk and are copied to --metrics-jsonl (GCS FUSE) once at the end.
    # A per-step append on GCS FUSE re-uploads the object on every close (rank 0 only), and every other rank then
    # waits for rank 0 at the next gradient AllReduce.
    local_metrics_path = Path(f"/tmp/{args.stage_name}_metrics.local.jsonl") if args.local_metrics else None
    if local_metrics_path is not None and local_metrics_path.exists():
        local_metrics_path.unlink()
    step_records: list[dict[str, Any]] = []
    _t_prev_step_start = None

    for step in range(start_step, args.max_steps):
        t_p0 = time.perf_counter()
        step_period_ms = None if _t_prev_step_start is None else (t_p0 - _t_prev_step_start) * 1000.0
        _t_prev_step_start = t_p0
        batch = next(data_iter)
        prior_s = time.perf_counter() - t_p0

        t_s0 = time.perf_counter()
        bd_on = bd_lo <= step + 1 <= bd_hi
        bd: dict[str, float] = {}
        bd_t = [time.perf_counter()]

        def _mark(name: str) -> None:
            if not bd_on:
                return
            torch.accelerator.synchronize(dev)
            now = time.perf_counter()
            bd[name] = bd.get(name, 0.0) + (now - bd_t[0]) * 1000.0
            bd_t[0] = now

        if bd_on:
            torch.accelerator.synchronize(dev)
            bd_t[0] = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if args.device_metrics:
            loss_dev = torch.zeros((), device=dev, dtype=torch.float32)
            acc_dev = torch.zeros((), device=dev, dtype=torch.float32)

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
            _mark("pad_h2d")

            if fwd_loss_c is not None and not args.regression:
                loss, acc_c = fwd_loss_c(mb_X_dev, mb_y_dev, int(b_tr))
            else:
              with torch.amp.autocast("tpu", dtype=torch.bfloat16):
                if args.fixed_num_classes and not args.regression:
                    pred = fwd_model(mb_X_dev, y_train, None, num_classes=cfg.max_classes)
                else:
                    pred = fwd_model(mb_X_dev, y_train, None)
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
            _mark("fwd")
            scaled_loss.backward()
            _mark("bwd")

            with torch.no_grad():
                if args.device_metrics:
                    loss_dev = loss_dev + scaled_loss.detach().float()
                    if not args.regression:
                        if fwd_loss_c is not None:
                            acc_dev = acc_dev + acc_c / num_micro_batches
                        else:
                            acc_dev = acc_dev + (logits_2d.argmax(dim=1) == targets_1d).float().mean() / num_micro_batches
                else:
                    total_loss_val += float(scaled_loss.detach().cpu().item())
                    if not args.regression:
                        acc = (logits_2d.argmax(dim=1) == targets_1d).float().mean()
                        total_acc_val += float(acc.detach().cpu().item()) / num_micro_batches
            _mark("metrics")

        # AllReduce gradients across all TPU v5p chips via ICI
        if args.coalesce_allreduce:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            off = 0
            for g in grads:
                n_el = g.numel()
                g.copy_(flat[off : off + n_el].view_as(g))
                off += n_el
            del flat
        else:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        _mark("allreduce")

        if cfg.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)
        _mark("clip")

        optimizer.step()
        scheduler.step()
        _mark("opt")
        torch.accelerator.synchronize(dev)
        train_s = time.perf_counter() - t_s0
        step_wall_s = prior_s + train_s

        # Aggregate metrics across all ranks every step for exact cluster-wide loss/accuracy/throughput
        if args.device_metrics:
            host_part = torch.tensor(
                [float(total_rows_local), step_wall_s, train_s, prior_s], device=dev, dtype=torch.float32
            )
            stat_vec = torch.cat([torch.stack([loss_dev, acc_dev]).float(), host_part])
        else:
            stat_vec = torch.tensor(
                [total_loss_val, total_acc_val, float(total_rows_local), step_wall_s, train_s, prior_s],
                device=dev,
                dtype=torch.float32,
            )
        dist.all_reduce(stat_vec, op=dist.ReduceOp.AVG)
        torch.accelerator.synchronize(dev)
        train_ms_max = train_ms_min = None
        if args.rank_stats:
            try:
                mm = torch.tensor([train_s, -train_s], device=dev, dtype=torch.float32)
                dist.all_reduce(mm, op=dist.ReduceOp.MAX)
                torch.accelerator.synchronize(dev)
                _mm = mm.cpu().tolist()
                train_ms_max, train_ms_min = _mm[0] * 1000.0, -_mm[1] * 1000.0
            except Exception as _exc:  # noqa: BLE001
                if rank == 0 and step == start_step:
                    print(f"[TabICLv2-TPU] rank-stats MAX all-reduce unavailable: {_exc}", flush=True)
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
            rec["tpu_train_ms_rank0"] = round(train_s * 1000.0, 2)
            if step_period_ms is not None:
                rec["step_period_ms_rank0"] = round(step_period_ms, 2)
            if train_ms_max is not None:
                rec["tpu_train_ms_max"] = round(train_ms_max, 2)
                rec["tpu_train_ms_min"] = round(train_ms_min, 2)
            if bd_on:
                rec.update({f"bd_{k}_ms": round(v, 2) for k, v in bd.items()})
            step_records.append(rec)
            with (local_metrics_path if local_metrics_path is not None else metrics_path).open("a", encoding="utf-8") as mf:
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

            if (not args.no_save) and (cur_step % args.save_every == 0 or cur_step == args.max_steps):
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

    if rank == 0 and local_metrics_path is not None and local_metrics_path.exists():
        import shutil

        shutil.copyfile(local_metrics_path, metrics_path)
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
            "latest_ckpt": "" if args.no_save else str(ckpt_dir / f"step-{args.max_steps}.ckpt"),
            "steady_p90_wall_ms": round(sorted(r["step_wall_ms"] for r in steady)[int(0.9 * len(steady))], 2) if steady else 0.0,
            "steady_median_period_ms_rank0": round(statistics.median([r["step_period_ms_rank0"] for r in steady if "step_period_ms_rank0" in r]), 2) if any("step_period_ms_rank0" in r for r in steady) else None,
            "steady_median_train_ms_rank0": round(statistics.median([r["tpu_train_ms_rank0"] for r in steady if "tpu_train_ms_rank0" in r]), 2) if any("tpu_train_ms_rank0" in r for r in steady) else None,
            "steady_median_train_ms_max": round(statistics.median([r["tpu_train_ms_max"] for r in steady if "tpu_train_ms_max" in r]), 2) if any("tpu_train_ms_max" in r for r in steady) else None,
            "steady_median_train_ms_min": round(statistics.median([r["tpu_train_ms_min"] for r in steady if "tpu_train_ms_min" in r]), 2) if any("tpu_train_ms_min" in r for r in steady) else None,
            "local_metrics": bool(args.local_metrics),
            "flags": {
                "coalesce_allreduce": bool(args.coalesce_allreduce),
                "device_metrics": bool(args.device_metrics),
                "compile_model": bool(args.compile_model),
                "breakdown_steps": args.breakdown_steps,
                "compiled_muon": bool(getattr(args, "compiled_muon", False)),
            },
        }
        bd_recs = [r for r in step_records if any(k.startswith("bd_") for k in r)]
        if bd_recs:
            keys = sorted({k for r in bd_recs for k in r if k.startswith("bd_")})
            summary_obj["breakdown_median_ms"] = {
                k: round(statistics.median([r[k] for r in bd_recs if k in r]), 2) for k in keys
            }
            summary_obj["breakdown_steps_counted"] = len(bd_recs)
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

    global_bs = args.global_batch_size if args.global_batch_size > 0 else (16 * world_size if args.preset == "best" else 64)
    micro_bs = args.micro_batch_size
    num_workers = args.num_workers

    extra_flags: list[str] = []
    if args.preset == "best":
        extra_flags.extend([
            "--coalesce-allreduce",
            "--device-metrics",
            "--compiled-muon",
            "--grouped-muon",
            "--compile-step-fn",
            "--precompile-splits",
            "--local-metrics",
            "--rank-stats",
            "--sdpa-math",
            "--mha=pk",
            "--mha-rope=mm",
            "--mha-packed",
        ])
    else:
        if args.coalesce_allreduce:
            extra_flags.append("--coalesce-allreduce")
        if args.device_metrics:
            extra_flags.append("--device-metrics")
        if args.compile_model:
            extra_flags.append("--compile-model")
        if args.compiled_muon:
            extra_flags.append("--compiled-muon")
        if args.grouped_muon:
            extra_flags.append("--grouped-muon")
        if args.fixed_num_classes:
            extra_flags.append("--fixed-num-classes")
        if args.compile_step_fn:
            extra_flags.append("--compile-step-fn")
        if args.precompile_splits:
            extra_flags.append("--precompile-splits")
        if args.local_metrics:
            extra_flags.append("--local-metrics")
        if args.rank_stats:
            extra_flags.append("--rank-stats")
        if args.sdpa_math:
            extra_flags.append("--sdpa-math")
        if args.mha:
            extra_flags.append(f"--mha={args.mha}")
        if args.mha_rope:
            extra_flags.append(f"--mha-rope={args.mha_rope}")
        if args.mha_packed:
            extra_flags.append("--mha-packed")
    if args.no_save:
        extra_flags.append("--no-save")
    if args.breakdown_steps:
        extra_flags.append(f"--breakdown-steps={args.breakdown_steps}")

    sb_port = 18480
    sb_addrs = ",".join(f"{ip}:{sb_port + d_i}" for ip in hosts_ips for d_i in range(args.nproc_per_node))
    stage_json = out_root / f"{args.stage_name}_summary.json"
    metrics_jsonl = out_root / f"{args.stage_name}_metrics.jsonl"
    ckpt_dir = out_root / "checkpoints" / args.stage_name

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
        f"--rdzv_id=tabiclv2_{args.stage_name}",
        f"--rdzv_endpoint={hosts_ips[0]}:{args.base_port}",
        f"--node_rank={node_rank}",
        __file__,
        "--worker",
        f"--stage-name={args.stage_name}",
        f"--max-steps={args.max_steps}",
        f"--save-every={args.save_every}",
        f"--log-every={args.log_every}",
        f"--global-batch-size={global_bs}",
        f"--micro-batch-size={micro_bs}",
        f"--batch-size-per-gp={args.batch_size_per_gp}",
        f"--lr={args.lr}",
        f"--grad-clip={args.grad_clip}",
        f"--min-seq-len={args.min_seq_len}",
        f"--max-seq-len={args.max_seq_len}",
        f"--min-train-size={args.min_train_size}",
        f"--max-train-size={args.max_train_size}",
        f"--num-workers={num_workers}",
        f"--ckpt-dir={ckpt_dir}",
        f"--metrics-jsonl={metrics_jsonl}",
        f"--stage-summary-json={stage_json}",
        *extra_flags,
    ]
    if args.log_seq_len:
        cmd.append("--log-seq-len")
    if args.regression:
        cmd.append("--regression")
    if args.recompute:
        cmd.append("--recompute")
    if args.init_ckpt:
        cmd.append(f"--init-ckpt={args.init_ckpt}")

    if node_rank == 0:
        print(f"\n=== Launching {args.stage_name} (global_bs={global_bs}, micro_bs={micro_bs}) ===", flush=True)

    rc = subprocess.call(cmd, env=stage_env)
    if rc == 0 and node_rank == 0 and stage_json.exists():
        summary = json.loads(stage_json.read_text(encoding="utf-8"))
        _write_overall_summary(out_root, args.slice_name, world_size, [summary])
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--preset", choices=["default", "best"], default="best",
                    help="Use 'best' to enable compiled forward+loss, grouped compiled Muon, and fused Pallas small-head + packed-projection kernels")
    # Multi-host orchestrator args
    ap.add_argument("--nnodes", type=int, default=4)
    ap.add_argument("--nproc-per-node", type=int, default=4)
    ap.add_argument("--master-addr", type=str, default="localhost")
    ap.add_argument("--base-port", type=int, default=29610)
    ap.add_argument("--slice-name", type=str, default="v5p-16")
    ap.add_argument("--output-dir", type=str, default="/tmp/tabiclv2-v5p-train")
    # Stage / worker args
    ap.add_argument("--stage-name", type=str, default="clf_stage1_1k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--global-batch-size", type=int, default=0, help="0 = 256 (16/chip) for --preset best on 16 chips, else 64")
    ap.add_argument("--micro-batch-size", type=int, default=8)
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
    ap.add_argument("--num-workers", type=int, default=24)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--init-ckpt", type=str, default="")
    ap.add_argument("--ckpt-dir", type=str, default="/tmp/ckpt")
    ap.add_argument("--metrics-jsonl", type=str, default="/tmp/metrics.jsonl")
    ap.add_argument("--stage-summary-json", type=str, default="")
    ap.add_argument("--coalesce-allreduce", action="store_true")
    ap.add_argument("--device-metrics", action="store_true")
    ap.add_argument("--compile-model", action="store_true")
    ap.add_argument("--breakdown-steps", type=str, default="")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--compiled-muon", action="store_true")
    ap.add_argument("--fixed-num-classes", action="store_true")
    ap.add_argument("--compile-step-fn", action="store_true")
    ap.add_argument("--precompile-splits", action="store_true")
    ap.add_argument("--local-metrics", action="store_true")
    ap.add_argument("--rank-stats", action="store_true")
    ap.add_argument("--sdpa-math", action="store_true")
    ap.add_argument("--grouped-muon", action="store_true")
    ap.add_argument("--mha", type=str, default="")
    ap.add_argument("--mha-rope", type=str, default="")
    ap.add_argument("--mha-packed", action="store_true")
    args = ap.parse_args()

    if args.worker:
        if args.global_batch_size <= 0:
            args.global_batch_size = 256
        return run_worker_stage(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())

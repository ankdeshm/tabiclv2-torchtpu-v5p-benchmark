#!/usr/bin/env python3
"""CPU equivalence test: upstream Muon.step vs per-parameter compiled Muon update (Dynamo fullgraph, eager backend).

Builds the real TabICLv2 classifier config (same args as the v5p training script), assigns identical random grads,
runs N optimizer steps with both implementations (lr changing every step like the cosine schedule), and compares
parameters + momentum buffers. Also reports the number of parameter tensors and distinct 2D NS shapes.
"""
import copy
import math
import sys

import torch

sys.path.insert(0, "/workspace/tabicl/src")
from tabicl._model.tabicl import TabICL  # noqa: E402
from tabicl.train._muon import Muon, zeropower_via_newtonschulz5 as _ns5  # noqa: E402

torch.manual_seed(0)
cfg = dict(max_classes=10, num_quantiles=999, embed_dim=128, col_num_blocks=3, col_nhead=8, col_num_inds=128,
           col_affine=False, col_feature_group="same", col_feature_group_size=3, col_target_aware=True,
           col_ssmax="qassmax-mlp-elementwise", row_num_blocks=3, row_nhead=8, row_num_cls=4, row_rope_base=100000,
           row_rope_interleaved=False, icl_num_blocks=12, icl_nhead=8, icl_ssmax="qassmax-mlp-elementwise",
           ff_factor=2, dropout=0.0, activation="gelu", norm_first=True, bias_free_ln=False, zero_init=False,
           recompute=False)
try:
    model = TabICL(**cfg)
except TypeError as exc:  # tolerate minor signature differences
    print("TabICL kwargs issue:", exc)
    for k in ("recompute", "bias_free_ln", "zero_init"):
        cfg.pop(k, None)
    model = TabICL(**cfg)

params = [p for p in model.parameters() if p.requires_grad]
shapes = {}
for p in params:
    rows = p.shape[0]
    cols = p.numel() // rows
    shapes[(rows, cols)] = shapes.get((rows, cols), 0) + 1
print(f"param tensors={len(params)} total={sum(p.numel() for p in params):,} distinct NS shapes={len(shapes)}")

kw = dict(lr=8e-4, weight_decay=0.01, matched_adamw_rms=0.2, momentum=0.9, nesterov=True, ns_steps=5,
          adamw_betas=(0.9, 0.95), adamw_eps=1e-8, use_cautious_wd=False)
m_ref = model
m_new = copy.deepcopy(model)
opt_ref = Muon(param_groups=[dict(params=[p for p in m_ref.parameters() if p.requires_grad], use_muon=True)], **kw)
opt_new = Muon(param_groups=[dict(params=[p for p in m_new.parameters() if p.requires_grad], use_muon=True)], **kw)


def _muon_param_update(p, g, buf, lr_t, momentum: float, nesterov: bool, ns_steps: int, weight_decay: float,
                       adj_ratio: float):
    buf.mul_(momentum).add_(g)
    ns_in = g.add(buf, alpha=momentum) if nesterov else buf
    ns_in = ns_in.reshape(g.shape[0], -1)
    upd = _ns5(ns_in, steps=ns_steps).view(g.shape)
    p.mul_(1.0 - lr_t * weight_decay)
    p.sub_(upd * (lr_t * adj_ratio))


torch._dynamo.config.recompile_limit = 1024
if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
    torch._dynamo.config.accumulated_recompile_limit = 8192
upd_c = torch.compile(_muon_param_update, backend="eager", dynamic=False, fullgraph=True)


def compiled_step(optimizer, dev):
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
            upd_c(p.data, g, st["momentum_buffer"], lr_t, float(group["momentum"]), bool(group["nesterov"]),
                  int(group["ns_steps"]), float(group["weight_decay"]), adj)


worst = 0.0
init_ref = [p.detach().clone() for p in m_ref.parameters() if p.requires_grad]
n_frozen = sum(1 for p in m_ref.parameters() if not p.requires_grad)
print(f"frozen (requires_grad=False) tensors={n_frozen}")
for step in range(3):
    lr = 8e-4 * (1.0 - 0.3 * step)
    for o in (opt_ref, opt_new):
        for gr in o.param_groups:
            gr["lr"] = lr
    gen = torch.Generator().manual_seed(100 + step)
    for p_r, p_n in zip(m_ref.parameters(), m_new.parameters()):
        g = torch.randn(p_r.shape, generator=gen) * 1e-2
        p_r.grad = g.clone()
        p_n.grad = g.clone()
    opt_ref.step()
    compiled_step(opt_new, torch.device("cpu"))
    with torch.no_grad():
        pairs = [(a, b) for a, b in zip(m_ref.parameters(), m_new.parameters()) if a.requires_grad]
        md = max(float((a - b).abs().max()) for a, b in pairs)
        bd = max(float((opt_ref.state[a]["momentum_buffer"] - opt_new.state[b]["momentum_buffer"]).abs().max())
                 for a, b in pairs)
        rel = max(float(((a - b).abs().max() / (a.abs().max() + 1e-12))) for a, b in pairs)
        upd = max(float((a - a0).abs().max()) for (a, _), a0 in zip(pairs, init_ref))
    worst = max(worst, md)
    print(f"step {step}: lr={lr:.2e} max|dparam|={md:.3e} max rel={rel:.3e} max|dbuf|={bd:.3e} (max |param change from init| = {upd:.3e})")
print("RESULT", "PASS" if worst < 1e-5 else "CHECK", f"worst={worst:.3e}")

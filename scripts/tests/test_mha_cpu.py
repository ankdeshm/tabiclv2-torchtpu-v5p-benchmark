#!/usr/bin/env python3
"""CPU equivalence test for tabicl_tpu_mha (small-head attention layouts) against upstream attention.

Runs on a CPU pod (TABICL_TPU_MHA_ANY_DEVICE=1 lets the TPU-only wrapper run on CPU):
 1. RoPE in merged layout vs RotaryEmbedding.rotate_queries_or_keys on the head-split layout.
 2. small_head_attention impls vs the head-split reference with masks (fp32).
 3. Full TabICL train forward + backward (logits, loss, every parameter grad) for each mode vs upstream.
 4. Dynamo fullgraph trace (aot_eager) of the train forward+backward with mode=auto (no graph breaks).
 5. Inference: TabICLClassifier.predict_proba (masks, ISAB train_size) with mode=auto vs upstream.
"""
from __future__ import annotations

import os
import sys
import time

os.environ["TABICL_TPU_MHA_ANY_DEVICE"] = "1"
sys.path.insert(0, "/workspace/run")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tabicl import TabICLClassifier  # noqa: E402
from tabicl._model import attention as att  # noqa: E402
from tabicl._model import layers as lay  # noqa: E402
import tabicl_tpu_mha as mha  # noqa: E402

torch.manual_seed(0)
ORIG = att.multi_head_attention_forward
FAIL = []


def check(name, val, tol):
    ok = val <= tol
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {val:.3e} (tol {tol:.0e})", flush=True)
    if not ok:
        FAIL.append(name)


def set_mode(mode: str) -> None:
    att.multi_head_attention_forward = ORIG
    lay.multi_head_attention_forward = ORIG
    att._v5p_mha_installed = False
    proj = "packed"
    if mode.endswith("+split"):
        mode, proj = mode[: -len("+split")], "split"
    if mode:
        assert mha.install(att, lay, mode=mode, proj=proj)


rng = np.random.RandomState(0)
X0 = rng.randn(64, 8).astype(np.float32)
clf0 = TabICLClassifier(n_estimators=1, random_state=0, device="cpu")
clf0.fit(X0, (X0[:, 0] > 0).astype(np.int64))
model = clf0.model_.float().train()

# 1. RoPE
rope = model.row_interactor.tf_row.rope
for L in (4, 104):
    x = torch.randn(3, 5, L, 128)
    xh = x.unflatten(-1, (8, 16)).transpose(-3, -2)
    ref = rope.rotate_queries_or_keys(xh).transpose(-3, -2).flatten(-2)
    check(f"rope merged L={L}", (mha.rope_merged(rope, x, 8) - ref).abs().max().item(), 1e-5)

# 2. core impls with masks
for (bs, Lq, Lk) in (((6,), 128, 300), ((6,), 512, 128), ((2, 3), 104, 104), ((2, 3), 4, 104)):
    q = torch.randn(*bs, Lq, 128) * 0.25
    k = torch.randn(*bs, Lk, 128)
    v = torch.randn(*bs, Lk, 128)
    mask = torch.zeros(*bs, 1, 1, Lk)
    mask[..., -7:] = torch.finfo(torch.float32).min
    ref = mha.small_head_attention(q, k, v, 8, mask, "std")
    for impl in ("hxq", "hxk", "thl"):
        out = mha.small_head_attention(q, k, v, 8, mask, impl)
        check(f"core {impl} bs={bs} {Lq}x{Lk} masked", (out - ref).abs().max().item(), 2e-5)

# 3. train forward/backward
NC, B, T, H, b_tr = 10, 2, 256, 20, 160
X = torch.randn(B, T, H)
y = torch.randint(0, NC, (B, T)).float()
y_tr, y_te = y[:, :b_tr], y[:, b_tr:]


def train_step(mode: str):
    set_mode(mode)
    model.zero_grad(set_to_none=True)
    logits = model(X, y_tr, None, num_classes=NC)
    loss = F.cross_entropy(logits.float().flatten(end_dim=-2), y_te.long().flatten())
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), float(loss), grads


ref_logits, ref_loss, ref_grads = train_step("")
for mode in ("std", "hxq", "hxk", "thl", "hx", "auto", "auto+split", "thl+split"):
    t0 = time.time()
    lg, ls, gr = train_step(mode)
    gdiff = max(((gr[n] - ref_grads[n]).abs().max() / (ref_grads[n].abs().max() + 1e-12)).item() for n in ref_grads)
    check(f"train {mode} logits", (lg - ref_logits).abs().max().item(), 1e-3)
    check(f"train {mode} loss", abs(ls - ref_loss), 1e-4)
    check(f"train {mode} grads (max rel to tensor max)", gdiff, 1e-3)
    check(f"train {mode} grad set", float(len(set(gr) ^ set(ref_grads))), 0)
    print(f"   ({time.time() - t0:.1f}s)", flush=True)

# 4. dynamo fullgraph trace with aot_eager
set_mode("auto+split")


def fwd_loss(X_, y_tr_, y_te_):
    logits = model(X_, y_tr_, None, num_classes=NC)
    return F.cross_entropy(logits.float().flatten(end_dim=-2), y_te_.long().flatten())


try:
    torch._dynamo.reset()
    cf = torch.compile(fwd_loss, backend="aot_eager", fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    lc = cf(X, y_tr, y_te)
    lc.backward()
    check("dynamo fullgraph auto loss", abs(float(lc) - ref_loss), 1e-4)
except Exception as exc:  # noqa: BLE001
    print(f"[FAIL] dynamo fullgraph auto: {type(exc).__name__}: {str(exc)[:800]}", flush=True)
    FAIL.append("dynamo")

# 5. inference predict_proba (categorical-free numeric table with varying features)
Xi = rng.randn(600, 13).astype(np.float32)
yi = (Xi[:, 0] + 0.5 * Xi[:, 1] > 0).astype(np.int64) + (Xi[:, 2] > 1).astype(np.int64)
Xtr, ytr, Xte = Xi[:450], yi[:450], Xi[450:]
probs = {}
for mode in ("", "auto", "hx", "thl", "auto+split"):
    set_mode(mode)
    c = TabICLClassifier(n_estimators=4, random_state=0, device="cpu")
    c.fit(Xtr, ytr)
    probs[mode] = c.predict_proba(Xte)
for mode in ("auto", "hx", "thl", "auto+split"):
    check(f"inference predict_proba {mode}", float(np.abs(probs[mode] - probs[""]).max()), 1e-4)

print("RESULT", "PASS" if not FAIL else f"FAIL {FAIL}", flush=True)

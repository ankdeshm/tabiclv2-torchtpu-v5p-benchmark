#!/usr/bin/env python3
"""CPU check of the folded first-ISAB attention algebra (tabicl_tpu_mha.folded_mab1) against upstream attention.

Uses the real TabICLv2 column ISAB blocks (trained weights, SSMax layer) and the upstream multi_head_attention_forward
with a key-padding mask for keys >= valid. The Pallas kernel is replaced by a torch reference of the same contract
(U = softmax(Qf x^T) x over the valid keys), so this checks the folding itself:
  S_g = (q_g W_k,g) x^T + const_row,   O_g = (P_g x) W_v,g^T + b_v,g.
"""
from __future__ import annotations

import os
import sys

os.environ["TABICL_TPU_MHA_ANY_DEVICE"] = "1"
sys.path.insert(0, "/workspace/run")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tabicl import TabICLClassifier  # noqa: E402
from tabicl._model import attention as att  # noqa: E402
import tabicl_tpu_mha as mha  # noqa: E402


class FakePK:
    @staticmethod
    def hxf_supported(hq: int, Lk: int, E: int) -> bool:
        return True

    @staticmethod
    def hxf_attention(qf, x, valid_len):  # noqa: ANN001
        Lk = x.shape[-2]
        s = torch.matmul(qf, x.transpose(-1, -2))
        keep = torch.arange(Lk) < valid_len.reshape(())
        s = s.masked_fill(~keep, float("-inf"))
        return torch.matmul(torch.softmax(s, dim=-1), x)


torch.manual_seed(0)
rng = np.random.RandomState(0)
X0 = rng.randn(64, 8).astype(np.float32)
clf0 = TabICLClassifier(n_estimators=1, random_state=0, device="cpu")
clf0.fit(X0, (X0[:, 0] > 0).astype(np.int64))
model = clf0.model_.float().eval()
FAIL = []
for bi, block in enumerate(model.col_embedder.tf_col.blocks):
    attn = block.multihead_attn1.attn
    E = attn.in_proj_weight.shape[1]
    h = attn.num_heads
    for (B, Lk, valid) in ((6, 300, 200), (3, 1024, 1000), (2, 4096, 3000)):
        x = torch.randn(B, Lk, E)
        ind = block.ind_vectors.expand(B, block.num_inds, E)
        vt = torch.tensor(valid)
        kpm = torch.zeros(B, Lk).masked_fill(torch.arange(Lk).expand(B, Lk) >= valid, float("-inf"))
        with torch.no_grad():
            ref = att.multi_head_attention_forward(
                ind, h, attn.in_proj_weight, attn.in_proj_bias, 0.0, attn.out_proj.weight, attn.out_proj.bias,
                key=x, value=x, training=False, key_padding_mask=kpm, ssmax_layer=attn.ssmax_layer,
                src_len_override=vt)
            for qkind, q in (("expanded", ind), ("per-batch", ind.contiguous())):
                o = mha.folded_mab1(q, x, vt, h, attn.in_proj_weight, attn.in_proj_bias, attn.ssmax_layer, vt,
                                    pk=FakePK)
                out = F.linear(o, attn.out_proj.weight, attn.out_proj.bias)
                err = (out - ref).abs().max().item()
                scale = max(ref.abs().max().item(), 1e-6)
                ok = err <= 1e-4 * max(1.0, scale)
                print(f"[{'ok' if ok else 'FAIL'}] block{bi} B={B} Lk={Lk} valid={valid} q={qkind} ssmax="
                      f"{type(attn.ssmax_layer).__name__}: max|d|={err:.3e} (ref max {scale:.3e})", flush=True)
                if not ok:
                    FAIL.append((bi, B, Lk, qkind))
print("FOLD_TEST", "PASS" if not FAIL else f"FAIL {FAIL}", flush=True)

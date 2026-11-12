"""Small-head multi-head attention layouts for TPU (TabICLv2 column / row stages: 8 heads x 16 dims).

Why: upstream ``multi_head_attention_forward`` splits heads with ``q.view(..., L, h, d).transpose(-3, -2)``, which
puts the 16-wide head dim on the TPU lane axis (padded to 128 lanes, so q/k/v/out cost 8x their size in HBM traffic
plus layout copies), and TorchTPU's SDPA kernel is slow for these many short sequences. XProf of a one-chip train
step: copies ~25% of device time even with explicit-matmul attention. The layouts below keep a wide axis (the
128-wide embedding or the sequence) on lanes. Each is mathematically the same attention as upstream
(softmax(Q_h K_h^T / sqrt(d) + mask) V_h per head, fp32 softmax); only the matmul accumulation order differs.

  "hxq"  query head expansion. Q_exp[..., g*Lq + i, :] = Q[..., i, :] * M[g] (M[g] = 0/1 mask of head g's dims)
         S = Q_exp @ K^T -> [..., h, Lq, Lk];  O_full = softmax(S) @ V -> [..., h, Lq, E];
         O = sum_g O_full[..., g, :, :] * M[g]  -> [..., Lq, E]. Good when Lq is small (inducing points, CLS).
  "hxk"  key head expansion. K_exp / V_exp[..., g*Lk + j, :] = K / V[..., j, :] * M[g]
         S = Q @ K_exp^T -> [..., Lq, h, Lk] (softmax per head block);  O = P @ V_exp -> [..., Lq, E].
         Good when Lk is small (the 128 inducing points as keys).
  "thl"  transposed head layout. Q^T / K^T / V^T as [..., h, d, L] (sequence on lanes, one 2-D transpose each);
         S = Q_h K_h^T -> [..., h, Lq, Lk];  O^T = V_h^T P^T -> [..., h, d, Lq] -> [..., Lq, E].

Install with ``install(attention_module, layers_module)`` (patches ``multi_head_attention_forward`` in both, since
layers.py imports it by name). Knobs:
  TABICL_TPU_MHA=hxq|hxk|thl|std|pk|hx|auto|<rules>  (hx = hxq if Lq <= Lk else hxk; auto = shape rules below;
     pk = fused Pallas kernel tabicl_tpu_pallas_attn.py: scores stay in VMEM; falls back to std if unsupported)
     <rules> example "col1=hxq,col2=hxk,row=thl,cls=hxq" (col1: Lq < Lk; col2: Lq > Lk; row: Lq == Lk; cls: Lq <= 16)
  TABICL_TPU_MHA_MAX_HEAD_DIM (default 32): only heads up to this size use these layouts (ICL heads are 64 wide).
  TABICL_TPU_MHA_PROJ=packed|split (default packed): "split" computes q, k, v with three linears on the weight
     slices instead of torch's packed in-projection, whose [.., L, 3, E] -> [3, .., L, E] transpose is a full copy
     (XProf: as_strided f32[4096,108,3,128,1] ~0.68 s per inference forward in the row stage). Same dot products.
Falls back to the upstream function for: KV-cache use/build (cached_kv / need_kv), bool masks, dropout in
training, keys >= TABICL_TPU_SPLASH_MIN_KV (Splash / tiled online path), or a registered flash fn.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

_MODE = os.environ.get("TABICL_TPU_MHA", "").strip().lower()
_MAX_HEAD_DIM = int(os.environ.get("TABICL_TPU_MHA_MAX_HEAD_DIM", "32") or 32)
_ANY_DEVICE = os.environ.get("TABICL_TPU_MHA_ANY_DEVICE", "") == "1"  # CPU equivalence tests only
_PROJ = os.environ.get("TABICL_TPU_MHA_PROJ", "packed").strip().lower()
_PK = None  # tabicl_tpu_pallas_attn, imported by install() (never inside a traced region)
# RoPE rotate-half in merged layout: "roll" (lane rolls + selects; default) or "mm" (exact signed-permutation matmul:
# fp32 inputs are split into three bf16 parts, so every product and the sum are exact; lane-dense, no unaligned slices).
_ROPE = os.environ.get("TABICL_TPU_MHA_ROPE", "roll").strip().lower()
# Inference column ISAB first attention (inducing points over the train rows, Splash path otherwise) through the
# query-expansion flash kernel (tabicl_tpu_pallas_attn.hxq_attention) with K/V read from the packed projection.
_HXQ = os.environ.get("TABICL_TPU_MHA_HXQ", "") == "1"
# Packed-projection kernels (tabicl_tpu_pallas_attn.pk_self_attention / pk_cross_attention): one in-projection matmul,
# q/k/v read as lane blocks, RoPE + scale in VMEM, packed gradients. Applies where the "pk" kernel would be used.
_PACKED = os.environ.get("TABICL_TPU_MHA_PACKED", "") == "1"
# Inference column ISAB first attention, folded (tabicl_tpu_pallas_attn.hxf_attention): the key projection is folded into
# the (shared) inducing-point queries and the value projection is applied after the attention, so the [.., Lk, 2E] K/V
# projection of the train rows is never materialized. Takes precedence over TABICL_TPU_MHA_HXQ where both apply.
_HXF = os.environ.get("TABICL_TPU_MHA_HXF", "") == "1"


def folded_mab1(query: Tensor, key: Tensor, valid_kv_len: Tensor, num_heads: int, in_proj_weight: Tensor,
                in_proj_bias: Optional[Tensor], ssmax_layer=None, src_len_override=None, pk=None) -> Optional[Tensor]:  # noqa: ANN001
    """softmax(q_g K_g^T) V_g for all heads g, merged [..., Lq, E] (before the output projection), with
    K = key W_k^T + b_k, V = key W_v^T + b_v, computed as U = softmax(Qf key^T) key, O_g = U_g W_v,g^T + b_v,g where
    Qf_g = q_g W_k,g (q pre-scaled; the q_g . b_k,g term is constant per query row). None when not applicable."""
    pk = pk if pk is not None else _PK
    E = query.shape[-1]
    hd = E // num_heads
    qsrc = query
    while qsrc.dim() > 2 and qsrc.stride(0) == 0:  # inducing points are expanded over the batch: fold them once
        qsrc = qsrc[0]
    Lq = qsrc.shape[-2]
    nb_q = 1
    for s_ in qsrc.shape[:-2]:
        nb_q *= s_
    nb_k = 1
    for s_ in key.shape[:-2]:
        nb_k *= s_
    if pk is None or nb_q not in (1, nb_k) or not pk.hxf_supported(num_heads * Lq, key.shape[-2], E):
        return None
    w_q, w_k, w_v = in_proj_weight.split([E, E, E])
    b_q, _b_k, b_v = (None, None, None) if in_proj_bias is None else in_proj_bias.split([E, E, E])
    q = F.linear(qsrc, w_q, b_q)
    if ssmax_layer is not None:
        n = src_len_override if src_len_override is not None else valid_kv_len
        qh = q.unflatten(-1, (num_heads, hd)).transpose(-3, -2)
        q = ssmax_layer(qh.reshape(-1, *qh.shape[-3:]), n).view(qh.shape).transpose(-3, -2).flatten(-2, -1)
    q = q * (hd ** -0.5)
    qf = torch.einsum("...qhd,hde->...hqe", q.unflatten(-1, (num_heads, hd)), w_k.view(num_heads, hd, E))
    u = pk.hxf_attention(qf.flatten(-3, -2), key, valid_kv_len)  # [*key_lead, h*Lq, E]
    o = torch.einsum("...hqe,hde->...qhd", u.unflatten(-2, (num_heads, Lq)), w_v.view(num_heads, hd, E))
    o = o.flatten(-2, -1)
    return o + b_v if b_v is not None else o


def in_projection(query: Tensor, key: Tensor, value: Tensor, w: Tensor, b: Optional[Tensor], proj: str):
    if proj == "views":  # one matmul per distinct input, q/k/v as 128-lane-aligned views (no transposed copy)
        E = query.shape[-1]
        if query is key and key is value:
            return F.linear(query, w, b).chunk(3, dim=-1)
        if key is value:
            w_q, w_kv = w.split([E, 2 * E])
            b_q, b_kv = (None, None) if b is None else b.split([E, 2 * E])
            k_p, v_p = F.linear(key, w_kv, b_kv).chunk(2, dim=-1)
            return F.linear(query, w_q, b_q), k_p, v_p
        proj = "split"
    if proj != "split":
        return F._in_projection_packed(query, key, value, w, b)
    E = query.shape[-1]
    w_q, w_k, w_v = w.split(E, dim=0)
    b_q, b_k, b_v = (None, None, None) if b is None else b.split(E)
    return F.linear(query, w_q, b_q), F.linear(key, w_k, b_k), F.linear(value, w_v, b_v)


def _head_mask(h: int, E: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    d = E // h
    col_head = torch.arange(E, device=device) // d
    return (col_head.unsqueeze(0) == torch.arange(h, device=device).unsqueeze(1)).to(dtype)  # [h, E]


def _parse_rules(mode: str) -> dict:
    rules = {"col1": "hxq", "col2": "hxk", "row": "thl", "cls": "hxq"}
    for part in mode.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            rules[k.strip()] = v.strip()
    return rules


def pick_impl(mode: str, Lq: int, Lk: int) -> str:
    if mode == "pk":  # fused kernel for the column / row attention; the 4 CLS-query rows stay on the XLA path
        return "std" if Lq <= 16 else "pk"
    if mode in ("hxq", "hxk", "thl", "std"):
        return mode
    if mode == "hx":
        return "hxq" if Lq <= Lk else "hxk"
    rules = _parse_rules(mode if "=" in mode else "")
    if Lq <= 16:
        return rules["cls"]
    if Lq == Lk:
        return rules["row"]
    return rules["col1"] if Lq < Lk else rules["col2"]


def small_head_attention(q: Tensor, k: Tensor, v: Tensor, h: int, mask: Optional[Tensor], impl: str) -> Tensor:
    """q [..., Lq, E] (already scaled), k / v [..., Lk, E]; mask additive, broadcastable to [..., h|1, Lq|1, Lk]."""
    Lq, E = q.shape[-2], q.shape[-1]
    Lk = k.shape[-2]
    d = E // h
    if impl == "hxq":
        M = _head_mask(h, E, q.dtype, q.device)
        q_exp = (q.unsqueeze(-3) * M.unsqueeze(-2)).flatten(-3, -2)  # [..., h*Lq, E]
        s = torch.matmul(q_exp, k.transpose(-2, -1)).float().unflatten(-2, (h, Lq))  # [..., h, Lq, Lk]
        if mask is not None:
            s = s + mask
        p = torch.softmax(s, dim=-1).to(v.dtype).flatten(-3, -2)  # [..., h*Lq, Lk]
        o_full = torch.matmul(p, v).unflatten(-2, (h, Lq))  # [..., h, Lq, E]
        return (o_full * M.to(o_full.dtype).unsqueeze(-2)).sum(dim=-3)
    if impl == "hxk":
        Mk = _head_mask(h, E, k.dtype, k.device)
        k_exp = (k.unsqueeze(-3) * Mk.unsqueeze(-2)).flatten(-3, -2)  # [..., h*Lk, E]
        v_exp = (v.unsqueeze(-3) * Mk.to(v.dtype).unsqueeze(-2)).flatten(-3, -2)
        s = torch.matmul(q, k_exp.transpose(-2, -1)).float().unflatten(-1, (h, Lk))  # [..., Lq, h, Lk]
        if mask is not None:
            s = s + mask.transpose(-3, -2)
        p = torch.softmax(s, dim=-1).to(v.dtype).flatten(-2, -1)  # [..., Lq, h*Lk]
        return torch.matmul(p, v_exp)
    if impl == "thl":
        qt = q.transpose(-2, -1).unflatten(-2, (h, d))  # [..., h, d, Lq]
        kt = k.transpose(-2, -1).unflatten(-2, (h, d))
        vt = v.transpose(-2, -1).unflatten(-2, (h, d))
        s = torch.matmul(qt.transpose(-2, -1), kt).float()  # [..., h, Lq, Lk]
        if mask is not None:
            s = s + mask
        p = torch.softmax(s, dim=-1).to(v.dtype)
        ot = torch.matmul(vt, p.transpose(-2, -1))  # [..., h, d, Lq]
        return ot.flatten(-3, -2).transpose(-2, -1)
    if impl == "pk":  # fused Pallas kernel (tabicl_tpu_pallas_attn.py); key-only masks
        _pk = _PK
        if _pk is None:
            return small_head_attention(q, k, v, h, mask, "std")
        kb = None
        if mask is not None:
            kb = mask.reshape(*mask.shape[:-3], mask.shape[-1]) if mask.shape[-3] == 1 and mask.shape[-2] == 1 else None
            if kb is None:
                return small_head_attention(q, k, v, h, mask, "std")
        if not _pk.supported(Lq, Lk, E, h):
            return small_head_attention(q, k, v, h, mask, "std")
        return _pk.hx_attention(q, k, v, h, kb)
    if impl == "std":  # upstream head-split layout with explicit matmul (reference)
        qh = q.unflatten(-1, (h, d)).transpose(-3, -2)
        kh = k.unflatten(-1, (h, d)).transpose(-3, -2)
        vh = v.unflatten(-1, (h, d)).transpose(-3, -2)
        s = torch.matmul(qh, kh.transpose(-2, -1)).float()
        if mask is not None:
            s = s + mask
        p = torch.softmax(s, dim=-1).to(vh.dtype)
        return torch.matmul(p, vh).transpose(-3, -2).flatten(-2, -1)
    raise ValueError(f"unknown small-head attention impl {impl!r}")


def rope_merged(rope, x: Tensor, h: int) -> Tensor:  # noqa: ANN001
    """RotaryEmbedding.rotate_queries_or_keys for merged-head x [..., L, E] (same rotation per head, no head split)."""
    L, E = x.shape[-2], x.shape[-1]
    d = E // h
    seq = rope.get_seq_pos(L, device=x.device, dtype=torch.float32)
    if rope.interleaved:
        freqs = rope.forward(seq, seq_len=L).float()  # [L, rot]
    else:
        f = seq.float().unsqueeze(-1) * rope.freqs.float()
        freqs = torch.cat([f, f], dim=-1)
    rot = freqs.shape[-1]
    cos, sin = freqs.cos(), freqs.sin()
    if rot < d:
        cos = torch.cat([cos, torch.ones(L, d - rot, device=x.device)], dim=-1)
        sin = torch.cat([sin, torch.zeros(L, d - rot, device=x.device)], dim=-1)
    cos_f, sin_f = cos.repeat(1, h), sin.repeat(1, h)  # [L, E]
    xf = x.float()
    j = torch.arange(E, device=x.device)
    if _ROPE == "mm":
        # R[i, j] = +-1 maps x[i] to rotate_half(x)[j] within each head; rot_x = x @ R exactly.
        jd = j % d
        if rope.interleaved:
            src = torch.where(jd % 2 == 0, j + 1, j - 1)
            sgn = torch.where(jd % 2 == 0, -1.0, 1.0)
        else:
            half = rot // 2
            src = torch.where(jd < half, j + half, j - half)
            sgn = torch.where(jd < half, -1.0, 1.0)
        valid = (jd < rot).float()
        R = (torch.arange(E, device=x.device).unsqueeze(1) == src.unsqueeze(0)).float() * (sgn * valid).unsqueeze(0)
        Rb = R.to(torch.bfloat16)
        if x.dtype == torch.bfloat16:
            rot_x = torch.matmul(x, Rb).float()
        else:
            x1 = xf.to(torch.bfloat16)
            r1 = xf - x1.float()
            x2 = r1.to(torch.bfloat16)
            x3 = (r1 - x2.float()).to(torch.bfloat16)
            rot_x = torch.matmul(x1, Rb).float() + torch.matmul(x2, Rb).float() + torch.matmul(x3, Rb).float()
        return (xf * cos_f + rot_x * sin_f).to(x.dtype)
    if rope.interleaved:  # out[2i] = -x[2i+1], out[2i+1] = x[2i] (pairs never cross a head boundary)
        rot_x = torch.where((j % 2) == 0, -torch.roll(xf, -1, dims=-1), torch.roll(xf, 1, dims=-1))
    else:
        half = rot // 2
        jd = j % d
        rot_x = torch.where(jd < half, -torch.roll(xf, -half, dims=-1),
                            torch.where(jd < rot, torch.roll(xf, half, dims=-1), torch.zeros_like(xf)))
    return (xf * cos_f + rot_x * sin_f).to(x.dtype)


def _combined_mask(attn_mask, key_padding_mask, batch_shape, nh: int, Lq: int, Lk: int, qdtype):  # noqa: ANN001
    """Upstream mask processing (attention.py) -> additive mask broadcastable to (*batch, nh, Lq, Lk); None if absent.
    Returns False for unsupported forms (caller falls back)."""
    nb = len(batch_shape)
    correct_nd = (*batch_shape, nh, Lq, Lk)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            return False
        if attn_mask.dim() == 2:
            if tuple(attn_mask.shape) != (Lq, Lk):
                return False
            attn_mask = attn_mask.view(*([1] * nb), 1, Lq, Lk)
        elif attn_mask.dim() == 4 and nb != 1 and tuple(attn_mask.shape[:3]) == (1, 1, 1) and attn_mask.shape[-1] == Lk:
            attn_mask = attn_mask.view(*([1] * nb), 1, 1, Lk)
        elif attn_mask.dim() == len(correct_nd):
            for g, e in zip(attn_mask.shape, correct_nd):
                if g not in (1, e):
                    return False
        else:
            return False
    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.bool:
            return False
        if tuple(key_padding_mask.shape) == (1, Lk) or (key_padding_mask.dim() == nb + 1
                                                        and key_padding_mask.shape[-1] == Lk):
            for g, e in zip(key_padding_mask.shape[:-1], batch_shape):
                if g not in (1, e):
                    return False
            if key_padding_mask.dim() == 2 and nb != 1:
                kpm = key_padding_mask.view(*([1] * nb), 1, 1, Lk)
            else:
                kpm = key_padding_mask.view(*key_padding_mask.shape[:-1], 1, 1, Lk)
        else:
            return False
        attn_mask = kpm if attn_mask is None else attn_mask + kpm
    if attn_mask is not None:
        if attn_mask.dtype != qdtype:
            attn_mask = attn_mask.to(dtype=qdtype)
        attn_mask = attn_mask.float()
    return attn_mask


def install(att_module, layers_module=None, mode: Optional[str] = None, proj: Optional[str] = None) -> bool:  # noqa: ANN001
    mode = (_MODE if mode is None else mode).strip().lower()
    proj = (_PROJ if proj is None else proj).strip().lower()
    if not mode or mode in ("0", "off", "none"):
        return False
    if getattr(att_module, "_v5p_mha_installed", False):
        return True
    if "pk" in mode or _HXQ or _PACKED or _HXF:  # register the Pallas ops now (eagerly), never while torch.compile is tracing
        global _PK
        import tabicl_tpu_pallas_attn as _pk_mod

        _PK = _pk_mod
    orig = att_module.multi_head_attention_forward

    def multi_head_attention_forward(query, num_heads, in_proj_weight, in_proj_bias, dropout_p,  # noqa: ANN001
                                     out_proj_weight, out_proj_bias, key=None, value=None, cached_kv=None,
                                     training=True, key_padding_mask=None, attn_mask=None, rope=None,
                                     ssmax_layer=None, need_kv=False, src_len_override=None, valid_kv_len=None):
        def _orig():
            return orig(query, num_heads, in_proj_weight, in_proj_bias, dropout_p, out_proj_weight, out_proj_bias,
                        key=key, value=value, cached_kv=cached_kv, training=training,
                        key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=rope, ssmax_layer=ssmax_layer,
                        need_kv=need_kv, src_len_override=src_len_override, valid_kv_len=valid_kv_len)

        E = query.shape[-1]
        hd = E // num_heads
        if (_HXF and _PK is not None and isinstance(valid_kv_len, Tensor) and valid_kv_len.numel() == 1
                and not torch.is_grad_enabled() and cached_kv is None and not need_kv and key is not None
                and key is value and rope is None and attn_mask is None and key_padding_mask is None
                and hd <= _MAX_HEAD_DIM and query.device.type == "tpu"
                and (not isinstance(src_len_override, Tensor) or src_len_override.numel() == 1)):
            o = folded_mab1(query, key, valid_kv_len, num_heads, in_proj_weight, in_proj_bias, ssmax_layer,
                            src_len_override)
            if o is not None:
                return F.linear(o, out_proj_weight, out_proj_bias)
        if (_HXQ and _PK is not None and isinstance(valid_kv_len, Tensor) and not torch.is_grad_enabled()
                and cached_kv is None and not need_kv and key is not None and key is value and rope is None
                and attn_mask is None and key_padding_mask is None and hd <= _MAX_HEAD_DIM
                and query.device.type == "tpu"
                and _PK.hxq_supported(query.shape[-2], key.shape[-2], E, num_heads)):
            w_q, w_kv = in_proj_weight.split([E, 2 * E])
            b_q, b_kv = (None, None) if in_proj_bias is None else in_proj_bias.split([E, 2 * E])
            q = F.linear(query, w_q, b_q)
            kv = F.linear(key, w_kv, b_kv)  # [.., Lk, 2E] packed: K lanes [0, E), V lanes [E, 2E)
            if ssmax_layer is not None:
                n = src_len_override if src_len_override is not None else valid_kv_len
                qh = q.unflatten(-1, (num_heads, hd)).transpose(-3, -2)
                q = ssmax_layer(qh.reshape(-1, *qh.shape[-3:]), n).view(qh.shape).transpose(-3, -2).flatten(-2, -1)
            out = _PK.hxq_attention(q * (hd ** -0.5), kv, valid_kv_len, num_heads)
            return F.linear(out, out_proj_weight, out_proj_bias)
        if ((query.device.type != "tpu" and not _ANY_DEVICE) or cached_kv is not None or need_kv or key is None or value is None
                or hd > _MAX_HEAD_DIM or (training and dropout_p > 0.0) or valid_kv_len is not None
                or getattr(att_module, "_tpu_flash_attention_fn", None) is not None
                or key.shape[-2] >= att_module._tpu_splash_min_kv()):
            return _orig()
        *batch_shape, Lq, _ = query.shape
        Lk = key.shape[-2]
        if _PACKED and _PK is not None and pick_impl(mode, Lq, Lk) == "pk" and (key is value):
            self_attn = query is key
            if _PK.packed_supported(Lq, Lk, E, num_heads, self_attn):
                mask = _combined_mask(attn_mask, key_padding_mask, batch_shape, num_heads, Lq, Lk, query.dtype)
                kb = None
                if mask is not None and mask is not False:
                    kb = mask.reshape(*mask.shape[:-3], mask.shape[-1]) if mask.shape[-3] == 1 and mask.shape[-2] == 1 else False
                if mask is not False and kb is not False:
                    if self_attn:
                        qkv = F.linear(query, in_proj_weight, in_proj_bias)
                        out = _PK.pk_self_attention(qkv, num_heads, kb, rope, hd ** -0.5)
                    else:
                        w_q, w_kv = in_proj_weight.split([E, 2 * E])
                        b_q, b_kv = (None, None) if in_proj_bias is None else in_proj_bias.split([E, 2 * E])
                        qp = F.linear(query, w_q, b_q)
                        kv = F.linear(key, w_kv, b_kv)
                        if ssmax_layer is not None:
                            n = src_len_override if src_len_override is not None else Lk
                            qh = qp.unflatten(-1, (num_heads, hd)).transpose(-3, -2)
                            qp = ssmax_layer(qh.reshape(-1, *qh.shape[-3:]), n).view(qh.shape).transpose(-3, -2).flatten(-2, -1)
                        out = _PK.pk_cross_attention(qp, kv, num_heads, kb, rope, hd ** -0.5)
                    return F.linear(out, out_proj_weight, out_proj_bias)
        q, k, v = in_projection(query, key, value, in_proj_weight, in_proj_bias, proj)
        mask = _combined_mask(attn_mask, key_padding_mask, batch_shape, num_heads, Lq, Lk, q.dtype)
        if mask is False:
            return _orig()
        if rope is not None:
            q = rope_merged(rope, q, num_heads)
            k = rope_merged(rope, k, num_heads)
        if ssmax_layer is not None:  # only the first ISAB attention (q = inducing points: small); head-split layout
            n = src_len_override if src_len_override is not None else Lk
            qh = q.unflatten(-1, (num_heads, hd)).transpose(-3, -2)
            q = ssmax_layer(qh.reshape(-1, *qh.shape[-3:]), n).view(qh.shape).transpose(-3, -2).flatten(-2, -1)
        impl = pick_impl(mode, Lq, Lk)
        out = small_head_attention(q * (hd ** -0.5), k, v, num_heads, mask, impl)
        return F.linear(out, out_proj_weight, out_proj_bias)

    att_module.multi_head_attention_forward = multi_head_attention_forward
    if layers_module is not None:
        layers_module.multi_head_attention_forward = multi_head_attention_forward
    att_module._v5p_mha_installed = True
    att_module._v5p_mha_mode = mode
    att_module._v5p_mha_proj = proj
    return True

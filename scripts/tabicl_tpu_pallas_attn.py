"""Fused small-head attention for TPU v5p as a Pallas kernel (attention scores never leave VMEM).

TabICLv2's column / row stages use 8 heads x 16 dims. With XLA ops the per-head score / probability tensors are
materialized in HBM (XProf of a one-chip v6 train step: ~110 GB of 330 GB accessed per step, the step is memory bound
at ~1.2 TB/s per TensorCore). This kernel keeps them in VMEM. It uses "key head expansion" so that every matmul is
128 lanes wide instead of 16:

  K_exp[g*Lkp + j, :] = K[j, :] * M[g]     (M[g] = 0/1 mask of head g's 16 dims; built in VMEM, Lkp = keys padded)
  S = Q @ K_exp^T          -> [bq, h*Lkp]   (block g = head g's scores, 128-lane aligned)
  P = softmax per head block (fp32, + additive key bias; padded keys get a large negative bias)
  O = P @ V_exp            -> [bq, E]       (merged-head output, no transposes)

Backward recomputes S / P in VMEM (keys fit in VMEM, so no log-sum-exp is stored) and accumulates dK_exp / dV_exp over
the query blocks of each batch element:
  dP = dO @ V_exp^T;  D_g = sum_{c in head g} dO*O;  dS = P * (dP - D_g);  dQ = dS @ K_exp
  dK_exp += dS^T @ Q;  dV_exp += P^T @ dO;  dK = sum_g dK_exp[g] * M[g];  dV likewise.
Same math as softmax(Q_h K_h^T + bias) V_h per head (q is passed pre-scaled by 1/sqrt(d)); 8x the minimal matmul FLOPs,
which the MXU absorbs (a 16-deep contraction would use 1/8 of it anyway).

torch entry point: hx_attention(q, k, v, num_heads, key_bias=None) with q [..., Lq, E], k/v [..., Lk, E] and an optional
additive key bias broadcastable to [..., Lk]; autograd supported (torch_tpu jax_op custom ops).
"""
# (no `from __future__ import annotations`: torch_tpu jax_op evaluates the kernel signatures in their own scope)
import functools
import os
from typing import Optional

import torch
from torch import Tensor

_NEG = -1e30
_VMEM_LIMIT = int(os.environ.get("TABICL_TPU_PK_VMEM_MB", "64") or 64) * 1024 * 1024
_MAX_LKP = int(os.environ.get("TABICL_TPU_PK_MAX_LK", "1024") or 1024)


def _pick_bq(Lq: int, lkp: int) -> int:
    """Query block: divides Lq, multiple of 8 (or == Lq), and keeps a [bq, h*lkp] fp32 tile <= ~2 MiB per head set."""
    cap = max(8, min(512, 65536 // lkp))
    if Lq <= cap:
        return Lq
    for bq in range(cap - cap % 8, 7, -8):
        if Lq % bq == 0:
            return bq
    return 0


def _jax_impl(h: int, lk: int, lkp: int, bq: int, prec: int):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    precision = {0: None, 1: jax.lax.Precision.HIGH, 2: jax.lax.Precision.HIGHEST}[prec]
    nt = (((1,), (1,)), ((), ()))  # A @ B^T

    def _expand(x, col, d):  # x [lk, E] -> [h*lkp, E] (zero rows for padded keys)
        if lkp > lk:
            x = jnp.concatenate([x, jnp.zeros((lkp - lk, x.shape[1]), x.dtype)], axis=0)
        return jnp.concatenate([jnp.where(col == g, x, jnp.zeros_like(x)) for g in range(h)], axis=0)

    def _bias(b_ref):  # [1, lk] -> [1, lkp] fp32 with padded keys masked
        b = b_ref[...].astype(jnp.float32)
        if lkp > lk:
            b = jnp.concatenate([b, jnp.full((1, lkp - lk), _NEG, jnp.float32)], axis=1)
        return b

    def _probs(s, bias):  # s [bq, h*lkp] fp32 -> per-head softmax
        ps = []
        for g in range(h):
            sg = s[:, g * lkp:(g + 1) * lkp] + bias
            m = jnp.max(sg, axis=-1, keepdims=True)
            e = jnp.exp(sg - m)
            ps.append(e / jnp.sum(e, axis=-1, keepdims=True))
        return jnp.concatenate(ps, axis=-1)

    def fwd_kernel(q_ref, k_ref, v_ref, b_ref, o_ref):
        q = q_ref[...]
        E = q.shape[-1]
        col = jax.lax.broadcasted_iota(jnp.int32, (1, E), 1) // (E // h)
        kexp = _expand(k_ref[...], col, E // h)
        vexp = _expand(v_ref[...], col, E // h)
        s = jax.lax.dot_general(q, kexp, nt, precision=precision, preferred_element_type=jnp.float32)
        p = _probs(s, _bias(b_ref))
        o = jnp.dot(p.astype(vexp.dtype), vexp, precision=precision, preferred_element_type=jnp.float32)
        o_ref[...] = o.astype(o_ref.dtype)

    def bwd_kernel(q_ref, k_ref, v_ref, b_ref, o_ref, do_ref, dq_ref, dk_ref, dv_ref, dk_acc, dv_acc):
        i = pl.program_id(1)

        @pl.when(i == 0)
        def _init():
            dk_acc[...] = jnp.zeros_like(dk_acc)
            dv_acc[...] = jnp.zeros_like(dv_acc)

        q = q_ref[...]
        do = do_ref[...]
        E = q.shape[-1]
        col = jax.lax.broadcasted_iota(jnp.int32, (1, E), 1) // (E // h)
        kexp = _expand(k_ref[...], col, E // h)
        vexp = _expand(v_ref[...], col, E // h)
        s = jax.lax.dot_general(q, kexp, nt, precision=precision, preferred_element_type=jnp.float32)
        p = _probs(s, _bias(b_ref))
        dp = jax.lax.dot_general(do, vexp, nt, precision=precision, preferred_element_type=jnp.float32)
        od = do.astype(jnp.float32) * o_ref[...].astype(jnp.float32)
        dss = []
        for g in range(h):
            dg = jnp.sum(jnp.where(col == g, od, 0.0), axis=-1, keepdims=True)
            dss.append(p[:, g * lkp:(g + 1) * lkp] * (dp[:, g * lkp:(g + 1) * lkp] - dg))
        ds = jnp.concatenate(dss, axis=-1).astype(q.dtype)
        dq = jnp.dot(ds, kexp, precision=precision, preferred_element_type=jnp.float32)
        dq_ref[...] = dq.astype(dq_ref.dtype)
        dk_acc[...] += jnp.dot(ds.T, q, precision=precision, preferred_element_type=jnp.float32)
        dv_acc[...] += jnp.dot(p.astype(do.dtype).T, do, precision=precision, preferred_element_type=jnp.float32)

        @pl.when(i == pl.num_programs(1) - 1)
        def _fin():
            dke = dk_acc[...]
            dve = dv_acc[...]
            dk = sum(jnp.where(col == g, dke[g * lkp:(g + 1) * lkp], 0.0) for g in range(h))
            dv = sum(jnp.where(col == g, dve[g * lkp:(g + 1) * lkp], 0.0) for g in range(h))
            dk_ref[...] = dk[:lk].astype(dk_ref.dtype)
            dv_ref[...] = dv[:lk].astype(dv_ref.dtype)

    def _bias_spec(nb_bias):
        if nb_bias == 1:
            return pl.BlockSpec((None, 1, lk), lambda b, i: (0, 0, 0))
        return pl.BlockSpec((None, 1, lk), lambda b, i: (b, 0, 0))

    def fwd(q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array) -> jax.Array:
        B, Lq, E = q.shape
        return pl.pallas_call(
            fwd_kernel,
            grid=(B, Lq // bq),
            in_specs=[pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
                      pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, 0)),
                      pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, 0)),
                      _bias_spec(bias.shape[0])],
            out_specs=pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"),
                                                 vmem_limit_bytes=_VMEM_LIMIT),
        )(q, k, v, bias)

    def bwd(q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array, o: jax.Array,
            do: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        B, Lq, E = q.shape
        return pl.pallas_call(
            bwd_kernel,
            grid=(B, Lq // bq),
            in_specs=[pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
                      pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, 0)),
                      pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, 0)),
                      _bias_spec(bias.shape[0]),
                      pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
                      pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0))],
            out_specs=[pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
                       pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, 0)),
                       pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, 0))],
            out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct(k.shape, k.dtype),
                       jax.ShapeDtypeStruct(v.shape, v.dtype)],
            scratch_shapes=[pltpu.VMEM((h * lkp, E), jnp.float32), pltpu.VMEM((h * lkp, E), jnp.float32)],
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"),
                                                 vmem_limit_bytes=_VMEM_LIMIT),
        )(q, k, v, bias, o, do)

    return fwd, bwd


@functools.lru_cache(maxsize=None)
def _impl_cached(h: int, lk: int, lkp: int, bq: int, prec: int):
    return _jax_impl(h, lk, lkp, bq, prec)


# One generic op pair, registered at import time (never while torch.compile is tracing). The ints are static
# arguments of the JAX function (one compiled kernel per distinct shape / int combination).
import jax  # noqa: E402  (needed in module globals: jax_op evaluates the annotations below)


def _hx_fwd(q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array, h: int, lkp: int, bq: int,
            prec: int) -> jax.Array:
    fwd, _ = _impl_cached(h, k.shape[1], lkp, bq, prec)
    return fwd(q, k, v, bias)


def _hx_bwd(q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array, o: jax.Array, do: jax.Array, h: int,
            lkp: int, bq: int, prec: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    _, bwd = _impl_cached(h, k.shape[1], lkp, bq, prec)
    return bwd(q, k, v, bias, o, do)


def _register_ops():
    import torch_tpu._internal.pallas as tpu_pallas

    fwd_op = tpu_pallas.jax_op("tabicl_pk::hx_fwd", _hx_fwd)
    bwd_op = tpu_pallas.jax_op("tabicl_pk::hx_bwd", _hx_bwd)

    def setup_context(ctx, inputs, output):  # noqa: ANN001
        q, k, v, bias, h, lkp, bq, prec = inputs
        ctx.save_for_backward(q, k, v, bias, output)
        ctx.cfg = (h, lkp, bq, prec)

    def backward(ctx, do):  # noqa: ANN001
        q, k, v, bias, o = ctx.saved_tensors
        dq, dk, dv = bwd_op(q, k, v, bias, o, do.contiguous(), *ctx.cfg)
        return dq, dk, dv, None, None, None, None, None

    fwd_op.register_autograd(backward, setup_context=setup_context)
    return fwd_op, bwd_op


try:
    HX_FWD, HX_BWD = _register_ops()
except Exception as _exc:  # noqa: BLE001  (e.g. no torch_tpu on a CPU host)
    HX_FWD = HX_BWD = None
    _REGISTER_ERROR = _exc


# ---- Query head expansion with online softmax over key blocks (inference only): the column stage's first ISAB
# attention, 128 inducing-point queries over up to ~50k train rows (keys >= valid are masked). K and V are read
# straight from the packed [.., Lk, 2E] in-projection output (lane blocks 0 / 1): no head split, no transposed copy.
#   Q_exp[g*Lq + i, :] = Q[i, :] * M[g];  per key block: S = Q_exp @ K_blk^T, online softmax per row, acc += P @ V_blk;
#   O = sum_g (acc / l)[g*Lq:(g+1)*Lq] * M[g].
def _jax_hxq(h: int, Lq: int, E: int, bk: int, prec: int):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    precision = {0: None, 1: jax.lax.Precision.HIGH, 2: jax.lax.Precision.HIGHEST}[prec]
    nt = (((1,), (1,)), ((), ()))
    hq = h * Lq

    def kernel(valid_ref, q_ref, k_ref, v_ref, o_ref, qexp_scr, m_scr, l_scr, acc_scr):
        j = pl.program_id(1)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, E), 1) // (E // h)
        valid = valid_ref[0]

        @pl.when(j == 0)
        def _init():
            q = q_ref[...]
            qexp_scr[...] = jnp.concatenate([jnp.where(col == g, q, jnp.zeros_like(q)) for g in range(h)], axis=0)
            m_scr[...] = jnp.full(m_scr.shape, _NEG, jnp.float32)
            l_scr[...] = jnp.zeros(l_scr.shape, jnp.float32)
            acc_scr[...] = jnp.zeros(acc_scr.shape, jnp.float32)

        @pl.when(j * bk < valid)
        def _step():
            k = k_ref[...]
            v = v_ref[...]
            s = jax.lax.dot_general(qexp_scr[...], k, nt, precision=precision, preferred_element_type=jnp.float32)
            kidx = j * bk + jax.lax.broadcasted_iota(jnp.int32, (1, bk), 1)
            s = jnp.where(kidx < valid, s, _NEG)
            m_prev = m_scr[:, 0:1]
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
            alpha = jnp.exp(m_prev - m_new)
            pr = jnp.exp(s - m_new)
            l_new = alpha * l_scr[:, 0:1] + jnp.sum(pr, axis=-1, keepdims=True)
            acc_scr[...] = alpha * acc_scr[...] + jnp.dot(pr.astype(v.dtype), v, precision=precision,
                                                          preferred_element_type=jnp.float32)
            m_scr[...] = jnp.broadcast_to(m_new, m_scr.shape)
            l_scr[...] = jnp.broadcast_to(l_new, l_scr.shape)

        @pl.when(j == pl.num_programs(1) - 1)
        def _fin():
            o_full = acc_scr[...] / l_scr[:, 0:1]
            o = sum(jnp.where(col == g, o_full[g * Lq:(g + 1) * Lq], 0.0) for g in range(h))
            o_ref[...] = o.astype(o_ref.dtype)

    def fwd(valid, q, kv):
        B, Lk, _ = kv.shape

        def kv_map(blk):
            # past the valid keys, keep pointing at the last valid block (no new DMA for skipped steps)
            return lambda b, j, valid_ref: (b, jnp.minimum(j, jnp.maximum(valid_ref[0] - 1, 0) // bk), blk)

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, Lk // bk),
            in_specs=[pl.BlockSpec((None, Lq, E), lambda b, j, valid_ref: (b, 0, 0)),
                      pl.BlockSpec((None, bk, E), kv_map(0)),
                      pl.BlockSpec((None, bk, E), kv_map(1))],
            out_specs=pl.BlockSpec((None, Lq, E), lambda b, j, valid_ref: (b, 0, 0)),
            scratch_shapes=[pltpu.VMEM((hq, E), q.dtype), pltpu.VMEM((hq, 128), jnp.float32),
                            pltpu.VMEM((hq, 128), jnp.float32), pltpu.VMEM((hq, E), jnp.float32)],
        )
        return pl.pallas_call(
            kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"),
                                                 vmem_limit_bytes=_VMEM_LIMIT),
        )(valid, q, kv, kv)

    return fwd


@functools.lru_cache(maxsize=None)
def _hxq_cached(h: int, Lq: int, E: int, bk: int, prec: int):
    return _jax_hxq(h, Lq, E, bk, prec)


def _hxq_fwd(valid: jax.Array, q: jax.Array, kv: jax.Array, h: int, bk: int, prec: int) -> jax.Array:
    return _hxq_cached(h, q.shape[1], q.shape[2], bk, prec)(valid, q, kv)


try:
    import torch_tpu._internal.pallas as _tpu_pallas_mod

    HXQ_FWD = _tpu_pallas_mod.jax_op("tabicl_pk::hxq_fwd", _hxq_fwd)
except Exception as _exc2:  # noqa: BLE001
    HXQ_FWD = None
    _REGISTER_ERROR_HXQ = _exc2


def _pick_bk(Lk: int) -> int:
    for bk in (512, 1024, 256, 128):
        if Lk % bk == 0:
            return bk
    return 0


def hxq_supported(Lq: int, Lk: int, E: int, h: int) -> bool:
    return HXQ_FWD is not None and E % 128 == 0 and Lq % 8 == 0 and Lq <= 256 and _pick_bk(Lk) > 0


def hxq_attention(q: Tensor, kv: Tensor, valid_len: Tensor, h: int) -> Tensor:
    """Inference only. q [..., Lq, E] (pre-scaled), kv [..., Lk, 2E] packed (K = lanes [0, E), V = [E, 2E));
    keys at positions >= valid_len are ignored."""
    lead = q.shape[:-2]
    Lq, E = q.shape[-2], q.shape[-1]
    Lk = kv.shape[-2]
    q3 = q.reshape(-1, Lq, E).contiguous()
    kv3 = kv.reshape(-1, Lk, 2 * E).contiguous()
    if q3.shape[0] != kv3.shape[0]:
        q3 = q3.expand(kv3.shape[0], Lq, E).contiguous()
    valid = valid_len.to(device=q.device, dtype=torch.int32).reshape(1)
    prec = _fp32_prec_code() if q.dtype == torch.float32 else 0
    out = HXQ_FWD(valid, q3, kv3, h, _pick_bk(Lk), prec)
    return out.reshape(*lead, Lq, E) if len(lead) else out.reshape(Lq, E)


# ---- Folded first ISAB attention (inference only, TABICL_TPU_MHA_HXF=1). With K = x W_k^T + b_k and V = x W_v^T + b_v:
#   S_g = q_g K_g^T = (q_g W_k,g) x^T + (q_g . b_k,g)      (the last term is constant per query row: softmax-invariant)
#   O_g = P_g V_g   = (P_g x) W_v,g^T + b_v,g              (rows of P sum to 1)
# so with folded queries Qf = [q_g W_k,g]_g ([h*Lq, E], dense) the kernel is a plain flash attention with d = E over the
# raw inputs: U = softmax(Qf x^T) x ([h*Lq, E]); the per-head value projection is applied to U afterwards (tiny). The
# [.., Lk, 2E] K/V projection is never materialized and every matmul is a full 128-deep contraction.
def _jax_hxf(hq: int, E: int, bk: int, prec: int, shared_q: bool):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    precision = {0: None, 1: jax.lax.Precision.HIGH, 2: jax.lax.Precision.HIGHEST}[prec]
    nt = (((1,), (1,)), ((), ()))

    def kernel(valid_ref, q_ref, x_ref, o_ref, m_scr, l_scr, acc_scr):
        j = pl.program_id(1)
        valid = valid_ref[0]

        @pl.when(j == 0)
        def _init():
            m_scr[...] = jnp.full(m_scr.shape, _NEG, jnp.float32)
            l_scr[...] = jnp.zeros(l_scr.shape, jnp.float32)
            acc_scr[...] = jnp.zeros(acc_scr.shape, jnp.float32)

        @pl.when(j * bk < valid)
        def _step():
            x = x_ref[...]
            s = jax.lax.dot_general(q_ref[...], x, nt, precision=precision, preferred_element_type=jnp.float32)
            kidx = j * bk + jax.lax.broadcasted_iota(jnp.int32, (1, bk), 1)
            s = jnp.where(kidx < valid, s, _NEG)
            m_prev = m_scr[:, 0:1]
            m_new = jnp.maximum(m_prev, jnp.max(s, axis=-1, keepdims=True))
            alpha = jnp.exp(m_prev - m_new)
            pr = jnp.exp(s - m_new)
            l_new = alpha * l_scr[:, 0:1] + jnp.sum(pr, axis=-1, keepdims=True)
            acc_scr[...] = alpha * acc_scr[...] + jnp.dot(pr.astype(x.dtype), x, precision=precision,
                                                          preferred_element_type=jnp.float32)
            m_scr[...] = jnp.broadcast_to(m_new, m_scr.shape)
            l_scr[...] = jnp.broadcast_to(l_new, l_scr.shape)

        @pl.when(j == pl.num_programs(1) - 1)
        def _fin():
            o_ref[...] = (acc_scr[...] / l_scr[:, 0:1]).astype(o_ref.dtype)

    def fwd(valid, q, x):
        B, Lk, _ = x.shape

        def x_map(b, j, valid_ref):  # past the valid keys, keep pointing at the last valid block (no new DMA)
            return (b, jnp.minimum(j, jnp.maximum(valid_ref[0] - 1, 0) // bk), 0)

        q_map = (lambda b, j, valid_ref: (0, 0, 0)) if shared_q else (lambda b, j, valid_ref: (b, 0, 0))
        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(B, Lk // bk),
            in_specs=[pl.BlockSpec((None, hq, E), q_map), pl.BlockSpec((None, bk, E), x_map)],
            out_specs=pl.BlockSpec((None, hq, E), lambda b, j, valid_ref: (b, 0, 0)),
            scratch_shapes=[pltpu.VMEM((hq, 128), jnp.float32), pltpu.VMEM((hq, 128), jnp.float32),
                            pltpu.VMEM((hq, E), jnp.float32)],
        )
        return pl.pallas_call(
            kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((B, hq, E), x.dtype),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"),
                                                 vmem_limit_bytes=_VMEM_LIMIT),
        )(valid, q, x)

    return fwd


@functools.lru_cache(maxsize=None)
def _hxf_cached(hq: int, E: int, bk: int, prec: int, shared_q: bool):
    return _jax_hxf(hq, E, bk, prec, shared_q)


def _hxf_fwd(valid: jax.Array, q: jax.Array, x: jax.Array, bk: int, prec: int) -> jax.Array:
    return _hxf_cached(q.shape[1], q.shape[2], bk, prec, q.shape[0] == 1)(valid, q, x)


try:
    import torch_tpu._internal.pallas as _tpu_pallas_mod2

    HXF_FWD = _tpu_pallas_mod2.jax_op("tabicl_pk::hxf_fwd", _hxf_fwd)
except Exception as _exc4:  # noqa: BLE001
    HXF_FWD = None
    _REGISTER_ERROR_HXF = _exc4


def hxf_supported(hq: int, Lk: int, E: int) -> bool:
    return HXF_FWD is not None and E % 128 == 0 and hq % 8 == 0 and hq <= 2048 and _pick_bk(Lk) > 0


def hxf_attention(qf: Tensor, x: Tensor, valid_len: Tensor) -> Tensor:
    """Inference only. qf [hq, E] (shared by every batch element) or [..., hq, E]: folded, pre-scaled queries;
    x [..., Lk, E]: raw key/value input (positions >= valid_len ignored). Returns softmax(qf x^T) x: [..., hq, E]."""
    lead = x.shape[:-2]
    Lk, E = x.shape[-2], x.shape[-1]
    hq = qf.shape[-2]
    x3 = x.reshape(-1, Lk, E).contiguous()
    q3 = qf.reshape(-1, hq, E).contiguous()
    if q3.shape[0] not in (1, x3.shape[0]):
        q3 = q3.expand(x3.shape[0], hq, E).contiguous()
    valid = valid_len.to(device=x.device, dtype=torch.int32).reshape(1)
    prec = _fp32_prec_code() if x.dtype == torch.float32 else 0
    out = HXF_FWD(valid, q3, x3, _pick_bk(Lk), prec)
    return out.reshape(*lead, hq, E)


# ---- Packed-projection kernels with in-kernel RoPE (TABICL_TPU_MHA_PACKED=1). q / k / v are read as 128-lane blocks of
# the in-projection output (self-attention: qkv [B, L, 3E]; cross: q [B, Lq, E] + kv [B, Lk, 2E]), so the packed
# projection is never transposed or split in HBM, and RoPE (rotate-half via lane rolls) + the 1/sqrt(d) scale are applied
# in VMEM. Backward writes d(qkv) / d(kv) packed. RoPE: out = x*cos + R(x)*sin with R a signed permutation per head
# (R^T = -R), so the backward is dx = dy*cos - R(dy*sin).
def _jax_packed(h: int, E: int, Lq: int, lk: int, lkp: int, bq: int, rope: int, half: int, rot: int, prec: int,
                self_attn: bool, scale: float):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    precision = {0: None, 1: jax.lax.Precision.HIGH, 2: jax.lax.Precision.HIGHEST}[prec]
    nt = (((1,), (1,)), ((), ()))
    d = E // h
    qb, kb, vb = (0, 1, 2) if self_attn else (0, 0, 1)

    def _iota():
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, E), 1)
        return lane // d, lane % d

    def _rot(x, jd):  # R(x): rotate-half within each head (pairs or halves), zero beyond the rotary dims
        if rope == 1:
            r = jnp.where(jd % 2 == 0, -pltpu.roll(x, E - 1, 1), pltpu.roll(x, 1, 1))
        else:
            r = jnp.where(jd < half, -pltpu.roll(x, E - half, 1), pltpu.roll(x, half, 1))
        return jnp.where(jd < rot, r, 0.0)

    def _rope_fwd(x, c, s_, jd):
        return x if rope == 0 else x * c + _rot(x, jd) * s_

    def _rope_bwd(y, c, s_, jd):
        return y if rope == 0 else y * c - _rot(y * s_, jd)

    def _expand(x, col):
        if lkp > lk:
            x = jnp.concatenate([x, jnp.zeros((lkp - lk, x.shape[1]), x.dtype)], axis=0)
        return jnp.concatenate([jnp.where(col == g, x, jnp.zeros_like(x)) for g in range(h)], axis=0)

    def _bias(b_ref):
        b = b_ref[...].astype(jnp.float32)
        if lkp > lk:
            b = jnp.concatenate([b, jnp.full((1, lkp - lk), _NEG, jnp.float32)], axis=1)
        return b

    def _probs(sc, bias):
        ps = []
        for g in range(h):
            sg = sc[:, g * lkp:(g + 1) * lkp] + bias
            m = jnp.max(sg, axis=-1, keepdims=True)
            e = jnp.exp(sg - m)
            ps.append(e / jnp.sum(e, axis=-1, keepdims=True))
        return jnp.concatenate(ps, axis=-1)

    def _prep(q_ref, k_ref, v_ref, cq_ref, sq_ref, ck_ref, sk_ref, col, jd):
        dt = q_ref.dtype
        q = (_rope_fwd(q_ref[...].astype(jnp.float32), cq_ref[...], sq_ref[...], jd) * scale).astype(dt)
        k = _rope_fwd(k_ref[...].astype(jnp.float32), ck_ref[...], sk_ref[...], jd).astype(dt)
        return q, _expand(k, col), _expand(v_ref[...], col)

    def fwd_kernel(q_ref, k_ref, v_ref, b_ref, cq_ref, sq_ref, ck_ref, sk_ref, o_ref):
        col, jd = _iota()
        q, kexp, vexp = _prep(q_ref, k_ref, v_ref, cq_ref, sq_ref, ck_ref, sk_ref, col, jd)
        sc = jax.lax.dot_general(q, kexp, nt, precision=precision, preferred_element_type=jnp.float32)
        pr = _probs(sc, _bias(b_ref))
        o = jnp.dot(pr.astype(vexp.dtype), vexp, precision=precision, preferred_element_type=jnp.float32)
        o_ref[...] = o.astype(o_ref.dtype)

    def _bwd_core(q_ref, k_ref, v_ref, b_ref, cq_ref, sq_ref, ck_ref, sk_ref, o_ref, do_ref, col, jd):
        q, kexp, vexp = _prep(q_ref, k_ref, v_ref, cq_ref, sq_ref, ck_ref, sk_ref, col, jd)
        do = do_ref[...]
        sc = jax.lax.dot_general(q, kexp, nt, precision=precision, preferred_element_type=jnp.float32)
        pr = _probs(sc, _bias(b_ref))
        dp = jax.lax.dot_general(do, vexp, nt, precision=precision, preferred_element_type=jnp.float32)
        od = do.astype(jnp.float32) * o_ref[...].astype(jnp.float32)
        dss = []
        for g in range(h):
            dg = jnp.sum(jnp.where(col == g, od, 0.0), axis=-1, keepdims=True)
            dss.append(pr[:, g * lkp:(g + 1) * lkp] * (dp[:, g * lkp:(g + 1) * lkp] - dg))
        ds = jnp.concatenate(dss, axis=-1).astype(q.dtype)
        dq_r = jnp.dot(ds, kexp, precision=precision, preferred_element_type=jnp.float32)
        dq = _rope_bwd(dq_r * scale, cq_ref[...], sq_ref[...], jd)
        dke = jnp.dot(ds.T, q, precision=precision, preferred_element_type=jnp.float32)
        dve = jnp.dot(pr.astype(do.dtype).T, do, precision=precision, preferred_element_type=jnp.float32)
        return dq, dke, dve

    def _collapse(xe, col):
        return sum(jnp.where(col == g, xe[g * lkp:(g + 1) * lkp], 0.0) for g in range(h))[:lk]

    def bwd_kernel_cross(q_ref, k_ref, v_ref, b_ref, cq_ref, sq_ref, ck_ref, sk_ref, o_ref, do_ref, dq_ref, dkv_ref,
                         dk_acc, dv_acc):
        i = pl.program_id(1)
        col, jd = _iota()

        @pl.when(i == 0)
        def _init():
            dk_acc[...] = jnp.zeros_like(dk_acc)
            dv_acc[...] = jnp.zeros_like(dv_acc)

        dq, dke, dve = _bwd_core(q_ref, k_ref, v_ref, b_ref, cq_ref, sq_ref, ck_ref, sk_ref, o_ref, do_ref, col, jd)
        dq_ref[...] = dq.astype(dq_ref.dtype)
        dk_acc[...] += dke
        dv_acc[...] += dve

        @pl.when(i == pl.num_programs(1) - 1)
        def _fin():
            dk = _rope_bwd(_collapse(dk_acc[...], col), ck_ref[...], sk_ref[...], jd)
            dkv_ref[:, 0:E] = dk.astype(dkv_ref.dtype)
            dkv_ref[:, E:2 * E] = _collapse(dv_acc[...], col).astype(dkv_ref.dtype)

    def bwd_kernel_self(q_ref, k_ref, v_ref, b_ref, cq_ref, sq_ref, ck_ref, sk_ref, o_ref, do_ref, dqkv_ref):
        col, jd = _iota()
        dq, dke, dve = _bwd_core(q_ref, k_ref, v_ref, b_ref, cq_ref, sq_ref, ck_ref, sk_ref, o_ref, do_ref, col, jd)
        dk = _rope_bwd(_collapse(dke, col), ck_ref[...], sk_ref[...], jd)
        dqkv_ref[:, 0:E] = dq.astype(dqkv_ref.dtype)
        dqkv_ref[:, E:2 * E] = dk.astype(dqkv_ref.dtype)
        dqkv_ref[:, 2 * E:3 * E] = _collapse(dve, col).astype(dqkv_ref.dtype)

    def _in_specs(nb_bias):
        bias_map = (lambda b, i: (0, 0, 0)) if nb_bias == 1 else (lambda b, i: (b, 0, 0))
        return [pl.BlockSpec((None, bq, E), lambda b, i: (b, i, qb)),
                pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, kb)),
                pl.BlockSpec((None, lk, E), lambda b, i: (b, 0, vb)),
                pl.BlockSpec((None, 1, lk), bias_map),
                pl.BlockSpec((bq, E), lambda b, i: (i, 0)),
                pl.BlockSpec((bq, E), lambda b, i: (i, 0)),
                pl.BlockSpec((lk, E), lambda b, i: (0, 0)),
                pl.BlockSpec((lk, E), lambda b, i: (0, 0))]

    def _cp(sem):
        return pltpu.CompilerParams(dimension_semantics=sem, vmem_limit_bytes=_VMEM_LIMIT)

    def fwd(q, kv, bias, cq, sq, ck, sk):
        B = q.shape[0]
        return pl.pallas_call(
            fwd_kernel, grid=(B, Lq // bq), in_specs=_in_specs(bias.shape[0]),
            out_specs=pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
            out_shape=jax.ShapeDtypeStruct((B, Lq, E), q.dtype), compiler_params=_cp(("parallel", "parallel")),
        )(q, kv, kv, bias, cq, sq, ck, sk)

    def bwd(q, kv, bias, cq, sq, ck, sk, o, do):
        B = q.shape[0]
        specs = _in_specs(bias.shape[0]) + [pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
                                            pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0))]
        if self_attn:
            return pl.pallas_call(
                bwd_kernel_self, grid=(B, 1), in_specs=specs,
                out_specs=pl.BlockSpec((None, Lq, 3 * E), lambda b, i: (b, 0, 0)),
                out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype), compiler_params=_cp(("parallel", "arbitrary")),
            )(q, kv, kv, bias, cq, sq, ck, sk, o, do)
        return pl.pallas_call(
            bwd_kernel_cross, grid=(B, Lq // bq), in_specs=specs,
            out_specs=[pl.BlockSpec((None, bq, E), lambda b, i: (b, i, 0)),
                       pl.BlockSpec((None, lk, 2 * E), lambda b, i: (b, 0, 0))],
            out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype), jax.ShapeDtypeStruct(kv.shape, kv.dtype)],
            scratch_shapes=[pltpu.VMEM((h * lkp, E), jnp.float32), pltpu.VMEM((h * lkp, E), jnp.float32)],
            compiler_params=_cp(("parallel", "arbitrary")),
        )(q, kv, kv, bias, cq, sq, ck, sk, o, do)

    return fwd, bwd


@functools.lru_cache(maxsize=None)
def _packed_cached(*args):
    return _jax_packed(*args)


def _pk_self_fwd(qkv: jax.Array, bias: jax.Array, cos: jax.Array, sin: jax.Array, h: int, lkp: int, rope: int,
                 half: int, rot: int, prec: int, scale: float) -> jax.Array:
    B, L, E3 = qkv.shape
    fwd, _ = _packed_cached(h, E3 // 3, L, L, lkp, L, rope, half, rot, prec, True, scale)
    return fwd(qkv, qkv, bias, cos, sin, cos, sin)


def _pk_self_bwd(qkv: jax.Array, bias: jax.Array, cos: jax.Array, sin: jax.Array, o: jax.Array, do: jax.Array, h: int,
                 lkp: int, rope: int, half: int, rot: int, prec: int, scale: float) -> jax.Array:
    B, L, E3 = qkv.shape
    _, bwd = _packed_cached(h, E3 // 3, L, L, lkp, L, rope, half, rot, prec, True, scale)
    return bwd(qkv, qkv, bias, cos, sin, cos, sin, o, do)


def _pk_cross_fwd(q: jax.Array, kv: jax.Array, bias: jax.Array, cq: jax.Array, sq: jax.Array, ck: jax.Array,
                  sk: jax.Array, h: int, lkp: int, bq: int, rope: int, half: int, rot: int, prec: int,
                  scale: float) -> jax.Array:
    B, Lq, E = q.shape
    fwd, _ = _packed_cached(h, E, Lq, kv.shape[1], lkp, bq, rope, half, rot, prec, False, scale)
    return fwd(q, kv, bias, cq, sq, ck, sk)


def _pk_cross_bwd(q: jax.Array, kv: jax.Array, bias: jax.Array, cq: jax.Array, sq: jax.Array, ck: jax.Array,
                  sk: jax.Array, o: jax.Array, do: jax.Array, h: int, lkp: int, bq: int, rope: int, half: int,
                  rot: int, prec: int, scale: float) -> tuple[jax.Array, jax.Array]:
    B, Lq, E = q.shape
    _, bwd = _packed_cached(h, E, Lq, kv.shape[1], lkp, bq, rope, half, rot, prec, False, scale)
    return bwd(q, kv, bias, cq, sq, ck, sk, o, do)


def _register_packed_ops():
    import torch_tpu._internal.pallas as tpu_pallas

    self_fwd = tpu_pallas.jax_op("tabicl_pk::pk_self_fwd", _pk_self_fwd)
    self_bwd = tpu_pallas.jax_op("tabicl_pk::pk_self_bwd", _pk_self_bwd)
    cross_fwd = tpu_pallas.jax_op("tabicl_pk::pk_cross_fwd", _pk_cross_fwd)
    cross_bwd = tpu_pallas.jax_op("tabicl_pk::pk_cross_bwd", _pk_cross_bwd)

    def self_setup(ctx, inputs, output):  # noqa: ANN001
        qkv, bias, cos, sin, *cfg = inputs
        ctx.save_for_backward(qkv, bias, cos, sin, output)
        ctx.cfg = tuple(cfg)

    def self_backward(ctx, do):  # noqa: ANN001
        qkv, bias, cos, sin, o = ctx.saved_tensors
        dqkv = self_bwd(qkv, bias, cos, sin, o, do.contiguous(), *ctx.cfg)
        return (dqkv,) + (None,) * (3 + len(ctx.cfg))

    def cross_setup(ctx, inputs, output):  # noqa: ANN001
        q, kv, bias, cq, sq, ck, sk, *cfg = inputs
        ctx.save_for_backward(q, kv, bias, cq, sq, ck, sk, output)
        ctx.cfg = tuple(cfg)

    def cross_backward(ctx, do):  # noqa: ANN001
        q, kv, bias, cq, sq, ck, sk, o = ctx.saved_tensors
        dq, dkv = cross_bwd(q, kv, bias, cq, sq, ck, sk, o, do.contiguous(), *ctx.cfg)
        return (dq, dkv) + (None,) * (5 + len(ctx.cfg))

    self_fwd.register_autograd(self_backward, setup_context=self_setup)
    cross_fwd.register_autograd(cross_backward, setup_context=cross_setup)
    return self_fwd, cross_fwd


try:
    PK_SELF, PK_CROSS = _register_packed_ops()
except Exception as _exc3:  # noqa: BLE001
    PK_SELF = PK_CROSS = None
    _REGISTER_ERROR_PACKED = _exc3


def rope_tables(rope_mod, L: int, E: int, h: int, device) -> tuple:  # noqa: ANN001
    """(cos, sin) [L, E] fp32 tiled per head (cos=1, sin=0 beyond the rotary dims), rope code, half, rot."""
    d = E // h
    if rope_mod is None:
        return (torch.ones(L, E, device=device), torch.zeros(L, E, device=device), 0, 0, 0)
    seq = rope_mod.get_seq_pos(L, device=device, dtype=torch.float32)
    if rope_mod.interleaved:
        freqs = rope_mod.forward(seq, seq_len=L).float()
    else:
        f = seq.float().unsqueeze(-1) * rope_mod.freqs.float()
        freqs = torch.cat([f, f], dim=-1)
    rot = freqs.shape[-1]
    cos, sin = freqs.cos(), freqs.sin()
    if rot < d:
        cos = torch.cat([cos, torch.ones(L, d - rot, device=device)], dim=-1)
        sin = torch.cat([sin, torch.zeros(L, d - rot, device=device)], dim=-1)
    return (cos.repeat(1, h).contiguous(), sin.repeat(1, h).contiguous(), 1 if rope_mod.interleaved else 2,
            rot // 2, rot)


def _bias3(key_bias, lead, Lk: int, device) -> Tensor:
    if key_bias is None:
        return torch.zeros(1, 1, Lk, device=device, dtype=torch.float32)
    kb = key_bias.float()
    while kb.dim() < len(lead) + 1:
        kb = kb.unsqueeze(0)
    if all(s_ == 1 for s_ in kb.shape[:-1]):
        return kb.reshape(1, 1, Lk).contiguous()
    B = 1
    for s_ in lead:
        B *= s_
    return kb.expand(*lead, Lk).reshape(B, 1, Lk).contiguous()


def packed_supported(Lq: int, Lk: int, E: int, h: int, self_attn: bool) -> bool:
    lkp = -(-Lk // 128) * 128
    if PK_SELF is None or E % 128 or E % h or lkp > _MAX_LKP:
        return False
    return (Lq <= 1024) if self_attn else _pick_bq(Lq, lkp) > 0


def pk_self_attention(qkv: Tensor, h: int, key_bias: Optional[Tensor], rope_mod, scale: float) -> Tensor:  # noqa: ANN001
    """qkv [..., L, 3E] (packed in-projection), RoPE (optional) on q and k, returns [..., L, E]."""
    lead = qkv.shape[:-2]
    L, E3 = qkv.shape[-2], qkv.shape[-1]
    E = E3 // 3
    lkp = -(-L // 128) * 128
    cos, sin, rc, half, rot = rope_tables(rope_mod, L, E, h, qkv.device)
    prec = _fp32_prec_code() if qkv.dtype == torch.float32 else 0
    out = PK_SELF(qkv.reshape(-1, L, E3).contiguous(), _bias3(key_bias, lead, L, qkv.device), cos, sin, h, lkp, rc,
                  half, rot, prec, float(scale))
    return out.reshape(*lead, L, E)


def pk_cross_attention(q: Tensor, kv: Tensor, h: int, key_bias: Optional[Tensor], rope_mod,  # noqa: ANN001
                       scale: float) -> Tensor:
    """q [..., Lq, E], kv [..., Lk, 2E] (packed K | V projection); RoPE (optional) with positions 0..L-1 each."""
    lead = q.shape[:-2]
    Lq, E = q.shape[-2], q.shape[-1]
    Lk = kv.shape[-2]
    lkp = -(-Lk // 128) * 128
    bq = _pick_bq(Lq, lkp)
    cq, sq, rc, half, rot = rope_tables(rope_mod, Lq, E, h, q.device)
    ck, sk, _, _, _ = rope_tables(rope_mod, Lk, E, h, q.device)
    prec = _fp32_prec_code() if q.dtype == torch.float32 else 0
    q3 = q.reshape(-1, Lq, E)
    kv3 = kv.reshape(-1, Lk, 2 * E)
    if q3.shape[0] != kv3.shape[0]:
        q3 = q3.expand(kv3.shape[0], Lq, E)
    out = PK_CROSS(q3.contiguous(), kv3.contiguous(), _bias3(key_bias, lead, Lk, q.device), cq, sq, ck, sk, h, lkp,
                   bq, rc, half, rot, prec, float(scale))
    return out.reshape(*lead, Lq, E)


@torch.compiler.assume_constant_result  # evaluated at trace time (Dynamo cannot trace the precision getter)
def _fp32_prec_code() -> int:
    """fp32 matmul precision of the current stage (set by the caller before tracing): highest=2, high=1, else 0."""
    p = torch.get_float32_matmul_precision()
    return 2 if p == "highest" else 1 if p == "high" else 0


def supported(Lq: int, Lk: int, E: int, h: int) -> bool:
    lkp = -(-Lk // 128) * 128
    return HX_FWD is not None and E % 128 == 0 and E % h == 0 and lkp <= _MAX_LKP and _pick_bq(Lq, lkp) > 0


def hx_attention(q: Tensor, k: Tensor, v: Tensor, h: int, key_bias: Optional[Tensor] = None) -> Tensor:
    """q [..., Lq, E] (pre-scaled), k / v [..., Lk, E]; key_bias additive, broadcastable to [..., Lk] (no Lq/head dims)."""
    lead = q.shape[:-2]
    Lq, E = q.shape[-2], q.shape[-1]
    Lk = k.shape[-2]
    lkp = -(-Lk // 128) * 128
    bq = _pick_bq(Lq, lkp)
    q3 = q.reshape(-1, Lq, E)
    k3 = k.reshape(-1, Lk, E)
    v3 = v.reshape(-1, Lk, E)
    B = q3.shape[0]
    if key_bias is None:
        bias = torch.zeros(1, 1, Lk, device=q.device, dtype=torch.float32)
    else:
        kb = key_bias.float()
        while kb.dim() < len(lead) + 1:
            kb = kb.unsqueeze(0)
        if all(s == 1 for s in kb.shape[:-1]):
            bias = kb.reshape(1, 1, Lk)
        else:
            bias = kb.expand(*lead, Lk).reshape(B, 1, Lk)
    prec = _fp32_prec_code() if q.dtype == torch.float32 else 0
    out = HX_FWD(q3.contiguous(), k3.contiguous(), v3.contiguous(), bias.contiguous(), h, lkp, bq, prec)
    return out.reshape(*lead, Lq, E)

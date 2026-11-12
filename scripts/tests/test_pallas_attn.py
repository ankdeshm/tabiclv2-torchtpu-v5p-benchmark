#!/usr/bin/env python3
"""TPU test of tabicl_tpu_pallas_attn.hx_attention: forward/backward vs an fp32 reference, then fwd+bwd timing vs the
explicit-matmul XLA path at the training shapes (one chip). Prints JSON lines."""
from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401

sys.path.insert(0, "/workspace/run")
import tabicl_tpu_pallas_attn as pk  # noqa: E402

dev = torch.device("tpu")
H = 8


def ref_attn(q, k, v, bias=None):
    B, Lq, E = q.shape
    d = E // H
    qh = q.float().view(B, Lq, H, d).transpose(1, 2)
    kh = k.float().view(B, -1, H, d).transpose(1, 2)
    vh = v.float().view(B, -1, H, d).transpose(1, 2)
    s = qh @ kh.transpose(-1, -2)
    if bias is not None:
        s = s + bias.float().view(bias.shape[0], 1, 1, -1)
    p = torch.softmax(s, dim=-1)
    return (p @ vh).transpose(1, 2).reshape(B, Lq, E)


def math_attn(q, k, v, bias=None):  # the v6 training path (head split, bf16 matmuls, fp32 softmax)
    B, Lq, E = q.shape
    d = E // H
    qh = q.view(B, Lq, H, d).transpose(1, 2)
    kh = k.view(B, -1, H, d).transpose(1, 2)
    vh = v.view(B, -1, H, d).transpose(1, 2)
    s = torch.matmul(qh, kh.transpose(-1, -2)).float()
    if bias is not None:
        s = s + bias.float().view(bias.shape[0], 1, 1, -1)
    p = torch.softmax(s, dim=-1).to(vh.dtype)
    return torch.matmul(p, vh).transpose(1, 2).reshape(B, Lq, E)


def emit(rec):
    print("[pk-test] " + json.dumps(rec), flush=True)


def check(name, B, Lq, Lk, dtype, with_bias=False, grad=True):
    torch.manual_seed(0)
    q = (torch.randn(B, Lq, 128, device=dev) * 0.25).to(dtype)
    k = torch.randn(B, Lk, 128, device=dev).to(dtype)
    v = torch.randn(B, Lk, 128, device=dev).to(dtype)
    bias = None
    if with_bias:
        bias = torch.zeros(B, Lk, device=dev)
        bias[:, Lk - 9:] = -1e30
    rec = {"check": name, "B": B, "Lq": Lq, "Lk": Lk, "dtype": str(dtype), "bias": with_bias}
    try:
        qr, kr, vr = (t.detach().clone().requires_grad_(grad) for t in (q, k, v))
        out = pk.hx_attention(qr, kr, vr, H, bias)
        qf, kf, vf = (t.detach().float().clone().requires_grad_(grad) for t in (q, k, v))
        ref = ref_attn(qf, kf, vf, bias)
        rec["fwd_max_abs"] = float((out.float() - ref).abs().max())
        rec["fwd_ref_max"] = float(ref.abs().max())
        if grad:
            go = torch.randn_like(ref)
            out.float().backward(go)
            ref.backward(go)
            for nm, a, b in (("dq", qr, qf), ("dk", kr, kf), ("dv", vr, vf)):
                rec[f"{nm}_max_abs"] = float((a.grad.float() - b.grad).abs().max())
                rec[f"{nm}_ref_max"] = float(b.grad.abs().max())
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    emit(rec)


def bench(name, B, Lq, Lk, dtype=torch.bfloat16, iters=10, with_bias=False):
    torch.manual_seed(0)
    q = (torch.randn(B, Lq, 128, device=dev) * 0.25).to(dtype).requires_grad_(True)
    k = torch.randn(B, Lk, 128, device=dev).to(dtype).requires_grad_(True)
    v = torch.randn(B, Lk, 128, device=dev).to(dtype).requires_grad_(True)
    bias = None
    if with_bias:
        bias = torch.zeros(B, Lk, device=dev)
        bias[:, Lk - 9:] = -1e30
    rec = {"bench": name, "B": B, "Lq": Lq, "Lk": Lk, "dtype": str(dtype)}
    for impl, fn in (("pallas", lambda a, b, c: pk.hx_attention(a, b, c, H, bias)),
                     ("xla_math", lambda a, b, c: math_attn(a, b, c, bias))):
        try:
            cf = torch.compile(lambda a, b, c: fn(a, b, c).float().square().mean(), backend="tpu", dynamic=False)
            for _ in range(2):
                cf(q, k, v).backward()
            torch.accelerator.synchronize(dev)
            ts = []
            for _ in range(iters):
                q.grad = k.grad = v.grad = None
                torch.accelerator.synchronize(dev)
                t0 = time.perf_counter()
                cf(q, k, v).backward()
                torch.accelerator.synchronize(dev)
                ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            rec[f"{impl}_fwd_bwd_ms"] = round(ts[len(ts) // 2], 3)
        except Exception as exc:  # noqa: BLE001
            rec[f"{impl}_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    emit(rec)


def bench_fwd(name, B, Lq, Lk, dtype=torch.float32, iters=5, prec="high"):
    torch.set_float32_matmul_precision(prec)
    q = (torch.randn(B, Lq, 128, device=dev) * 0.25).to(dtype)
    k = torch.randn(B, Lk, 128, device=dev).to(dtype)
    v = torch.randn(B, Lk, 128, device=dev).to(dtype)
    rec = {"bench_fwd": name, "B": B, "Lq": Lq, "Lk": Lk, "dtype": str(dtype), "prec": prec}

    def sdpa(a, b, c):
        Bq, L, E = a.shape
        ah = a.view(Bq, L, H, 16).transpose(1, 2)
        bh = b.view(Bq, -1, H, 16).transpose(1, 2)
        ch = c.view(Bq, -1, H, 16).transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(ah, bh, ch, scale=1.0).transpose(1, 2).reshape(Bq, L, E)

    with torch.no_grad():
        for impl, fn in (("pallas", lambda a, b, c: pk.hx_attention(a, b, c, H)), ("torchtpu_sdpa", sdpa)):
            try:
                cf = torch.compile(lambda a, b, c: fn(a, b, c) + 0, backend="tpu", dynamic=False)
                for _ in range(2):
                    cf(q, k, v)
                torch.accelerator.synchronize(dev)
                ts = []
                for _ in range(iters):
                    torch.accelerator.synchronize(dev)
                    t0 = time.perf_counter()
                    cf(q, k, v)
                    torch.accelerator.synchronize(dev)
                    ts.append((time.perf_counter() - t0) * 1000)
                ts.sort()
                rec[f"{impl}_fwd_ms"] = round(ts[len(ts) // 2], 3)
            except Exception as exc:  # noqa: BLE001
                rec[f"{impl}_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    torch.set_float32_matmul_precision("highest")
    emit(rec)


def check_hxq(name, B, Lq, Lk, valid, dtype=torch.float32, prec="high", iters=5):
    """Query-expansion flash kernel (inference) vs fp32 reference, and timing vs head split + Splash."""
    import os
    os.environ.setdefault("TABICL_TPU_SPLASH_BKV", "4096")
    os.environ.setdefault("TABICL_TPU_SPLASH_BKV_COMPUTE", "256")
    import tabicl_tpu_bucketed as tb
    from tabicl._model import attention as att
    tb._install_splash_knobs()
    torch.set_float32_matmul_precision(prec)
    torch.manual_seed(0)
    q = (torch.randn(B, Lq, 128, device=dev) * 0.25).to(dtype)
    kv = torch.randn(B, Lk, 256, device=dev).to(dtype)
    vt = torch.tensor(valid, device=dev, dtype=torch.int32)
    rec = {"check_hxq": name, "B": B, "Lq": Lq, "Lk": Lk, "valid": valid, "dtype": str(dtype), "prec": prec}
    try:
        with torch.no_grad():
            out = pk.hxq_attention(q, kv, vt, H)
            ref = ref_attn(q.float(), kv[:, :valid, :128].float(), kv[:, :valid, 128:].float())
            rec["fwd_max_abs"] = float((out.float() - ref).abs().max())
            rec["fwd_ref_max"] = float(ref.abs().max())

            def splash_path(a, b):
                Bq, L, E = a.shape
                qh = a.view(Bq, L, H, 16).transpose(1, 2)
                kh = b[..., :128].reshape(Bq, -1, H, 16).transpose(1, 2)
                vh = b[..., 128:].reshape(Bq, -1, H, 16).transpose(1, 2)
                return att._tpu_pallas_splash_sdpa(qh, kh, vh, valid_kv_len=vt).transpose(1, 2).reshape(Bq, L, E)

            sp = splash_path(q, kv)
            rec["splash_vs_ref_max_abs"] = float((sp.float() - ref).abs().max())
            for impl, fn in (("hxq", lambda a, b: pk.hxq_attention(a, b, vt, H)), ("headsplit_splash", splash_path)):
                cf = torch.compile(lambda a, b: fn(a, b) + 0, backend="tpu", dynamic=False)
                for _ in range(2):
                    cf(q, kv)
                torch.accelerator.synchronize(dev)
                ts = []
                for _ in range(iters):
                    torch.accelerator.synchronize(dev)
                    t0 = time.perf_counter()
                    cf(q, kv)
                    torch.accelerator.synchronize(dev)
                    ts.append((time.perf_counter() - t0) * 1000)
                ts.sort()
                rec[f"{impl}_fwd_ms"] = round(ts[len(ts) // 2], 3)
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    torch.set_float32_matmul_precision("highest")
    emit(rec)


def check_packed(name, B, Lq, Lk, dtype, self_attn, interleaved=None, with_bias=False):
    """Packed-projection kernels (in-kernel RoPE) vs fp32 reference: forward and grads w.r.t. the packed inputs."""
    from tabicl._model.rope import RotaryEmbedding
    import tabicl_tpu_mha as mha
    torch.manual_seed(0)
    rope = None if interleaved is None else RotaryEmbedding(dim=16, theta=100000, interleaved=interleaved).to(dev)
    scale = 16 ** -0.5
    rec = {"check_packed": name, "B": B, "Lq": Lq, "Lk": Lk, "dtype": str(dtype), "self": self_attn,
           "rope": interleaved, "bias": with_bias}
    try:
        bias = None
        if with_bias:
            bias = torch.zeros(B, Lk, device=dev)
            bias[:, Lk - 7:] = -1e30
        if self_attn:
            x = torch.randn(B, Lq, 384, device=dev).to(dtype)
            xr = x.detach().clone().requires_grad_(True)
            out = pk.pk_self_attention(xr, H, bias, rope, scale)
            xf = x.detach().float().clone().requires_grad_(True)
            q, k, v = xf[..., :128], xf[..., 128:256], xf[..., 256:]
        else:
            x = torch.randn(B, Lq, 128, device=dev).to(dtype)
            y = torch.randn(B, Lk, 256, device=dev).to(dtype)
            xr = x.detach().clone().requires_grad_(True)
            yr = y.detach().clone().requires_grad_(True)
            out = pk.pk_cross_attention(xr, yr, H, bias, rope, scale)
            xf = x.detach().float().clone().requires_grad_(True)
            yf = y.detach().float().clone().requires_grad_(True)
            q, k, v = xf, yf[..., :128], yf[..., 128:]
        if rope is not None:
            mha._ROPE = "roll"
            q = mha.rope_merged(rope, q, H)
            k = mha.rope_merged(rope, k, H)
        ref = ref_attn(q * scale, k, v, bias)
        rec["fwd_max_abs"] = float((out.float() - ref).abs().max())
        rec["fwd_ref_max"] = float(ref.abs().max())
        go = torch.randn_like(ref)
        out.float().backward(go)
        ref.backward(go)
        rec["dx_max_abs"] = float((xr.grad.float() - xf.grad).abs().max())
        rec["dx_ref_max"] = float(xf.grad.abs().max())
        if not self_attn:
            rec["dy_max_abs"] = float((yr.grad.float() - yf.grad).abs().max())
            rec["dy_ref_max"] = float(yf.grad.abs().max())
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    emit(rec)


def bench_packed(name, B, L, iters=10):
    """Row-stage self-attention fwd+bwd: packed kernel (+RoPE in VMEM) vs projection split + XLA RoPE + kernel."""
    from tabicl._model.rope import RotaryEmbedding
    import tabicl_tpu_mha as mha
    torch.manual_seed(0)
    rope = RotaryEmbedding(dim=16, theta=100000, interleaved=False).to(dev)
    scale = 16 ** -0.5
    x = torch.randn(B, L, 384, device=dev).to(torch.bfloat16).requires_grad_(True)
    rec = {"bench_packed": name, "B": B, "L": L}

    def unpacked(t):
        q, k, v = t[..., :128], t[..., 128:256], t[..., 256:]
        q = mha.rope_merged(rope, q, H)
        k = mha.rope_merged(rope, k, H)
        return pk.hx_attention(q * scale, k, v, H)

    for impl, fn in (("packed", lambda t: pk.pk_self_attention(t, H, None, rope, scale)), ("unpacked", unpacked)):
        try:
            cf = torch.compile(lambda t: fn(t).float().square().mean(), backend="tpu", dynamic=False)
            for _ in range(2):
                cf(x).backward()
            torch.accelerator.synchronize(dev)
            ts = []
            for _ in range(iters):
                x.grad = None
                torch.accelerator.synchronize(dev)
                t0 = time.perf_counter()
                cf(x).backward()
                torch.accelerator.synchronize(dev)
                ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            rec[f"{impl}_fwd_bwd_ms"] = round(ts[len(ts) // 2], 3)
        except Exception as exc:  # noqa: BLE001
            rec[f"{impl}_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    emit(rec)


def _mab1_inputs(B, Lq, Lk):
    E = 128
    g = torch.Generator(device="cpu").manual_seed(0)
    ind = (torch.randn(Lq, E, generator=g) * 0.5).to(dev)
    x = torch.randn(B, Lk, E, generator=g).to(dev)
    W = (torch.randn(3 * E, E, generator=g) * E ** -0.5).to(dev)
    b = (torch.randn(3 * E, generator=g) * 0.1).to(dev)
    return E, ind, x, W, b


def check_hxf(name, B, Lq, Lk, valid):
    """Folded first-ISAB attention (tabicl_tpu_mha.folded_mab1 + hxf kernel) vs the unfolded fp32 reference."""
    import tabicl_tpu_mha as mha
    rec = {"check_hxf": name, "B": B, "Lq": Lq, "Lk": Lk, "valid": valid}
    try:
        E, ind, x, W, b = _mab1_inputs(B, Lq, Lk)
        vt = torch.tensor(valid, device=dev)
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            qp = F.linear(ind, W[:E], b[:E]) * (E // H) ** -0.5
            k = F.linear(x, W[E:2 * E], b[E:2 * E])
            v = F.linear(x, W[2 * E:], b[2 * E:])
            bias = torch.zeros(B, Lk, device=dev)
            bias[:, valid:] = -1e30
            ref = ref_attn(qp.expand(B, Lq, E), k, v, bias)
            rec["ref_max"] = float(ref.abs().max())
            for prec in ("highest", "high"):
                torch.set_float32_matmul_precision(prec)
                for qkind, q in (("expanded", ind.expand(B, Lq, E)), ("per_batch", ind.expand(B, Lq, E).contiguous())):
                    o = mha.folded_mab1(q, x, vt, H, W, b, None, None, pk=pk)
                    rec[f"{prec}_{qkind}_max_abs"] = None if o is None else float((o.float() - ref).abs().max())
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    torch.set_float32_matmul_precision("highest")
    emit(rec)


def bench_hxf(name, B, Lq, Lk, valid, iters=10):
    """Inference column first attention, fwd only at fp32 'high': folded (hxf) vs packed K/V projection + hxq kernel."""
    import tabicl_tpu_mha as mha
    rec = {"bench_hxf": name, "B": B, "Lq": Lq, "Lk": Lk, "valid": valid}
    torch.set_float32_matmul_precision("high")
    E, ind, x, W, b = _mab1_inputs(B, Lq, Lk)
    vt = torch.tensor(valid, device=dev)
    scale = (E // H) ** -0.5

    def folded(xx):
        return mha.folded_mab1(ind.expand(B, Lq, E), xx, vt, H, W, b, None, None, pk=pk)

    def unfolded(xx):
        q = F.linear(ind, W[:E], b[:E]) * scale
        kv = F.linear(xx, W[E:], b[E:])
        return pk.hxq_attention(q.expand(B, Lq, E), kv, vt, H)

    outs = {}
    for impl, fn in (("folded", folded), ("hxq", unfolded)):
        try:
            cf = torch.compile(fn, backend="tpu", dynamic=False)
            with torch.no_grad():
                for _ in range(2):
                    outs[impl] = cf(x)
                torch.accelerator.synchronize(dev)
                ts = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    outs[impl] = cf(x)
                    torch.accelerator.synchronize(dev)
                    ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            rec[f"{impl}_ms"] = round(ts[len(ts) // 2], 3)
        except Exception as exc:  # noqa: BLE001
            rec[f"{impl}_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    if "folded" in outs and "hxq" in outs:
        rec["folded_vs_hxq_max_abs"] = float((outs["folded"].float() - outs["hxq"].float()).abs().max())
        rec["hxq_max"] = float(outs["hxq"].abs().max())
    torch.set_float32_matmul_precision("highest")
    emit(rec)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "hxf"):
        check_hxf("small", 4, 128, 4096, 3000)
        check_hxf("infer_tile54", 54, 128, 53248, 45000)
        bench_hxf("infer_tile54", 54, 128, 53248, 45000)
    if what in ("all", "check"):
        check("col2_bf16", 4, 1024, 128, torch.bfloat16)
        check("col2_fp32", 4, 1024, 128, torch.float32)
        check("col1_bf16", 4, 128, 576, torch.bfloat16)
        check("row_bf16", 16, 104, 104, torch.bfloat16)
        check("row_fp32_bias", 16, 108, 108, torch.float32, with_bias=True)
        check("cls_bf16", 16, 4, 104, torch.bfloat16)
        check("col2_inf_fp32", 2, 53248, 128, torch.float32, grad=False)
    if what in ("all", "bench"):
        bench("col2_train_B4", 416, 1024, 128)
        bench("col1_train_B4_k576", 416, 128, 576)
        bench("row_train_B4", 4096, 104, 104)
        bench("cls_train_B4", 4096, 4, 104)
        bench_fwd("col2_infer_tile54", 54, 53248, 128)
    if what in ("all", "hxq"):
        check_hxq("col1_small", 4, 128, 4096, 3000)
        check_hxq("col1_infer_tile54", 54, 128, 53248, 45000)
    if what in ("all", "packed"):
        check_packed("self_bf16_rope_halves", 16, 104, 104, torch.bfloat16, True, interleaved=False)
        check_packed("self_fp32_rope_halves_bias", 16, 108, 108, torch.float32, True, interleaved=False, with_bias=True)
        check_packed("self_fp32_rope_pairs", 8, 104, 104, torch.float32, True, interleaved=True)
        check_packed("cross_cls_bf16_rope", 16, 4, 104, torch.bfloat16, False, interleaved=False)
        check_packed("cross_col2_bf16", 4, 1024, 128, torch.bfloat16, False)
        check_packed("cross_col1_bf16", 4, 128, 576, torch.bfloat16, False)
        check_packed("cross_col2_fp32", 2, 4096, 128, torch.float32, False)
        bench_packed("row_train_B4", 4096, 104)
    print("PK_TEST_DONE", flush=True)

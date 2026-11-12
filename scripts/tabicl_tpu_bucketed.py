#!/usr/bin/env python3
"""Bucketed `torch.compile(backend="tpu", dynamic=False, fullgraph=True)` Engine for TabICLv2 on TorchTPU.

Provides:
  - Fixed power-of-two bucket grids along `(n_rows, n_features, batch_tables)` so TPU XLA/Pallas
    compiles each bucket once and serves arbitrary dataset shapes with zero recompilations and zero
    Dynamo graph breaks (`fullgraph=True`).
  - Mathematically exact masked padding:
      * Circular feature grouping modulo `d_real` (`real_features`)
      * Masked `RowInteraction` feature attention over `d_real + num_cls` columns
      * Masked `SetTransformer` inducing-point attention and `QASSMaxMLP` `log(train_size_real)`
      * Masked `ICLPredictor` context attention via TorchTPU Pallas rectangular Splash Attention
        (`tabicl::tpu_splash_sdpa_seg`) for `T_kv >= 16,384` and fused/tiled XLA SDPA for `< 16,384`
  - Both `kv_cache=False` (end-to-end single-graph fused OR automatic HBM-safe stage-tiled execution
    up to `500K+` rows x `500` features) and `kv_cache=True` (`prefill` + `query` pure-tensor
    `fullgraph=True` compiled graphs supporting `bf16` and `fp32` KV caches).
  - Seamless drop-in attachment to `TabICLClassifier` and `TabICLRegressor` via `attach_compiled_runner`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from tabicl._model.kv_cache import KVCacheEntry

# Mandatory Step 1.4 bucket grids (+ Step 1.5 ultra-long 128K / 256K / 512K buckets)
ROW_BUCKETS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
)
TEST_ROW_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384)
FEATURE_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
ESTIMATOR_BUCKETS: tuple[int, ...] = (1, 4, 8, 16, 32)
BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def pick_bucket(val: int, buckets: tuple[int, ...]) -> int:
    """Return smallest bucket >= val (or round up to multiple of 1024 if above max bucket)."""
    if val <= 0:
        return buckets[0]
    for b in buckets:
        if val <= b:
            return b
    return int(math.ceil(val / 1024.0) * 1024)


# ---- v5p tuning knobs (environment variables; defaults reproduce the original v7x runner exactly) ----
import os as _os


def _env_prec(name: str, default: str) -> str:
    """Matmul precision override for a stage: highest | high | medium (default keeps the original choice)."""
    v = _os.environ.get(name, "").strip().lower()
    return v if v in ("highest", "high", "medium") else default


def pick_row_bucket(val: int) -> int:
    """Row bucket: powers of two by default; with TABICL_TPU_ROW_BUCKET_STEP=N (>0), rows above 16384 round up to a
    multiple of N instead (e.g. N=8192: 45,000 -> 49,152 and 50,000 -> 57,344 instead of 65,536)."""
    step = int(_os.environ.get("TABICL_TPU_ROW_BUCKET_STEP", "0") or 0)
    if step > 0 and val > 16384:
        return int(math.ceil(val / step) * step)
    return pick_bucket(val, ROW_BUCKETS)


_ICL_BSTEP1_MIN_T = 65536 if int(_os.environ.get("TABICL_TPU_ROW_BUCKET_STEP", "0") or 0) <= 0 else 40960


def pick_feature_bucket(val: int) -> int:
    """Feature bucket: powers of two by default (FEATURE_BUCKETS); with TABICL_TPU_FEATURE_STEP=N (>0), round up to a
    multiple of N (minimum 16) instead, e.g. N=8: 100 -> 104 rather than 128 (padded columns are masked downstream)."""
    step = int(_os.environ.get("TABICL_TPU_FEATURE_STEP", "0") or 0)
    if step > 0:
        return max(16, int(math.ceil(val / step) * step))
    return pick_bucket(val, FEATURE_BUCKETS)


def _auto_col_group_tile(hc: int, default: int) -> int:
    """With TABICL_TPU_COL_GROUP_TILE=auto, use the largest divisor of the column count hc in [16, 66] (no padded
    column groups), e.g. 132 -> 66, 108 -> 54; otherwise the configured/default tile."""
    if _os.environ.get("TABICL_TPU_COL_GROUP_TILE", "").strip().lower() != "auto":
        return default
    for g in range(min(66, hc), 15, -1):
        if hc % g == 0:
            return g
    return default


# ---- v5p round-2 knobs (all default OFF, i.e. the original behaviour) ----
# Per-stage wall-clock timing with device syncs at stage boundaries (adds a few ms of sync overhead per stage).
STAGE_T: dict[str, float] = {}
_STAGE_TIMING = _os.environ.get("TABICL_TPU_STAGE_TIMING", "") == "1"


def stage_timing_reset() -> None:
    STAGE_T.clear()


def _stage_add(name: str, dt: float) -> None:
    STAGE_T[name] = STAGE_T.get(name, 0.0) + dt


# Static ICL key/value length. With a tensor `train_size`, the patched ICL blocks use k = v = ALL T_bkt rows and mask
# keys >= train_size inside the Splash kernel (FullMask + segment ids: masked blocks are still computed), so padded
# rows and test rows cost full attention compute. With TABICL_TPU_ICL_KV_TRIM=1 the nocache path slices K/V to a
# static length L = ceil(train_real / TABICL_TPU_ICL_KV_STEP) * STEP (<= T_bkt) and keeps exact masking inside L via
# valid_kv_len = train_size (and SSMax uses src_len_override = train_size, as before).
_ICL_KV_TRIM = _os.environ.get("TABICL_TPU_ICL_KV_TRIM", "") == "1"
_ICL_KV_STEP = int(_os.environ.get("TABICL_TPU_ICL_KV_STEP", "1024") or 1024)
_ICL_KV_STATIC_LEN: Optional[int] = None  # set around ICL calls; read at trace time (Dynamo guards on its value)


def _icl_kv_len(train_real: int, T_bkt: int) -> Optional[int]:
    if not _ICL_KV_TRIM:
        return None
    L = int(math.ceil(train_real / _ICL_KV_STEP) * _ICL_KV_STEP)
    return L if L < T_bkt else None


def _install_icl_kv_trim() -> None:
    from tabicl._model import layers as _layers

    cls = _layers.MultiheadAttentionBlock
    if getattr(cls, "_v5p_kv_trim_installed", False):
        return
    orig_forward = cls.forward

    def forward(self, q, k=None, v=None, *args, train_size=None, **kwargs):  # noqa: ANN001
        L = _ICL_KV_STATIC_LEN
        if (
            L is not None
            and isinstance(train_size, Tensor)
            and k is None
            and v is None
            and kwargs.get("cached_kv") is None
            and L < q.shape[-2]
        ):
            kv = q[..., :L, :]
            if kwargs.get("src_len_override") is None:
                kwargs["src_len_override"] = train_size
            if kwargs.get("valid_kv_len") is None:
                kwargs["valid_kv_len"] = train_size
            return orig_forward(self, q, kv, kv, *args, train_size=None, **kwargs)
        return orig_forward(self, q, k, v, *args, train_size=train_size, **kwargs)

    cls.forward = forward
    cls._v5p_kv_trim_installed = True


# Splash block sizes (defaults in the patched attention.py: block_q 128/512/1024 by T_q, block_kv 1024,
# block_kv_compute min(block_kv, 512)). Override with TABICL_TPU_SPLASH_BQ / _BKV / _BKV_COMPUTE.
_SPLASH_BQ = int(_os.environ.get("TABICL_TPU_SPLASH_BQ", "0") or 0)
_SPLASH_BKV = int(_os.environ.get("TABICL_TPU_SPLASH_BKV", "0") or 0)
_SPLASH_BKVC = int(_os.environ.get("TABICL_TPU_SPLASH_BKV_COMPUTE", "0") or 0)


def _install_splash_knobs() -> None:
    if not (_SPLASH_BQ or _SPLASH_BKV or _SPLASH_BKVC):
        return
    from tabicl._model import attention as _att

    if getattr(_att, "_v5p_splash_knobs_installed", False):
        return
    cache: dict = {}

    def _builder(H: int, T_q_pad: int, T_kv_pad: int, D: int, bq: int, bkv: int, has_segment_ids: bool):
        bkvc = _SPLASH_BKVC if _SPLASH_BKVC else min(bkv, 512)
        bkvc = min(bkvc, bkv)
        key = (H, T_q_pad, T_kv_pad, D, bq, bkv, bkvc, has_segment_ids)
        if key in cache:
            return cache[key]
        import jax
        import jax.numpy as jnp
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask
        import torch_tpu._internal.pallas as tpu_pallas

        mask = splash_attention_mask.MultiHeadMask([splash_attention_mask.FullMask((T_q_pad, T_kv_pad)) for _ in range(H)])
        block_sizes = splash_attention_kernel.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkvc)
        splash_single = splash_attention_kernel.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
        if has_segment_ids:
            vmapped = jax.vmap(splash_single, in_axes=(0, 0, 0, None))

            def _jax_splash_seg(q, k, v, valid_kv):  # noqa: ANN001
                q_seg = jnp.ones((T_q_pad,), dtype=jnp.int32)
                kv_idx = jnp.arange(T_kv_pad, dtype=jnp.int32)
                kv_seg = jnp.where(kv_idx < valid_kv, 1, 0).astype(jnp.int32)
                return vmapped(q, k, v, splash_attention_kernel.SegmentIds(q=q_seg, kv=kv_seg))

            kernel = tpu_pallas.custom_jax_kernel(
                _jax_splash_seg, f"tabicl::v5p_splash_seg_{H}_{T_q_pad}_{T_kv_pad}_{D}_{bq}_{bkv}_{bkvc}", warn_deprecated=False
            )
        else:
            vmapped = jax.vmap(splash_single, in_axes=(0, 0, 0))

            def _jax_splash_full(q, k, v):  # noqa: ANN001
                return vmapped(q, k, v)

            kernel = tpu_pallas.custom_jax_kernel(
                _jax_splash_full, f"tabicl::v5p_splash_full_{H}_{T_q_pad}_{T_kv_pad}_{D}_{bq}_{bkv}_{bkvc}", warn_deprecated=False
            )
        cache[key] = kernel
        return kernel

    def _sdpa(q: Tensor, k: Tensor, v: Tensor, valid_kv_len=None) -> Tensor:  # noqa: ANN001
        _, H, T_q, D = q.shape
        T_kv = k.shape[-2]
        bkv = _SPLASH_BKV if _SPLASH_BKV else 1024
        bq = 128 if T_q <= 256 else (512 if T_q < 1024 else 1024)
        if _SPLASH_BQ and T_q >= _SPLASH_BQ:
            bq = _SPLASH_BQ
        T_q_pad = ((T_q + bq - 1) // bq) * bq
        T_kv_pad = ((T_kv + bkv - 1) // bkv) * bkv
        q_scaled = (q * (D**-0.5)).contiguous()
        q_in = F.pad(q_scaled, (0, 0, 0, T_q_pad - T_q)).contiguous() if T_q_pad != T_q else q_scaled
        k_in = F.pad(k, (0, 0, 0, T_kv_pad - T_kv)).contiguous() if T_kv_pad != T_kv else k.contiguous()
        v_in = F.pad(v, (0, 0, 0, T_kv_pad - T_kv)).contiguous() if T_kv_pad != T_kv else v.contiguous()
        if isinstance(valid_kv_len, Tensor):
            valid_kv_t = valid_kv_len.to(device=q.device, dtype=torch.int32).reshape(())
            out = _att._tpu_splash_sdpa_seg_op(q_in, k_in, v_in, valid_kv_t, bq, bkv)
        else:
            eff_kv = T_kv if valid_kv_len is None else int(valid_kv_len)
            if eff_kv == T_kv_pad:
                out = _att._tpu_splash_sdpa_full_op(q_in, k_in, v_in, bq, bkv)
            else:
                valid_kv_t = torch.full((), eff_kv, dtype=torch.int32, device=q.device)
                out = _att._tpu_splash_sdpa_seg_op(q_in, k_in, v_in, valid_kv_t, bq, bkv)
        if T_q_pad != T_q:
            out = out[:, :, :T_q, :]
        return out

    _att._get_tpu_rect_splash_kernel = torch.compiler.assume_constant_result(_builder)
    _att._tpu_pallas_splash_sdpa = _sdpa
    _att._v5p_splash_knobs_installed = True


# Small attention problems (no mask, short key length) via explicit matmul + fp32 softmax + matmul instead of
# F.scaled_dot_product_attention. The XProf traces show TorchTPU's SDPA kernel ("scaled_dot_product_cudnn_attention")
# is very slow for the many short sequences with 16-dim heads of the column / row stages (training step: SDPA fwd+bwd
# ~53% of device time, plus head-transpose copies). Default off. TABICL_TPU_SDPA_MATH_MAX_ELEMS bounds q_len * k_len
# per (batch, head) (default 2^20: training-sized problems; the inference column ISAB needs ~6.8M).
_SDPA_MATH = _os.environ.get("TABICL_TPU_SDPA_MATH", "") == "1"
_SDPA_MATH_MAX = int(_os.environ.get("TABICL_TPU_SDPA_MATH_MAX_ELEMS", str(1 << 20)) or (1 << 20))


def _install_sdpa_math() -> None:
    if not _SDPA_MATH:
        return
    from tabicl._model import attention as _att

    if getattr(_att, "_v5p_sdpa_math_installed", False):
        return
    orig = _att.sdpa_with_flattened_batch

    def sdpa_with_flattened_batch(q, k, v, attn_mask=None, dropout_p=0.0, ssmax_layer=None,  # noqa: ANN001
                                  src_len_override=None, valid_kv_len=None):
        if (q.device.type == "tpu" and dropout_p == 0.0 and attn_mask is None and valid_kv_len is None
                and getattr(_att, "_tpu_flash_attention_fn", None) is None
                and k.shape[-2] < _att._tpu_splash_min_kv() and q.shape[-2] * k.shape[-2] <= _SDPA_MATH_MAX):
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


# Packed QKV in-projection without the transposed .contiguous() copy: torch's F._in_projection_packed materializes
# proj.unflatten(-1, (3, E)).transpose(...).contiguous() (XProf: as_strided f32[4096,108,3,128,1] ~0.68 s per headline
# forward in the row stage, plus the K/V variant in the column stage). The same linear outputs are returned as chunk()
# views instead (identical values). Default off.
_FAST_INPROJ = _os.environ.get("TABICL_TPU_FAST_INPROJ", "") == "1"


def _install_fast_inproj() -> None:
    if not _FAST_INPROJ:
        return
    import torch.nn.functional as _F

    if getattr(_F, "_v5p_fast_inproj", False):
        return
    orig = _F._in_projection_packed

    def _in_projection_packed(q, k, v, w, b=None):  # noqa: ANN001
        E = q.size(-1)
        if k is v:
            if q is k:
                return _F.linear(q, w, b).chunk(3, dim=-1)
            w_q, w_kv = w.split([E, E * 2])
            b_q, b_kv = (None, None) if b is None else b.split([E, E * 2])
            k_p, v_p = _F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
            return _F.linear(q, w_q, b_q), k_p, v_p
        return orig(q, k, v, w, b)

    _F._in_projection_packed = _in_projection_packed
    _F._v5p_fast_inproj = True


# bf16 autocast for the column / row stages of the no-cache path (default off: those stages run fp32 activations).
_COL_AUTOCAST = _os.environ.get("TABICL_TPU_COL_AUTOCAST", "") == "1"
_ROW_AUTOCAST = _os.environ.get("TABICL_TPU_ROW_AUTOCAST", "") == "1"
# Column ISAB stack: compute the skip mask (columns whose values all equal skip_value) once per chunk instead of once per
# block. Skipped columns are re-filled with skip_value after every block, so the mask is the same at every block (a live
# column would need every one of its outputs to equal skip_value exactly). Default off.
_COL_SKIP_ONCE = _os.environ.get("TABICL_TPU_COL_SKIP_ONCE", "") == "1"


def _tf_col_skip_once(tf_col, src: Tensor, train_size) -> Tensor:  # noqa: ANN001
    blocks = list(tf_col.blocks)
    sv = blocks[0].skip_value
    skip_mask = (src == sv).all(dim=(-2, -1), keepdim=True)
    for block in blocks:
        out = block.induced_attention(src, train_size)
        src = torch.where(skip_mask, torch.full_like(out, sv), out)
    return src


def _skip_once_ok(tf_col) -> bool:  # noqa: ANN001
    blocks = list(getattr(tf_col, "blocks", []))
    return (_COL_SKIP_ONCE and bool(blocks) and not getattr(tf_col, "recompute", False)
            and all(hasattr(b, "induced_attention") and getattr(b, "skip_value", None) == blocks[0].skip_value
                    for b in blocks))


# Circular feature grouping with per-table feature counts (eager, before the compiled column chunks): upstream computes
# the gather indices (idx % rf + 2**i) % rf on the full [B, T, H] int64 grid (64-bit remainders are emulated on TPU;
# 62-221 ms per headline forward in XProf). The indices do not depend on the row, so compute them on [B, 1, H] and
# broadcast: same indices, same output. Default off.
_FAST_GROUP = _os.environ.get("TABICL_TPU_FAST_GROUP", "") == "1"


def _install_fast_group(col_emb) -> None:  # noqa: ANN001
    cls = type(col_emb)
    if not _FAST_GROUP or getattr(cls, "_v5p_fast_group", False) or not hasattr(cls, "feature_grouping"):
        return
    orig = cls.feature_grouping

    def feature_grouping(self, X: Tensor, real_features: Optional[Tensor] = None) -> Tensor:  # noqa: ANN001
        mode = "same" if self.feature_group is True else self.feature_group
        if not self.feature_group or mode != "same" or real_features is None or real_features.dim() == 0:
            return orig(self, X, real_features=real_features)
        B, T, H = X.shape
        idxs = torch.arange(H, dtype=torch.long, device=X.device)
        rf = real_features.view(B, 1, 1).long()
        base = idxs.view(1, 1, H) % rf  # [B, 1, H]
        return torch.stack([torch.gather(X, 2, ((base + 2 ** i) % rf).expand(B, T, H))
                            for i in range(self.feature_group_size)], dim=-1)

    cls.feature_grouping = feature_grouping
    cls._v5p_fast_group = True


def _stage_autocast(enabled: bool):
    """bf16 autocast when enabled, otherwise a no-op context (keeps the default traced graph unchanged)."""
    import contextlib

    return torch.autocast(device_type="tpu", dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


def active_knobs() -> dict[str, str]:
    """All tuning env knobs (for recording in result rows)."""
    names = (
        "TABICL_TPU_COL_PRECISION", "TABICL_TPU_ROW_PRECISION", "TABICL_TPU_ROW_BUCKET_STEP", "TABICL_TPU_COL_GROUP_TILE",
        "TABICL_TPU_ICL_KV_TRIM", "TABICL_TPU_ICL_KV_STEP", "TABICL_TPU_SPLASH_BQ", "TABICL_TPU_SPLASH_BKV",
        "TABICL_TPU_SPLASH_BKV_COMPUTE", "TABICL_TPU_SPLASH_MIN_KV", "TABICL_TPU_ROW_TILE", "TABICL_TPU_STAGE_TIMING",
        "TABICL_FAST_PREPROCESS", "TABICL_REUSE_MODEL", "TABICL_PREPROCESS_THREADS",
        "TABICL_TPU_COL_AUTOCAST", "TABICL_TPU_ROW_AUTOCAST", "TABICL_TPU_FEATURE_STEP",
        "TABICL_TPU_SDPA_MATH", "TABICL_TPU_SDPA_MATH_MAX_ELEMS", "TABICL_TPU_FAST_INPROJ",
        "TABICL_TPU_MHA", "TABICL_TPU_MHA_MAX_HEAD_DIM", "TABICL_TPU_MHA_PROJ",
        "TABICL_TPU_MHA_ROPE", "TABICL_TPU_MHA_HXQ", "TABICL_TPU_MHA_PACKED", "TABICL_TPU_COL_SKIP_ONCE", "TABICL_TPU_MHA_HXF", "TABICL_TPU_FAST_GROUP",
        "TABICL_PREPROCESS_PROCS",
    )
    return {n: _os.environ.get(n, "") for n in names if _os.environ.get(n, "")}


# Small-head attention layouts (tabicl_tpu_mha.py): head expansion / transposed head layout keep a wide axis on the
# TPU lanes instead of the 16-wide head dim (the upstream head split pads it to 128 lanes). Default off;
# TABICL_TPU_MHA=hx|thl|auto|<rules> (see tabicl_tpu_mha.py). KV-cache build/use keeps the upstream path.
def _install_mha() -> None:
    if not _os.environ.get("TABICL_TPU_MHA", "").strip():
        return
    from tabicl._model import attention as _att
    from tabicl._model import layers as _lay
    import tabicl_tpu_mha as _mha

    _mha.install(_att, _lay)


@dataclass
class BucketedKVCache:
    """Pure-tensor bucket-aligned KV cache on TPU HBM (`bf16` or `fp32`)."""

    col_keys: tuple[Tensor, ...]
    col_values: tuple[Tensor, ...]
    icl_keys: tuple[Tensor, ...]
    icl_values: tuple[Tensor, ...]
    real_batch: int
    padded_batch: int
    real_train_size: int
    padded_train_size: int
    real_features: int
    padded_features: int
    num_classes: int
    cache_dtype: torch.dtype

    def nbytes(self) -> int:
        total = 0
        for seq in (self.col_keys, self.col_values, self.icl_keys, self.icl_values):
            for t in seq:
                total += t.numel() * t.element_size()
        return total


class BucketedCompiledTabICL:
    """Manages `torch.compile(backend='tpu', dynamic=False, fullgraph=True)` kernels for a `TabICL` model."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool = True,
        kv_cache_dtype: torch.dtype = torch.bfloat16,
        max_fused_elements: int = 4_500_000,
        row_tile_size: int = int(_os.environ.get("TABICL_TPU_ROW_TILE", "4096") or 4096),
        col_group_tile: int = int(_os.environ.get("TABICL_TPU_COL_GROUP_TILE", "32"))
        if _os.environ.get("TABICL_TPU_COL_GROUP_TILE", "32").strip().isdigit() else 32,
    ) -> None:
        if _ICL_KV_TRIM:
            _install_icl_kv_trim()
        _install_splash_knobs()
        _install_sdpa_math()
        _install_fast_inproj()
        _install_mha()
        self.model = model.to(device).eval()
        if hasattr(self.model, "col_embedder"):
            _install_fast_group(self.model.col_embedder)
        self.device = device
        self.use_amp = bool(use_amp)
        self.kv_cache_dtype = kv_cache_dtype if use_amp else torch.float32
        if self.kv_cache_dtype == torch.float16:
            raise ValueError("fp16 is strictly forbidden on TorchTPU; use torch.bfloat16 or torch.float32.")
        self.max_fused_elements = int(max_fused_elements)
        self.row_tile_size = int(row_tile_size)
        self.col_group_tile = int(col_group_tile)

        torch._dynamo.config.recompile_limit = max(
            getattr(torch._dynamo.config, "recompile_limit", 8), 256
        )
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = max(
                getattr(torch._dynamo.config, "accumulated_recompile_limit", 64), 1024
            )

        self.compile_events: list[dict[str, Any]] = []
        self._seen_keys: set[tuple[Any, ...]] = set()

        self._nocache_fn = self._make_nocache_fn()
        self._nocache_compiled = torch.compile(
            self._nocache_fn, backend="tpu", dynamic=False, fullgraph=True
        )

        self._prefill_bfloat16_fn = self._make_prefill_fn(torch.bfloat16, icl_amp=True)
        self._prefill_float32_fn = self._make_prefill_fn(torch.float32, icl_amp=False)
        self._prefill_bfloat16_compiled = torch.compile(
            self._prefill_bfloat16_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._prefill_float32_compiled = torch.compile(
            self._prefill_float32_fn, backend="tpu", dynamic=False, fullgraph=True
        )

        self._query_fn = self._make_query_fn()
        self._query_compiled = torch.compile(
            self._query_fn, backend="tpu", dynamic=False, fullgraph=True
        )

        # Stage-tiled fullgraph=True kernels for ultra-large tables (100K-500K rows x 100-500 features)
        self._col_chunk_fn = self._make_col_chunk_fn()
        self._col_chunk_compiled = torch.compile(
            self._col_chunk_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._col_chunk_prefill_bfloat16_fn = self._make_col_chunk_prefill_fn(torch.bfloat16)
        self._col_chunk_prefill_float32_fn = self._make_col_chunk_prefill_fn(torch.float32)
        self._col_chunk_prefill_bfloat16_compiled = torch.compile(
            self._col_chunk_prefill_bfloat16_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._col_chunk_prefill_float32_compiled = torch.compile(
            self._col_chunk_prefill_float32_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._row_tile_fn = self._make_row_tile_fn()
        self._row_tile_compiled = torch.compile(
            self._row_tile_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._icl_only_fn = self._make_icl_only_fn()
        self._icl_only_compiled = torch.compile(
            self._icl_only_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._icl_prefill_bfloat16_fn = self._make_icl_prefill_fn(torch.bfloat16, icl_amp=True)
        self._icl_prefill_float32_fn = self._make_icl_prefill_fn(torch.float32, icl_amp=False)
        self._icl_prefill_bfloat16_compiled = torch.compile(
            self._icl_prefill_bfloat16_fn, backend="tpu", dynamic=False, fullgraph=True
        )
        self._icl_prefill_float32_compiled = torch.compile(
            self._icl_prefill_float32_fn, backend="tpu", dynamic=False, fullgraph=True
        )

    def _sync(self) -> None:
        try:
            torch.accelerator.synchronize(self.device)
        except Exception:
            torch.accelerator.synchronize()

    def _make_nocache_fn(self) -> Callable[..., Tensor]:
        model = self.model
        use_amp = self.use_amp
        fixed_nc = 0 if model.max_classes == 0 else model.max_classes
        col_ac = use_amp and _COL_AUTOCAST
        row_ac = use_amp and _ROW_AUTOCAST

        def _fn(X: Tensor, y_train: Tensor, d_vec: Tensor, train_sz: Tensor) -> Tensor:
            with _stage_autocast(col_ac):
                col_emb = model.col_embedder._train_forward(
                    X,
                    y_train=y_train,
                    d=d_vec,
                    embed_with_test=False,
                    num_classes=fixed_nc,
                    train_size_override=train_sz,
                )
            row_d = model._resolve_row_d(d_vec)
            with _stage_autocast(row_ac):
                reprs = model.row_interactor._train_forward(col_emb, d=row_d)
            if row_ac:
                reprs = reprs.float()
            if use_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    out = model.icl_predictor._icl_predictions(
                        reprs, y_train, train_size_override=train_sz
                    )
            else:
                out = model.icl_predictor._icl_predictions(
                    reprs, y_train, train_size_override=train_sz
                )
            return out.float()

        return _fn

    def _make_col_chunk_fn(self) -> Callable[..., Tensor]:
        col_emb = self.model.col_embedder
        col_ac = self.use_amp and _COL_AUTOCAST
        skip_once = _skip_once_ok(col_emb.tf_col)

        def _fn(feat_chunk: Tensor, y_chunk: Tensor, train_sz: Tensor) -> Tensor:
            with _stage_autocast(col_ac):
                src = col_emb.in_linear(feat_chunk)
                if col_emb.max_classes > 0:
                    y_e = col_emb.y_encoder(y_chunk.float())
                else:
                    y_e = col_emb.y_encoder(y_chunk.unsqueeze(-1))
                src = col_emb._add_train_target_emb(src, y_e, train_sz)
                if skip_once:
                    src = _tf_col_skip_once(col_emb.tf_col, src, train_sz)
                else:
                    src = col_emb.tf_col(src, train_size=train_sz)
                if col_emb.affine:
                    weights = col_emb.ln_w(col_emb.out_w(src))
                    biases = col_emb.ln_b(col_emb.out_b(src))
                    out_c = feat_chunk * weights + biases
                    return (out_c.float() if col_ac else out_c) + 0
                return (src.float() if col_ac else src) + 0

        return _fn

    def _make_col_chunk_prefill_fn(self, cache_dtype: torch.dtype) -> Callable[..., tuple[Tensor, ...]]:
        col_emb = self.model.col_embedder

        def _fn(feat_chunk: Tensor, y_chunk: Tensor, train_sz: Tensor) -> tuple[Tensor, ...]:
            src = col_emb.in_linear(feat_chunk)
            if col_emb.max_classes > 0:
                y_e = col_emb.y_encoder(y_chunk.float())
            else:
                y_e = col_emb.y_encoder(y_chunk.unsqueeze(-1))
            src = col_emb._add_train_target_emb(src, y_e, train_sz)
            ks: list[Tensor] = []
            vs: list[Tensor] = []
            for block in col_emb.tf_col.blocks:
                skip_mask = (src == block.skip_value).all(dim=(-2, -1), keepdim=True)
                ind_vec = block.ind_vectors.expand(*src.shape[:-2], block.num_inds, src.shape[-1])
                hidden = block.multihead_attn1(
                    ind_vec, src, src, src_len_override=train_sz, valid_kv_len=train_sz
                )
                out_b, k_b, v_b = block.multihead_attn2(src, hidden, hidden, need_kv=True)
                src = torch.where(skip_mask, torch.full_like(out_b, block.skip_value), out_b)
                ks.append(k_b.to(cache_dtype) + 0)
                vs.append(v_b.to(cache_dtype) + 0)
            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                src = feat_chunk * weights + biases
            return (src + 0, *ks, *vs)

        return _fn

    def _make_row_tile_fn(self) -> Callable[..., Tensor]:
        model = self.model
        row_ac = self.use_amp and _ROW_AUTOCAST

        def _fn(col_emb_tile: Tensor, d_vec: Tensor) -> Tensor:
            row_d = model._resolve_row_d(d_vec)
            if row_ac:
                with _stage_autocast(True):
                    r = model.row_interactor._train_forward(col_emb_tile, d=row_d)
                return r.float() + 0
            return model.row_interactor._train_forward(col_emb_tile, d=row_d) + 0

        return _fn

    def _make_icl_only_fn(self) -> Callable[..., Tensor]:
        model = self.model
        use_amp = self.use_amp

        def _fn(reprs: Tensor, y_train: Tensor, train_sz: Tensor) -> Tensor:
            if use_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    out = model.icl_predictor._icl_predictions(
                        reprs, y_train, train_size_override=train_sz
                    )
            else:
                out = model.icl_predictor._icl_predictions(
                    reprs, y_train, train_size_override=train_sz
                )
            return out.float() + 0

        return _fn

    def _make_icl_prefill_fn(self, cache_dtype: torch.dtype, icl_amp: bool) -> Callable[..., tuple[Tensor, ...]]:
        icl = self.model.icl_predictor

        def _fn(reprs: Tensor, y_train: Tensor, train_sz: Tensor) -> tuple[Tensor, ...]:
            icl_ks: list[Tensor] = []
            icl_vs: list[Tensor] = []
            if icl_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    if icl.max_classes > 0:
                        ry = icl.y_encoder(y_train.float())
                    else:
                        ry = icl.y_encoder(y_train.unsqueeze(-1))
                    r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                    for block in icl.tf_icl.blocks:
                        r_in, k_l, v_l = block(
                            q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                        )
                        icl_ks.append(k_l.to(cache_dtype) + 0)
                        icl_vs.append(v_l.to(cache_dtype) + 0)
            else:
                if icl.max_classes > 0:
                    ry = icl.y_encoder(y_train.float())
                else:
                    ry = icl.y_encoder(y_train.unsqueeze(-1))
                r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                for block in icl.tf_icl.blocks:
                    r_in, k_l, v_l = block(
                        q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                    )
                    icl_ks.append(k_l.to(cache_dtype) + 0)
                    icl_vs.append(v_l.to(cache_dtype) + 0)
            return (*icl_ks, *icl_vs)

        return _fn

    def _make_prefill_fn(self, cache_dtype: torch.dtype, icl_amp: bool) -> Callable[..., tuple[Tensor, ...]]:
        model = self.model
        col_emb = model.col_embedder
        icl = model.icl_predictor

        def _fn(X_tr: Tensor, y_tr: Tensor, d_vec: Tensor, train_sz: Tensor) -> tuple[Tensor, ...]:
            x_grp = col_emb.feature_grouping(X_tr, real_features=d_vec)
            if col_emb.reserve_cls_tokens > 0:
                x_grp = F.pad(x_grp, (0, 0, col_emb.reserve_cls_tokens, 0), value=-100.0)
            features = x_grp.transpose(1, 2)
            y_exp = y_tr.unsqueeze(1).expand(-1, features.shape[1], -1)

            src = col_emb.in_linear(features)
            if col_emb.max_classes > 0:
                y_e = col_emb.y_encoder(y_exp.float())
            else:
                y_e = col_emb.y_encoder(y_exp.unsqueeze(-1))
            src = col_emb._add_train_target_emb(src, y_e, train_sz)

            col_ks: list[Tensor] = []
            col_vs: list[Tensor] = []
            for block in col_emb.tf_col.blocks:
                skip_mask = (src == block.skip_value).all(dim=(-2, -1), keepdim=True)
                ind_vec = block.ind_vectors.expand(*src.shape[:-2], block.num_inds, src.shape[-1])
                hidden = block.multihead_attn1(
                    ind_vec, src, src, src_len_override=train_sz, valid_kv_len=train_sz
                )
                out_b, k_b, v_b = block.multihead_attn2(src, hidden, hidden, need_kv=True)
                src = torch.where(skip_mask, torch.full_like(out_b, block.skip_value), out_b)
                col_ks.append(k_b.to(cache_dtype) + 0)
                vs_val = v_b.to(cache_dtype) + 0
                col_vs.append(vs_val)

            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                src = features * weights + biases

            reprs = model.row_interactor._train_forward(
                src.transpose(1, 2), d=model._resolve_row_d(d_vec)
            )

            icl_ks: list[Tensor] = []
            icl_vs: list[Tensor] = []
            if icl_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    if icl.max_classes > 0:
                        ry = icl.y_encoder(y_tr.float())
                    else:
                        ry = icl.y_encoder(y_tr.unsqueeze(-1))
                    r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                    for block in icl.tf_icl.blocks:
                        r_in, k_l, v_l = block(
                            q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                        )
                        icl_ks.append(k_l.to(cache_dtype) + 0)
                        icl_vs.append(v_l.to(cache_dtype) + 0)
            else:
                if icl.max_classes > 0:
                    ry = icl.y_encoder(y_tr.float())
                else:
                    ry = icl.y_encoder(y_tr.unsqueeze(-1))
                r_in = icl._add_train_target_repr(reprs, ry, train_sz)
                for block in icl.tf_icl.blocks:
                    r_in, k_l, v_l = block(
                        q=r_in, train_size=train_sz, rope=icl.tf_icl.rope, need_kv=True
                    )
                    icl_ks.append(k_l.to(cache_dtype) + 0)
                    icl_vs.append(v_l.to(cache_dtype) + 0)

            return (*col_ks, *col_vs, *icl_ks, *icl_vs)

        return _fn

    def _make_query_fn(self) -> Callable[..., Tensor]:
        model = self.model
        col_emb = model.col_embedder
        icl = model.icl_predictor
        use_amp = self.use_amp
        n_col = len(col_emb.tf_col.blocks)
        n_icl = len(icl.tf_icl.blocks)

        def _fn(X_te: Tensor, d_vec: Tensor, train_sz: Tensor, *flat_kv: Tensor) -> Tensor:
            col_ks = flat_kv[:n_col]
            col_vs = flat_kv[n_col : 2 * n_col]
            icl_ks = flat_kv[2 * n_col : 2 * n_col + n_icl]
            icl_vs = flat_kv[2 * n_col + n_icl : 2 * n_col + 2 * n_icl]

            x_grp = col_emb.feature_grouping(X_te, real_features=d_vec)
            if col_emb.reserve_cls_tokens > 0:
                x_grp = F.pad(x_grp, (0, 0, col_emb.reserve_cls_tokens, 0), value=-100.0)
            features = x_grp.transpose(1, 2)

            src = col_emb.in_linear(features)
            for idx, block in enumerate(col_emb.tf_col.blocks):
                skip_mask = (src == block.skip_value).all(dim=(-2, -1), keepdim=True)
                entry = KVCacheEntry(key=col_ks[idx], value=col_vs[idx])
                out_b = block.multihead_attn2(src, cached_kv=entry)
                src = torch.where(skip_mask, torch.full_like(out_b, block.skip_value), out_b)

            if col_emb.affine:
                weights = col_emb.ln_w(col_emb.out_w(src))
                biases = col_emb.ln_b(col_emb.out_b(src))
                src = features * weights + biases

            reprs = model.row_interactor._train_forward(
                src.transpose(1, 2), d=model._resolve_row_d(d_vec)
            )

            if use_amp:
                with torch.autocast(device_type="tpu", dtype=torch.bfloat16):
                    r_q = reprs
                    for idx, block in enumerate(icl.tf_icl.blocks):
                        entry = KVCacheEntry(key=icl_ks[idx], value=icl_vs[idx])
                        r_q = block(
                            q=r_q,
                            rope=icl.tf_icl.rope,
                            cached_kv=entry,
                            src_len_override=train_sz,
                            valid_kv_len=train_sz,
                        )
                    if icl.norm_first:
                        r_q = icl.ln(r_q)
                    out = icl.decoder(r_q)
            else:
                r_q = reprs
                for idx, block in enumerate(icl.tf_icl.blocks):
                    entry = KVCacheEntry(key=icl_ks[idx], value=icl_vs[idx])
                    r_q = block(
                        q=r_q,
                        rope=icl.tf_icl.rope,
                        cached_kv=entry,
                        src_len_override=train_sz,
                        valid_kv_len=train_sz,
                    )
                if icl.norm_first:
                    r_q = icl.ln(r_q)
                out = icl.decoder(r_q)
            return out.float() + 0

        return _fn

    def explain_all(self, B: int = 4, T: int = 1024, H: int = 32, T_test: int = 256) -> dict[str, Any]:
        """Run `torch._dynamo.explain` on nocache, prefill, and query paths and return graph/break counts."""
        results: dict[str, Any] = {}
        with torch.no_grad():
            X = torch.randn(B, T, H, device=self.device, dtype=torch.float32)
            y = torch.zeros(B, T, device=self.device, dtype=torch.float32)
            d_vec = torch.full((B,), max(1, H - 4), device=self.device, dtype=torch.long)
            tsz = torch.tensor(int(T * 0.75), device=self.device, dtype=torch.long)

            exp_nc = torch._dynamo.explain(self._nocache_fn)(X, y, d_vec, tsz)
            results["nocache"] = {
                "shape": (B, T, H),
                "graph_count": int(exp_nc.graph_count),
                "graph_break_count": int(exp_nc.graph_break_count),
                "break_reasons": [str(r) for r in exp_nc.break_reasons],
            }

            pf_fn = (
                self._prefill_bfloat16_fn
                if (self.use_amp and self.kv_cache_dtype == torch.bfloat16)
                else self._prefill_float32_fn
            )
            exp_pf = torch._dynamo.explain(pf_fn)(X, y, d_vec, tsz)
            results["kv_prefill"] = {
                "shape": (B, T, H),
                "graph_count": int(exp_pf.graph_count),
                "graph_break_count": int(exp_pf.graph_break_count),
                "break_reasons": [str(r) for r in exp_pf.break_reasons],
            }

            flat_kv = pf_fn(X, y, d_vec, tsz)
            X_te = torch.randn(B, T_test, H, device=self.device, dtype=torch.float32)
            exp_q = torch._dynamo.explain(self._query_fn)(X_te, d_vec, tsz, *flat_kv)
            results["kv_query"] = {
                "shape": (B, T_test, H),
                "graph_count": int(exp_q.graph_count),
                "graph_break_count": int(exp_q.graph_break_count),
                "break_reasons": [str(r) for r in exp_q.break_reasons],
            }
        return results

    def _run_stage_tiled_col_row(
        self,
        X_bkt: Tensor,
        y_bkt: Tensor,
        d_vec: Tensor,
        train_sz: Tensor,
        store_col_cache: bool = False,
        cache_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Compute `(B, T, 512)` row representations for massive `(B, T, H)` using fixed compiled stage tiles."""
        import os

        multi_host_stage = int(os.environ.get("WORLD_SIZE", "1")) > 8
        col_emb = self.model.col_embedder
        B, T, H = X_bkt.shape
        C = col_emb.reserve_cls_tokens

        # Micro-batch B=1 per step when T >= 32768 to cap XLA static HBM well below 16 GiB even up to 500K x 500
        b_step = 1 if T >= 32768 else min(B, 2)
        reprs_list: list[Tensor] = []
        col_k_batches: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]
        col_v_batches: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]

        prev_prec = torch.get_float32_matmul_precision()
        try:
            for b0 in range(0, B, b_step):
                if multi_host_stage:
                    xb_cpu = X_bkt[b0 : b0 + b_step].contiguous()
                    yb_cpu = y_bkt[b0 : b0 + b_step].contiguous()
                    db_cpu = d_vec[b0 : b0 + b_step].contiguous()
                    db = db_cpu.to(self.device)
                    x_grp = col_emb.feature_grouping(xb_cpu, real_features=db_cpu)
                    if C > 0:
                        x_grp = F.pad(x_grp, (0, 0, C, 0), value=-100.0)
                    features_cpu = x_grp.transpose(1, 2).contiguous()
                    del x_grp
                    HC = features_cpu.shape[1]
                    max_g = 8 if T >= 262144 else (16 if T >= 131072 else _auto_col_group_tile(HC, self.col_group_tile))
                    g_tile = min(max_g, HC)
                    HC_pad = int(math.ceil(HC / g_tile) * g_tile)
                    if HC_pad > HC:
                        features_cpu = F.pad(features_cpu, (0, 0, 0, 0, 0, HC_pad - HC), value=-100.0).contiguous()
                else:
                    xb = X_bkt[b0 : b0 + b_step]
                    yb = y_bkt[b0 : b0 + b_step]
                    db = d_vec[b0 : b0 + b_step]

                    x_grp = col_emb.feature_grouping(xb, real_features=db)  # (bs, T, H, 3)
                    if C > 0:
                        x_grp = F.pad(x_grp, (0, 0, C, 0), value=-100.0)
                    features = x_grp.transpose(1, 2).contiguous()  # (bs, H+C, T, 3)
                    del x_grp

                    HC = features.shape[1]
                    max_g = 8 if T >= 262144 else (16 if T >= 131072 else _auto_col_group_tile(HC, self.col_group_tile))
                    g_tile = min(max_g, HC)
                    HC_pad = int(math.ceil(HC / g_tile) * g_tile)
                    if HC_pad > HC:
                        features = F.pad(features, (0, 0, 0, 0, 0, HC_pad - HC), value=-100.0)

                if _STAGE_TIMING:
                    self._sync()
                    _t_col0 = time.perf_counter()
                torch.set_float32_matmul_precision(_env_prec("TABICL_TPU_COL_PRECISION", "highest"))
                col_chunks: list[Tensor] = []
                k_chunks: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]
                v_chunks: list[list[Tensor]] = [[] for _ in range(len(col_emb.tf_col.blocks))]

                ultra_cells = T * HC
                use_bf16_buf = self.use_amp and (ultra_cells > 25_000_000)
                stage_cpu = (ultra_cells > 100_000_000) or multi_host_stage

                for g0 in range(0, HC_pad, g_tile):
                    if multi_host_stage:
                        f_sub = features_cpu[:, g0 : g0 + g_tile].contiguous().to(self.device)
                        y_sub = yb_cpu.unsqueeze(1).expand(-1, g_tile, -1).contiguous().to(self.device)
                    else:
                        f_sub = features[:, g0 : g0 + g_tile].contiguous()
                        y_sub = yb.unsqueeze(1).expand(-1, g_tile, -1).contiguous()
                    if store_col_cache:
                        fn_pf = (
                            self._col_chunk_prefill_bfloat16_compiled
                            if cache_dtype == torch.bfloat16
                            else self._col_chunk_prefill_float32_compiled
                        )
                        res = fn_pf(f_sub, y_sub, train_sz)
                        if stage_cpu:
                            r0_cpu = res[0].cpu()
                            col_chunks.append(r0_cpu.to(torch.bfloat16) if use_bf16_buf else r0_cpu)
                        else:
                            col_chunks.append(res[0].to(torch.bfloat16) if use_bf16_buf else res[0])
                        n_blk = len(col_emb.tf_col.blocks)
                        for bi in range(n_blk):
                            k_chunks[bi].append(res[1 + bi].cpu() if multi_host_stage else res[1 + bi])
                            v_chunks[bi].append(res[1 + n_blk + bi].cpu() if multi_host_stage else res[1 + n_blk + bi])
                    else:
                        ch_raw = self._col_chunk_compiled(f_sub, y_sub, train_sz)
                        if stage_cpu:
                            ch_cpu = ch_raw.cpu()
                            col_chunks.append(ch_cpu.to(torch.bfloat16) if use_bf16_buf else ch_cpu)
                        else:
                            col_chunks.append(ch_raw.to(torch.bfloat16) if use_bf16_buf else ch_raw)
                if multi_host_stage:
                    del features_cpu
                else:
                    del features

                if store_col_cache:
                    for bi in range(len(col_emb.tf_col.blocks)):
                        col_k_batches[bi].append(torch.cat(k_chunks[bi], dim=1)[:, :HC].contiguous())
                        col_v_batches[bi].append(torch.cat(v_chunks[bi], dim=1)[:, :HC].contiguous())

                if _STAGE_TIMING:
                    self._sync()
                    _t_row0 = time.perf_counter()
                    _stage_add("col_ms", (_t_row0 - _t_col0) * 1000.0)
                torch.set_float32_matmul_precision(_env_prec("TABICL_TPU_ROW_PRECISION", "high" if self.use_amp else "highest"))
                r_tile = min(self.row_tile_size, T)
                row_outs: list[Tensor] = []
                for t0 in range(0, T, r_tile):
                    if stage_cpu:
                        c_sub = (
                            torch.cat([ch[:, :, t0 : t0 + r_tile, :] for ch in col_chunks], dim=1)[:, :HC]
                            .transpose(1, 2)
                            .to(dtype=torch.float32)
                            .contiguous()
                            .to(self.device)
                        )
                    else:
                        tile_slices = [ch[:, :, t0 : t0 + r_tile, :].float() for ch in col_chunks]
                        c_sub = torch.cat(tile_slices, dim=1)[:, :HC].transpose(1, 2).contiguous()
                        del tile_slices
                    ro = self._row_tile_compiled(c_sub, db)
                    row_outs.append(ro.cpu() if multi_host_stage else ro)
                    del c_sub
                del col_chunks
                reprs_list.append(torch.cat(row_outs, dim=1))
                if _STAGE_TIMING:
                    self._sync()
                    _stage_add("row_ms", (time.perf_counter() - _t_row0) * 1000.0)
        finally:
            torch.set_float32_matmul_precision(prev_prec)

        reprs = torch.cat(reprs_list, dim=0).contiguous()
        if store_col_cache:
            col_ks = tuple(torch.cat(col_k_batches[bi], dim=0).contiguous() for bi in range(len(col_emb.tf_col.blocks)))
            col_vs = tuple(torch.cat(col_v_batches[bi], dim=0).contiguous() for bi in range(len(col_emb.tf_col.blocks)))
            return reprs, col_ks, col_vs
        return reprs, (), ()

    def forward_nocache_bucketed(
        self,
        X: Tensor,
        y_train: Tensor,
        num_classes: int = 0,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
    ) -> tuple[Tensor, float]:
        """Run bucket-padded compiled forward (`kv_cache=False`) and return `(out_real, compile_s)`."""
        if (self.model.max_classes > 0 and num_classes > self.model.max_classes) or (
            not self.model.col_embedder.feature_group
        ):
            with torch.no_grad():
                out_fb = self.model(
                    X.to(self.device, dtype=torch.float32),
                    y_train.to(self.device, dtype=torch.float32),
                    return_logits=return_logits,
                    softmax_temperature=softmax_temperature,
                    num_classes=num_classes if num_classes > 0 else None,
                )
                self._sync()
            return out_fb.float(), 0.0

        import os

        multi_host_stage = int(os.environ.get("WORLD_SIZE", "1")) > 8

        B_real, T_real, H_real = X.shape
        train_real = y_train.shape[1]
        test_real = T_real - train_real

        B_bkt = pick_bucket(B_real, BATCH_BUCKETS)
        T_bkt = pick_row_bucket(T_real)
        H_bkt = pick_feature_bucket(H_real)

        pad_dev = torch.device("cpu") if multi_host_stage else self.device
        X_bkt = torch.zeros((B_bkt, T_bkt, H_bkt), device=pad_dev, dtype=torch.float32)
        X_bkt[:B_real, :T_real, :H_real] = X.to(pad_dev, dtype=torch.float32)

        y_bkt = torch.zeros((B_bkt, T_bkt), device=pad_dev, dtype=torch.float32)
        y_bkt[:B_real, :train_real] = y_train.to(pad_dev, dtype=torch.float32)
        if B_bkt > B_real:
            X_bkt[B_real:] = X_bkt[0:1]
            y_bkt[B_real:] = y_bkt[0:1]

        d_vec = torch.full((B_bkt,), H_real, device=pad_dev, dtype=torch.long)
        train_sz = torch.tensor(train_real, dtype=torch.long).to(self.device)

        use_tiled = (B_bkt * T_bkt * (H_bkt + 4)) > self.max_fused_elements
        kv_len = _icl_kv_len(train_real, T_bkt)
        key = ("nocache_tiled" if use_tiled else "nocache_fused", B_bkt, T_bkt, H_bkt, self.use_amp, kv_len)
        is_first = key not in self._seen_keys

        global _ICL_KV_STATIC_LEN
        prev_kv_len = _ICL_KV_STATIC_LEN
        _ICL_KV_STATIC_LEN = kv_len
        try:
          with torch.no_grad():
            t0 = time.perf_counter()
            if not use_tiled:
                prev_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision(_env_prec("TABICL_TPU_COL_PRECISION", "highest"))
                try:
                    if multi_host_stage:
                        raw_full = self._nocache_compiled(
                            X_bkt.to(self.device), y_bkt.to(self.device), d_vec.to(self.device), train_sz
                        ).cpu()
                    else:
                        raw_full = self._nocache_compiled(X_bkt, y_bkt, d_vec, train_sz)
                finally:
                    torch.set_float32_matmul_precision(prev_prec)
                if _STAGE_TIMING:
                    self._sync()
                    _stage_add("fused_ms", (time.perf_counter() - t0) * 1000.0)
            else:
                reprs, _, _ = self._run_stage_tiled_col_row(X_bkt, y_bkt, d_vec, train_sz, store_col_cache=False)
                if _STAGE_TIMING:
                    self._sync()
                    _t_icl0 = time.perf_counter()
                icl_b_step = 1 if T_bkt >= _ICL_BSTEP1_MIN_T else (min(B_bkt, 2) if T_bkt >= 32768 else B_bkt)
                if icl_b_step == B_bkt:
                    if multi_host_stage:
                        raw_full = self._icl_only_compiled(
                            reprs.to(self.device), y_bkt.to(self.device), train_sz
                        ).cpu()
                    else:
                        raw_full = self._icl_only_compiled(reprs, y_bkt, train_sz)
                else:
                    parts = [
                        (
                            self._icl_only_compiled(
                                reprs[i : i + icl_b_step].contiguous().to(self.device),
                                y_bkt[i : i + icl_b_step].contiguous().to(self.device),
                                train_sz,
                            ).cpu()
                            if multi_host_stage
                            else self._icl_only_compiled(reprs[i : i + icl_b_step], y_bkt[i : i + icl_b_step], train_sz)
                        )
                        for i in range(0, B_bkt, icl_b_step)
                    ]
                    raw_full = torch.cat(parts, dim=0)
                if _STAGE_TIMING:
                    self._sync()
                    _stage_add("icl_ms", (time.perf_counter() - _t_icl0) * 1000.0)

            out = raw_full[:B_real, train_real : train_real + test_real]
            if self.model.max_classes > 0:
                out = out[..., :num_classes]
                if not return_logits:
                    out = torch.softmax(out / softmax_temperature, dim=-1)
            self._sync()
            elapsed = time.perf_counter() - t0
        finally:
            _ICL_KV_STATIC_LEN = prev_kv_len

        compile_s = 0.0
        if is_first:
            self._seen_keys.add(key)
            compile_s = elapsed
            self.compile_events.append({"key": key, "first_call_s": round(compile_s, 4)})

        return out, compile_s

    def prefill_kv_bucketed(
        self,
        X_train: Tensor,
        y_train: Tensor,
        num_classes: int = 0,
        cache_dtype: Optional[torch.dtype] = None,
    ) -> tuple[BucketedKVCache, float]:
        """Compile & execute bucket-padded KV-cache prefill (`store_cache=True`)."""
        import os

        multi_host_stage = int(os.environ.get("WORLD_SIZE", "1")) > 8

        B_real, T_real, H_real = X_train.shape
        B_bkt = pick_bucket(B_real, BATCH_BUCKETS)
        T_bkt = pick_row_bucket(T_real)
        H_bkt = pick_feature_bucket(H_real)

        eff_dtype = cache_dtype if cache_dtype is not None else self.kv_cache_dtype
        if eff_dtype == torch.float16:
            raise ValueError("fp16 KV cache is forbidden on TorchTPU.")

        pad_dev = torch.device("cpu") if multi_host_stage else self.device
        X_bkt = torch.zeros((B_bkt, T_bkt, H_bkt), device=pad_dev, dtype=torch.float32)
        X_bkt[:B_real, :T_real, :H_real] = X_train.to(pad_dev, dtype=torch.float32)

        y_bkt = torch.zeros((B_bkt, T_bkt), device=pad_dev, dtype=torch.float32)
        y_bkt[:B_real, :T_real] = y_train.to(pad_dev, dtype=torch.float32)
        if B_bkt > B_real:
            X_bkt[B_real:] = X_bkt[0:1]
            y_bkt[B_real:] = y_bkt[0:1]

        d_vec = torch.full((B_bkt,), H_real, device=pad_dev, dtype=torch.long)
        train_sz = torch.tensor(T_real, dtype=torch.long).to(self.device)

        use_tiled = (B_bkt * T_bkt * (H_bkt + 4)) > self.max_fused_elements
        key = ("prefill_tiled" if use_tiled else "prefill_fused", B_bkt, T_bkt, H_bkt, str(eff_dtype), self.use_amp)
        is_first = key not in self._seen_keys

        n_col = len(self.model.col_embedder.tf_col.blocks)
        n_icl = len(self.model.icl_predictor.tf_icl.blocks)

        with torch.no_grad():
            t0 = time.perf_counter()
            if not use_tiled:
                prev_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision(_env_prec("TABICL_TPU_COL_PRECISION", "highest"))
                try:
                    pf_fn = (
                        self._prefill_bfloat16_compiled
                        if (self.use_amp and eff_dtype == torch.bfloat16)
                        else self._prefill_float32_compiled
                    )
                    if multi_host_stage:
                        flat_kv = pf_fn(X_bkt.to(self.device), y_bkt.to(self.device), d_vec.to(self.device), train_sz)
                        flat_kv = tuple(t.cpu() for t in flat_kv)
                    else:
                        flat_kv = pf_fn(X_bkt, y_bkt, d_vec, train_sz)
                finally:
                    torch.set_float32_matmul_precision(prev_prec)
                col_ks = tuple(flat_kv[:n_col])
                col_vs = tuple(flat_kv[n_col : 2 * n_col])
                icl_ks = tuple(flat_kv[2 * n_col : 2 * n_col + n_icl])
                icl_vs = tuple(flat_kv[2 * n_col + n_icl : 2 * n_col + 2 * n_icl])
            else:
                reprs, col_ks, col_vs = self._run_stage_tiled_col_row(
                    X_bkt, y_bkt, d_vec, train_sz, store_col_cache=True, cache_dtype=eff_dtype
                )
                icl_pf_fn = (
                    self._icl_prefill_bfloat16_compiled
                    if (self.use_amp and eff_dtype == torch.bfloat16)
                    else self._icl_prefill_float32_compiled
                )
                icl_b_step = 1 if T_bkt >= _ICL_BSTEP1_MIN_T else (min(B_bkt, 2) if T_bkt >= 32768 else B_bkt)
                if icl_b_step == B_bkt:
                    if multi_host_stage:
                        icl_flat = icl_pf_fn(reprs.to(self.device), y_bkt.to(self.device), train_sz)
                        icl_flat = tuple(t.cpu() for t in icl_flat)
                    else:
                        icl_flat = icl_pf_fn(reprs, y_bkt, train_sz)
                    icl_ks = tuple(icl_flat[:n_icl])
                    icl_vs = tuple(icl_flat[n_icl : 2 * n_icl])
                else:
                    ks_acc: list[list[Tensor]] = [[] for _ in range(n_icl)]
                    vs_acc: list[list[Tensor]] = [[] for _ in range(n_icl)]
                    for i in range(0, B_bkt, icl_b_step):
                        if multi_host_stage:
                            sub_flat = icl_pf_fn(
                                reprs[i : i + icl_b_step].contiguous().to(self.device),
                                y_bkt[i : i + icl_b_step].contiguous().to(self.device),
                                train_sz,
                            )
                        else:
                            sub_flat = icl_pf_fn(reprs[i : i + icl_b_step], y_bkt[i : i + icl_b_step], train_sz)
                        for li in range(n_icl):
                            ks_acc[li].append(sub_flat[li].cpu() if multi_host_stage else sub_flat[li])
                            vs_acc[li].append(sub_flat[n_icl + li].cpu() if multi_host_stage else sub_flat[n_icl + li])
                    icl_ks = tuple(torch.cat(ks_acc[li], dim=0).contiguous() for li in range(n_icl))
                    icl_vs = tuple(torch.cat(vs_acc[li], dim=0).contiguous() for li in range(n_icl))

                if multi_host_stage:
                    # Promote finished full-batch KV cache buffers once onto TPU for fast repeated queries when B_bkt == icl_b_step_q
                    if T_bkt < 65536:
                        col_ks = tuple(t.to(self.device) for t in col_ks)
                        col_vs = tuple(t.to(self.device) for t in col_vs)
                        icl_ks = tuple(t.to(self.device) for t in icl_ks)
                        icl_vs = tuple(t.to(self.device) for t in icl_vs)
            if multi_host_stage and not use_tiled and T_bkt < 65536:
                col_ks = tuple(t.to(self.device) for t in col_ks)
                col_vs = tuple(t.to(self.device) for t in col_vs)
                icl_ks = tuple(t.to(self.device) for t in icl_ks)
                icl_vs = tuple(t.to(self.device) for t in icl_vs)

            self._sync()
            elapsed = time.perf_counter() - t0

        compile_s = 0.0
        if is_first:
            self._seen_keys.add(key)
            compile_s = elapsed
            self.compile_events.append({"key": key, "first_call_s": round(compile_s, 4)})

        cache = BucketedKVCache(
            col_keys=col_ks,
            col_values=col_vs,
            icl_keys=icl_ks,
            icl_values=icl_vs,
            real_batch=B_real,
            padded_batch=B_bkt,
            real_train_size=T_real,
            padded_train_size=T_bkt,
            real_features=H_real,
            padded_features=H_bkt,
            num_classes=num_classes,
            cache_dtype=eff_dtype,
        )
        return cache, compile_s

    def query_kv_bucketed(
        self,
        X_test: Tensor,
        cache: BucketedKVCache,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
    ) -> tuple[Tensor, float]:
        """Compile & execute bucket-padded KV-cache query (`use_cache=True`)."""
        import os

        multi_host_stage = int(os.environ.get("WORLD_SIZE", "1")) > 8

        B_real, T_te_real, H_real = X_test.shape
        B_bkt = cache.padded_batch
        H_bkt = cache.padded_features
        T_te_bkt = pick_bucket(T_te_real, TEST_ROW_BUCKETS)

        pad_dev = torch.device("cpu") if multi_host_stage else self.device
        X_te_bkt = torch.zeros((B_bkt, T_te_bkt, H_bkt), device=pad_dev, dtype=torch.float32)
        X_te_bkt[:B_real, :T_te_real, :H_real] = X_test.to(pad_dev, dtype=torch.float32)
        if B_bkt > B_real:
            X_te_bkt[B_real:] = X_te_bkt[0:1]

        d_vec = torch.full((B_bkt,), cache.real_features, device=pad_dev, dtype=torch.long)
        train_sz = torch.tensor(cache.real_train_size, dtype=torch.long).to(self.device)

        key = (
            "query_fused",
            B_bkt,
            T_te_bkt,
            cache.padded_train_size,
            H_bkt,
            str(cache.cache_dtype),
            self.use_amp,
        )
        is_first = key not in self._seen_keys

        with torch.no_grad():
            t0 = time.perf_counter()
            prev_prec = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision(_env_prec("TABICL_TPU_COL_PRECISION", "highest"))
            try:
                icl_b_step = (
                    1
                    if (cache.padded_train_size >= _ICL_BSTEP1_MIN_T or B_bkt * T_te_bkt * (H_bkt + 4) > self.max_fused_elements)
                    else B_bkt
                )
                if icl_b_step == B_bkt:
                    raw_out = self._query_compiled(
                        X_te_bkt.to(self.device) if multi_host_stage else X_te_bkt,
                        d_vec.to(self.device) if multi_host_stage else d_vec,
                        train_sz,
                        *[t.to(self.device) if t.device.type == "cpu" else t for t in cache.col_keys],
                        *[t.to(self.device) if t.device.type == "cpu" else t for t in cache.col_values],
                        *[t.to(self.device) if t.device.type == "cpu" else t for t in cache.icl_keys],
                        *[t.to(self.device) if t.device.type == "cpu" else t for t in cache.icl_values],
                    )
                    if multi_host_stage:
                        raw_out = raw_out.cpu()
                else:
                    parts = []
                    for i in range(0, B_bkt, icl_b_step):
                        if multi_host_stage:
                            ck = [t[i : i + icl_b_step].contiguous().to(self.device) for t in cache.col_keys]
                            cv = [t[i : i + icl_b_step].contiguous().to(self.device) for t in cache.col_values]
                            ik = [t[i : i + icl_b_step].contiguous().to(self.device) for t in cache.icl_keys]
                            iv = [t[i : i + icl_b_step].contiguous().to(self.device) for t in cache.icl_values]
                            q_res = self._query_compiled(
                                X_te_bkt[i : i + icl_b_step].contiguous().to(self.device),
                                d_vec[i : i + icl_b_step].contiguous().to(self.device),
                                train_sz,
                                *ck,
                                *cv,
                                *ik,
                                *iv,
                            )
                            parts.append(q_res.cpu())
                        else:
                            ck = [t[i : i + icl_b_step] for t in cache.col_keys]
                            cv = [t[i : i + icl_b_step] for t in cache.col_values]
                            ik = [t[i : i + icl_b_step] for t in cache.icl_keys]
                            iv = [t[i : i + icl_b_step] for t in cache.icl_values]
                            q_res = self._query_compiled(
                                X_te_bkt[i : i + icl_b_step],
                                d_vec[i : i + icl_b_step],
                                train_sz,
                                *ck,
                                *cv,
                                *ik,
                                *iv,
                            )
                            parts.append(q_res)
                    raw_out = torch.cat(parts, dim=0)
            finally:
                torch.set_float32_matmul_precision(prev_prec)

            out = raw_out[:B_real, :T_te_real]
            if self.model.max_classes > 0:
                out = out[..., : cache.num_classes]
                if not return_logits:
                    out = torch.softmax(out / softmax_temperature, dim=-1)
            self._sync()
            elapsed = time.perf_counter() - t0

        compile_s = 0.0
        if is_first:
            self._seen_keys.add(key)
            compile_s = elapsed
            self.compile_events.append({"key": key, "first_call_s": round(compile_s, 4)})

        return out, compile_s


def attach_compiled_runner(
    estimator: Any,
    runner: Optional[BucketedCompiledTabICL] = None,
) -> BucketedCompiledTabICL:
    """Attach `BucketedCompiledTabICL` to a fitted or about-to-be-fitted `TabICLClassifier` / `TabICLRegressor`."""
    import types
    from tabicl import TabICLClassifier, TabICLRegressor

    is_cls = isinstance(estimator, TabICLClassifier)
    is_reg = isinstance(estimator, TabICLRegressor)
    if not (is_cls or is_reg):
        raise TypeError(f"Expected TabICLClassifier or TabICLRegressor, got {type(estimator)}")

    # Optional CPU-side fit speedups (default off; numerics unchanged):
    if _os.environ.get("TABICL_FAST_PREPROCESS", "") == "1":
        try:
            import tabicl_fast_preprocess

            tabicl_fast_preprocess.install()
        except Exception as exc:  # noqa: BLE001
            print(f"[tabicl_tpu_bucketed] fast preprocess not installed: {exc}", flush=True)
    if _os.environ.get("TABICL_REUSE_MODEL", "") == "1":
        orig_load_model = estimator._load_model

        def _load_model_reuse(self: Any) -> None:
            key = (str(getattr(self, "model_path", None)), str(getattr(self, "checkpoint_version", None)))
            if getattr(self, "model_", None) is not None and getattr(self, "_v5p_loaded_key_", None) == key:
                return  # same pretrained checkpoint already loaded (and already on device)
            orig_load_model()
            self._v5p_loaded_key_ = key

        estimator._load_model = types.MethodType(_load_model_reuse, estimator)

    orig_fit = estimator.fit

    def _ensure_runner(est: Any) -> BucketedCompiledTabICL:
        nonlocal runner
        use_amp, _ = est._resolve_amp_fa3()
        kv_dt = est._resolve_kv_cache_dtype() or (torch.bfloat16 if use_amp else torch.float32)
        if runner is None or runner.model is not est.model_ or runner.use_amp != use_amp:
            runner = BucketedCompiledTabICL(
                model=est.model_,
                device=est.device_,
                use_amp=use_amp,
                kv_cache_dtype=kv_dt,
            )
        else:
            runner.kv_cache_dtype = kv_dt
        est._compiled_runner_ = runner
        return runner

    if is_cls:

        def _cls_build_kv_cache(self: Any) -> None:
            r = _ensure_runner(self)
            train_data = self.ensemble_generator_.transform(X=None, mode="train")
            self.model_kv_cache_ = OrderedDict()
            cache_dt = self._resolve_kv_cache_dtype() or r.kv_cache_dtype
            for norm_method, (Xs, ys) in train_data.items():
                X_t = torch.from_numpy(Xs).float()
                y_t = torch.from_numpy(ys).float()
                cache_obj, _ = r.prefill_kv_bucketed(
                    X_t, y_t, num_classes=self.n_classes_, cache_dtype=cache_dt
                )
                self.model_kv_cache_[norm_method] = cache_obj

        def _cls_batch_forward(
            self: Any, Xs: np.ndarray, ys: np.ndarray, feature_shuffles: Optional[np.ndarray] = None
        ) -> np.ndarray:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            y_t = torch.from_numpy(ys).float()
            out, _ = r.forward_nocache_bucketed(
                X_t,
                y_t,
                num_classes=self.n_classes_,
                return_logits=self.average_logits,
                softmax_temperature=self.softmax_temperature,
            )
            return out.float().cpu().numpy()

        def _cls_batch_forward_with_cache(
            self: Any, Xs: np.ndarray, kv_cache: BucketedKVCache
        ) -> np.ndarray:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            out, _ = r.query_kv_bucketed(
                X_t,
                cache=kv_cache,
                return_logits=self.average_logits,
                softmax_temperature=self.softmax_temperature,
            )
            return out.float().cpu().numpy()

        estimator._build_kv_cache = types.MethodType(_cls_build_kv_cache, estimator)
        estimator._batch_forward = types.MethodType(_cls_batch_forward, estimator)
        estimator._batch_forward_with_cache = types.MethodType(_cls_batch_forward_with_cache, estimator)

    else:

        def _reg_quantiles_to_stats(
            model_obj: Any, raw_q: Tensor, output_type: str | list[str], alphas: Optional[list[float]]
        ) -> dict[str, np.ndarray] | np.ndarray:
            dist = model_obj.quantile_dist(raw_q.float())
            mono_q = dist.quantiles
            out_types = [output_type] if isinstance(output_type, str) else list(output_type)
            res: dict[str, np.ndarray] = {}
            if "mean" in out_types:
                res["mean"] = mono_q.mean(dim=-1).cpu().numpy()
            if "variance" in out_types:
                res["variance"] = mono_q.var(dim=-1).cpu().numpy()
            if "median" in out_types:
                med = dist.icdf(alpha=torch.tensor(0.5, device=mono_q.device, dtype=mono_q.dtype))
                res["median"] = med.cpu().numpy()
            if "quantiles" in out_types:
                lvls = alphas if alphas is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                qs = dist.icdf(alpha=torch.tensor(lvls, device=mono_q.device, dtype=mono_q.dtype))
                res["quantiles"] = qs.cpu().numpy()
            if "raw_quantiles" in out_types:
                res["raw_quantiles"] = mono_q.cpu().numpy()
            if len(out_types) == 1:
                return res[out_types[0]]
            return res

        def _reg_build_kv_cache(self: Any) -> None:
            r = _ensure_runner(self)
            train_data = self.ensemble_generator_.transform(X=None, mode="train")
            self.model_kv_cache_ = OrderedDict()
            cache_dt = self._resolve_kv_cache_dtype() or r.kv_cache_dtype
            for norm_method, (Xs, ys) in train_data.items():
                X_t = torch.from_numpy(Xs).float()
                y_t = torch.from_numpy(ys).float()
                cache_obj, _ = r.prefill_kv_bucketed(
                    X_t, y_t, num_classes=0, cache_dtype=cache_dt
                )
                self.model_kv_cache_[norm_method] = cache_obj

        def _reg_batch_forward(
            self: Any,
            Xs: np.ndarray,
            ys: np.ndarray,
            output_type: str | list[str] = "mean",
            alphas: Optional[list[float]] = None,
        ) -> np.ndarray | dict[str, np.ndarray]:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            y_t = torch.from_numpy(ys).float()
            raw_q, _ = r.forward_nocache_bucketed(X_t, y_t, num_classes=0)
            return _reg_quantiles_to_stats(self.model_, raw_q, output_type, alphas)

        def _reg_batch_forward_with_cache(
            self: Any,
            Xs: np.ndarray,
            kv_cache: BucketedKVCache,
            output_type: str | list[str] = "mean",
            alphas: Optional[list[float]] = None,
        ) -> np.ndarray | dict[str, np.ndarray]:
            r = _ensure_runner(self)
            X_t = torch.from_numpy(Xs).float()
            raw_q, _ = r.query_kv_bucketed(X_t, cache=kv_cache)
            return _reg_quantiles_to_stats(self.model_, raw_q, output_type, alphas)

        estimator._build_kv_cache = types.MethodType(_reg_build_kv_cache, estimator)
        estimator._batch_forward = types.MethodType(_reg_batch_forward, estimator)
        estimator._batch_forward_with_cache = types.MethodType(_reg_batch_forward_with_cache, estimator)

    if getattr(estimator, "model_", None) is not None and getattr(estimator, "device_", None) is not None:
        return _ensure_runner(estimator)

    def _wrapped_fit(self: Any, X: Any, y: Any) -> Any:
        out = orig_fit(X, y)
        _ensure_runner(self)
        return out

    estimator.fit = types.MethodType(_wrapped_fit, estimator)
    return runner  # type: ignore[return-value]

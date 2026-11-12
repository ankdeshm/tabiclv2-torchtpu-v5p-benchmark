"""Bit-exact parallel Yeo-Johnson fitting for TabICL's `power` preprocessing (CPU side of `fit`).

Why: at 45k x 100 the `power` normalizer (sklearn PowerTransformer, method="yeo-johnson") spends ~86% of
TabICLClassifier.fit in the per-column lambda search (100 columns x ~30 scipy-brent evaluations, each a vectorized
transform over all rows), one column at a time on one core.

How: lambdas for all non-constant columns are computed concurrently in a thread pool (numpy releases the GIL inside
the vectorized transform), with the SAME function (`_yeo_johnson_optimize`) on the SAME column data, so each lambda is
bit-identical to the serial result. sklearn's own `_fit` then runs unchanged with the optimizer swapped for a lookup
of the precomputed values (same order, constant columns skipped exactly as sklearn does), so the transformed data and
the fitted scaler are identical too.

Enable with `install()` (or TABICL_FAST_PREPROCESS=1 via tabicl_tpu_bucketed.attach_compiled_runner).
Threads: TABICL_PREPROCESS_THREADS (default min(32, cpu_count)).
Processes: TABICL_PREPROCESS_PROCS=N (> 1) uses a persistent process pool instead (the brent loop holds the GIL, so threads
stop scaling at ~4x). The column matrix goes to the workers through shared memory with the SAME memory layout, and each
worker calls the same sklearn routine on the same strided column view, so lambdas stay bit-identical.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from sklearn.preprocessing import PowerTransformer

try:  # sklearn >= 1.2
    from sklearn.preprocessing._data import _is_constant_feature
except Exception:  # noqa: BLE001
    _is_constant_feature = None


_POOL = None
_POOL_N = 0


def _n_procs() -> int:
    return int(os.environ.get("TABICL_PREPROCESS_PROCS", "0") or 0)


def _get_pool(n: int):
    """Persistent forkserver pool (workers import numpy/sklearn once; no fork of the TPU process)."""
    global _POOL, _POOL_N
    if _POOL is None or _POOL_N != n:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(["numpy", "scipy.optimize", "sklearn.preprocessing"])
        except Exception:  # noqa: BLE001
            pass
        _POOL = ProcessPoolExecutor(max_workers=n, mp_context=ctx)
        _POOL_N = n
    return _POOL


def _yj_worker(shm_name: str, shape: tuple, dtype: str, strides: tuple, cols: list) -> list:
    """Attach to the shared column matrix (same strides as the parent's Xc) and run sklearn's optimizer per column."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        Xc = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf, strides=strides)
        pt = PowerTransformer(method="yeo-johnson")
        with np.errstate(invalid="ignore"):
            out = [float(pt._yeo_johnson_optimize(Xc.T[i])) for i in cols]
        del Xc
        return out
    finally:
        shm.close()


class _HiddenMain:
    """Hide the parent's __main__ path while workers are spawned: forkserver children otherwise re-run the main script as
    __mp_main__ (for a TPU benchmark that means importing torch / torch_tpu in every CPU worker)."""

    def __enter__(self):
        import sys

        self._m = sys.modules.get("__main__")
        self._saved = {k: getattr(self._m, k) for k in ("__file__", "__spec__") if self._m is not None and hasattr(self._m, k)}
        if self._m is not None:
            if "__file__" in self._saved:
                del self._m.__file__
            self._m.__spec__ = None
        return self

    def __exit__(self, *exc):
        if self._m is not None:
            for k, v in self._saved.items():
                setattr(self._m, k, v)
        return False


def _lambdas_procs(Xc, todo: list, n: int) -> list:
    from multiprocessing import shared_memory

    nbytes = Xc.nbytes
    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
    try:
        # same layout as Xc (C or F order) so every column view has the same strides as in the parent
        order = "F" if (Xc.flags.f_contiguous and not Xc.flags.c_contiguous) else "C"
        Xs = np.ndarray(Xc.shape, dtype=Xc.dtype, buffer=shm.buf, order=order)
        Xs[...] = Xc
        if Xs.strides != Xc.strides:  # non-contiguous parent: fall back to threads (layout cannot be replicated)
            return None
        chunks = [todo[k::n] for k in range(min(n, len(todo)))]
        pool = _get_pool(n)
        with _HiddenMain():  # workers are spawned on demand inside submit()
            futs = [pool.submit(_yj_worker, shm.name, Xc.shape, Xc.dtype.str, Xc.strides, c) for c in chunks]
        res = {}
        for c, f in zip(chunks, futs):
            for i, lam in zip(c, f.result()):
                res[i] = lam
        del Xs
        return [res[i] for i in todo]
    finally:
        shm.close()
        shm.unlink()


def _n_threads() -> int:
    v = int(os.environ.get("TABICL_PREPROCESS_THREADS", "0") or 0)
    return v if v > 0 else max(1, min(32, os.cpu_count() or 1))


class ParallelPowerTransformer(PowerTransformer):
    """PowerTransformer whose Yeo-Johnson lambda search runs column-parallel; results are bit-identical."""

    def _fit(self, X, y=None, force_transform=False):  # noqa: ANN001
        n_thr = _n_threads()
        if self.method != "yeo-johnson" or n_thr <= 1 or _is_constant_feature is None:
            return super()._fit(X, y, force_transform)
        Xc = self._check_input(X, in_fit=True, check_positive=True)
        n_samples = Xc.shape[0]
        mean = np.mean(Xc, axis=0, dtype=np.float64)
        var = np.var(Xc, axis=0, dtype=np.float64)
        todo = [i for i in range(Xc.shape[1]) if not _is_constant_feature(var[i], mean[i], n_samples)]
        if len(todo) <= 1:
            return super()._fit(X, y, force_transform)
        cols = Xc.T  # same strided column views sklearn iterates over
        lams = _lambdas_procs(Xc, todo, _n_procs()) if _n_procs() > 1 else None
        if lams is None:
            with np.errstate(invalid="ignore"), ThreadPoolExecutor(max_workers=min(n_thr, len(todo))) as ex:
                lams = list(ex.map(lambda i: self._yeo_johnson_optimize(cols[i]), todo))
        precomputed = iter(lams)

        def _lookup(col):  # noqa: ANN001  - called by sklearn's loop, once per non-constant column, in order
            return next(precomputed)

        self._yeo_johnson_optimize = _lookup  # instance attribute shadows the method for this call only
        try:
            return super()._fit(X, y, force_transform)
        finally:
            del self._yeo_johnson_optimize


def install() -> bool:
    """Make TabICL's preprocessing use ParallelPowerTransformer. Returns True if patched."""
    try:
        from tabicl._sklearn import preprocessing as tp
    except Exception:  # noqa: BLE001
        return False
    if getattr(tp, "PowerTransformer", None) is ParallelPowerTransformer:
        return True
    tp.PowerTransformer = ParallelPowerTransformer
    return True

# Appendix: Detailed Experiments, Kernel Design, and Ablations (TabICLv2 on TPU v5p)

This document records the full optimization progressions, TorchTPU porting changes, custom Pallas kernel design, accuracy-gate rejections, and experiments that did not help or hit hardware limits during our TPU v5p evaluation. For headline numbers and reproduction commands, see [`README.md`](./README.md).

---

## 1. TorchTPU Compatibility Changes & Custom Pallas Kernels

Applying [`results/tpu_patch.diff`](./results/tpu_patch.diff) alongside the modules in [`scripts/`](./scripts/) addresses three architectural mismatches between stock CUDA TabICLv2 and TPU v5p:

1. **Precision and long-context attention:** Replaces `float16` AMP (unsupported on TorchTPU) with `bfloat16` mixed precision (`float32` master weights in training; `high` 3-pass `bfloat16` matmuls in the column stage and `float32` activations in the row stage for inference accuracy) and routes long in-context sequences (`≥16,384` keys) to TPU Pallas Splash Attention.
2. **Static shape bucketing ([`scripts/tabicl_tpu_bucketed.py`](./scripts/tabicl_tpu_bucketed.py)):** Removes dynamic host-synchronizing ops (`.item()`, `torch.unique`, `nonzero`) and pads row counts, train/test split positions, and feature counts onto fixed buckets so `torch.compile(backend="tpu", fullgraph=True)` caches executables without per-table recompilations.
3. **Fused VMEM Pallas kernels for 16-wide attention heads ([`scripts/tabicl_tpu_pallas_attn.py`](./scripts/tabicl_tpu_pallas_attn.py), [`scripts/tabicl_tpu_mha.py`](./scripts/tabicl_tpu_mha.py)):**
   - Column and row stages use `d = 128` with `8` heads of size `16`. Standard XLA head-splitting pads the 16-wide minor axis to `128` TPU lanes and writes per-head score/probability tensors to HBM (`~110 GB` of `330 GB` HBM traffic per training step).
   - Our Pallas kernel expands key heads (`K_exp[g·Lkp + j] = K[j] ⊙ mask_g`) inside on-chip VMEM so `Q @ K_exp^T` produces 128-lane-aligned blocks, computes `fp32` softmax in VMEM, and writes merged outputs directly without transposes (recomputing probabilities in VMEM during backward).
   - For inference on large tables (`128` inducing-point queries attending over up to `53,248` training rows), a dedicated query-expansion Pallas flash kernel (`hxq_attention`) computes online softmax over key blocks while reading K/V directly from packed projections.
   - Rotary embeddings (RoPE) are computed via an exact 3-part `bfloat16` signed-permutation matmul rather than TPU lane rolls, and Q/K/V are read as lane blocks of a single packed projection with RoPE applied in VMEM.

| Single-Chip Training Block Benchmark (Fwd + Bwd at 4 Tables) | Explicit XLA Attention | Fused Pallas Kernel |
| :--- | ---: | ---: |
| Column ISAB 2nd attention (`416 × 1024` queries, `128` keys) | 12.48 ms | 3.72 ms |
| Column ISAB 1st attention (`128` inducing queries, `576` keys) | 5.60 ms | 2.69 ms |
| Row interaction attention (`4,096 × 104` tokens) | 15.48 ms | 4.91 ms |
| Full model fwd + bwd (`4` / `8` / `16` tables on 1 chip) | 167.5 ms / 323.3 ms / OOM | 120.8 ms / 225.6 ms / 446.2 ms |

---

## 2. Full Stage-1 Training Optimization Progression

Measured on `v5p-16` (`16` chips) and `v5p-32` (`32` chips) using `1,024`-row tables (`≤100` features, `10` classes, `6` static train/test split buckets, Muon optimizer, 40–80 steps, warmup third excluded). Raw summaries are in [`results/training/stage1_optimization_summary.json`](./results/training/stage1_optimization_summary.json).

| Step (Cumulative) | Slice | Tables / Chip (`micro × accum`) | Global Batch | Step Period (ms) | Tables/s / Chip | Peak HBM / Chip |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Eager baseline | `v5p-16` | 4 (`4 × 1`) | 64 | 1,477.8 | 2.71 | 26.7 GiB |
| + Coalesced AllReduce + on-device metrics | `v5p-16` | 4 (`4 × 1`) | 64 | 1,486.7 | 2.69 | 26.7 GiB |
| + Compiled Muon step (`torch.compile` per parameter) | `v5p-16` | 4 (`4 × 1`) | 64 | 601.4 | 6.65 | 25.9 GiB |
| + Compiled forward + loss per split bucket | `v5p-16` | 4 (`4 × 1`) | 64 | 524.6 | 7.63 | 25.9 GiB |
| + Precompile all 6 split buckets upfront | `v5p-16` | 4 (`4 × 1`) | 64 | 508.5 | 7.87 | 25.9 GiB |
| + Write per-step metrics to local disk instead of GCS FUSE | `v5p-16` | 4 (`4 × 1`) | 64 | 383.7 | 10.42 | 25.9 GiB |
| + Batch scaling to 8 tables/chip | `v5p-16` | 8 (`8 × 1`) | 128 | 673.1 | 11.89 | 49.3 GiB |
| + Batch scaling to 16 tables/chip (`16 × 1` / `8 × 2`) | `v5p-16` | 16 (`8 × 2`) | 256 | 1,270.1 | 12.60 | 76.1 GiB |
| + Explicit matmul + fp32 softmax for short attention | `v5p-16` | 16 (`4 × 4`) | 256 | 865.8 | 18.48 | 89.9 GiB |
| + Increase synthetic-prior CPU workers (`8 → 24` per rank) | `v5p-16` | 16 (`4 × 4`) | 256 | 853.5 | 18.75 | 91.3 GiB |
| + Same configuration on 32 chips | `v5p-32` | 16 (`4 × 4`) | 512 | 865.3 | 18.49 | 91.4 GiB |
| + Single-graph grouped compiled Muon (`32` chips) | `v5p-32` | 16 (`4 × 4`) | 512 | 816.4 | 19.60 | 91.4 GiB |
| + Single-graph grouped compiled Muon (`16` chips) | `v5p-16` | 16 (`4 × 4`) | 256 | 804.4 | 19.89 | 91.4 GiB |
| + Fused Pallas small-head VMEM attention (`micro-batch 4`) | `v5p-16` | 16 (`4 × 4`) | 256 | 607.4 | 26.34 | 34.5 GiB |
| + Exact signed-permutation matmul RoPE (`micro-batch 4`) | `v5p-16` | 16 (`4 × 4`) | 256 | 540.7 | 29.59 | 33.5 GiB |
| + Packed-projection kernels with in-kernel RoPE (`micro-batch 4`) | `v5p-16` | 16 (`4 × 4`) | 256 | 507.7 | 31.51 | 31.9 GiB |
| **Best: Same kernels at `micro-batch 8`** | **`v5p-16`** | **16 (`8 × 2`)** | **256** | **449.6** | **35.58** | **60.5 GiB** |

### Synchronized Per-Phase Step Timing Breakdown (Rank 0 Median, ms)

| Phase | Compiled Step (`4`/chip) | Explicit Attn + Grouped Muon (`16`/chip, `v5p-32`) | + Pallas Attention (`16`/chip, `4×4`) | Best: + Matmul RoPE + Packed Kernels (`16`/chip, `8×2`) |
| :--- | ---: | ---: | ---: | ---: |
| Forward (+ loss) | 103.4 | 303.1 | 195.7 | 126.0 |
| Backward | 190.6 | 445.6 | 357.8 | 254.9 |
| Optimizer (Muon) | 44.3 | 19.1 | 19.1 | 18.8 |
| Gradient AllReduce | 14.8 | 89.1 | 47.0 | 61.5 |
| Clip + metrics + pad/H2D | 11.3 | 22.2 | 23.4 | 20.3 |

*(Synchronized breakdowns insert device barriers between phases, adding ~5–10% overhead relative to uninstrumented runs.)*

### Multi-Stage Curriculum Validation & Stage-3 Compile Limit

In eager validation runs at global batch 64 ([`results/pretraining/summary.json`](./results/pretraining/summary.json)):
- **Stage 1 (`1,024` rows, 200 steps):** `1,479.3 ms/step`, `43.0 tables/s`, loss `2.291 → 1.149`, accuracy `6.7% → 56.6%`, `26.7 GiB` HBM.
- **Stage 2 (`1,024–10,240` rows log-uniform, 120 steps, activation checkpointing):** `4,276.8 ms/step`, `14.8 tables/s`, loss `0.912 → 0.195`, accuracy `73.8% → 93.1%`, `38.4 GiB` HBM.
- **Stage 3 (`2,048–32,768` rows):** Failed at XLA compile with `CompileTimeScopedVmemOom` in `scaled_dot_product_cudnn_attention_backward` for `bf16[104, 8, 13312, 16]` (column attention backward, head dim 16). Because our Pallas training backward currently caps key length at `1,024` (Stage 1), unblocking Stage 3 requires adding a key-block-tiled backward pass for long 16-dim column attention.

---

## 3. Full Single-Chip Inference Progression & Accuracy Verification

Measured on one `v5p` chip (`45,000` train + `5,000` test rows × `100` features, binary classification, `8` estimators, warm `p50`). Raw logs are in [`results/inference/`](./results/inference/).

| Step (Cumulative) | Model Forward `p50` (ms) | End-to-End `fit + predict_proba` `p50` (s) | CPU `fit` Configuration |
| :--- | ---: | ---: | :--- |
| Power-of-two buckets (`50k → 65,536` rows, `100 → 128` features) | 18,545 | 24.20 | Stock single-thread `sklearn` (`5.59 s`) |
| + `4,096`-row buckets (`→ 53,248`), column tile 44, trim ICL K/V to train rows | 13,190 | 18.86 | Stock single-thread `sklearn` |
| + Multi-threaded Yeo-Johnson `fit` + model reuse across `fit()` calls | 13,190 | 15.01 | Thread pool (`1.73 s`; Python GIL limits scaling beyond ~4×) |
| + `high` (3-pass bf16) column matmul precision | 12,767 | 14.62 | Thread pool |
| + Feature bucket step 8 (`100 → 104` features) + auto divisor column tile | 10,691 | — | — |
| + Tuned Splash block sizes (`block_kv=4096`, `block_kv_compute=256`) | 10,132 | 11.93 | Thread pool (`1.71 s`) |
| + Fused Pallas small-head attention (column + row) | 8,065 | — | — |
| + Exact matmul RoPE + CLS on Pallas kernel + lane-aligned Q/K/V views | 7,623 | 9.42 | Thread pool (`1.72 s`) |
| + Column 1st-attention Pallas flash kernel (`128` inducing queries × `53k` rows) | 6,748 | 7.52 | 32-process shared-memory pool (`0.69 s`) |
| + Packed-projection kernels with in-kernel RoPE | 6,291 | 7.04 | 32-process shared-memory pool (`0.69 s`) |
| + Feature bucket step 4 (`100` features unpadded) + 64-process `fit` | 5,887 | 6.58 | 64-process shared-memory pool (`0.61 s`) |
| **Best: + Per-table feature-grouping gather indices (`[B, 1, H]` instead of `[B, T, H]` int64 modulo)** | **5,681** *(5,672 in quick sweep)* | **6.43** *(11.31 with stock 1-core `fit`)* | 64-process shared-memory pool (`0.65 s`) / Stock `sklearn` (`5.55 s`) |

### XProf Time Breakdown of the Best 5.68 s Model Forward
- **In-context learning stage:** `2.41 s` (`93%` inside `head_dim=64` Splash Attention, MXU-bound)
- **Column embedding stage:** `1.83 s` (`~1.00 s` first/second Pallas attention kernels, `~0.35 s` linear projections)
- **Row interaction stage:** `1.32 s` (`0.55 s` row attention kernel, `0.15 s` CLS attention, `0.23 s` projections)
- **Other / host-device overhead:** `0.07 s`

---

## 4. Summary of Experiments That Did Not Help or Were Rejected

### Training
- **Direct whole-model `torch.compile(model)`:** Failed on step 2 with `RuntimeProgramInputMismatch` when the train/test split size changed; compiling a wrapper taking the split index as a static integer succeeded.
- **Coalesced AllReduce alone:** Negligible change (`1,486.7 ms` vs `1,477.8 ms`) because per-parameter AllReduce took only 15–27 ms.
- **XLA-only head layout rearrangements (without Pallas VMEM fusion):** 172.8–217.4 ms vs 167.5 ms at 4 tables on 1 chip (no gain).
- **Split Q/K/V linear layers or view-only Q/K/V without packed kernels:** Slower than a single packed projection (`123.3 ms` vs `120.8 ms` at 4 tables).
- **Micro-batch 8 without Pallas VMEM fusion:** Exceeded 95 GiB HBM and hung; with Pallas VMEM attention, micro-batch 8 runs inside 60.5 GiB HBM.

### Inference & Accuracy Gate
- **Column matmul precision `medium` (single-pass bf16):** Rejected — failed the OpenML accuracy gate on 3 datasets (`max |ΔAUC| = 0.228` on `kc1`).
- **Row-stage `bfloat16` autocast:** Rejected — failed the OpenML accuracy gate on 2 datasets (`max |ΔAUC| = 0.158` on `kc1`).
- **Row matmul precision `medium`:** Passed AUC/RMSE thresholds (`−4.4%` latency), but increased max per-sample probability deviation vs CPU from `0.021` to `0.100`; not used.
- **`1,024`-row buckets above 16k rows:** Slower than `4,096`-row buckets (`11,382 ms` vs `11,078 ms`) when paired with `block_kv = 4,096`.
- **Larger row tiles (`8k` / `16k`):** `≤1%` change vs `4,096`.
- **Single-tile column group (`132`):** Triggered an XLA compiler check error (`Scaled dimension has non-zero pad high`).
- **Computing column skip masks once per chunk / folding key projection into inducing queries:** Measured `0–18 ms` difference (`5,905 ms` vs `5,887 ms`), within run-to-run noise.
- **FP8 quantization:** Not applicable — TPU v5p does not have native FP8 matmul hardware.

---

## 5. Comparison Caveats Against Upstream H100 Figures

1. **Upstream H100 numbers are published estimates, not measured in this study:** Training speed (`0.7–0.8 s/step` fp16, `1.0 s/step` fp32 on `4× H100` at batch size 64) and inference speed ("under 10 s" in text; `~7.5 s` on the `H100 NVL 94GB` curve at 50k samples) are taken from [`soda-inria/tabicl`](https://github.com/soda-inria/tabicl).
2. **Global batch size in Stage-1 training:** Matching 16 tables per device across 16 `v5p` chips increases global batch size from 64 (`4× H100`) to 256 (`16× v5p`), and train/test splits are discretized to 6 buckets.
3. **Host CPU `fit` and warm compilation in inference:** Reported TPU latencies are warm `p50` runs after compilation (~32–36 s cold compile). Reaching `6.43 s` end-to-end pairs TPU `v5p` inference (`5.78 s`) with a persistent 64-process shared-memory CPU pool for scikit-learn's Yeo-Johnson transform (`0.65 s`, bit-identical to serial `5.55 s`).

# TabICLv2 on Google Cloud TPU v5p

Training, single-chip inference, and multi-worker serving benchmarks for [TabICLv2](https://github.com/soda-inria/tabicl) on Google Cloud TPU v5p with PyTorch and TorchTPU.

**Summary:** With static shape bucketing, compiled Muon updates, and custom Pallas attention kernels for TabICLv2's 16-dimensional heads, **TPU v5p achieves 35.58 tables/s per chip on Stage-1 training (1.55–1.78× higher than the published H100 fp16 reference of 20.0–22.9 tables/s/GPU)** and **6.43 s end-to-end on 50k-row single-chip inference (14% faster than the ~7.5 s published H100 figure when paired with parallel host CPU preprocessing, or 11.31 s with single-threaded scikit-learn preprocessing)** while passing all 18 accuracy checks against CPU `fp32`.

---

## 1. Model, Framework, and Hardware

- **Model:** TabICLv2 (*Tabular In-Context Learning v2*, [arXiv:2602.11139](https://arxiv.org/abs/2602.11139)) is an open-source tabular foundation model for classification and regression that predicts test labels from labeled context rows in a single forward pass without per-dataset gradient updates.
  - **Parameters & Weights:** **27.55M parameters** for the classifier (28.54M for the quantile regressor). Pretraining uses `float32` master weights with `bfloat16` mixed precision; inference runs `bfloat16` mixed precision (`high` 3-pass `bfloat16` column matmuls, `float32` row activations, `bfloat16` in-context blocks) averaged across an **8-estimator ensemble**.
  - **Architecture:** Three sequential transformer stages — (1) a 3-block column-wise Set Transformer (`width 128`, `8 heads × 16 dim`, `128` inducing points) that embeds per-column distributions, (2) a 3-block row-wise interaction transformer (`width 128`, `8 heads × 16 dim`, RoPE + `4` CLS tokens) that compresses each row into a 512-dim vector, and (3) a 12-block dataset-wise in-context transformer (`width 512`, `8 heads × 64 dim`) where test rows attend over training rows.
- **Framework:** Python 3.12, PyTorch 2.15.0, `torch_tpu` 0.2.0, `libtpu` 0.0.50 (`tpu_dist` ICI backend), `tabicl` 2.2.0, and JAX/Pallas custom TPU kernels ([`docker/Dockerfile`](./docker/Dockerfile)).
- **Hardware:** Google Cloud TPU v5p slices on GKE (`95 GiB` HBM2e per chip, 1 TorchTPU device per chip):
  - `v5p-16` (`2x2x4` ICI topology: 4 hosts × 4 chips = 16 chips)
  - `v5p-32` (`2x4x4` ICI topology: 8 hosts × 4 chips = 32 chips)
  - Single-chip and single-host (`4` chips) configurations for inference and HTTP serving.

---

## 2. Main Results

### A. Training Throughput (`v5p-16`, Stage 1: `1,024` Rows × `≤100` Features)

| Configuration | Per-Device Load | Step Time | Throughput / Device | Comparison |
| :--- | :--- | ---: | ---: | :--- |
| Initial eager TPU v5p baseline | 4 tables (`BS 64`) | 1,477.8 ms | 2.71 tables/s/chip | Baseline (`1.0×`) |
| Upstream H100 reference (`fp32`, published) | 16 tables (`4× H100`, `BS 64`) | ~1,000 ms | 16.0 tables/s/GPU | Published `4× H100` fp32 estimate |
| Upstream H100 reference (`fp16` AMP, published) | 16 tables (`4× H100`, `BS 64`) | 700–800 ms | 20.0–22.9 tables/s/GPU | Published `4× H100` fp16 estimate |
| **Optimized TPU v5p (`v5p-16`, this repo)** | **16 tables (`16× v5p`, `BS 256`)** | **449.6 ms** | **35.58 tables/s/chip** (`569 tables/s` total) | **13.1×** over initial TPU baseline; **1.55–1.78× higher** than published H100 fp16 reference |

*Notes:* H100 figures are the upstream maintainers' published estimates from the [`soda-inria/tabicl` README](https://github.com/soda-inria/tabicl) (not measured here). Matching 16 tables/device across 16 `v5p` chips uses global batch 256 over 60 steady-state steps. Stage 1 and Stage 2 pretraining were validated on `v5p-16`; Stage 3 (`up to 32k` rows) is not yet enabled due to a VMEM compile limit (see [`APPENDIX.md`](./APPENDIX.md)).

### B. Single-Chip Inference (`1× TPU v5p` Chip, `8` Estimators, Warm `p50`)

All optimized TPU results below pass accuracy verification against CPU `fp32` across **15 OpenML datasets + 3 large synthetic datasets with 0 failures** (`max |ΔAUC| = 0.00085`, `max RMSE diff = 0.185%`).

| Workload (`Train + Test × Features`) | Initial TPU v5p Baseline | Best Optimized TPU v5p | Upstream H100 Reference (Published) | Comparison |
| :--- | ---: | ---: | :--- | :--- |
| **`45k + 5k × 100` — Model forward only** (no KV cache) | 18.55 s | **5.68 s** (`18.2 GiB` HBM) | Not reported separately | **3.26× faster** than initial TPU baseline |
| **`45k + 5k × 100` — End-to-end `fit + predict_proba`** (with 64-process CPU `fit`) | 24.20 s | **6.43 s** (`0.65 s` `fit` + `5.78 s` TPU) | "Under 10 s" in README; **~7.5 s** on `1× H100 NVL` plot | **3.76× faster** than initial TPU baseline; **14% faster** than ~7.5 s H100 plot |
| **`45k + 5k × 100` — End-to-end `fit + predict_proba`** (with stock 1-core CPU `fit`) | 24.20 s | **11.31 s** (`5.55 s` `fit` + `5.77 s` TPU) | ~7.5 s on `1× H100 NVL` plot | **2.14× faster** than initial TPU baseline; **1.51× slower** than ~7.5 s H100 plot due to single-core CPU Yeo-Johnson `fit` |
| **`45k + 5k × 100` — Repeated query with KV cache** | 1.32 s query (`8.49 s` build) | **0.94 s** forward / **0.98 s** `predict_proba` (`5.78 s` build) | No published H100 figure | **6.0× faster** per repeated query than a full 5.68 s uncached forward pass |
| **`10k + 1k × 100` (10 classes)** — Model forward | 2.71 s | **1.37 s** uncached / **0.10 s** KV query | No published H100 figure | **1.98× faster** than tuned non-Pallas TPU run |
| **`1k + 1k × 100` (10 classes)** — Model forward | 0.29 s | **0.18 s** uncached | No published H100 figure | Sub-200 ms single-chip ensemble forward |

*Notes:* Warm `p50` latencies after compilation (cold first-call compilation takes ~32–36 s). Raw outputs are in [`results/inference/`](./results/inference/).

### C. Single-Host Multi-Worker Serving (`4× TPU v5p` Chips on One Host)

FastAPI HTTP router dispatching across 4 worker processes (1 per chip) on a mixed tabular stream (`70%` `<5k` rows, `25%` `5k–50k` rows, `5%` `50k–122k` rows, `8` estimators, pre-serialized `float32` arrays, **0 errors** at every concurrency; [`results/serving/`](./results/serving/)):

| Mode | Concurrency 1 | Concurrency 4 | Concurrency 16 | Concurrency 64 |
| :--- | :--- | :--- | :--- | :--- |
| **Stateless (`kv_cache=False`)** | 0.22 datasets/s (`p50 318 ms`) | 1.14 datasets/s (`p50 313 ms`) | 0.82 datasets/s (`p50 2.69 s`) | **2.36 datasets/s** (`1,601 test rows/s`) |
| **Cached Context (`kv_cache=True`)** | 14.27 datasets/s (`p50 36.8 ms`) | 28.81 datasets/s (`p50 44.4 ms`) | 43.50 datasets/s (`p50 208 ms`) | **47.08 datasets/s** (**`31,880 test rows/s`**, `p50 874 ms`) |

---

## 3. Reproducing the Main Results

Build the image in [`docker/Dockerfile`](./docker/Dockerfile) and apply [`results/tpu_patch.diff`](./results/tpu_patch.diff) once to `/workspace/tabicl`:

```bash
git -C /workspace/tabicl apply /path/to/results/tpu_patch.diff
```

### Stage-1 Distributed Training (`35.58 tables/s/chip` on `v5p-16`)
Script: [`scripts/train_stage1_optimized.py`](./scripts/train_stage1_optimized.py) (`--preset best` enables compiled split buckets, grouped Muon, local-disk metrics, 24 prior workers, and the fused Pallas attention + matmul RoPE + packed projection kernels):

```bash
python3 scripts/train_stage1_optimized.py \
  --preset best \
  --nnodes=4 \
  --nproc-per-node=4 \
  --max-steps=60 \
  --no-save \
  --output-dir=/tmp/tabiclv2_stage1_best
```

### Single-Chip Inference (`5.68 s` Forward / `6.43 s` End-to-End) & Accuracy Check
Scripts: [`scripts/benchmark_inference.py`](./scripts/benchmark_inference.py) and [`scripts/check_accuracy.py`](./scripts/check_accuracy.py):

```bash
# Model forward (5.68 s) + end-to-end with standard single-core sklearn fit (11.31 s)
python3 scripts/benchmark_inference.py --preset best --out /tmp/best_inference.jsonl

# End-to-end with 64-process shared-memory Yeo-Johnson fit (6.43 s)
python3 scripts/benchmark_inference.py --preset best --fast-fit-procs 64 --e2e-only --out /tmp/best_e2e_parallel.jsonl

# Accuracy verification against CPU fp32 (15 OpenML + 3 large synthetic datasets)
python3 scripts/build_openml_reference_bundle.py --results-dir /tmp/cpu_ref
python3 scripts/build_large_reference_bundle.py --out /tmp/cpu_ref/ref_cpu_bundle_large.pt
python3 scripts/check_accuracy.py --preset best --bundle /tmp/cpu_ref/ref_cpu_bundle.pt --out /tmp/accuracy_openml15.jsonl
python3 scripts/check_accuracy.py --preset best --bundle /tmp/cpu_ref/ref_cpu_bundle_large.pt --out /tmp/accuracy_large.jsonl
```

### Single-Host 4-Chip Serving (`47.08 datasets/s` KV-Cached)
Scripts: [`scripts/serving_server.py`](./scripts/serving_server.py) and [`scripts/serving_load_test.py`](./scripts/serving_load_test.py):

```bash
python3 scripts/serving_server.py --preset best --mode launch-host --num-workers 4 --port 8000 --base-worker-port 8100 --prewarm fast &

python3 scripts/serving_load_test.py \
  --tag v5p_single_host_4chips \
  --devices-label tpu_v5p:4 \
  --direct-urls http://127.0.0.1:8000 \
  --target-urls http://127.0.0.1:8000 \
  --concurrencies 1,4,16,64 \
  --modes nocache,kv \
  --out-csv /tmp/serving.csv \
  --out-json /tmp/serving.json
```

---

## 4. Detailed Ablations, Kernel Design, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Step-by-step training and inference optimization ladders, TorchTPU porting details, custom Pallas kernel design, per-phase timing breakdowns, experiments that did not improve performance or failed accuracy/compile checks, and full comparison notes.
- **[`results/`](./results/):** Compatibility patch (`tpu_patch.diff`) and raw measured outputs under [`training/`](./results/training/), [`pretraining/`](./results/pretraining/), [`inference/`](./results/inference/), and [`serving/`](./results/serving/).

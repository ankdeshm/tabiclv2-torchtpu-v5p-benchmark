FROM python:3.12

ENV PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl jq procps libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

# 1. Install PyTorch 2.15.0 CPU release wheel first so downstream deps never pull CUDA wheels
RUN pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu

# 2. Install TorchTPU 0.2.0 GA + libtpu / Pallas runtime support (for torch_tpu custom ops) + torchao
RUN pip install \
      "torch_tpu==0.2.0" \
      "libtpu==0.0.50" \
      "jax==0.12.0.1" \
      "jaxlib==0.12.0.1" \
      "tokamax==0.0.17" \
      "torchao==0.17.0" \
      portpicker absl-py frozendict \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
      --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Install pinned tabicl==2.2.0 plus tabular data, serving, testing, and GCS/HF utilities
RUN pip install \
      "tabicl==2.2.0" \
      "pandas==2.2.3" \
      "scikit-learn==1.6.1" \
      "scipy==1.15.2" \
      "openml==0.15.1" \
      "pyarrow==19.0.1" \
      "fastapi==0.115.11" \
      "uvicorn[standard]==0.34.0" \
      "httpx==0.28.1" \
      "aiohttp==3.11.13" \
      "pytest==8.3.5" \
      "huggingface_hub[hf_transfer]<2.0.0" \
      "google-cloud-storage==3.1.0" \
      "psutil>=6.0.0" \
      "einops>=0.8.0" \
      "tqdm>=4.66.0" \
      --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Clone soda-inria/tabicl at pinned commit SHA d83091b845a8c5f7c087bd54964f755a9e560f45
ARG TABICL_COMMIT=d83091b845a8c5f7c087bd54964f755a9e560f45
RUN git clone https://github.com/soda-inria/tabicl.git /workspace/tabicl && \
    git -C /workspace/tabicl checkout ${TABICL_COMMIT} && \
    git -C /workspace/tabicl rev-parse HEAD > /workspace/TABICL_COMMIT_SHA.txt

# 5. Verify versions at build time
RUN python3 -c "import importlib.metadata as m; print('BUILD_VERSIONS:', {k: m.version(k) for k in ['torch', 'torch-tpu', 'libtpu', 'tabicl', 'pandas', 'scikit-learn', 'openml', 'fastapi', 'uvicorn', 'pytest', 'torchao']})"

WORKDIR /workspace
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"

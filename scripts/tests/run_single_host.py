#!/usr/bin/env python3
"""Run a script on this host as an independent single-host TPU (same env setup as the benches)."""
import subprocess
import sys

sys.path.insert(0, "/workspace/run")
from tpu_single_host_setup import build_single_host_env  # noqa: E402

env = build_single_host_env(port=18080, cache_dir="/tmp/torchtpu_cc_dev")
env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_dev"
sys.exit(subprocess.call([sys.executable, "-u", *sys.argv[1:]], env=env))

#!/usr/bin/env python3
"""Single-Host PJRT Metadata Proxy + LD_PRELOAD Hook for TPU v5p (4 chips per host, one TorchTPU device per chip).

Allows single-host pods on a multi-host GKE TPU v5p slice (v5p-16 `2x2x4` / v5p-32 `2x4x4`) to run one host as an
independent `v5p-8` (`2x2x1`, 4 chips):
  - 1-process single-device TorchTPU (`torch.device("tpu")`)
  - 4-process full-host TorchTPU (`torchrun --nnodes=1 --nproc_per_node=4`, `backend="tpu_dist"`)
  - 4 independent FastAPI worker processes (1 process per chip)
completely independently on each host, while forwarding non-TPU GCE metadata requests (Workload Identity OAuth
tokens for GCS) transparently to the real `169.254.169.254` metadata server.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        const char *port_s = getenv("MOCK_METADATA_PORT");
        if (port_s && port_s[0] != '\0') {
            struct sockaddr_in copy_addr;
            memcpy(&copy_addr, addr, sizeof(struct sockaddr_in));
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(copy_addr.sin_addr), ip, INET_ADDRSTRLEN);
            if (strcmp(ip, "169.254.169.254") == 0) {
                int port = atoi(port_s);
                copy_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
                copy_addr.sin_port = htons(port);
                return real_connect(sockfd, (const struct sockaddr *)&copy_addr, sizeof(struct sockaddr_in));
            }
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""


def _real_meta(rel: str, default: str = "") -> str:
    try:
        req = urllib.request.Request(
            f"http://169.254.169.254/computeMetadata/v1/{rel.lstrip('/')}",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=3.0) as r:
            return r.read().decode("utf-8").strip()
    except Exception:
        return default


def build_single_host_env(
    port: int = 18080,
    use_metadata_proxy: bool = True,
    cache_dir: str | None = "/tmp/torchtpu_compile_cache",
    slicebuilder_base_port: int | None = None,
    nproc: int = 4,
) -> dict[str, str]:
    """Returns a clean child process environment for single-host TPU v5p (`v5p-8`, `2x2x1`) execution."""
    child_env = os.environ.copy()
    for strip_k in (
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "CLOUD_TPU_TASK_ID",
        "MEGASCALE_NUM_SLICES",
        "MEGASCALE_SLICE_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    ):
        child_env.pop(strip_k, None)

    if use_metadata_proxy and __import__("shutil").which("gcc"):
        shm = Path("/dev/shm")
        shm.mkdir(parents=True, exist_ok=True)
        c_path = shm / "metadata_hook.c"
        so_path = shm / "metadata_hook.so"
        if not so_path.exists():
            c_path.write_text(HOOK_C, encoding="utf-8")
            subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_path), str(c_path), "-ldl"], check=True)

        host_short = _real_meta("instance/hostname", "tpu-v5p-node").split(".")[0]
        zone = _real_meta("instance/zone", "projects/0/zones/us-central1-c").rsplit("/", 1)[-1]
        accel = "v5p-8"

        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                already_up = True
        except OSError:
            already_up = False

        if not already_up:
            class _Handler(BaseHTTPRequestHandler):
                def log_message(self, fmt: str, *args) -> None:
                    return

                def do_GET(self) -> None:
                    if self.headers.get("Metadata-Flavor") != "Google":
                        self.send_response(400)
                        self.end_headers()
                        return
                    path = urllib.parse.urlparse(self.path).path
                    tpu_kv = {
                        "TPU_WORKER_ID": "0",
                        "ALT": "false",
                        "CHIPS_PER_HOST_BOUNDS": "2,2,1",
                        "CONSUMER_PROJECT_ID": "YOUR_GCP_PROJECT_ID",
                        "ENABLE_ICI_RESILIENCY": "false",
                        "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
                        "HOST_BOUNDS": "1,1,1",
                        "INJECT_SLICE_BUILDER_FAULT": "",
                        "NODE_ID": "0",
                        "TOPOLOGY": "2x2x1",
                        "TPU_ACCELERATOR_TYPE": accel,
                        "WRAP": "false,false,false",
                        "ZONE": zone,
                    }
                    body: bytes | None = None
                    attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
                    if attr in tpu_kv:
                        body = tpu_kv[attr].encode()
                    elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
                        lines = [f"{k}: '{v}'" for k, v in tpu_kv.items()]
                        lines.append("TPU_RUNTIME_METRICS_PORTS: '8431,8432,8433,8434'")
                        body = ("\n".join(lines) + "\n").encode()
                    elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints"):
                        body = host_short.encode()
                    elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
                        body = accel.encode()
                    elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
                        body = b"0"

                    if body is not None:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(body)
                        return

                    real_req = urllib.request.Request(f"http://169.254.169.254{self.path}")
                    for k, v in self.headers.items():
                        if k.lower() != "host":
                            real_req.add_header(k, v)
                    try:
                        with urllib.request.urlopen(real_req, timeout=5.0) as resp:
                            self.send_response(resp.status)
                            for k, v in resp.headers.items():
                                self.send_header(k, v)
                            self.end_headers()
                            self.wfile.write(resp.read())
                    except urllib.error.HTTPError as e:
                        self.send_response(e.code)
                        self.end_headers()
                        self.wfile.write(e.read())
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(str(e).encode())

            srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            time.sleep(0.2)

        tcmalloc_so = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
        preload_items = [str(so_path)]
        if os.path.exists(tcmalloc_so):
            preload_items.append(tcmalloc_so)
        prev_ld = child_env.get("LD_PRELOAD", "").strip()
        if prev_ld:
            preload_items.extend(p for p in prev_ld.split(":") if p and p not in preload_items)
        child_env["LD_PRELOAD"] = ":".join(preload_items)
        child_env["MOCK_METADATA_PORT"] = str(port)

    child_env.update(
        {
            "PJRT_DEVICE": "TPU",
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "127.0.0.1",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": "2x2x1",
            "TPU_TOPOLOGY_ALT": "false",
            "TPU_TOPOLOGY_WRAP": "false,false,false",
            "TPU_ACCELERATOR_TYPE": "v5p-8",
            "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434",
        }
    )

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        child_env["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = str(cache_dir)
        child_env["TORCH_TPU_CACHE_DIR"] = str(cache_dir)
        child_env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    if slicebuilder_base_port is not None:
        child_env["TORCH_TPU_TOPOLOGY"] = "2,2,1" if nproc == 4 else ("2,1,1" if nproc == 2 else "1,1,1")
        child_env["TORCH_TPU_DEVICES_PER_HOST"] = str(nproc)
        child_env["TORCH_TPU_BASE_PORT"] = str(slicebuilder_base_port)
        child_env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
            f"127.0.0.1:{slicebuilder_base_port + i}" for i in range(nproc)
        )

    return child_env

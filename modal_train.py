"""
Deploy MiniGPT training to Modal with GPU acceleration via CUDA/cuBLAS.

Usage:
  modal run modal_train.py              # full 5000-step run
  modal run modal_train.py --steps 100  # quick test run
  modal run modal_train.py --fresh      # ignore checkpoints, start fresh
  modal run modal_train.py --probe      # probe GPU / CUDA availability

After training, download the model:
  modal volume get mini-gpt-vol model-final.json
"""
import subprocess
import os

import modal

app = modal.App("mini-gpt-training")

vol = modal.Volume.from_name("mini-gpt-gpu-vol", create_if_missing=True)

# NVIDIA CUDA runtime image — includes libcudart.so + libcublas.so
# We install Node.js 22 on top so our JS training code can call cuBLAS
# via koffi FFI.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.12"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .add_local_dir(
        "framework/packages/framework",
        "/app/framework/packages/framework",
        copy=True,
    )
    .add_local_file("package.json", "/app/package.json", copy=True)
    .run_commands("cd /app && npm install --omit=dev")
    .add_local_file("prepare.js", "/app/prepare.js", copy=True)
    .run_commands("cd /app && node prepare.js")
    .add_local_file("train.js", "/app/train.js")
    .add_local_file("gpu_probe.js", "/app/gpu_probe.js")
)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    cpu=4,
    memory=16384,
    gpu="A100",
    volumes={"/app/out": vol},
)
def train(steps: int = 0, fresh: bool = False):
    env = dict(os.environ)
    env["MODEL_DIR"] = "/app/out"
    if fresh:
        env["NO_RESUME"] = "1"
    if steps > 0:
        env["MAX_ITERS"] = str(steps)
    result = subprocess.run(["node", "train.js"], cwd="/app", env=env)
    vol.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.function(image=image, gpu="A100", timeout=300)
def probe_gpu():
    result = subprocess.run(
        ["node", "gpu_probe.js"], cwd="/app", env=dict(os.environ)
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(steps: int = 0, fresh: bool = False, probe: bool = False):
    if probe:
        probe_gpu.remote()
        return
    train.remote(steps=steps, fresh=fresh)
    print("\nModel saved to Modal volume 'mini-gpt-gpu-vol'.")
    print("Download with:  modal volume get mini-gpt-gpu-vol model-final.json")

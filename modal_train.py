"""
Deploy MiniGPT training to Modal with Rust+CUDA native backend on A100 GPU.

Uses cuBLAS for matmul, FlashAttention, GPU-resident data pipeline,
block-parallel LayerNorm/Softmax, fused ops, and mixed-precision support.

Usage:
  modal run modal_train.py              # full run (CUDA backend, A100)
  modal run modal_train.py --steps 100  # quick test run
  modal run modal_train.py --fresh      # ignore checkpoints, start fresh
  modal run --detach modal_train.py     # run detached (survives client disconnect)

After training, download the model:
  modal volume get mini-gpt-rust-cuda-vol model-final.json
"""
import subprocess
import os

import modal

app = modal.App("mini-gpt-optimized-test")

vol = modal.Volume.from_name("mini-gpt-optimized-test-vol", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends curl ca-certificates build-essential",
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .add_local_dir(
        "framework/packages/framework",
        "/app/framework/packages/framework",
        copy=True,
    )
    .add_local_file("package.json", "/app/package.json", copy=True)
    .run_commands(
        'export PATH="/root/.cargo/bin:$PATH" && '
        "cd /app/framework/packages/framework/native && "
        "cargo build --release --no-default-features --features cuda 2>&1 && "
        "cp target/release/libmni_framework_native.so mni-framework-native.linux-x64-gnu.node"
    )
    .run_commands("cd /app && npm install --omit=dev")
    .run_commands(
        "cd /app/framework/packages/framework && "
        "npm install typescript --save-dev && npx tsc"
    )
    .add_local_file("prepare.js", "/app/prepare.js", copy=True)
    .run_commands("cd /app && node prepare.js")
    .add_local_file("train.js", "/app/train.js")
)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    gpu="A100-80GB",
    memory=32768,
    volumes={"/app/out": vol},
)
def train(steps: int = 0, fresh: bool = False):
    env = dict(os.environ)
    env["MODEL_DIR"] = "/app/out"
    # nanoGPT-scale model: 6 layers, 6 heads, 384-dim, block_size=256
    # ~10.7M params — feasible with A100 CUDA backend
    env["N_EMBD"] = "384"
    env["N_HEAD"] = "6"
    env["N_LAYER"] = "6"
    env["BLOCK_SIZE"] = "256"
    env["BATCH_SIZE"] = "8"
    env["GRAD_ACCUM_STEPS"] = "4"
    env["CHECKPOINT_EVERY"] = "500"
    env["MAX_ITERS"] = "5000"
    if fresh:
        env["NO_RESUME"] = "1"
    if steps > 0:
        env["MAX_ITERS"] = str(steps)
    result = subprocess.run(["node", "train.js"], cwd="/app", env=env)
    vol.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(steps: int = 0, fresh: bool = False):
    train.remote(steps=steps, fresh=fresh)
    print("\nModel saved to Modal volume 'mini-gpt-optimized-test-vol'.")
    print("Download with:  modal volume get mini-gpt-optimized-test-vol model-final.json")

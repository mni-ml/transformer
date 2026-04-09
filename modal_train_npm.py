"""
Train MiniGPT using @mni-ml/framework installed from npm (CUDA backend).

No local framework files — just npm install on a fresh CUDA machine.

Usage:
  modal run modal_train_npm.py              # full run
  modal run modal_train_npm.py --steps 20   # quick test
  modal run --detach modal_train_npm.py     # detached
"""
import subprocess
import os
import modal

app = modal.App("mini-gpt-npm")

vol = modal.Volume.from_name("mini-gpt-rust-cuda-vol", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        'mkdir -p /app && cd /app && npm init -y'
        ' && node -e "const p=require(\'./package.json\');p.type=\'module\';require(\'fs\').writeFileSync(\'package.json\',JSON.stringify(p,null,2))"'
        ' && npm install @mni-ml/framework@0.3.1 @mni-ml/framework-linux-x64-gnu-cuda@0.3.1'
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
    print("\nModel saved to Modal volume 'mini-gpt-rust-cuda-vol'.")

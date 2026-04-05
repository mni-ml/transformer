"""
Deploy MiniGPT training to Modal with GPU acceleration.

Usage:
  modal run modal_train.py              # full 5000-step run
  modal run modal_train.py --steps 100  # quick test run

After training, download the model:
  modal volume get mini-gpt-vol model-final.json
"""
import subprocess
import os

import modal

app = modal.App("mini-gpt-training")

vol = modal.Volume.from_name("mini-gpt-vol", create_if_missing=True)

image = (
    modal.Image.from_registry("node:22-slim", add_python="3.12")
    .apt_install("libvulkan1", "mesa-vulkan-drivers")
    .add_local_file("package.json", "/app/package.json", copy=True)
    .add_local_file("package-lock.json", "/app/package-lock.json", copy=True)
    .run_commands("cd /app && npm ci --omit=dev --ignore-scripts")
    .add_local_file("prepare.js", "/app/prepare.js", copy=True)
    .run_commands("cd /app && node prepare.js")
    .add_local_file("train.js", "/app/train.js")
)


@app.function(
    image=image,
    timeout=8 * 60 * 60,
    gpu="T4",
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


@app.local_entrypoint()
def main(steps: int = 0, fresh: bool = False):
    train.remote(steps=steps, fresh=fresh)
    print("\nModel saved to Modal volume 'mini-gpt-vol'.")
    print("Download with:  modal volume get mini-gpt-vol model-final.json")

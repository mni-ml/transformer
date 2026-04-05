"""
Deploy MiniGPT training to Modal.

Usage:
  modal run modal_train.py              # full 3000-step run (~2-3 hours)
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
    cpu=4,
    volumes={"/app/out": vol},
)
def train(steps: int = 0):
    env = dict(os.environ)
    env["MODEL_DIR"] = "/app/out"
    if steps > 0:
        env["MAX_ITERS"] = str(steps)
    result = subprocess.run(["node", "train.js"], cwd="/app", env=env)
    vol.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(steps: int = 0):
    train.remote(steps=steps)
    print("\nModel saved to Modal volume 'mini-gpt-vol'.")
    print("Download with:  modal volume get mini-gpt-vol model-final.json")

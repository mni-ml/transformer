"""
Train MiniGPT on TinyStories with Multi-head Latent Attention (MLA).

Mirrors modal_train_tinystories.py exactly except:
  - Entrypoint is train_tinystories_mla.js (MLA attention with low-rank K/V).
  - Uses a separate Modal volume so MLA checkpoints don't collide with the baseline.
  - Adds D_KV=128 (latent dim d_c). KV cache per token = dKv floats at inference.

Usage:
  modal run modal_train_tinystories_mla.py                  # full run (7500 steps)
  modal run modal_train_tinystories_mla.py --steps 20       # quick smoke test
  modal run modal_train_tinystories_mla.py --fresh          # ignore checkpoints
  modal run --detach modal_train_tinystories_mla.py         # detached
"""
import subprocess
import os
import signal
import modal

app = modal.App("mini-gpt-tinystories-mla")

vol = modal.Volume.from_name("mini-gpt-tinystories-mla-vol", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    # ── System deps ──────────────────────────────────────────────
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    # ── Node.js: install @mni-ml/framework with CUDA ─────────────
    .run_commands(
        'mkdir -p /app && cd /app && npm init -y'
        ' && node -e "const p=require(\'./package.json\');p.type=\'module\';require(\'fs\').writeFileSync(\'package.json\',JSON.stringify(p,null,2))"'
        ' && npm install @mni-ml/framework@0.3.3 @mni-ml/framework-linux-x64-gnu-cuda@0.3.3'
    )
    # ── Python: install HuggingFace deps for data download ───────
    .pip_install("datasets", "tokenizers")
    # ── Data prep: download stories, train BPE, tokenize ─────────
    .add_local_file("prepare_tinystories.py", "/app/prepare_tinystories.py", copy=True)
    .run_commands(
        "cd /app && DATA_DIR=/app/data python prepare_tinystories.py"
    )
    # ── Training code ────────────────────────────────────────────
    .add_local_file("bpe.js", "/app/bpe.js")
    .add_local_file("train_tinystories_mla.js", "/app/train_tinystories_mla.js")
)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    gpu="A100-80GB",
    memory=32768,
    volumes={"/app/out": vol},
)
def train(steps: int = 0, fresh: bool = False):
    # Flush volume on SIGTERM so checkpoints survive preemption
    def _flush(signum, frame):
        print("\n  SIGTERM received — flushing volume...")
        vol.commit()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _flush)

    env = dict(os.environ)
    env["MODEL_DIR"] = "/app/out"
    env["DATA_DIR"] = "/app/data"
    env["N_EMBD"] = "384"
    env["N_HEAD"] = "6"
    env["N_LAYER"] = "6"
    env["D_KV"] = "128"
    env["BLOCK_SIZE"] = "256"
    env["BATCH_SIZE"] = "8"
    env["GRAD_ACCUM_STEPS"] = "4"
    env["CHECKPOINT_EVERY"] = "500"
    env["LR"] = "3e-4"
    env["MAX_ITERS"] = "7500"
    if fresh:
        env["NO_RESUME"] = "1"
    if steps > 0:
        env["MAX_ITERS"] = str(steps)
    result = subprocess.run(["node", "train_tinystories_mla.js"], cwd="/app", env=env)
    vol.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(steps: int = 0, fresh: bool = False):
    train.remote(steps=steps, fresh=fresh)
    print("\nModel saved to Modal volume 'mini-gpt-tinystories-mla-vol'.")

"""
Train a SMALL draft model for speculative decoding.

The target model lives in mini-gpt-tinystories-v4-vol (6L / 6H / 384d).
This trains a much smaller draft model that shares the *exact* same
tokenizer as the target (vocab=4096, EOT id 0). Sharing the vocab is
the only hard requirement of speculative decoding.

Architecture (≈10x smaller than target):
  nLayer  = 2
  nHead   = 4
  nEmbd   = 128       # head_dim = 32
  blockSize = 256     # same context window as target

Image build:
  - bake the local out/tokenizer.json into /app/data so prepare_tinystories.py
    skips re-training BPE and just re-tokenizes 500K stories with it
  - npm install @mni-ml/framework with CUDA backend

Usage:
  modal run modal_train_draft.py                # full run
  modal run modal_train_draft.py --steps 50     # smoke test
  modal run --detach modal_train_draft.py       # detached
"""
import subprocess
import os
import signal
import modal

app = modal.App("mini-gpt-tinystories-draft")

vol = modal.Volume.from_name("mini-gpt-tinystories-draft-v1-vol", create_if_missing=True)

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
    # ── Python deps for data prep ────────────────────────────────
    .pip_install("datasets", "tokenizers")
    # ── Bake the TARGET model's tokenizer in so vocab matches ────
    # prepare_tinystories.py is patched to reuse this file when present.
    .add_local_file("out/tokenizer.json", "/app/data/tokenizer.json", copy=True)
    .add_local_file("prepare_tinystories.py", "/app/prepare_tinystories.py", copy=True)
    .run_commands(
        "cd /app && DATA_DIR=/app/data NUM_STORIES=500000 VOCAB_SIZE=4096"
        " python prepare_tinystories.py"
    )
    # ── Training code ────────────────────────────────────────────
    .add_local_file("bpe.js", "/app/bpe.js")
    .add_local_file("train_youtube.js", "/app/train_youtube.js")
)


@app.function(
    image=image,
    timeout=24 * 60 * 60,
    gpu="A100-80GB",
    memory=32768,
    volumes={"/app/out": vol},
)
def train(steps: int = 0, fresh: bool = False):
    def _flush(signum, frame):
        print("\n  SIGTERM received — flushing volume...")
        vol.commit()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _flush)

    env = dict(os.environ)
    env["MODEL_DIR"] = "/app/out"
    env["DATA_DIR"] = "/app/data"
    # ── Draft model architecture (≈10x smaller than target) ──
    env["N_EMBD"] = "128"
    env["N_HEAD"] = "4"
    env["N_LAYER"] = "2"
    env["BLOCK_SIZE"] = "256"
    # Smaller model → can fit a bigger batch comfortably
    env["BATCH_SIZE"] = "32"
    env["GRAD_ACCUM_STEPS"] = "1"
    env["CHECKPOINT_EVERY"] = "500"
    env["LR"] = "6e-4"
    env["MAX_ITERS"] = "5000"
    if fresh:
        env["NO_RESUME"] = "1"
    if steps > 0:
        env["MAX_ITERS"] = str(steps)
    result = subprocess.run(["node", "train_youtube.js"], cwd="/app", env=env)
    vol.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(steps: int = 0, fresh: bool = False):
    train.remote(steps=steps, fresh=fresh)
    print("\nDraft model saved to Modal volume 'mini-gpt-tinystories-draft-v1-vol'.")

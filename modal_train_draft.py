"""
Train a draft model for speculative decoding (v2 — bigger, longer).

The target model is a 6L / 6H / 384d MiniGPT (~12.3M params). This
trains a draft model that shares the *exact* same tokenizer as the
target (vocab=4096, EOT id 0). Sharing the vocab is the only hard
requirement of speculative decoding.

Architecture (≈5.7x smaller than target → ~2.17M params):
  nLayer  = 3
  nHead   = 6                # head_dim = 32
  nEmbd   = 192
  blockSize = 256            # same context window as target

Why bigger than v1 (which was 1L=2,H=4,E=128, ~960K params):
  v1 reached val loss 2.61 → only 33% draft acceptance → 0.85x speedup.
  Sitting in the textbook 5–10x smaller band (vs v1's ~13x) and
  3x more training steps should push acceptance into the 50–70%
  range where speculative decoding actually wins.

Image build:
  - bake the local out/tokenizer.json into /app/data so prepare_tinystories.py
    skips re-training BPE and just re-tokenizes 500K stories with it
  - npm install @mni-ml/framework with CUDA backend

Usage:
  modal run modal_train_draft.py                # full run (7500 steps)
  modal run modal_train_draft.py --steps 50     # smoke test
  modal run --detach modal_train_draft.py       # detached

Resume:
  Re-running without --fresh picks up from the latest checkpoint-NNNN.json
  found in the volume. To extend training past the original MAX_ITERS,
  bump MAX_ITERS in this file (or pass --steps NNNN) and re-run.
"""
import subprocess
import os
import signal
import modal

app = modal.App("mini-gpt-tinystories-draft-v2")

vol = modal.Volume.from_name("mini-gpt-tinystories-draft-v2-vol", create_if_missing=True)

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
    # ── Draft model architecture (≈5.7x smaller than target) ──
    # 3L / 6H / 192d → 2,174,656 params (vs target 12,322,816)
    env["N_EMBD"] = "192"
    env["N_HEAD"] = "6"
    env["N_LAYER"] = "3"
    env["BLOCK_SIZE"] = "256"
    # Still small enough for a comfortable single-pass batch on A100
    env["BATCH_SIZE"] = "32"
    env["GRAD_ACCUM_STEPS"] = "1"
    env["CHECKPOINT_EVERY"] = "250"
    env["LR"] = "6e-4"
    env["MAX_ITERS"] = "7500"
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
    print("\nDraft model saved to Modal volume 'mini-gpt-tinystories-draft-v2-vol'.")

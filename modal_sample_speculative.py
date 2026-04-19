"""
Speculative-decoding demo on Modal.

Mounts:
  - mini-gpt-tinystories-v4-vol         → /app/target  (the big model)
  - mini-gpt-tinystories-draft-v1-vol   → /app/draft   (the small model)

Both volumes hold a tokenizer.json. Since modal_train_draft.py was built
with the target's tokenizer baked in, the two tokenizers are identical
and speculative decoding is valid.

Usage:
  modal run modal_sample_speculative.py \\
      --target-checkpoint checkpoint-6500.json \\
      --draft-checkpoint  model-final.json \\
      --prompt "Once upon a time" \\
      --num-tokens 200 \\
      --temperature 0.8 \\
      --k 4
"""
import subprocess
import shutil
import modal

app = modal.App("mini-gpt-spec-decode")

target_vol = modal.Volume.from_name("mini-gpt-tinystories-v4-vol", create_if_missing=False)
draft_vol = modal.Volume.from_name("mini-gpt-tinystories-draft-v1-vol", create_if_missing=False)

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
        ' && npm install @mni-ml/framework@0.3.3 @mni-ml/framework-linux-x64-gnu-cuda@0.3.3'
    )
    .add_local_file("bpe.js", "/app/bpe.js")
    .add_local_file("generate_speculative.js", "/app/generate_speculative.js")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    memory=32768,
    timeout=60 * 60,
    volumes={"/app/target": target_vol, "/app/draft": draft_vol},
)
def sample(
    target_checkpoint: str = "checkpoint-6500.json",
    draft_checkpoint: str = "model-final.json",
    prompt: str = "Once upon a time",
    num_tokens: int = 200,
    temperature: float = 0.8,
    k: int = 4,
):
    # The tokenizer baked into the draft volume is the canonical one
    # (it was copied from the target before training). Use it for both.
    shutil.copyfile("/app/draft/tokenizer.json", "/app/tokenizer.json")

    cmd = [
        "node",
        "/app/generate_speculative.js",
        f"/app/target/{target_checkpoint}",
        f"/app/draft/{draft_checkpoint}",
        prompt,
        str(num_tokens),
        str(temperature),
        str(k),
        "/app/tokenizer.json",
    ]
    result = subprocess.run(cmd, cwd="/app")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    target_checkpoint: str = "checkpoint-6500.json",
    draft_checkpoint: str = "model-final.json",
    prompt: str = "Once upon a time",
    num_tokens: int = 200,
    temperature: float = 0.8,
    k: int = 4,
):
    sample.remote(
        target_checkpoint=target_checkpoint,
        draft_checkpoint=draft_checkpoint,
        prompt=prompt,
        num_tokens=num_tokens,
        temperature=temperature,
        k=k,
    )

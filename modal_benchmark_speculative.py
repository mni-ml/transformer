"""
Run benchmark_speculative.js on Modal so we get real A100 numbers
(local CPU has no flashAttention kernel and is unrealistically slow).

Mounts:
  - mini-gpt-tinystories-v4-vol         → /app/target  (the big model)
  - mini-gpt-tinystories-draft-v1-vol   → /app/draft   (the small model)

Usage:
  modal run modal_benchmark_speculative.py
  modal run modal_benchmark_speculative.py --num-tokens 400 --k 6 --num-runs 5
"""
import subprocess
import shutil
import modal

app = modal.App("mini-gpt-spec-decode-bench")

target_vol = modal.Volume.from_name("mini-gpt-tinystories-target-vol", create_if_missing=False)
draft_vol = modal.Volume.from_name("mini-gpt-tinystories-draft-v2-vol", create_if_missing=False)

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
    .add_local_file("benchmark_speculative.js", "/app/benchmark_speculative.js")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    memory=32768,
    timeout=60 * 60,
    volumes={"/app/target": target_vol, "/app/draft": draft_vol},
)
def bench(
    target_checkpoint: str = "model-final.json",
    draft_checkpoint: str = "model-final.json",
    num_tokens: int = 200,
    temperature: float = 0.0,
    k: int = 4,
    num_runs: int = 3,
):
    # Both models share the target's tokenizer (the draft was trained
    # with that exact tokenizer baked into its image at build time).
    shutil.copyfile("/app/target/tokenizer.json", "/app/tokenizer.json")
    cmd = [
        "node",
        "/app/benchmark_speculative.js",
        f"/app/target/{target_checkpoint}",
        f"/app/draft/{draft_checkpoint}",
        str(num_tokens),
        str(temperature),
        str(k),
        str(num_runs),
        "/app/tokenizer.json",
    ]
    result = subprocess.run(cmd, cwd="/app")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    target_checkpoint: str = "model-final.json",
    draft_checkpoint: str = "model-final.json",
    num_tokens: int = 200,
    temperature: float = 0.0,
    k: int = 4,
    num_runs: int = 3,
):
    bench.remote(
        target_checkpoint=target_checkpoint,
        draft_checkpoint=draft_checkpoint,
        num_tokens=num_tokens,
        temperature=temperature,
        k=k,
        num_runs=num_runs,
    )

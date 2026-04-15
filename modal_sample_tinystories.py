import subprocess
import shutil
import modal

app = modal.App("mini-gpt-tinystories-sample")
vol = modal.Volume.from_name("mini-gpt-tinystories-v5-vol", create_if_missing=False)

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
    .pip_install("datasets", "tokenizers")
    .add_local_file("prepare_tinystories.py", "/app/prepare_tinystories.py", copy=True)
    .run_commands("cd /app && DATA_DIR=/app/data NUM_STORIES=500000 VOCAB_SIZE=2048 python prepare_tinystories.py")
    .add_local_file("bpe.js", "/app/bpe.js")
    .add_local_file("generate.js", "/app/generate.js")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    memory=32768,
    timeout=60 * 60,
    volumes={"/app/out": vol},
)
def sample(
    checkpoint: str = "checkpoint-4500.json",
    prompt: str = "Once upon a time",
    num_tokens: int = 220,
    temperature: float = 0.0,
):
    # Persist the exact tokenizer used in this Modal environment to the shared volume
    # so future local sampling can use the same vocab/merges.
    shutil.copyfile("/app/data/tokenizer.json", "/app/out/tokenizer.json")
    vol.commit()

    model_path = f"/app/out/{checkpoint}"
    cmd = [
        "node",
        "/app/generate.js",
        model_path,
        prompt,
        str(num_tokens),
        str(temperature),
        "/app/out/tokenizer.json",
    ]
    result = subprocess.run(cmd, cwd="/app")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@app.local_entrypoint()
def main(
    checkpoint: str = "checkpoint-4500.json",
    prompt: str = "Once upon a time",
    num_tokens: int = 220,
    temperature: float = 0.0,
):
    sample.remote(
        checkpoint=checkpoint,
        prompt=prompt,
        num_tokens=num_tokens,
        temperature=temperature,
    )

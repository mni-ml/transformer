"""
One-off: sample from mini-gpt-tinystories-v4-vol (4096-vocab run) on GPU.
Does not run prepare_tinystories — uses tokenizer.json already on the volume.
"""
import subprocess
import modal

app = modal.App("mini-gpt-sample-v4-checkpt")
vol = modal.Volume.from_name("mini-gpt-tinystories-v4-vol", create_if_missing=False)

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
    .add_local_file("generate.js", "/app/generate.js")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    memory=16384,
    timeout=30 * 60,
    volumes={"/app/out": vol},
)
def sample(
    checkpoint: str = "checkpoint-5000.json",
    prompt: str = "Once upon a time",
    num_tokens: int = 220,
    temperature: float = 0.0,
):
    cmd = [
        "node",
        "/app/generate.js",
        f"/app/out/{checkpoint}",
        prompt,
        str(num_tokens),
        str(temperature),
        "/app/out/tokenizer.json",
    ]
    subprocess.run(cmd, cwd="/app", check=True)


@app.local_entrypoint()
def main(
    checkpoint: str = "checkpoint-5000.json",
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

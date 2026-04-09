"""
Test that `npm install @mni-ml/framework` works on a fresh machine.

Installs the published package from npm (no Rust, no compilation),
then runs the test suite and XOR training example.

Usage:
  modal run modal_npm_test.py
"""
import subprocess
import modal

app = modal.App("mni-framework-npm-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update",
        "apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        "mkdir -p /app && cd /app && npm init -y && npm install @mni-ml/framework"
    )
    .add_local_file(
        "framework/packages/framework/test/ops.test.js",
        "/app/test/ops.test.js",
        copy=True,
    )
    .add_local_file(
        "framework/packages/framework/examples/xor.js",
        "/app/examples/xor.js",
        copy=True,
    )
    .run_commands(
        "sed -i 's|../dist/index.js|@mni-ml/framework|g' /app/test/ops.test.js /app/examples/xor.js"
    )
)


@app.function(image=image, timeout=120)
def test_npm_install():
    print("=== Checking installed packages ===")
    subprocess.run(["npm", "ls", "@mni-ml/framework"], cwd="/app", check=True)
    print()

    print("=== Running test suite ===")
    result = subprocess.run(["node", "test/ops.test.js"], cwd="/app")
    if result.returncode != 0:
        raise SystemExit(f"Tests failed with exit code {result.returncode}")
    print()

    print("=== Running XOR training example ===")
    result = subprocess.run(["node", "examples/xor.js"], cwd="/app")
    if result.returncode != 0:
        raise SystemExit(f"XOR example failed with exit code {result.returncode}")

    print()
    print("✅ Everything works! npm install @mni-ml/framework is fully functional.")


@app.local_entrypoint()
def main():
    test_npm_install.remote()

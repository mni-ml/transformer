"""
Profile cudaMalloc, kernel launch overhead, and "cached vs uncached" workload
on Modal with Nsight Systems.

This is a teaching benchmark. It compiles a small CUDA C++ program (bench.cu)
on a CUDA devel image, runs it once on its own to print the host-side
timings, then runs it again under `nsys profile` to capture a timeline you
can either skim from `nsys stats` (text) or open in the Nsight Systems GUI
locally for visual exploration.

Sections in the bench (printed by bench.cu):
  A. First cudaMalloc after context init (cold-driver cost)
  B. Repeated cudaMalloc/cudaFree loop  (steady-state malloc cost)
  C. Empty-kernel launch loop           (launch overhead)
  D. "No-cache" workload                (malloc + kernel + free, repeated)
  E. "Cached" workload                  (malloc once, reuse buffer, kernel)

Usage:
  modal run modal_nsys_bench.py

After it runs, the profile is saved to a Modal volume. Download it locally
to open in the Nsight Systems desktop app:
  modal volume get nsys-bench-vol bench.nsys-rep
  open bench.nsys-rep                 # macOS, with Nsight Systems installed
"""
import base64
import subprocess
from pathlib import Path

import modal

app = modal.App("nsys-bench")
vol = modal.Volume.from_name("nsys-bench-vol", create_if_missing=True)

# Read the CUDA source and embed it as base64 so we can write it into the
# image without worrying about shell quoting / heredoc edge cases.
# Modal also imports this module *inside* the container, where bench.cu
# isn't shipped, so guard the read with an existence check.
_BENCH_PATH = Path(__file__).with_name("bench.cu")
CUDA_SRC = _BENCH_PATH.read_text() if _BENCH_PATH.exists() else ""
CUDA_SRC_B64 = base64.b64encode(CUDA_SRC.encode()).decode()

image = (
    # 22.04 base — NVIDIA's devtools repo (which hosts nsight-systems-cli)
    # has reliable Jammy packages. The CUDA toolkit version is independent
    # of the Ubuntu version for our purposes.
    modal.Image.from_registry(
        # Match the nsys 2026.x injector: it needs CUDA 12.8+ symbols.
        # Stay on jammy (22.04) so the devtools apt repo for nsys works.
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .run_commands(
        # Add NVIDIA developer-tools apt repo and install Nsight Systems CLI.
        "apt-get update",
        "apt-get install -y --no-install-recommends wget gnupg",
        "wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub "
        "| gpg --dearmor -o /etc/apt/trusted.gpg.d/nvidia-devtools.gpg",
        "echo 'deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "> /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-get update",
        "apt-get install -y --no-install-recommends nsight-systems-cli",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
        "nsys --version",
    )
    .run_commands(
        # Materialize bench.cu into the image and compile it.
        f"mkdir -p /bench && echo {CUDA_SRC_B64} | base64 -d > /bench/bench.cu",
        "nvcc -O2 -lineinfo -o /bench/bench /bench/bench.cu",
    )
)


@app.function(
    image=image,
    timeout=10 * 60,
    gpu="A10G",  # cheap + plenty for this microbench; any NVIDIA GPU works
    volumes={"/out": vol},
)
def run_bench():
    # 1. Sanity: where is nsys, what version, what GPU did we land on?
    print("=== Environment ===")
    subprocess.run(
        ["bash", "-lc", "which nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys"],
        check=False,
    )
    subprocess.run(["nsys", "--version"], check=False)
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"], check=False)
    print()

    # 2. Run the bench unprofiled first so we have clean host-timer numbers
    #    without any nsys instrumentation overhead.
    print("=== Bench (unprofiled host timings) ===")
    subprocess.run(["/bench/bench"], check=True)
    print()

    # 3. Run again under nsys profile.
    #    -t cuda,nvtx           : trace CUDA runtime+driver and NVTX ranges
    #                             (osrt sometimes destabilizes the injector)
    #    --cuda-graph-trace=node: explicit setting — the default 'graph' mode
    #                             in newer nsys can crash on simple workloads
    #    --cuda-memory-usage=true: capture cudaMalloc/cudaFree bytes/duration
    #    -o /out/bench          : write /out/bench.nsys-rep to the volume
    print("=== nsys profile ===")
    subprocess.run(
        [
            "nsys", "profile",
            "-t", "cuda,nvtx",
            "--cuda-graph-trace=node",
            "--cuda-memory-usage=true",
            "-o", "/out/bench",
            "--force-overwrite=true",
            "/bench/bench",
        ],
        check=False,  # don't blow up if the injector misbehaves; we still
                     # try to extract whatever stats made it into the report
    )
    print()

    # 4. Re-print stats. With no --report flag, nsys prints all default
    #    summaries: CUDA API calls, GPU kernels, GPU memory ops, etc.
    #    The CUDA API summary is the headline — it shows total time spent
    #    in each cudaMalloc/cudaFree/cudaLaunchKernel call so you can
    #    see allocation overhead at a glance.
    print("=== nsys stats (all default reports) ===")
    subprocess.run(
        ["nsys", "stats", "--force-export=true", "/out/bench.nsys-rep"],
        check=False,
    )

    vol.commit()

    print()
    print("Profile saved. To open in the GUI locally:")
    print("  modal volume get nsys-bench-vol bench.nsys-rep")
    print("  # then open bench.nsys-rep with Nsight Systems")


@app.local_entrypoint()
def main():
    run_bench.remote()

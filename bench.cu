// Tiny CUDA microbenchmark for understanding cudaMalloc cost,
// kernel launch overhead, and the "no-cache vs cached" workload pattern
// that motivates the framework's caching allocator.
//
// Each section prints a one-line summary measured with std::chrono on the
// host side, with explicit cudaDeviceSynchronize() barriers so we're not
// measuring async-launch return time.
//
// Run under `nsys profile` to get per-call timestamps in addition to the
// totals printed below.

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while(0)

__global__ void noop_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;
}

using clk = std::chrono::high_resolution_clock;

static double us(clk::time_point t0, clk::time_point t1) {
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

int main() {
    // Force CUDA context init so the first real measurement isn't
    // contaminated by driver setup and JIT.
    CHECK(cudaFree(0));

    // ---- A. First cudaMalloc after context init ----
    {
        float* p;
        auto t0 = clk::now();
        CHECK(cudaMalloc(&p, 1024 * sizeof(float)));
        auto t1 = clk::now();
        printf("A. First cudaMalloc(4 KB):              %8.2f us\n", us(t0, t1));
        CHECK(cudaFree(p));
    }

    // ---- B. Repeated cudaMalloc/cudaFree (steady state) ----
    {
        const int N = 2000;
        const size_t SZ = 4096 * sizeof(float);  // 16 KB
        float* p;
        auto t0 = clk::now();
        for (int i = 0; i < N; i++) {
            CHECK(cudaMalloc(&p, SZ));
            CHECK(cudaFree(p));
        }
        auto t1 = clk::now();
        double total = us(t0, t1);
        printf("B. cudaMalloc+cudaFree (16 KB) avg:     %8.2f us  (n=%d, total=%.0f us)\n",
               total / N, N, total);
    }

    // ---- C. Empty-kernel launch overhead ----
    // A trivial kernel body so what we mostly measure is launch overhead.
    {
        const int N = 10000;
        float* p;
        CHECK(cudaMalloc(&p, 256 * sizeof(float)));

        // Warmup so first launch's PTX upload isn't in the timing.
        for (int i = 0; i < 100; i++) noop_kernel<<<1, 256>>>(p, 256);
        CHECK(cudaDeviceSynchronize());

        auto t0 = clk::now();
        for (int i = 0; i < N; i++) {
            noop_kernel<<<1, 256>>>(p, 256);
        }
        CHECK(cudaDeviceSynchronize());
        auto t1 = clk::now();
        double total = us(t0, t1);
        printf("C. Kernel launch+exec (1 block) avg:    %8.2f us  (n=%d)\n",
               total / N, N);
        CHECK(cudaFree(p));
    }

    // ---- D. "No-cache" workload: malloc + kernel + free, repeated ----
    {
        const int N = 2000;
        const size_t SZ = 4096 * sizeof(float);
        auto t0 = clk::now();
        for (int i = 0; i < N; i++) {
            float* p;
            CHECK(cudaMalloc(&p, SZ));
            noop_kernel<<<16, 256>>>(p, 4096);
            CHECK(cudaFree(p));
        }
        CHECK(cudaDeviceSynchronize());
        auto t1 = clk::now();
        double total = us(t0, t1);
        printf("D. NO-cache step (malloc+launch+free):  %8.2f us/step  (n=%d)\n",
               total / N, N);
    }

    // ---- E. "Cached" workload: alloc once, reuse ----
    // Models what the framework's caching allocator gives you: after the
    // first iteration, every alloc is a free-list pop, so effectively zero
    // cost compared to cudaMalloc.
    {
        const int N = 2000;
        const size_t SZ = 4096 * sizeof(float);
        float* p;
        CHECK(cudaMalloc(&p, SZ));   // amortized once

        auto t0 = clk::now();
        for (int i = 0; i < N; i++) {
            noop_kernel<<<16, 256>>>(p, 4096);
        }
        CHECK(cudaDeviceSynchronize());
        auto t1 = clk::now();
        double total = us(t0, t1);
        printf("E. CACHED step (kernel only):           %8.2f us/step  (n=%d)\n",
               total / N, N);

        CHECK(cudaFree(p));
    }

    return 0;
}

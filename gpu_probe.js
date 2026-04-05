import { existsSync } from 'node:fs';
import { execSync } from 'node:child_process';

console.log('=== GPU / CUDA Probe ===\n');

// nvidia-smi
console.log('--- nvidia-smi ---');
try {
  const smi = execSync('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1').toString().trim();
  console.log(smi);
} catch { console.log('Not available'); }

// Check CUDA libraries
console.log('\n--- CUDA libraries ---');
for (const lib of ['libcudart.so', 'libcublas.so', 'libcuda.so.1']) {
  try {
    const path = execSync(`ldconfig -p 2>/dev/null | grep "${lib}" | head -1 || true`).toString().trim();
    console.log(`  ${lib}: ${path || 'NOT FOUND'}`);
  } catch { console.log(`  ${lib}: lookup failed`); }
}

// Check /dev/dri
console.log('\n--- /dev/dri ---');
console.log(existsSync('/dev/dri') ? 'EXISTS' : 'NOT PRESENT (expected on CUDA-only)');

// Test cuBLAS via koffi
console.log('\n--- cuBLAS via koffi test ---');
try {
  const koffi = (await import('koffi')).default;
  console.log('  koffi loaded OK');

  const cudart = koffi.load('libcudart.so');
  console.log('  libcudart.so loaded OK');

  const cublas = koffi.load('libcublas.so');
  console.log('  libcublas.so loaded OK');

  // Init
  const cudaMalloc = cudart.func('int cudaMalloc(_Out_ void** devPtr, size_t size)');
  const cudaFree   = cudart.func('int cudaFree(void* devPtr)');
  const cudaMemcpy = cudart.func('int cudaMemcpy(void* dst, const void* src, size_t count, int kind)');
  const cublasCreate  = cublas.func('int cublasCreate_v2(_Out_ void** handle)');
  const cublasDestroy = cublas.func('int cublasDestroy_v2(void* handle)');
  const cublasSgemm   = cublas.func(
    'int cublasSgemm_v2(void* handle, int transa, int transb, ' +
    'int m, int n, int k, ' +
    'const float* alpha, void* A, int lda, void* B, int ldb, ' +
    'const float* beta, void* C, int ldc)'
  );

  const handleOut = [null];
  let status = cublasCreate(handleOut);
  console.log(`  cublasCreate: status=${status}`);
  if (status !== 0) throw new Error(`cublasCreate failed: ${status}`);

  // 2x2 matmul test:  [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
  const M = 2, K = 2, N = 2;
  const A = new Float32Array([1,2, 3,4]);  // row-major
  const B = new Float32Array([5,6, 7,8]);
  const C = new Float32Array(4);
  const alpha = new Float32Array([1.0]);
  const beta  = new Float32Array([0.0]);

  const dA = [null], dB = [null], dC = [null];
  cudaMalloc(dA, 4*4);
  cudaMalloc(dB, 4*4);
  cudaMalloc(dC, 4*4);

  cudaMemcpy(dA[0], A, 4*4, 1);  // H2D
  cudaMemcpy(dB[0], B, 4*4, 1);

  // Row-major trick: C^T = B^T * A^T
  status = cublasSgemm(handleOut[0], 0, 0, N, M, K, alpha, dB[0], N, dA[0], K, beta, dC[0], N);
  console.log(`  cublasSgemm: status=${status}`);

  cudaMemcpy(C, dC[0], 4*4, 2);  // D2H

  console.log(`  Result: [${C[0]}, ${C[1]}, ${C[2]}, ${C[3]}]`);
  console.log(`  Expected: [19, 22, 43, 50]`);
  const ok = C[0] === 19 && C[1] === 22 && C[2] === 43 && C[3] === 50;
  console.log(`  PASS: ${ok}`);

  cudaFree(dA[0]);
  cudaFree(dB[0]);
  cudaFree(dC[0]);
  cublasDestroy(handleOut[0]);

  if (!ok) process.exit(1);
} catch (e) {
  console.log(`  ERROR: ${e.message}`);
  process.exit(1);
}

// Test framework matmul
console.log('\n--- Framework matmul (should use CUDA) ---');
try {
  const fw = await import('@mni-ml/framework');
  const A = fw.Tensor.tensor([[1,2],[3,4]]);
  const B = fw.Tensor.tensor([[5,6],[7,8]]);
  const C = await A.matmul(B);
  console.log(`  Result: ${C.toString()}`);
  const expected = [[19,22],[43,50]];
  const pass = C.get([0,0]) === 19 && C.get([0,1]) === 22 && C.get([1,0]) === 43 && C.get([1,1]) === 50;
  console.log(`  PASS: ${pass}`);
  await fw.destroyDevice();
} catch (e) {
  console.log(`  ERROR: ${e.message}`);
}

console.log('\n=== Probe complete ===');

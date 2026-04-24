# mni-ml/transformer

A 12M paramater LLM in **Node.js**, trained with [`@mni-ml/framework`](https://www.npmjs.com/package/@mni-ml/framework) on **BPE-tokenized** text. You can build the dataset from **TinyStories** or **YouTube-Commons** transcripts using the included Python scripts; training and sampling run in JavaScript.
<img width="2132" height="1200" alt="Logo" src="https://github.com/user-attachments/assets/053142ce-810c-4cf1-8335-b96efc676b14" />

## Layout

| Path | Role |
|------|------|
| `src/train.js` | **CPU** training (default npm native build; matmul attention fallback) |
| `src/train_gpu.js` | **GPU** training (`createDataset` / `forwardGpu` / `crossEntropyLossGpu`; needs platform CUDA/WebGPU optional package) |
| `src/generate.js` | **CPU-friendly** sampling (works everywhere) |
| `src/generate_gpu.js` | **GPU** sampling (`native.flashAttention` required) |
| `src/generate_kv.js` | **KV-cache** sampling with the local `framework` branch |
| `src/demo_kv.js` | baseline vs `kv-fp32` vs `kv-int8` benchmark/demo |
| `src/server.js` | Minimal HTTP inference server (`POST /generate`) |
| `src/benchmark_http.js` | HTTP load generator (p50/p95 latency, TTFT, throughput) |
| `src/bpe.js` | HuggingFace-style BPE JSON loader (ByteLevel) |
| `scripts/prepare_tinystories.py` | Download TinyStories, train BPE, write `data/*.bin` and metadata |
| `scripts/prepare_youtube.py` | Download YouTube-Commons transcripts, same output layout |
| `scripts/gpu_probe.js` | Optional CUDA / framework matmul check |
| `out/` | `model-final.json` and `tokenizer.json` are committed as defaults; other checkpoints stay local (gitignored) |

## Prerequisites

- **Node.js** ≥ **22.18** (required by `@mni-ml/framework`).
- **Python** 3.10+ with packages from `requirements-data.txt` (for dataset + tokenizer prep only).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-data.txt
```

Build the local `framework` checkout first if you want the KV-cache demo path:

```bash
git -C framework checkout kv-cache-quantization
npm --prefix framework install
npm --prefix framework run build:native
npm --prefix framework run build
```

Install JS dependencies:

```bash
npm install --omit=optional
```

This repo uses the local `framework/` checkout via `file:./framework`, so `npm install --omit=optional` keeps the demo on your local branch instead of the published npm package.

## Train on TinyStories

1. **Prepare data** (downloads stories, trains BPE, encodes to `data/train.bin`, `data/val.bin`, `data/meta.json`, `data/tokenizer.json`):

   ```bash
   npm run prepare:tinystories
   ```

   Tunables via environment variables (see `scripts/prepare_tinystories.py`): e.g. `NUM_STORIES`, `VOCAB_SIZE`, `DATA_DIR`.

2. **Train** (CPU or GPU):

   ```bash
   npm run train
   ```

   If you have installed the matching **`@mni-ml/framework-*`** optional native package (e.g. CUDA) so GPU symbols exist:

   ```bash
   npm run train:gpu
   ```

   Checkpoints go under `out/`; the last save is `out/model-final.json`.

3. **Generate** (tokenizer path is stored in the checkpoint; override with the 5th arg if needed):

   ```bash
   npm run generate
   ```

   With a GPU-capable native build and `native.flashAttention`:

   ```bash
   npm run generate:gpu
   ```

   ```bash
   node src/generate.js out/model-final.json "<|endoftext|>" 400 0.9 data/tokenizer.json
   ```

   With the local KV-cache branch:

   ```bash
   npm run generate:kv -- out/model-final.json "<|endoftext|>" 128 0 data/tokenizer.json fp32 fp32
   npm run generate:kv -- out/model-final.json "<|endoftext|>" 128 0 data/tokenizer.json int8 int8
   npm run demo:kv -- out/model-final.json "<|endoftext|>" 128 0 data/tokenizer.json int8
   ```

   Arguments after `tokenizer.json` are:
   - `cacheMode`: `fp32` or `int8` (KV cache precision)
   - `weightMode`: `fp32` or `int8` (checkpoint weights quantized to int8 then dequantized at load)

## HTTP inference + benchmark

Run the HTTP server:

```bash
npm run serve -- out/model-final.json data/tokenizer.json 3000 int8
```

Server endpoints:
- `GET /health`
- `GET /stats`
- `POST /generate` with JSON body:

```json
{
  "prompt": "Once upon a time",
  "maxTokens": 64,
  "temperature": 0,
  "cacheMode": "int8"
}
```

Run a simple load test:

```bash
npm run bench:http -- http://localhost:3000 40 8 64 0 int8
```

## Train on YouTube-Commons

1. **Prepare data** (transcripts from HuggingFace, then BPE + bins):

   ```bash
   npm run prepare:youtube
   ```

   Tunables: `NUM_TRANSCRIPTS`, `VOCAB_SIZE`, `DATA_DIR` (see `scripts/prepare_youtube.py`).

2. **Train** and **generate** — same as above (`npm run train`, `npm run generate`).

## Training options

`src/train.js` and `src/train_gpu.js` read the same hyperparameters from the environment, including:

`MAX_ITERS`, `BATCH_SIZE`, `LR`, `N_LAYER`, `N_HEAD`, `N_EMBD`, `BLOCK_SIZE`, `CHECKPOINT_EVERY`, `MODEL_DIR`, `DATA_DIR`, `GRAD_ACCUM_STEPS`, `NO_RESUME=1`.

## Development

- **`npm run probe:gpu`** — optional CUDA / `@mni-ml/framework` check (`scripts/gpu_probe.js`).

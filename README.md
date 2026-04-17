# mini-gpt

A small GPT-style transformer trained in **Node.js** with `@mni-ml/framework` (installed from a local path; see below), using Karpathy-style char-level modeling on Tiny Shakespeare by default, or BPE + larger corpora when you use the optional Python data scripts.

## Layout

| Path | Role |
|------|------|
| `src/train.js` | Train on `data/input.txt` (character-level tokenizer) |
| `src/train_youtube.js` | Train on BPE data under `data/` (see below) |
| `src/generate.js` | Sample from a saved checkpoint |
| `src/bpe.js` | Load HuggingFace-style BPE JSON for inference / YouTube training |
| `scripts/prepare.js` | Download Tiny Shakespeare into `data/input.txt` |
| `scripts/prepare_youtube.py` / `scripts/prepare_tinystories.py` | Optional: HF datasets → `data/train.bin`, `data/val.bin`, `tokenizer.json`, `meta.json` |
| `scripts/gpu_probe.js` | Debug CUDA / framework matmul (development helper) |
| `out/` | Checkpoints and `model-final.json` (created when you train; not shipped in the repo) |

## Prerequisites

1. **Node.js** (18+ recommended; uses `fetch` in `scripts/prepare.js`).

2. **`@mni-ml/framework`** is declared as a **local file dependency**:

   ```text
   framework/packages/framework
   ```

   Clone or symlink your copy of the framework into `framework/` at the repo root so `npm install` can resolve it. If this path does not exist, installs will fail until you add it.

3. **Optional Python** (3.10+) for BPE dataset preparation only:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements-data.txt
   ```

## Quick start (Tiny Shakespeare, char-level)

```bash
npm install
npm run download    # data/input.txt
npm run train       # writes checkpoints under out/
npm run generate    # loads out/model-final.json by default
```

`generate` accepts optional args: `[checkpoint] [prompt] [numTokens] [temperature] [tokenizerPath]`.

Training honors env vars such as `MAX_ITERS`, `BATCH_SIZE`, `LR`, `N_LAYER`, `N_HEAD`, `N_EMBD`, `BLOCK_SIZE`, `CHECKPOINT_EVERY`, `MODEL_DIR`, `NO_RESUME=1`. See `src/train.js` for the full list.

## BPE + larger data (optional)

1. Prepare binary token files and metadata (example: YouTube-Commons transcripts):

   ```bash
   npm run prepare:youtube
   # or: python3 scripts/prepare_youtube.py
   ```

   Or TinyStories:

   ```bash
   npm run prepare:tinystories
   ```

2. Train:

   ```bash
   npm run train:youtube
   ```

3. Generate with the same tokenizer as training (pass path to `data/tokenizer.json` if needed):

   ```bash
   node src/generate.js out/model-final.json "\n" 300 0.8 data/tokenizer.json
   ```

## Development

- **`npm run probe:gpu`** — optional; exercises CUDA libraries and `@mni-ml/framework` matmul (expects `koffi` where used in `scripts/gpu_probe.js`).


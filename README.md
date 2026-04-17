# mini-gpt

A small GPT-style transformer in **Node.js**, trained with [`@mni-ml/framework`](https://www.npmjs.com/package/@mni-ml/framework) on **BPE-tokenized** text. You can build the dataset from **TinyStories** or **YouTube-Commons** transcripts using the included Python scripts; training and sampling run in JavaScript.

## Layout

| Path | Role |
|------|------|
| `src/train.js` | Train on `data/train.bin` / `data/val.bin` + `data/tokenizer.json` |
| `src/generate.js` | Sample from a checkpoint |
| `src/bpe.js` | HuggingFace-style BPE JSON loader (ByteLevel) |
| `scripts/prepare_tinystories.py` | Download TinyStories, train BPE, write `data/*.bin` and metadata |
| `scripts/prepare_youtube.py` | Download YouTube-Commons transcripts, same output layout |
| `scripts/gpu_probe.js` | Optional CUDA / framework matmul check |
| `out/` | Checkpoints and `model-final.json` (created when you train; ignored by git) |

## Prerequisites

- **Node.js** ≥ **22.18** (required by `@mni-ml/framework`).
- **Python** 3.10+ with packages from `requirements-data.txt` (for dataset + tokenizer prep only).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-data.txt
```

Install JS dependencies:

```bash
npm install
```

## Train on TinyStories

1. **Prepare data** (downloads stories, trains BPE, encodes to `data/train.bin`, `data/val.bin`, `data/meta.json`, `data/tokenizer.json`):

   ```bash
   npm run prepare:tinystories
   ```

   Tunables via environment variables (see `scripts/prepare_tinystories.py`): e.g. `NUM_STORIES`, `VOCAB_SIZE`, `DATA_DIR`.

2. **Train**:

   ```bash
   npm run train
   ```

   Checkpoints go under `out/`; the last save is `out/model-final.json`.

3. **Generate** (tokenizer path is stored in the checkpoint; override with the 5th arg if needed):

   ```bash
   npm run generate
   ```

   ```bash
   node src/generate.js out/model-final.json "<|endoftext|>" 400 0.9 data/tokenizer.json
   ```

## Train on YouTube-Commons

1. **Prepare data** (transcripts from HuggingFace, then BPE + bins):

   ```bash
   npm run prepare:youtube
   ```

   Tunables: `NUM_TRANSCRIPTS`, `VOCAB_SIZE`, `DATA_DIR` (see `scripts/prepare_youtube.py`).

2. **Train** and **generate** — same as above (`npm run train`, `npm run generate`).

## Training options

`src/train.js` reads hyperparameters from the environment, including:

`MAX_ITERS`, `BATCH_SIZE`, `LR`, `N_LAYER`, `N_HEAD`, `N_EMBD`, `BLOCK_SIZE`, `CHECKPOINT_EVERY`, `MODEL_DIR`, `DATA_DIR`, `GRAD_ACCUM_STEPS`, `NO_RESUME=1`.

## Development

- **`npm run probe:gpu`** — optional CUDA / `@mni-ml/framework` check (`scripts/gpu_probe.js`).

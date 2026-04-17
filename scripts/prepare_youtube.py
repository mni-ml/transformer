"""
Download YouTube-Commons transcripts from HuggingFace, train a BPE tokenizer,
and save pre-tokenized binary data for training.

Run from the repo root: python3 scripts/prepare_youtube.py

Requirements: pip install -r requirements-data.txt (or datasets tokenizers huggingface_hub pyarrow)
"""
import os
import json
import array

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_ROOT, "data"))
NUM_TRANSCRIPTS = int(os.environ.get("NUM_TRANSCRIPTS", "10000"))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "4096"))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(os.path.join(DATA_DIR, "train.bin")):
        print("Data already prepared, skipping.")
        return

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    # ── Download transcripts ─────────────────────────────────────────
    print(f"Downloading YouTube-Commons transcripts (target: {NUM_TRANSCRIPTS})...")
    texts = []
    total_chars = 0
    file_idx = 0

    while len(texts) < NUM_TRANSCRIPTS:
        fname = f"cctube_{file_idx}.parquet"
        print(f"  Downloading {fname}...")
        local = hf_hub_download(
            repo_id="PleIAs/YouTube-Commons",
            filename=fname,
            repo_type="dataset",
        )
        table = pq.read_table(local, columns=["text", "original_language", "transcription_language"])
        for i in range(table.num_rows):
            orig_lang = table.column("original_language")[i].as_py() or ""
            trans_lang = table.column("transcription_language")[i].as_py() or ""
            if orig_lang != "en" or trans_lang != "en":
                continue
            text = (table.column("text")[i].as_py() or "").strip()
            if len(text) < 200:
                continue
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
            if ascii_ratio < 0.90:
                continue
            texts.append(text)
            total_chars += len(text)
            if len(texts) >= NUM_TRANSCRIPTS:
                break
        print(f"    Collected {len(texts)} transcripts so far ({total_chars / 1024 / 1024:.1f} MB)")
        file_idx += 1
        if file_idx > 20:
            break

    print(f"  Total: {len(texts)} transcripts, {total_chars / 1024 / 1024:.1f} MB")

    # ── Train BPE tokenizer ──────────────────────────────────────────
    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer)

    eot_id = tokenizer.token_to_id("<|endoftext|>")
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab_size}, <|endoftext|> ID = {eot_id}")

    tok_path = os.path.join(DATA_DIR, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"  Saved tokenizer to {tok_path}")

    # ── Encode all transcripts ───────────────────────────────────────
    print("Encoding transcripts...")
    all_tokens = []
    CHUNK = 500
    for i in range(0, len(texts), CHUNK):
        chunk = texts[i : i + CHUNK]
        outputs = tokenizer.encode_batch(chunk)
        for output in outputs:
            all_tokens.extend(output.ids)
            all_tokens.append(eot_id)
        done = min(i + CHUNK, len(texts))
        if done % 2000 == 0 or done == len(texts):
            print(f"  Encoded {done}/{len(texts)} ({len(all_tokens):,} tokens)")

    print(f"  Total tokens: {len(all_tokens):,}")

    # ── Train / validation split (90 / 10) ───────────────────────────
    split_idx = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    for name, data in [("train", train_tokens), ("val", val_tokens)]:
        path = os.path.join(DATA_DIR, f"{name}.bin")
        a = array.array("i", data)
        with open(path, "wb") as f:
            a.tofile(f)
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {name}: {len(data):,} tokens ({mb:.1f} MB)")

    # ── Metadata ─────────────────────────────────────────────────────
    meta = {
        "vocab_size": actual_vocab_size,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "num_transcripts": len(texts),
        "eot_token": eot_id,
    }
    meta_path = os.path.join(DATA_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {json.dumps(meta, indent=2)}")
    print("Done!")


if __name__ == "__main__":
    main()

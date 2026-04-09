/**
 * BPE tokenizer for Node.js — loads a HuggingFace tokenizers JSON file
 * (trained with ByteLevel pre-tokenizer) and provides encode / decode.
 *
 * Compatible with the output of prepare_youtube.py.
 */
import { readFileSync } from 'node:fs';

// ── GPT-2 byte encoder / decoder ──────────────────────────────────
// Maps every byte (0-255) to a printable unicode character so that BPE
// can operate on "visible" strings.  This is the standard mapping used
// by GPT-2 and the HuggingFace ByteLevel pre-tokenizer.

function buildByteEncoder() {
  const bs = [];
  for (let i = 0x21; i <= 0x7E; i++) bs.push(i); // ! to ~
  for (let i = 0xA1; i <= 0xAC; i++) bs.push(i); // ¡ to ¬
  for (let i = 0xAE; i <= 0xFF; i++) bs.push(i); // ® to ÿ

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const encoder = {};
  const decoder = {};
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    encoder[bs[i]] = ch;
    decoder[ch] = bs[i];
  }
  return { encoder, decoder };
}

const { encoder: BYTE_ENC, decoder: BYTE_DEC } = buildByteEncoder();

// GPT-2 pre-tokenizer regex (matches the HuggingFace ByteLevel default)
const GPT2_PAT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// ── BPETokenizer class ────────────────────────────────────────────

export class BPETokenizer {
  /**
   * @param {string} tokenizerJsonPath - path to tokenizer.json (HuggingFace format)
   */
  constructor(tokenizerJsonPath) {
    const raw = JSON.parse(readFileSync(tokenizerJsonPath, 'utf-8'));
    const model = raw.model;

    // vocab: unicode string → token id
    this.vocab = { ...model.vocab };

    // reverse vocab: token id → unicode string
    this.idToToken = {};
    for (const [tok, id] of Object.entries(this.vocab)) {
      this.idToToken[id] = tok;
    }

    // added tokens (e.g. <|endoftext|>)
    if (raw.added_tokens) {
      for (const at of raw.added_tokens) {
        if (at.id !== undefined && at.content) {
          this.vocab[at.content] = at.id;
          this.idToToken[at.id] = at.content;
        }
      }
    }

    this.vocabSize = Math.max(
      Object.keys(this.vocab).length,
      ...Object.keys(this.idToToken).map(Number),
    ) + 1;

    // merge priority: "tok1 tok2" → rank (lower = merge first)
    this.bpeRanks = {};
    if (model.merges) {
      for (let i = 0; i < model.merges.length; i++) {
        this.bpeRanks[model.merges[i]] = i;
      }
    }
  }

  // ── encode ──────────────────────────────────────────────────────

  encode(text) {
    if (!text) return [];
    const ids = [];
    const matches = text.match(GPT2_PAT);
    if (!matches) return ids;

    for (const word of matches) {
      const bytes = Buffer.from(word, 'utf-8');
      const unicoded = Array.from(bytes)
        .map((b) => BYTE_ENC[b])
        .join('');

      let pieces = [...unicoded]; // one char per element
      pieces = this._bpe(pieces);

      for (const piece of pieces) {
        const id = this.vocab[piece];
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  /** Apply BPE merges to an array of single-character strings. */
  _bpe(pieces) {
    while (pieces.length > 1) {
      let bestIdx = -1;
      let bestRank = Infinity;

      for (let i = 0; i < pieces.length - 1; i++) {
        const key = pieces[i] + ' ' + pieces[i + 1];
        const rank = this.bpeRanks[key];
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;

      const merged = pieces[bestIdx] + pieces[bestIdx + 1];
      pieces = [
        ...pieces.slice(0, bestIdx),
        merged,
        ...pieces.slice(bestIdx + 2),
      ];
    }
    return pieces;
  }

  // ── decode ──────────────────────────────────────────────────────

  decode(ids) {
    const strs = ids.map((id) => this.idToToken[id] || '');
    const joined = strs.join('');
    const bytes = [];
    for (const ch of joined) {
      const b = BYTE_DEC[ch];
      if (b !== undefined) bytes.push(b);
    }
    return Buffer.from(bytes).toString('utf-8');
  }

  // ── helpers ─────────────────────────────────────────────────────

  get eotToken() {
    return this.vocab['<|endoftext|>'] ?? -1;
  }
}

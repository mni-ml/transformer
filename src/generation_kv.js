import {
  Tensor, native,
  Module, Parameter,
  Linear, Embedding, KvCache,
  softmax, gelu, layerNorm, flashAttention,
} from '@mni-ml/framework';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { BPETokenizer } from './bpe.js';

class LayerNorm extends Module {
  constructor(dim) {
    super();
    const gammaData = new Float32Array(dim).fill(1.0);
    const betaData = new Float32Array(dim).fill(0.0);
    this.gamma = new Parameter(Tensor.fromFloat32(gammaData, [dim]).setRequiresGrad(true));
    this.beta = new Parameter(Tensor.fromFloat32(betaData, [dim]).setRequiresGrad(true));
  }

  forward(x) {
    return layerNorm(x, this.gamma.value, this.beta.value, 1e-5);
  }
}

const causalMaskCache = new Map();

function getCausalMask(size) {
  if (causalMaskCache.has(size)) {
    return causalMaskCache.get(size);
  }
  const storage = new Float32Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      storage[i * size + j] = j <= i ? 0.0 : -1e9;
    }
  }
  const mask = Tensor.fromFloat32(storage, [1, 1, size, size]);
  causalMaskCache.set(size, mask);
  return mask;
}

function reshapeForAttention(x, batch, seqLen, nHead, headDim) {
  return x.view(batch, seqLen, nHead, headDim).permute(0, 2, 1, 3).contiguous();
}

function sampleNextToken(logits, vocabSize, temperature = 0.8) {
  const data = logits.toFloat32();
  const offset = data.length - vocabSize;
  let nextToken = 0;

  if (temperature <= 0) {
    let best = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const value = data[offset + i];
      if (value > best) {
        best = value;
        nextToken = i;
      }
    }
    return nextToken;
  }

  const scaled = new Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    scaled[i] = data[offset + i] / temperature;
  }
  const maxLogit = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - maxLogit));
  const sumExps = exps.reduce((sum, value) => sum + value, 0);
  const probs = exps.map((value) => value / sumExps);

  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) {
      return i;
    }
  }
  return probs.length - 1;
}

function ensurePromptTokens(tokenizer, prompt) {
  const tokens = tokenizer.encode(prompt);
  if (tokens.length > 0) {
    return tokens;
  }
  if (tokenizer.eotToken >= 0) {
    return [tokenizer.eotToken];
  }
  throw new Error('prompt must encode to at least one token');
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} b`;
  const units = ['kb', 'mb', 'gb'];
  let value = bytes;
  let unitIndex = -1;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(2)} ${units[unitIndex]}`;
}

function cacheBytesForMode(config, cacheLen, cacheMode) {
  if (cacheMode === 'none') {
    return { usedBytes: 0, capacityBytes: 0 };
  }
  const headDim = config.nEmbd / config.nHead;
  const rowsPerToken = config.nLayer * config.nHead;
  const valueBytesPerRow = cacheMode === 'int8' ? headDim : headDim * 4;
  const scaleBytesPerRow = cacheMode === 'int8' ? 4 : 0;
  const bytesPerToken = 2 * rowsPerToken * (valueBytesPerRow + scaleBytesPerRow);
  return {
    usedBytes: cacheLen * bytesPerToken,
    capacityBytes: config.blockSize * bytesPerToken,
  };
}

class CausalSelfAttention extends Module {
  constructor(nEmbd, nHead, dropoutRate = 0) {
    super();
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.scale = 1.0 / Math.sqrt(this.headDim);
    this.dropoutRate = dropoutRate;
    this.queryProj = new Linear(nEmbd, nEmbd);
    this.keyProj = new Linear(nEmbd, nEmbd);
    this.valueProj = new Linear(nEmbd, nEmbd);
    this.outProj = new Linear(nEmbd, nEmbd);
  }

  forward(x) {
    const [batch, seqLen, embd] = x.shape;
    const { nHead, headDim, scale } = this;
    let q = this.queryProj.forward(x);
    let k = this.keyProj.forward(x);
    let v = this.valueProj.forward(x);

    q = q.view(batch, seqLen, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(batch * nHead, seqLen, headDim);
    k = k.view(batch, seqLen, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(batch * nHead, seqLen, headDim);
    v = v.view(batch, seqLen, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(batch * nHead, seqLen, headDim);

    let out;
    if (typeof native.flashAttention === 'function') {
      out = flashAttention(q, k, v, scale, true);
    } else {
      let scores = q.matmul(k.permute(0, 2, 1)).mul(scale);
      const mask = getCausalMask(seqLen).view(seqLen, seqLen);
      scores = scores.add(mask);
      const probs = softmax(scores, -1);
      out = probs.matmul(v);
    }

    out = out.view(batch, nHead, seqLen, headDim).permute(0, 2, 1, 3).contiguous().view(batch, seqLen, embd);
    return this.outProj.forward(out);
  }

  forwardStep(x, cache) {
    const [batch, seqLen, embd] = x.shape;
    if (seqLen !== 1) {
      throw new Error(`kv attention expects seq_len=1, got ${seqLen}`);
    }
    let q = reshapeForAttention(this.queryProj.forward(x), batch, seqLen, this.nHead, this.headDim);
    let k = reshapeForAttention(this.keyProj.forward(x), batch, seqLen, this.nHead, this.headDim);
    let v = reshapeForAttention(this.valueProj.forward(x), batch, seqLen, this.nHead, this.headDim);
    let out = cache.decodeStep(q, k, v, this.scale);
    out = out.permute(0, 2, 1, 3).contiguous().view(batch, seqLen, embd);
    return this.outProj.forward(out);
  }
}

class FeedForward extends Module {
  constructor(nEmbd, dropoutRate = 0) {
    super();
    this.dropoutRate = dropoutRate;
    this.fc1 = new Linear(nEmbd, 4 * nEmbd);
    this.fc2 = new Linear(4 * nEmbd, nEmbd);
  }

  forward(x) {
    let h = this.fc1.forward(x);
    h = gelu(h);
    return this.fc2.forward(h);
  }
}

class TransformerBlock extends Module {
  constructor(nEmbd, nHead, dropoutRate = 0) {
    super();
    this.ln1 = new LayerNorm(nEmbd);
    this.attn = new CausalSelfAttention(nEmbd, nHead, dropoutRate);
    this.ln2 = new LayerNorm(nEmbd);
    this.ffn = new FeedForward(nEmbd, dropoutRate);
  }

  forward(x) {
    x = x.add(this.attn.forward(this.ln1.forward(x)));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }

  forwardStep(x, cache) {
    x = x.add(this.attn.forwardStep(this.ln1.forward(x), cache));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }
}

export class MiniGPT extends Module {
  constructor(vocabSize, config) {
    super();
    const { nEmbd, nHead, nLayer, blockSize, dropoutRate = 0 } = config;
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);
    for (let i = 0; i < nLayer; i++) {
      this[`block${i}`] = new TransformerBlock(nEmbd, nHead, dropoutRate);
    }
    this.lnFinal = new LayerNorm(nEmbd);
    this.headBias = new Parameter(Tensor.zeros([vocabSize]).setRequiresGrad(true));
  }

  forward(indices) {
    const batch = indices.length;
    const seqLen = indices[0].length;
    let x = this.tokenEmb.forward(indices);
    const posIndices = [];
    for (let b = 0; b < batch; b++) {
      posIndices.push(Array.from({ length: seqLen }, (_, i) => i));
    }
    x = x.add(this.posEmb.forward(posIndices));
    for (let i = 0; i < this.config.nLayer; i++) {
      x = this[`block${i}`].forward(x);
    }
    x = this.lnFinal.forward(x);
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(wT).add(this.headBias.value);
  }

  createCaches(cacheMode) {
    if (typeof native.kvCacheCreate !== 'function') {
      throw new Error('kv cache APIs are unavailable in the current framework build');
    }
    if (cacheMode !== 'fp32' && cacheMode !== 'int8') {
      throw new Error(`unsupported cache mode: ${cacheMode}`);
    }
    const caches = [];
    const quantized = cacheMode === 'int8';
    const headDim = this.config.nEmbd / this.config.nHead;
    for (let i = 0; i < this.config.nLayer; i++) {
      caches.push(new KvCache(1, this.config.nHead, headDim, this.config.blockSize, quantized));
    }
    return caches;
  }

  forwardToken(tokenId, position, caches) {
    let x = this.tokenEmb.forward([[tokenId]]);
    x = x.add(this.posEmb.forward([[position]]));
    for (let i = 0; i < this.config.nLayer; i++) {
      x = this[`block${i}`].forwardStep(x, caches[i]);
    }
    x = this.lnFinal.forward(x);
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(wT).add(this.headBias.value);
  }
}

function resolveTokenizerPath(checkpoint, tokenizerOverridePath) {
  const tokPath = tokenizerOverridePath || checkpoint.tokenizerPath;
  if (!tokPath) {
    throw new Error('checkpoint has no tokenizerPath. pass data/tokenizer.json as the 5th CLI argument');
  }
  if (!existsSync(tokPath)) {
    throw new Error(`tokenizer not found: ${tokPath}`);
  }
  if (statSync(tokPath).isDirectory()) {
    throw new Error(
      `tokenizer path points to a directory: ${tokPath}. pass the tokenizer.json file path instead; if your shell wrapped the command, keep /tmp/transformer-weights/tokenizer.json on one line`
    );
  }
  return tokPath;
}

export function loadModel(path, tokenizerOverridePath = null) {
  const checkpoint = JSON.parse(readFileSync(path, 'utf-8'));
  const tokPath = resolveTokenizerPath(checkpoint, tokenizerOverridePath);
  const tokenizer = new BPETokenizer(tokPath);
  const model = new MiniGPT(tokenizer.vocabSize, checkpoint.config);

  for (const [name, param] of model.namedParameters()) {
    if (!checkpoint.parameters[name]) {
      continue;
    }
    const { shape, data } = checkpoint.parameters[name];
    const tensor = Tensor.fromFloat32(new Float32Array(data), shape);
    tensor.setRequiresGrad(true);
    param.update(tensor);
  }

  return { model, tokenizer, config: checkpoint.config };
}

export function generateBaseline(model, tokenizer, prompt, maxTokens, temperature = 0.8) {
  model.eval();
  const context = ensurePromptTokens(tokenizer, prompt);
  const blockSize = model.config.blockSize;
  const start = performance.now();
  native.noGradStart();
  try {
    for (let i = 0; i < maxTokens; i++) {
      const ctxWindow = context.slice(-blockSize);
      const logits = model.forward([ctxWindow]);
      context.push(sampleNextToken(logits, tokenizer.vocabSize, temperature));
    }
  } finally {
    native.noGradEnd();
  }
  const totalMs = performance.now() - start;
  return {
    mode: 'baseline',
    cacheMode: 'none',
    promptTokens: context.length - maxTokens,
    generatedTokens: maxTokens,
    totalMs,
    prefillMs: null,
    decodeMs: totalMs,
    decodeTokens: maxTokens,
    cacheLen: 0,
    usedBytes: 0,
    capacityBytes: 0,
    text: tokenizer.decode(context),
  };
}

export function generateWithKvCache(model, tokenizer, prompt, maxTokens, temperature = 0.8, cacheMode = 'fp32') {
  const promptTokens = ensurePromptTokens(tokenizer, prompt);
  const maxCacheLen = promptTokens.length + Math.max(0, maxTokens - 1);
  if (maxCacheLen > model.config.blockSize) {
    throw new Error(
      `kv cache demo requires prompt_tokens + max_new_tokens - 1 <= blockSize (${maxCacheLen} > ${model.config.blockSize})`
    );
  }

  const context = [...promptTokens];
  const caches = model.createCaches(cacheMode);
  const prefixTokens = promptTokens.slice(0, -1);
  let currentToken = promptTokens[promptTokens.length - 1];
  let currentPosition = promptTokens.length - 1;
  let prefillMs = 0;
  let decodeMs = 0;

  model.eval();
  native.noGradStart();
  try {
    const prefillStart = performance.now();
    for (let i = 0; i < prefixTokens.length; i++) {
      model.forwardToken(prefixTokens[i], i, caches);
    }
    prefillMs = performance.now() - prefillStart;

    const decodeStart = performance.now();
    for (let step = 0; step < maxTokens; step++) {
      const logits = model.forwardToken(currentToken, currentPosition, caches);
      const nextToken = sampleNextToken(logits, tokenizer.vocabSize, temperature);
      context.push(nextToken);
      currentToken = nextToken;
      currentPosition += 1;
    }
    decodeMs = performance.now() - decodeStart;
  } finally {
    native.noGradEnd();
    for (const cache of caches) {
      cache.free();
    }
  }

  const cacheLen = promptTokens.length - 1 + maxTokens;
  const bytes = cacheBytesForMode(model.config, cacheLen, cacheMode);
  return {
    mode: `kv-${cacheMode}`,
    cacheMode,
    promptTokens: promptTokens.length,
    generatedTokens: maxTokens,
    totalMs: prefillMs + decodeMs,
    prefillMs,
    decodeMs,
    decodeTokens: maxTokens,
    cacheLen,
    usedBytes: bytes.usedBytes,
    capacityBytes: bytes.capacityBytes,
    text: tokenizer.decode(context),
  };
}

function pad(value, width, alignRight = false) {
  const text = String(value);
  return alignRight ? text.padStart(width, ' ') : text.padEnd(width, ' ');
}

export function formatBenchmarkTable(results) {
  const rows = results.map((result) => {
    const decodeTokens = Math.max(result.decodeTokens, 1);
    const decodeMsPerToken = result.decodeMs / decodeTokens;
    const tokensPerSec = result.generatedTokens > 0 ? (result.generatedTokens * 1000) / result.totalMs : 0;
    return {
      mode: result.mode,
      prompt: result.promptTokens,
      generated: result.generatedTokens,
      prefillMs: result.prefillMs == null ? '-' : result.prefillMs.toFixed(1),
      totalMs: result.totalMs.toFixed(1),
      decodeMsPerToken: decodeMsPerToken.toFixed(2),
      tokensPerSec: tokensPerSec.toFixed(2),
      cacheLen: result.cacheLen || '-',
      bytesUsed: result.usedBytes ? formatBytes(result.usedBytes) : '-',
      bytesCapacity: result.capacityBytes ? formatBytes(result.capacityBytes) : '-',
    };
  });

  const widths = {
    mode: Math.max('mode'.length, ...rows.map((row) => row.mode.length)),
    prompt: Math.max('prompt'.length, ...rows.map((row) => String(row.prompt).length)),
    generated: Math.max('new'.length, ...rows.map((row) => String(row.generated).length)),
    prefillMs: Math.max('prefill_ms'.length, ...rows.map((row) => row.prefillMs.length)),
    totalMs: Math.max('total_ms'.length, ...rows.map((row) => row.totalMs.length)),
    decodeMsPerToken: Math.max('decode_ms/token'.length, ...rows.map((row) => row.decodeMsPerToken.length)),
    tokensPerSec: Math.max('tok/s'.length, ...rows.map((row) => row.tokensPerSec.length)),
    cacheLen: Math.max('cache_len'.length, ...rows.map((row) => String(row.cacheLen).length)),
    bytesUsed: Math.max('cache_used'.length, ...rows.map((row) => row.bytesUsed.length)),
    bytesCapacity: Math.max('cache_cap'.length, ...rows.map((row) => row.bytesCapacity.length)),
  };

  const header = [
    pad('mode', widths.mode),
    pad('prompt', widths.prompt, true),
    pad('new', widths.generated, true),
    pad('prefill_ms', widths.prefillMs, true),
    pad('total_ms', widths.totalMs, true),
    pad('decode_ms/token', widths.decodeMsPerToken, true),
    pad('tok/s', widths.tokensPerSec, true),
    pad('cache_len', widths.cacheLen, true),
    pad('cache_used', widths.bytesUsed, true),
    pad('cache_cap', widths.bytesCapacity, true),
  ].join('  ');

  const separator = '-'.repeat(header.length);
  const lines = rows.map((row) => [
    pad(row.mode, widths.mode),
    pad(row.prompt, widths.prompt, true),
    pad(row.generated, widths.generated, true),
    pad(row.prefillMs, widths.prefillMs, true),
    pad(row.totalMs, widths.totalMs, true),
    pad(row.decodeMsPerToken, widths.decodeMsPerToken, true),
    pad(row.tokensPerSec, widths.tokensPerSec, true),
    pad(row.cacheLen, widths.cacheLen, true),
    pad(row.bytesUsed, widths.bytesUsed, true),
    pad(row.bytesCapacity, widths.bytesCapacity, true),
  ].join('  '));

  return [header, separator, ...lines].join('\n');
}

/**
 * Speculative decoding demo.
 *
 *   target  = large MiniGPT (slow, accurate)   — out/model-final.json
 *   draft   = small MiniGPT (fast, approximate) — out/draft-final.json
 *
 * Both models MUST share the same tokenizer / vocab. The output distribution
 * is provably identical to running the target alone (Leviathan et al., 2023).
 *
 * Usage:
 *   node generate_speculative.js \
 *        out/model-final.json out/draft-final.json \
 *        "Once upon a time" 200 0.8 4
 *
 *   args: target_path draft_path prompt num_tokens temperature K [tokenizer_path]
 *   K = number of tokens drafted per outer step (default 4)
 */
import {
  Tensor, native,
  Module, Parameter,
  Linear, Embedding,
  softmax, gelu, layerNorm, flashAttention,
} from '@mni-ml/framework';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BPETokenizer } from './bpe.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Model architecture (matches train_youtube.js / generate.js) ─────

class LayerNorm extends Module {
  constructor(dim) {
    super();
    this.dim = dim;
    this.gamma = new Parameter(Tensor.fromFloat32(new Float32Array(dim).fill(1.0), [dim]).setRequiresGrad(true));
    this.beta = new Parameter(Tensor.fromFloat32(new Float32Array(dim).fill(0.0), [dim]).setRequiresGrad(true));
  }
  forward(x) { return layerNorm(x, this.gamma.value, this.beta.value, 1e-5); }
}

const _causalMaskCache = new Map();
function getCausalMask(size) {
  if (_causalMaskCache.has(size)) return _causalMaskCache.get(size);
  const storage = new Float32Array(size * size);
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++)
      storage[i * size + j] = j <= i ? 0.0 : -1e9;
  const mask = Tensor.fromFloat32(storage, [1, 1, size, size]);
  _causalMaskCache.set(size, mask);
  return mask;
}

class CausalSelfAttention extends Module {
  constructor(nEmbd, nHead) {
    super();
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.scale = 1.0 / Math.sqrt(this.headDim);
    this.queryProj = new Linear(nEmbd, nEmbd);
    this.keyProj = new Linear(nEmbd, nEmbd);
    this.valueProj = new Linear(nEmbd, nEmbd);
    this.outProj = new Linear(nEmbd, nEmbd);
  }
  forward(x) {
    const [B, S, E] = x.shape;
    const { nHead, headDim, scale } = this;
    let q = this.queryProj.forward(x);
    let k = this.keyProj.forward(x);
    let v = this.valueProj.forward(x);
    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    k = k.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);

    let out;
    if (typeof native.flashAttention === 'function') {
      out = flashAttention(q, k, v, scale, true);
    } else {
      let scores = q.matmul(k.permute(0, 2, 1)).mul(scale);
      const mask = getCausalMask(S).view(S, S);
      scores = scores.add(mask);
      const probs = softmax(scores, -1);
      out = probs.matmul(v);
    }
    out = out.view(B, nHead, S, headDim).permute(0, 2, 1, 3).contiguous().view(B, S, E);
    return this.outProj.forward(out);
  }
}

class FeedForward extends Module {
  constructor(nEmbd) {
    super();
    this.fc1 = new Linear(nEmbd, 4 * nEmbd);
    this.fc2 = new Linear(4 * nEmbd, nEmbd);
  }
  forward(x) { return this.fc2.forward(gelu(this.fc1.forward(x))); }
}

class TransformerBlock extends Module {
  constructor(nEmbd, nHead) {
    super();
    this.ln1 = new LayerNorm(nEmbd);
    this.attn = new CausalSelfAttention(nEmbd, nHead);
    this.ln2 = new LayerNorm(nEmbd);
    this.ffn = new FeedForward(nEmbd);
  }
  forward(x) {
    x = x.add(this.attn.forward(this.ln1.forward(x)));
    x = x.add(this.ffn.forward(this.ln2.forward(x)));
    return x;
  }
}

class MiniGPT extends Module {
  constructor(vocabSize, config) {
    super();
    const { nEmbd, nHead, nLayer, blockSize } = config;
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);
    for (let i = 0; i < nLayer; i++)
      this[`block${i}`] = new TransformerBlock(nEmbd, nHead);
    this.lnFinal = new LayerNorm(nEmbd);
    this.headBias = new Parameter(Tensor.zeros([vocabSize]).setRequiresGrad(true));
  }
  forward(indices) {
    const batch = indices.length;
    const seqLen = indices[0].length;
    let x = this.tokenEmb.forward(indices);
    const posIndices = [];
    for (let b = 0; b < batch; b++)
      posIndices.push(Array.from({ length: seqLen }, (_, i) => i));
    x = x.add(this.posEmb.forward(posIndices));
    for (let i = 0; i < this.config.nLayer; i++)
      x = this[`block${i}`].forward(x);
    x = this.lnFinal.forward(x);
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(wT).add(this.headBias.value);
  }
}

// ── Loader ──────────────────────────────────────────────────────────

function loadModel(path, tokenizerOverridePath = null) {
  const checkpoint = JSON.parse(readFileSync(path, 'utf-8'));
  let tokenizer;
  if (checkpoint.tokenizerPath || tokenizerOverridePath) {
    const tokPath = tokenizerOverridePath || checkpoint.tokenizerPath;
    tokenizer = new BPETokenizer(tokPath);
  } else {
    throw new Error(`Checkpoint at ${path} has no tokenizer metadata.`);
  }
  const model = new MiniGPT(tokenizer.vocabSize, checkpoint.config);
  for (const [name, param] of model.namedParameters()) {
    if (checkpoint.parameters[name]) {
      const { shape, data } = checkpoint.parameters[name];
      const t = Tensor.fromFloat32(new Float32Array(data), shape);
      t.setRequiresGrad(true);
      param.update(t);
    }
  }
  return { model, tokenizer, config: checkpoint.config };
}

// ── Math helpers ────────────────────────────────────────────────────

/** Numerically stable softmax of a Float32Array slice with temperature. */
function softmaxRow(logits, offset, vocabSize, temperature) {
  const out = new Float64Array(vocabSize);
  let maxL = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    const l = logits[offset + v] / temperature;
    out[v] = l;
    if (l > maxL) maxL = l;
  }
  let sum = 0;
  for (let v = 0; v < vocabSize; v++) {
    const e = Math.exp(out[v] - maxL);
    out[v] = e;
    sum += e;
  }
  if (!isFinite(sum) || sum === 0) {
    out.fill(1 / vocabSize);
    return out;
  }
  for (let v = 0; v < vocabSize; v++) out[v] /= sum;
  return out;
}

function sampleFromProbs(probs) {
  const r = Math.random();
  let cum = 0;
  for (let v = 0; v < probs.length; v++) {
    cum += probs[v];
    if (r < cum) return v;
  }
  return probs.length - 1;
}

function argmaxRow(logits, offset, vocabSize) {
  let best = -Infinity, idx = 0;
  for (let v = 0; v < vocabSize; v++) {
    const l = logits[offset + v];
    if (l > best) { best = l; idx = v; }
  }
  return idx;
}

// ── Plain (non-speculative) generation, for baseline timing ─────────

function generateBaseline(model, vocabSize, blockSize, contextInit, maxTokens, temperature) {
  let context = [...contextInit];
  for (let i = 0; i < maxTokens; i++) {
    const ctxWindow = context.slice(-blockSize);
    const logits = model.forward([ctxWindow]);
    const data = logits.toFloat32();
    const offset = (ctxWindow.length - 1) * vocabSize;
    let nextToken;
    if (temperature <= 0) {
      nextToken = argmaxRow(data, offset, vocabSize);
    } else {
      const probs = softmaxRow(data, offset, vocabSize, temperature);
      nextToken = sampleFromProbs(probs);
    }
    context.push(nextToken);
  }
  return context;
}

// ── Speculative decoding ────────────────────────────────────────────

/**
 * Run K draft steps, then verify with one target forward pass.
 * Returns the new context plus stats: { context, accepted, drafted }.
 *
 * Greedy path (temperature == 0): accept while draft.argmax == target.argmax.
 * Sampling path: standard Leviathan acceptance with corrected resampling.
 */
function speculativeStep({
  target, draft,
  vocabSize,
  blockSize,
  context,
  K,
  temperature,
}) {
  // ── 1. Draft K tokens autoregressively, recording q distributions ──
  const draftedTokens = [];
  const draftProbs = []; // softmax distributions (sampling path only)
  let dCtx = context.slice(-blockSize);

  for (let k = 0; k < K; k++) {
    const logits = draft.forward([dCtx]);
    const data = logits.toFloat32();
    const offset = (dCtx.length - 1) * vocabSize;
    let token;
    if (temperature <= 0) {
      token = argmaxRow(data, offset, vocabSize);
    } else {
      const q = softmaxRow(data, offset, vocabSize, temperature);
      token = sampleFromProbs(q);
      draftProbs.push(q);
    }
    draftedTokens.push(token);
    dCtx = dCtx.concat([token]);
    if (dCtx.length > blockSize) dCtx = dCtx.slice(-blockSize);
  }

  // ── 2. Verify with ONE target forward pass on context + drafts ─────
  let extCtx = context.concat(draftedTokens);
  if (extCtx.length > blockSize) extCtx = extCtx.slice(-blockSize);
  const T_len = extCtx.length;

  // The drafts occupy positions [T_len - K .. T_len - 1].
  // Distribution that predicts the i-th draft (0-indexed) lives at
  // logits row (T_len - K - 1 + i). The bonus distribution lives at row (T_len - 1).
  const targetLogits = target.forward([extCtx]).toFloat32();

  // ── 3. Verification loop ───────────────────────────────────────────
  let accepted = 0;
  if (temperature <= 0) {
    // Greedy verification: accept iff draft == target argmax at that position.
    for (let i = 0; i < K; i++) {
      const row = (T_len - K - 1 + i) * vocabSize;
      const tArg = argmaxRow(targetLogits, row, vocabSize);
      if (tArg === draftedTokens[i]) {
        context.push(draftedTokens[i]);
        accepted++;
      } else {
        context.push(tArg);
        return { context, accepted, drafted: K };
      }
    }
    // All K accepted → bonus token from row (T_len - 1)
    const bonusRow = (T_len - 1) * vocabSize;
    context.push(argmaxRow(targetLogits, bonusRow, vocabSize));
    return { context, accepted, drafted: K };
  }

  for (let i = 0; i < K; i++) {
    const row = (T_len - K - 1 + i) * vocabSize;
    const p = softmaxRow(targetLogits, row, vocabSize, temperature);
    const q = draftProbs[i];
    const dTok = draftedTokens[i];

    const ratio = p[dTok] / Math.max(q[dTok], 1e-30);
    if (Math.random() < Math.min(1, ratio)) {
      context.push(dTok);
      accepted++;
    } else {
      // Resample from normalised max(0, p - q)
      const adjusted = new Float64Array(vocabSize);
      let sum = 0;
      for (let v = 0; v < vocabSize; v++) {
        const diff = p[v] - q[v];
        if (diff > 0) { adjusted[v] = diff; sum += diff; }
      }
      let resampled;
      if (sum > 0) {
        for (let v = 0; v < vocabSize; v++) adjusted[v] /= sum;
        resampled = sampleFromProbs(adjusted);
      } else {
        // q already dominates p → fall back to p
        resampled = sampleFromProbs(p);
      }
      context.push(resampled);
      return { context, accepted, drafted: K };
    }
  }

  // All K accepted → bonus draw from p at row (T_len - 1)
  const bonusRow = (T_len - 1) * vocabSize;
  const pBonus = softmaxRow(targetLogits, bonusRow, vocabSize, temperature);
  context.push(sampleFromProbs(pBonus));
  return { context, accepted, drafted: K };
}

function generateSpeculative(target, draft, vocabSize, blockSize, contextInit, maxTokens, temperature, K) {
  let context = [...contextInit];
  let totalAccepted = 0;
  let totalDrafted = 0;
  let outerSteps = 0;
  const startLen = context.length;

  while (context.length - startLen < maxTokens) {
    const res = speculativeStep({
      target, draft,
      vocabSize, blockSize,
      context,
      K,
      temperature,
    });
    context = res.context;
    totalAccepted += res.accepted;
    totalDrafted += res.drafted;
    outerSteps++;
  }

  return {
    context: context.slice(0, startLen + maxTokens),
    outerSteps,
    totalAccepted,
    totalDrafted,
  };
}

// ── Main ────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const targetPath = args[0] || join(__dirname, 'out', 'model-final.json');
const draftPath = args[1] || join(__dirname, 'out', 'draft-final.json');
const prompt = args[2] || 'Once upon a time';
const numTokens = parseInt(args[3] || '200');
const temperature = parseFloat(args[4] || '0.8');
const K = parseInt(args[5] || '4');
const tokenizerOverride = args[6] || null;

if (!existsSync(targetPath)) {
  console.error(`Target model not found: ${targetPath}`);
  process.exit(1);
}
if (!existsSync(draftPath)) {
  console.error(`Draft model not found: ${draftPath}`);
  console.error('Train one with: modal run modal_train_draft.py');
  process.exit(1);
}

console.log(`Loading target from ${targetPath}...`);
const { model: target, tokenizer: tokT, config: cfgT } = loadModel(targetPath, tokenizerOverride);
console.log(`  target: ${cfgT.nLayer}L / ${cfgT.nHead}H / ${cfgT.nEmbd}d  (vocab=${tokT.vocabSize})`);

console.log(`Loading draft  from ${draftPath}...`);
const { model: draft, tokenizer: tokD, config: cfgD } = loadModel(draftPath, tokenizerOverride);
console.log(`  draft:  ${cfgD.nLayer}L / ${cfgD.nHead}H / ${cfgD.nEmbd}d  (vocab=${tokD.vocabSize})`);

if (tokT.vocabSize !== tokD.vocabSize) {
  console.error(`\nVOCAB MISMATCH: target=${tokT.vocabSize}, draft=${tokD.vocabSize}`);
  console.error('Speculative decoding requires identical vocabularies.');
  process.exit(1);
}
const blockSize = Math.min(cfgT.blockSize, cfgD.blockSize);
const vocabSize = tokT.vocabSize;

target.eval();
draft.eval();

const promptIds = tokT.encode(prompt);

// ── Baseline timing (target alone) ──────────────────────────────────
console.log('');
console.log('─── Baseline (target only) ─────────────────────────────────');
native.noGradStart();
const t0 = Date.now();
const baseline = generateBaseline(target, vocabSize, blockSize, promptIds, numTokens, temperature);
const t1 = Date.now();
native.noGradEnd();

const baselineSec = (t1 - t0) / 1000;
console.log(`  ${numTokens} tokens in ${baselineSec.toFixed(2)}s  (${(numTokens / baselineSec).toFixed(1)} tok/s)`);
console.log('');
console.log(tokT.decode(baseline).replaceAll('<|endoftext|>', '⏎').slice(0, 600));

// ── Speculative decoding ────────────────────────────────────────────
console.log('');
console.log(`─── Speculative decoding (K=${K}) ──────────────────────────`);
native.noGradStart();
const s0 = Date.now();
const spec = generateSpeculative(target, draft, vocabSize, blockSize, promptIds, numTokens, temperature, K);
const s1 = Date.now();
native.noGradEnd();

const specSec = (s1 - s0) / 1000;
const acceptanceRate = spec.totalAccepted / Math.max(spec.totalDrafted, 1);
const tokensPerOuterStep = (spec.context.length - promptIds.length) / spec.outerSteps;
console.log(`  ${numTokens} tokens in ${specSec.toFixed(2)}s  (${(numTokens / specSec).toFixed(1)} tok/s)`);
console.log(`  outer target passes: ${spec.outerSteps}`);
console.log(`  draft acceptance:    ${(acceptanceRate * 100).toFixed(1)}%  (${spec.totalAccepted}/${spec.totalDrafted})`);
console.log(`  tokens / target pass: ${tokensPerOuterStep.toFixed(2)}`);
console.log(`  speedup vs baseline: ${(baselineSec / specSec).toFixed(2)}x`);
console.log('');
console.log(tokT.decode(spec.context).replaceAll('<|endoftext|>', '⏎').slice(0, 600));

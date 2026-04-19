/**
 * Benchmark: target-only generation vs speculative decoding.
 *
 * Runs both decoding paths on a fixed prompt, timing wall-clock and
 * computing tokens/second. In greedy mode (temperature=0) it also
 * verifies that the two paths produce IDENTICAL token sequences, which
 * is the correctness check for speculative decoding being lossless.
 *
 *   node benchmark_speculative.js \
 *        [target] [draft] [num_tokens] [temperature] [K] [num_runs] [tokenizer]
 *
 * Defaults:
 *   target      out/model-final.json
 *   draft       out/draft-final.json
 *   num_tokens  200
 *   temperature 0      (greedy → reproducible + correctness verifiable)
 *   K           4
 *   num_runs    3      (1 warmup + 3 timed runs of each path)
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

// ── Model architecture (matches train_youtube.js) ───────────────────

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

function loadModel(path, tokenizerOverride) {
  const ckpt = JSON.parse(readFileSync(path, 'utf-8'));
  const tokPath = tokenizerOverride || ckpt.tokenizerPath;
  if (!tokPath) throw new Error(`No tokenizer for ${path}`);
  const tokenizer = new BPETokenizer(tokPath);
  const model = new MiniGPT(tokenizer.vocabSize, ckpt.config);
  for (const [name, param] of model.namedParameters()) {
    if (ckpt.parameters[name]) {
      const { shape, data } = ckpt.parameters[name];
      const t = Tensor.fromFloat32(new Float32Array(data), shape);
      t.setRequiresGrad(true);
      param.update(t);
    }
  }
  return { model, tokenizer, config: ckpt.config };
}

function paramCount(model) {
  let n = 0;
  for (const [, p] of model.namedParameters()) n += p.value.size;
  return n;
}

// ── Math helpers ────────────────────────────────────────────────────

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

// ── Baseline generation (target only) ───────────────────────────────

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

function speculativeStep({ target, draft, vocabSize, blockSize, context, K, temperature }) {
  const draftedTokens = [];
  const draftProbs = [];
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

  let extCtx = context.concat(draftedTokens);
  if (extCtx.length > blockSize) extCtx = extCtx.slice(-blockSize);
  const T_len = extCtx.length;
  const targetLogits = target.forward([extCtx]).toFloat32();

  let accepted = 0;
  if (temperature <= 0) {
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
        resampled = sampleFromProbs(p);
      }
      context.push(resampled);
      return { context, accepted, drafted: K };
    }
  }

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
    const res = speculativeStep({ target, draft, vocabSize, blockSize, context, K, temperature });
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

// ── Stats helpers ───────────────────────────────────────────────────

function mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }
function std(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
}

// ── Main ────────────────────────────────────────────────────────────

const FIXED_PROMPT = 'Once upon a time, there was a little girl who';

const args = process.argv.slice(2);
const targetPath = args[0] || join(__dirname, 'out', 'model-final.json');
const draftPath = args[1] || join(__dirname, 'out', 'draft-final.json');
const numTokens = parseInt(args[2] || '200');
const temperature = parseFloat(args[3] || '0');
const K = parseInt(args[4] || '4');
const numRuns = parseInt(args[5] || '3');
const tokenizerOverride = args[6] || null;

if (!existsSync(targetPath)) {
  console.error(`Target not found: ${targetPath}`); process.exit(1);
}
if (!existsSync(draftPath)) {
  console.error(`Draft not found: ${draftPath}`); process.exit(1);
}

console.log('═══════════════════════════════════════════════════════════════');
console.log('  Speculative-Decoding Benchmark');
console.log('═══════════════════════════════════════════════════════════════');

const { model: target, tokenizer: tok, config: cfgT } = loadModel(targetPath, tokenizerOverride);
const { model: draft, config: cfgD } = loadModel(draftPath, tokenizerOverride);

if (cfgT.blockSize !== cfgD.blockSize) {
  console.error(`blockSize mismatch: target=${cfgT.blockSize}, draft=${cfgD.blockSize}`);
  process.exit(1);
}
if (target.vocabSize !== draft.vocabSize) {
  console.error(`vocab mismatch: target=${target.vocabSize}, draft=${draft.vocabSize}`);
  process.exit(1);
}

const vocabSize = target.vocabSize;
const blockSize = cfgT.blockSize;

console.log('');
console.log(`  Target:  ${cfgT.nLayer}L / ${cfgT.nHead}H / ${cfgT.nEmbd}d  →  ${paramCount(target).toLocaleString()} params`);
console.log(`  Draft:   ${cfgD.nLayer}L / ${cfgD.nHead}H / ${cfgD.nEmbd}d  →  ${paramCount(draft).toLocaleString()} params`);
console.log(`  Vocab:   ${vocabSize}    blockSize: ${blockSize}`);
console.log(`  Backend: ${typeof native.flashAttention === 'function' ? 'CUDA (flashAttention)' : 'CPU fallback'}`);
console.log('');
console.log(`  Prompt:      "${FIXED_PROMPT}"`);
console.log(`  Tokens:      ${numTokens}`);
console.log(`  Temperature: ${temperature}    K=${K}    runs=${numRuns} (+1 warmup)`);
console.log('');

target.eval();
draft.eval();
const promptIds = tok.encode(FIXED_PROMPT);
console.log(`  Prompt tokens (${promptIds.length}): [${promptIds.slice(0, 12).join(', ')}${promptIds.length > 12 ? ', …' : ''}]`);
console.log('');

native.noGradStart();

// ── Warmup ─────────────────────────────────────────────────────────
console.log('  Warming up (eliminates JIT / kernel-load bias)...');
generateBaseline(target, vocabSize, blockSize, promptIds, 16, temperature);
generateSpeculative(target, draft, vocabSize, blockSize, promptIds, 16, temperature, K);
console.log('');

// ── Baseline runs ──────────────────────────────────────────────────
console.log('─── Baseline (target only) ───────────────────────────────────');
const baselineTimes = [];
let baselineOutput = null;
for (let r = 0; r < numRuns; r++) {
  const t0 = Date.now();
  const out = generateBaseline(target, vocabSize, blockSize, promptIds, numTokens, temperature);
  const sec = (Date.now() - t0) / 1000;
  baselineTimes.push(sec);
  if (r === 0) baselineOutput = out;
  console.log(`  run ${r + 1}/${numRuns}: ${sec.toFixed(2)}s   ${(numTokens / sec).toFixed(2)} tok/s`);
}
const baselineMean = mean(baselineTimes);
const baselineStd = std(baselineTimes);
const baselineTps = numTokens / baselineMean;
console.log(`  → ${baselineMean.toFixed(2)} ± ${baselineStd.toFixed(2)}s   ${baselineTps.toFixed(2)} tok/s`);
console.log('');

// ── Speculative runs ──────────────────────────────────────────────
console.log(`─── Speculative decoding (K=${K}) ────────────────────────────`);
const specTimes = [];
let specOutput = null;
let totAccepted = 0, totDrafted = 0, totOuterSteps = 0;
for (let r = 0; r < numRuns; r++) {
  const t0 = Date.now();
  const res = generateSpeculative(target, draft, vocabSize, blockSize, promptIds, numTokens, temperature, K);
  const sec = (Date.now() - t0) / 1000;
  specTimes.push(sec);
  if (r === 0) specOutput = res.context;
  totAccepted += res.totalAccepted;
  totDrafted += res.totalDrafted;
  totOuterSteps += res.outerSteps;
  const accRate = res.totalAccepted / Math.max(res.totalDrafted, 1);
  console.log(
    `  run ${r + 1}/${numRuns}: ${sec.toFixed(2)}s   ${(numTokens / sec).toFixed(2)} tok/s` +
    `   acc=${(accRate * 100).toFixed(1)}%   ${res.outerSteps} target passes`
  );
}
const specMean = mean(specTimes);
const specStd = std(specTimes);
const specTps = numTokens / specMean;
const acceptanceRate = totAccepted / Math.max(totDrafted, 1);
const tokensPerOuter = (numTokens * numRuns) / totOuterSteps;
console.log(`  → ${specMean.toFixed(2)} ± ${specStd.toFixed(2)}s   ${specTps.toFixed(2)} tok/s`);
console.log('');

native.noGradEnd();

// ── Verdict ───────────────────────────────────────────────────────
console.log('═══════════════════════════════════════════════════════════════');
console.log('  Results');
console.log('═══════════════════════════════════════════════════════════════');
console.log('');
console.log(`  Baseline      :  ${baselineTps.toFixed(2)} tok/s   (${baselineMean.toFixed(2)}s for ${numTokens} tokens)`);
console.log(`  Speculative   :  ${specTps.toFixed(2)} tok/s   (${specMean.toFixed(2)}s for ${numTokens} tokens)`);
console.log(`  Speedup       :  ${(baselineMean / specMean).toFixed(2)}x`);
console.log('');
console.log(`  Draft accept  :  ${(acceptanceRate * 100).toFixed(1)}%   (${totAccepted}/${totDrafted} draft tokens)`);
console.log(`  Tok / target pass: ${tokensPerOuter.toFixed(2)}   (max possible: ${K + 1})`);
console.log('');

// ── Correctness check (greedy mode is deterministic) ──────────────
if (temperature <= 0) {
  const tail = numTokens; // compare the generated tokens only
  const baseTail = baselineOutput.slice(-tail);
  const specTail = specOutput.slice(-tail);
  let firstDivergence = -1;
  for (let i = 0; i < tail; i++) {
    if (baseTail[i] !== specTail[i]) { firstDivergence = i; break; }
  }
  if (firstDivergence === -1) {
    console.log(`  ✓ Correctness: speculative output IDENTICAL to baseline (${tail}/${tail} tokens)`);
  } else {
    console.log(`  ✗ Correctness: outputs diverge at token ${firstDivergence} of ${tail}`);
    console.log(`    baseline[${firstDivergence}] = ${baseTail[firstDivergence]}, spec[${firstDivergence}] = ${specTail[firstDivergence]}`);
  }
} else {
  console.log(`  (correctness check skipped at temperature=${temperature} — outputs differ run-to-run by design)`);
}
console.log('');
console.log('─── Sample output (baseline run 1) ──────────────────────────');
console.log(tok.decode(baselineOutput).replaceAll('<|endoftext|>', '⏎').slice(0, 600));
console.log('');

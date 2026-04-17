/**
 * Train MiniGPT on YouTube-Commons transcripts using a BPE tokenizer.
 *
 * Expects pre-tokenized binary data produced by scripts/prepare_youtube.py:
 *   data/train.bin   – Int32 token IDs
 *   data/val.bin     – Int32 token IDs
 *   data/meta.json   – { vocab_size, eot_token, … }
 *   data/tokenizer.json – HuggingFace tokenizer file (for encode/decode)
 */
import {
  Tensor, native,
  Module, Parameter,
  Linear, Embedding,
  softmax, crossEntropyLoss, crossEntropyLossGpu,
  gelu, dropout, layerNorm, flashAttention,
  Adam,
} from '@mni-ml/framework';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BPETokenizer } from './bpe.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

// ════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════

const CONFIG = {
  nEmbd: 384,
  nHead: 6,
  nLayer: 6,
  blockSize: 256,
  batchSize: 64,
  gradAccumSteps: 1,
  maxIters: 5000,
  evalInterval: 500,
  evalIters: 3,
  logInterval: 10,
  generateEvery: 500,
  generateLen: 200,

  lr: 6e-4,
  beta1: 0.9,
  beta2: 0.95,
  weightDecay: 0.1,
  dropoutRate: 0.1,

  warmupSteps: 200,
  minLr: 6e-5,
  maxGradNorm: 1.0,

  checkpointEvery: 500,
  modelDir: process.env.MODEL_DIR || join(ROOT, 'out'),
  dataDir: process.env.DATA_DIR || join(ROOT, 'data'),
};

for (const [key, envKey] of [
  ['maxIters', 'MAX_ITERS'], ['batchSize', 'BATCH_SIZE'],
  ['nEmbd', 'N_EMBD'], ['nHead', 'N_HEAD'], ['nLayer', 'N_LAYER'],
  ['blockSize', 'BLOCK_SIZE'], ['lr', 'LR'], ['checkpointEvery', 'CHECKPOINT_EVERY'],
  ['gradAccumSteps', 'GRAD_ACCUM_STEPS'],
]) {
  if (process.env[envKey]) CONFIG[key] = Number(process.env[envKey]);
}
const NO_RESUME = process.env.NO_RESUME === '1';

// ════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════

function clipGradNorm(paramTensors, maxNorm) {
  const ids = paramTensors.map(t => t._id);
  const norm = native.gradNorm(ids);
  if (norm > maxNorm) native.clipGradNorm(ids, maxNorm);
  return norm;
}

function gcKeepParams(params) {
  const keepIds = [];
  for (const p of params) {
    keepIds.push(p.value._id);
    const g = native.getGrad(p.value._id);
    if (g !== null && g !== undefined) keepIds.push(g);
  }
  for (const mask of _causalMaskCache.values()) keepIds.push(mask._id);
  native.gcTensors(keepIds);
}

function getLR(step, warmup, total, maxLr, minLr) {
  if (step < warmup) return maxLr * (step + 1) / warmup;
  const progress = (step - warmup) / Math.max(1, total - warmup);
  return minLr + (maxLr - minLr) * 0.5 * (1 + Math.cos(Math.PI * progress));
}

// ════════════════════════════════════════════════════════════════
// Data Loading (pre-tokenized binary)
// ════════════════════════════════════════════════════════════════

function loadBinaryTokens(path) {
  const buf = readFileSync(path);
  const ab = new ArrayBuffer(buf.length);
  new Uint8Array(ab).set(buf);
  return new Int32Array(ab);
}

// ════════════════════════════════════════════════════════════════
// Model Serialization
// ════════════════════════════════════════════════════════════════

function saveModel(model, tokPath, path, { step = -1, optimizer = null, bestValLoss = Infinity } = {}) {
  const namedParams = model.namedParameters();
  const params = {};
  for (const [name, param] of namedParams) {
    const data = param.value.toFloat32();
    params[name] = { shape: [...param.value.shape], data: Array.from(data) };
  }
  const checkpoint = {
    config: model.config,
    tokenizerPath: tokPath,
    parameters: params,
    step,
    bestValLoss,
  };
  // We only save the step counter here. The per-parameter Adam m/v buffers
  // live inside the Rust native backend and can't be exported yet.
  // On resume we intentionally reset t=0 so bias correction compensates
  // for the fresh m/v, giving well-behaved updates from the first step.
  if (optimizer) checkpoint.optimizer = { t: optimizer.t };
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(checkpoint));
  const sizeMB = (Buffer.byteLength(JSON.stringify(checkpoint)) / 1024 / 1024).toFixed(1);
  console.log(`  Model saved to ${path} (${sizeMB} MB)`);
}

function loadModel(path, tokenizerPath) {
  const checkpoint = JSON.parse(readFileSync(path, 'utf-8'));
  const tokenizer = new BPETokenizer(tokenizerPath);
  const model = new MiniGPT(tokenizer.vocabSize, checkpoint.config);
  const namedParams = model.namedParameters();
  for (const [name, param] of namedParams) {
    if (checkpoint.parameters[name]) {
      const { shape, data } = checkpoint.parameters[name];
      const t = Tensor.fromFloat32(new Float32Array(data), shape);
      t.setRequiresGrad(true);
      param.update(t);
    }
  }
  console.log(`  Model loaded from ${path}`);
  return {
    model, tokenizer,
    config: checkpoint.config,
    step: checkpoint.step ?? -1,
    optimizerState: checkpoint.optimizer ?? null,
    bestValLoss: checkpoint.bestValLoss ?? Infinity,
  };
}

function findLatestCheckpoint(dir) {
  if (!existsSync(dir)) return null;
  const files = readdirSync(dir).filter(f => f.startsWith('checkpoint-') && f.endsWith('.json'));
  if (files.length === 0) return null;
  const byStep = files.map(f => {
    const m = f.match(/checkpoint-(\d+)\.json/);
    return m ? { file: f, step: parseInt(m[1], 10) } : null;
  }).filter(Boolean);
  byStep.sort((a, b) => b.step - a.step);
  return { path: join(dir, byStep[0].file), step: byStep[0].step };
}

// ════════════════════════════════════════════════════════════════
// Model Architecture (identical to train.js)
// ════════════════════════════════════════════════════════════════

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
  constructor(nEmbd, nHead, dropoutRate) {
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
    const [B, S, E] = x.shape;
    const { nHead, headDim, scale } = this;
    let q = this.queryProj.forward(x);
    let k = this.keyProj.forward(x);
    let v = this.valueProj.forward(x);
    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    k = k.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    let out = flashAttention(q, k, v, scale, true);
    out = out.view(B, nHead, S, headDim).permute(0, 2, 1, 3).contiguous().view(B, S, E);
    return this.outProj.forward(out);
  }
}

class FeedForward extends Module {
  constructor(nEmbd, dropoutRate) {
    super();
    this.dropoutRate = dropoutRate;
    this.fc1 = new Linear(nEmbd, 4 * nEmbd);
    this.fc2 = new Linear(4 * nEmbd, nEmbd);
  }
  forward(x) {
    let h = this.fc1.forward(x);
    h = gelu(h);
    h = this.fc2.forward(h);
    return dropout(h, this.dropoutRate, !this.training);
  }
}

class TransformerBlock extends Module {
  constructor(nEmbd, nHead, dropoutRate) {
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
}

class MiniGPT extends Module {
  constructor(vocabSize, config) {
    super();
    const { nEmbd, nHead, nLayer, blockSize, dropoutRate = 0 } = config;
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);
    for (let i = 0; i < nLayer; i++) this[`block${i}`] = new TransformerBlock(nEmbd, nHead, dropoutRate);
    this.lnFinal = new LayerNorm(nEmbd);
    this.headBias = new Parameter(Tensor.zeros([vocabSize]).setRequiresGrad(true));
  }
  forward(indices) {
    const batch = indices.length;
    const seqLen = indices[0].length;
    let x = this.tokenEmb.forward(indices);
    const posIndices = [];
    for (let b = 0; b < batch; b++) posIndices.push(Array.from({ length: seqLen }, (_, i) => i));
    x = x.add(this.posEmb.forward(posIndices));
    for (let i = 0; i < this.config.nLayer; i++) x = this[`block${i}`].forward(x);
    x = this.lnFinal.forward(x);
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(wT).add(this.headBias.value);
  }
  forwardGpu(inputsIntId, batch, seqLen) {
    let x = this.tokenEmb.forwardGpu(inputsIntId, batch, seqLen);
    const posIndices = [];
    for (let b = 0; b < batch; b++) posIndices.push(Array.from({ length: seqLen }, (_, i) => i));
    x = x.add(this.posEmb.forward(posIndices));
    for (let i = 0; i < this.config.nLayer; i++) x = this[`block${i}`].forward(x);
    x = this.lnFinal.forward(x);
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return x.matmul(wT).add(this.headBias.value);
  }
}

// ════════════════════════════════════════════════════════════════
// Batch helpers
// ════════════════════════════════════════════════════════════════

function getBatch(data, blockSize, batchSize) {
  const inputs = [];
  const targets = [];
  for (let i = 0; i < batchSize; i++) {
    const start = Math.floor(Math.random() * (data.length - blockSize - 1));
    inputs.push(Array.from(data.slice(start, start + blockSize)));
    targets.push(Array.from(data.slice(start + 1, start + blockSize + 1)));
  }
  return { inputs, targets };
}

function estimateLoss(model, trainData, valData, config, params) {
  model.eval();
  native.noGradStart();
  const losses = { train: 0, val: 0 };
  for (const [name, data] of [['train', trainData], ['val', valData]]) {
    let total = 0;
    for (let i = 0; i < config.evalIters; i++) {
      const { inputs, targets } = getBatch(data, config.blockSize, config.batchSize);
      const logits = model.forward(inputs);
      const loss = crossEntropyLoss(logits, targets);
      total += loss.item();
      gcKeepParams(params);
    }
    losses[name] = total / config.evalIters;
  }
  native.noGradEnd();
  model.train();
  return losses;
}

const GENERATION_PROMPTS = [
  "Once upon a time",
  "There was a little girl named",
  "One day, a boy named Tom",
  "The sun was shining and",
  "Lily and her mom went to the",
  "Once there was a big bear who",
];

function generate(model, tokenizer, prompt, maxTokens, temperature = 0.8, params = null) {
  model.eval();
  native.noGradStart();

  let context;
  if (typeof prompt === 'string') {
    context = tokenizer.encode(prompt);
  } else if (Array.isArray(prompt) && prompt.length > 0) {
    context = [...prompt];
  } else {
    const pick = GENERATION_PROMPTS[Math.floor(Math.random() * GENERATION_PROMPTS.length)];
    context = tokenizer.encode(pick);
  }
  const blockSize = model.config.blockSize;
  const eot = tokenizer.eotToken;
  const promptLen = context.length;

  for (let i = 0; i < maxTokens; i++) {
    const ctxWindow = context.slice(-blockSize);
    const logits = model.forward([ctxWindow]);

    const seqLen = ctxWindow.length;
    const vocabSize = model.vocabSize;
    const lastLogitsData = logits.toFloat32();
    const offset = (seqLen - 1) * vocabSize;
    const lastLogits = [];
    for (let v = 0; v < vocabSize; v++) lastLogits.push(lastLogitsData[offset + v] / temperature);

    // Suppress EOT for the first 40 generated tokens so the model
    // is forced to produce real text even when undertrained.
    if (i < 40 && eot >= 0) lastLogits[eot] = -Infinity;

    // Numerically stable softmax: subtract max, exponentiate, normalize.
    const maxLogit = Math.max(...lastLogits);
    const exps = lastLogits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);

    let nextToken;
    if (sumExps === 0 || !isFinite(sumExps)) {
      // All probs underflowed — pick a random non-EOT token.
      do { nextToken = Math.floor(Math.random() * vocabSize); }
      while (nextToken === eot);
    } else {
      const probs = exps.map(e => e / sumExps);
      const r = Math.random();
      let cumSum = 0;
      nextToken = vocabSize - 1; // fallback to last token, never 0 (EOT)
      for (let v = 0; v < vocabSize; v++) {
        cumSum += probs[v];
        if (r < cumSum) { nextToken = v; break; }
      }
    }

    context.push(nextToken);
    if (nextToken === eot && i >= 40) break;
    if (params) gcKeepParams(params);
  }

  native.noGradEnd();
  model.train();
  return tokenizer.decode(context);
}

// ════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════

function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log('  MiniGPT — BPE tokenizer');
  console.log('═══════════════════════════════════════════════════════');
  console.log('');

  const tokenizerPath = join(CONFIG.dataDir, 'tokenizer.json');
  const metaPath = join(CONFIG.dataDir, 'meta.json');

  if (!existsSync(metaPath)) {
    console.error('  ERROR: data/meta.json not found — run: python3 scripts/prepare_youtube.py');
    process.exit(1);
  }

  const meta = JSON.parse(readFileSync(metaPath, 'utf-8'));
  const vocabSize = meta.vocab_size;

  let model, tokenizer, startStep = 0;
  let savedOptimizerState = null;
  let bestValLoss = Infinity;

  const latest = NO_RESUME ? null : findLatestCheckpoint(CONFIG.modelDir);
  if (latest) {
    console.log(`  Resuming from checkpoint at step ${latest.step}`);
    const loaded = loadModel(latest.path, tokenizerPath);
    model = loaded.model;
    tokenizer = loaded.tokenizer;
    startStep = latest.step;
    savedOptimizerState = loaded.optimizerState;
    bestValLoss = loaded.bestValLoss;
    console.log(`  Best val loss so far: ${bestValLoss === Infinity ? 'N/A' : bestValLoss.toFixed(4)}`);
    console.log('');
  }

  if (!tokenizer) tokenizer = new BPETokenizer(tokenizerPath);

  // ── Load pre-tokenized binary data ──
  console.log('  Loading pre-tokenized data...');
  const trainData = loadBinaryTokens(join(CONFIG.dataDir, 'train.bin'));
  const valData = loadBinaryTokens(join(CONFIG.dataDir, 'val.bin'));

  console.log(`  Vocabulary:     ${vocabSize} BPE tokens`);
  console.log(`  Training data:  ${trainData.length.toLocaleString()} tokens (${(trainData.length * 4 / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`  Validation:     ${valData.length.toLocaleString()} tokens`);
  console.log(`  Architecture:   ${CONFIG.nLayer} blocks, ${CONFIG.nHead} heads, ${CONFIG.nEmbd}-dim`);
  console.log(`  Context window: ${CONFIG.blockSize} tokens`);
  console.log(`  Dropout:        ${CONFIG.dropoutRate}`);

  if (!model) model = new MiniGPT(vocabSize, CONFIG);
  const params = model.parameters();
  const totalParams = params.reduce((sum, p) => sum + p.value.size, 0);
  console.log(`  Parameters:     ${totalParams.toLocaleString()}`);
  const effectiveBatch = CONFIG.batchSize * CONFIG.gradAccumSteps;
  console.log(`  Optimizer:      Adam (lr=${CONFIG.lr}, β1=${CONFIG.beta1}, β2=${CONFIG.beta2})`);
  console.log(`  Schedule:       ${CONFIG.warmupSteps} warmup → cosine decay to ${CONFIG.minLr}`);
  console.log(`  Grad clipping:  max norm ${CONFIG.maxGradNorm}`);
  console.log(`  Training:       steps ${startStep}→${CONFIG.maxIters}, batch=${CONFIG.batchSize}, accum=${CONFIG.gradAccumSteps} (effective ${effectiveBatch})`);
  console.log('');
  console.log('─────────────────────────────────────────────────────────');
  console.log(`  Training${startStep > 0 ? ` (resuming from step ${startStep})` : ''}`);
  console.log('─────────────────────────────────────────────────────────');
  console.log('');

  const optimizer = new Adam(params, {
    lr: CONFIG.lr,
    beta1: CONFIG.beta1,
    beta2: CONFIG.beta2,
    weightDecay: CONFIG.weightDecay,
  });
  if (savedOptimizerState) {
    // Adam m/v buffers can't be restored from checkpoint (they live in Rust).
    // Keeping t=0 lets bias correction compensate for fresh m/v, avoiding
    // the instability that comes from t=N with m=0,v=0.
    console.log(`  Optimizer restarting fresh (checkpoint had t=${savedOptimizerState.t}, reset to 0 for stability)`);
  }

  const paramTensors = params.map(p => p.value);

  // Upload tokenized data to GPU
  const trainDatasetId = native.createDataset(new Int32Array(trainData.buffer, trainData.byteOffset, trainData.length));
  console.log(`  GPU dataset uploaded: ${trainData.length.toLocaleString()} tokens`);

  const startTime = Date.now();

  for (let iter = startStep; iter < CONFIG.maxIters; iter++) {
    const stepStart = Date.now();
    optimizer.lr = getLR(iter, CONFIG.warmupSteps, CONFIG.maxIters, CONFIG.lr, CONFIG.minLr);
    optimizer.zeroGrad();
    let accumLoss = 0;
    const shouldLog = (iter === startStep) || (iter % CONFIG.logInterval === 0) || (iter % CONFIG.evalInterval === 0);

    for (let microStep = 0; microStep < CONFIG.gradAccumSteps; microStep++) {
      const [inputsId, targetsId] = native.sampleBatch(trainDatasetId, CONFIG.blockSize, CONFIG.batchSize);
      const logits = model.forwardGpu(inputsId, CONFIG.batchSize, CONFIG.blockSize);
      const loss = crossEntropyLossGpu(logits, targetsId);
      const scaledLoss = loss.mul(1 / CONFIG.gradAccumSteps);
      scaledLoss.backward();
      if (shouldLog) accumLoss += loss.item();
      if (iter === startStep && microStep === 0) {
        if (!shouldLog) accumLoss = loss.item();
        console.log(`  [first micro-batch done: ${((Date.now() - stepStart) / 1000).toFixed(1)}s, loss=${accumLoss.toFixed(4)}]`);
      }
      gcKeepParams(params);
      native.freeIntBuffer(inputsId);
      native.freeIntBuffer(targetsId);
    }
    if (shouldLog) accumLoss /= CONFIG.gradAccumSteps;

    const gradNorm = optimizer.step(CONFIG.maxGradNorm);
    gcKeepParams(params);

    const stepMs = Date.now() - stepStart;
    const stepsCompleted = iter - startStep + 1;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);

    if (iter % CONFIG.evalInterval === 0) {
      console.log(
        `  step ${String(iter).padStart(5)} │` +
        ` loss ${accumLoss.toFixed(4)} │` +
        ` ${(stepMs / 1000).toFixed(1)}s/step │` +
        ` ${elapsed}s │ evaluating...`
      );
      const losses = estimateLoss(model, trainData, valData, CONFIG, params);
      const marker = losses.val < bestValLoss ? ' *' : '';
      if (losses.val < bestValLoss) bestValLoss = losses.val;
      const avgStepMs = (Date.now() - startTime) / stepsCompleted;
      const etaSeconds = (CONFIG.maxIters - iter - 1) * avgStepMs / 1000;
      const eta = etaSeconds > 3600
        ? `${(etaSeconds / 3600).toFixed(1)}h`
        : `${(etaSeconds / 60).toFixed(0)}m`;
      console.log(
        `         ${' '.repeat(5)} │` +
        ` train ${losses.train.toFixed(4)} │` +
        ` val ${losses.val.toFixed(4)}${marker} │` +
        ` lr ${optimizer.lr.toExponential(1)} │` +
        ` gnorm ${gradNorm.toFixed(1)} │` +
        ` eta ${eta}`
      );
    } else if (iter % CONFIG.logInterval === 0) {
      console.log(
        `  step ${String(iter).padStart(5)} │` +
        ` loss ${accumLoss.toFixed(4)} │` +
        ` ${(stepMs / 1000).toFixed(1)}s/step │` +
        ` ${elapsed}s`
      );
    }

    if (iter > 0 && iter % CONFIG.generateEvery === 0) {
      const sample = generate(model, tokenizer, null, CONFIG.generateLen, 0.8, params);
      console.log('');
      console.log('  ── sample ──────────────────────────────────────────');
      const cleaned = sample.replaceAll('<|endoftext|>', '').trim();
      for (const line of cleaned.split('\n').slice(0, 8)) {
        console.log(`  ${line}`);
      }
      console.log('  ───────────────────────────────────────────────────');
      console.log('');
    }

    if (iter > 0 && iter % CONFIG.checkpointEvery === 0) {
      saveModel(model, tokenizerPath, join(CONFIG.modelDir, `checkpoint-${iter}.json`), { step: iter, optimizer, bestValLoss });
    }
  }

  console.log('');
  console.log('─────────────────────────────────────────────────────────');
  console.log('  Final Generation (temperature=0.8)');
  console.log('─────────────────────────────────────────────────────────');
  console.log('');
  const finalText = generate(model, tokenizer, null, CONFIG.generateLen * 2, 0.8, params);
  const finalCleaned = finalText.replaceAll('<|endoftext|>', '').trim();
  for (const line of finalCleaned.split('\n').slice(0, 12)) {
    console.log(`  ${line}`);
  }
  saveModel(model, tokenizerPath, join(CONFIG.modelDir, 'model-final.json'), { step: CONFIG.maxIters, optimizer, bestValLoss });

  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Best val loss: ${bestValLoss.toFixed(4)}`);
  console.log(`  Total time:    ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`);
  console.log(`  Model saved:   ${join(CONFIG.modelDir, 'model-final.json')}`);
  console.log('═══════════════════════════════════════════════════════');
}

main();

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
  generateLen: 300,

  lr: 1e-3,
  beta1: 0.9,
  beta2: 0.99,
  weightDecay: 0.01,
  dropoutRate: 0.1,

  warmupSteps: 200,
  minLr: 1e-4,
  maxGradNorm: 1.0,

  checkpointEvery: 500,
  modelDir: process.env.MODEL_DIR || join(ROOT, 'out'),
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
// Gradient Clipping (legacy, kept for generate.js compat)
// ════════════════════════════════════════════════════════════════

function clipGradNorm(paramTensors, maxNorm) {
  const ids = paramTensors.map(t => t._id);
  const norm = native.gradNorm(ids);
  if (norm > maxNorm) {
    native.clipGradNorm(ids, maxNorm);
  }
  return norm;
}

function gcKeepParams(params) {
  const keepIds = [];
  for (const p of params) {
    keepIds.push(p.value._id);
    const g = native.getGrad(p.value._id);
    if (g !== null && g !== undefined) keepIds.push(g);
  }
  for (const mask of _causalMaskCache.values()) {
    keepIds.push(mask._id);
  }
  native.gcTensors(keepIds);
}

// ════════════════════════════════════════════════════════════════
// Learning Rate Schedule (linear warmup + cosine decay)
// ════════════════════════════════════════════════════════════════

function getLR(step, warmup, total, maxLr, minLr) {
  if (step < warmup) return maxLr * (step + 1) / warmup;
  const progress = (step - warmup) / Math.max(1, total - warmup);
  return minLr + (maxLr - minLr) * 0.5 * (1 + Math.cos(Math.PI * progress));
}

// ════════════════════════════════════════════════════════════════
// Data Loading
// ════════════════════════════════════════════════════════════════

const FALLBACK_TEXT = `twinkle twinkle little star how i wonder what you are
up above the world so high like a diamond in the sky
twinkle twinkle little star how i wonder what you are
the quick brown fox jumps over the lazy dog`.trim();

function loadTrainingText() {
  const dataPath = join(ROOT, 'data', 'input.txt');
  if (existsSync(dataPath)) {
    console.log(`  Loading data from ${dataPath}`);
    return readFileSync(dataPath, 'utf-8');
  }
  console.log('  WARNING: data/input.txt not found — run "npm run download" first');
  return FALLBACK_TEXT;
}

// ════════════════════════════════════════════════════════════════
// Model Serialization
// ════════════════════════════════════════════════════════════════

function saveModel(model, tokenizer, path, { step = -1, optimizer = null, bestValLoss = Infinity } = {}) {
  const namedParams = model.namedParameters();
  const params = {};
  for (const [name, param] of namedParams) {
    const data = param.value.toFloat32();
    params[name] = {
      shape: [...param.value.shape],
      data: Array.from(data),
    };
  }
  const checkpoint = {
    config: model.config,
    tokenizer: { stoi: tokenizer.stoi, itos: tokenizer.itos, vocabSize: tokenizer.vocabSize },
    parameters: params,
    step,
    bestValLoss,
  };
  if (optimizer) {
    checkpoint.optimizer = { t: optimizer.t };
  }
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(checkpoint));
  const sizeMB = (Buffer.byteLength(JSON.stringify(checkpoint)) / 1024 / 1024).toFixed(1);
  console.log(`  Model saved to ${path} (${sizeMB} MB)`);
}

function loadModel(path) {
  const checkpoint = JSON.parse(readFileSync(path, 'utf-8'));
  const tokenizer = new CharTokenizer('');
  Object.assign(tokenizer, checkpoint.tokenizer);

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
    model,
    tokenizer,
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
// Character-Level Tokenizer
// ════════════════════════════════════════════════════════════════

class CharTokenizer {
  constructor(text) {
    const chars = [...new Set(text)].sort();
    this.vocabSize = chars.length;
    this.stoi = Object.fromEntries(chars.map((c, i) => [c, i]));
    this.itos = Object.fromEntries(chars.map((c, i) => [i, c]));
  }
  encode(text) { return [...text].map(ch => this.stoi[ch]); }
  decode(indices) { return indices.map(i => this.itos[i]).join(''); }
}

// ════════════════════════════════════════════════════════════════
// Layer Normalization (uses native fused kernel)
// ════════════════════════════════════════════════════════════════

class LayerNorm extends Module {
  constructor(dim) {
    super();
    this.dim = dim;
    const gammaData = new Float32Array(dim).fill(1.0);
    this.gamma = new Parameter(Tensor.fromFloat32(gammaData, [dim]).setRequiresGrad(true));
    const betaData = new Float32Array(dim).fill(0.0);
    this.beta = new Parameter(Tensor.fromFloat32(betaData, [dim]).setRequiresGrad(true));
  }
  forward(x) {
    return layerNorm(x, this.gamma.value, this.beta.value, 1e-5);
  }
}

// ════════════════════════════════════════════════════════════════
// Causal Self-Attention (Multi-Head)
// ════════════════════════════════════════════════════════════════

const _causalMaskCache = new Map();
function getCausalMask(size) {
  if (_causalMaskCache.has(size)) return _causalMaskCache.get(size);
  const storage = new Float32Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      storage[i * size + j] = j <= i ? 0.0 : -1e9;
    }
  }
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

    // [B, S, nHead, D] -> [B*nHead, S, D] for flash attention
    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    k = k.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3).contiguous().view(B * nHead, S, headDim);

    let out = flashAttention(q, k, v, scale, true);
    out = out.view(B, nHead, S, headDim).permute(0, 2, 1, 3).contiguous().view(B, S, E);

    return this.outProj.forward(out);
  }
}

// ════════════════════════════════════════════════════════════════
// Feed-Forward Network (4x hidden expansion with GELU)
// ════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════
// Transformer Block (pre-norm with residual connections)
// ════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════
// MiniGPT Model (GPT-2 architecture)
// ════════════════════════════════════════════════════════════════

class MiniGPT extends Module {
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

  forwardGpu(inputsIntId, batch, seqLen) {
    let x = this.tokenEmb.forwardGpu(inputsIntId, batch, seqLen);

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
}

// ════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════

function getBatch(data, blockSize, batchSize) {
  const inputs = [];
  const targets = [];
  for (let i = 0; i < batchSize; i++) {
    const start = Math.floor(Math.random() * (data.length - blockSize - 1));
    inputs.push(data.slice(start, start + blockSize));
    targets.push(data.slice(start + 1, start + blockSize + 1));
  }
  return { inputs, targets };
}

function estimateLoss(model, tokenizer, trainData, valData, config, params) {
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

function generate(model, tokenizer, prompt, maxTokens, temperature = 0.8, params = null) {
  model.eval();
  native.noGradStart();
  let context = tokenizer.encode(prompt);
  const blockSize = model.config.blockSize;

  for (let i = 0; i < maxTokens; i++) {
    const ctxWindow = context.slice(-blockSize);
    const logits = model.forward([ctxWindow]);

    const seqLen = ctxWindow.length;
    const vocabSize = tokenizer.vocabSize;
    const lastLogitsData = logits.toFloat32();
    const offset = (seqLen - 1) * vocabSize;
    const lastLogits = [];
    for (let v = 0; v < vocabSize; v++) {
      lastLogits.push(lastLogitsData[offset + v] / temperature);
    }

    const maxLogit = Math.max(...lastLogits);
    const exps = lastLogits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    const probs = exps.map(e => e / sumExps);

    const r = Math.random();
    let cumSum = 0;
    let nextToken = 0;
    for (let v = 0; v < vocabSize; v++) {
      cumSum += probs[v];
      if (r < cumSum) { nextToken = v; break; }
    }

    context.push(nextToken);
    if (params) gcKeepParams(params);
  }

  native.noGradEnd();
  model.train();
  return tokenizer.decode(context);
}

// ════════════════════════════════════════════════════════════════
// Training
// ════════════════════════════════════════════════════════════════

function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log('  MiniGPT — Rust+CUDA Native Backend');
  console.log('═══════════════════════════════════════════════════════');
  console.log('');

  let model, tokenizer, startStep = 0;
  let savedOptimizerState = null;
  let bestValLoss = Infinity;

  const latest = NO_RESUME ? null : findLatestCheckpoint(CONFIG.modelDir);
  if (latest) {
    console.log(`  Resuming from checkpoint at step ${latest.step}`);
    const loaded = loadModel(latest.path);
    model = loaded.model;
    tokenizer = loaded.tokenizer;
    startStep = latest.step;
    savedOptimizerState = loaded.optimizerState;
    bestValLoss = loaded.bestValLoss;
    console.log(`  Best val loss so far: ${bestValLoss === Infinity ? 'N/A' : bestValLoss.toFixed(4)}`);
    console.log('');
  }

  const text = loadTrainingText();
  if (!tokenizer) tokenizer = new CharTokenizer(text);
  const allData = tokenizer.encode(text);

  const splitIdx = Math.floor(allData.length * 0.9);
  const trainData = allData.slice(0, splitIdx);
  const valData = allData.slice(splitIdx);

  console.log(`  Vocabulary:     ${tokenizer.vocabSize} unique characters`);
  console.log(`  Training data:  ${trainData.length.toLocaleString()} tokens (${(trainData.length / 1024).toFixed(1)} KB)`);
  console.log(`  Validation:     ${valData.length.toLocaleString()} tokens`);
  console.log(`  Architecture:   ${CONFIG.nLayer} blocks, ${CONFIG.nHead} heads, ${CONFIG.nEmbd}-dim`);
  console.log(`  Context window: ${CONFIG.blockSize} tokens`);
  console.log(`  Dropout:        ${CONFIG.dropoutRate}`);

  if (!model) model = new MiniGPT(tokenizer.vocabSize, CONFIG);
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
    optimizer.t = savedOptimizerState.t;
    console.log(`  Optimizer state restored (t=${optimizer.t})`);
  }

  const paramTensors = params.map(p => p.value);

  // Upload tokenized data to GPU (zero-copy batch sampling)
  const trainDatasetId = native.createDataset(new Int32Array(trainData));
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
      const losses = estimateLoss(model, tokenizer, trainData, valData, CONFIG, params);
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
      const sample = generate(model, tokenizer, '\n', CONFIG.generateLen, 0.8, params);
      console.log('');
      console.log('  ── sample ──────────────────────────────────────────');
      for (const line of sample.split('\n').slice(0, 6)) {
        console.log(`  ${line}`);
      }
      console.log('  ───────────────────────────────────────────────────');
      console.log('');
    }

    if (iter > 0 && iter % CONFIG.checkpointEvery === 0) {
      saveModel(model, tokenizer, join(CONFIG.modelDir, `checkpoint-${iter}.json`), { step: iter, optimizer, bestValLoss });
    }
  }

  console.log('');
  console.log('─────────────────────────────────────────────────────────');
  console.log('  Final Generation (temperature=0.8)');
  console.log('─────────────────────────────────────────────────────────');
  console.log('');
  const finalText = generate(model, tokenizer, '\n', CONFIG.generateLen * 2, 0.8, params);
  for (const line of finalText.split('\n').slice(0, 12)) {
    console.log(`  ${line}`);
  }
  saveModel(model, tokenizer, join(CONFIG.modelDir, 'model-final.json'), { step: CONFIG.maxIters, optimizer, bestValLoss });

  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Best val loss: ${bestValLoss.toFixed(4)}`);
  console.log(`  Total time:    ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`);
  console.log(`  Model saved:   ${join(CONFIG.modelDir, 'model-final.json')}`);
  console.log('═══════════════════════════════════════════════════════');
}

main();

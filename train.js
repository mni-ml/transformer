import {
  Tensor, TensorData,
  Module, Parameter,
  Linear, Embedding,
  softmax, crossEntropyLoss, destroyPool, destroyDevice, gelu, dropout,
} from '@mni-ml/framework';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════

const CONFIG = {
  nEmbd: 128,
  nHead: 4,
  nLayer: 4,
  blockSize: 64,
  batchSize: 16,
  maxIters: 5000,
  evalInterval: 100,
  evalIters: 5,
  generateEvery: 500,
  generateLen: 200,

  lr: 6e-4,
  beta1: 0.9,
  beta2: 0.95,
  weightDecay: 0.01,
  dropoutRate: 0.1,

  warmupSteps: 200,
  minLr: 6e-5,
  maxGradNorm: 1.0,

  checkpointEvery: 500,
  modelDir: process.env.MODEL_DIR || join(__dirname, 'out'),
};

for (const [key, envKey] of [
  ['maxIters', 'MAX_ITERS'], ['batchSize', 'BATCH_SIZE'],
  ['nEmbd', 'N_EMBD'], ['nHead', 'N_HEAD'], ['nLayer', 'N_LAYER'],
  ['blockSize', 'BLOCK_SIZE'], ['lr', 'LR'], ['checkpointEvery', 'CHECKPOINT_EVERY'],
]) {
  if (process.env[envKey]) CONFIG[key] = Number(process.env[envKey]);
}
const NO_RESUME = process.env.NO_RESUME === '1';

// ════════════════════════════════════════════════════════════════
// Adam Optimizer
// ════════════════════════════════════════════════════════════════

class Adam {
  constructor(parameters, { lr = 6e-4, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0 } = {}) {
    this.parameters = parameters;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.t = 0;
    this.m = parameters.map(p => new Float64Array(p.value.size));
    this.v = parameters.map(p => new Float64Array(p.value.size));
  }

  zeroGrad() {
    for (const p of this.parameters) {
      if (p.value && typeof p.value === 'object' && 'grad' in p.value && p.value.grad != null) {
        p.value.grad = null;
      }
    }
  }

  step() {
    this.t++;
    const { beta1, beta2, eps, weightDecay, lr } = this;
    const bc1 = 1 - beta1 ** this.t;
    const bc2 = 1 - beta2 ** this.t;

    for (let i = 0; i < this.parameters.length; i++) {
      const p = this.parameters[i];
      if (!(p.value instanceof Tensor) || !p.value.grad) continue;

      const grad = p.value.grad.contiguous();
      const val = p.value.contiguous();
      const gs = grad.data.storage;
      const vs = val.data.storage;
      const m = this.m[i];
      const v = this.v[i];
      const out = new Float64Array(val.size);

      for (let j = 0; j < val.size; j++) {
        const g = weightDecay > 0 ? gs[j] + weightDecay * vs[j] : gs[j];
        m[j] = beta1 * m[j] + (1 - beta1) * g;
        v[j] = beta2 * v[j] + (1 - beta2) * g * g;
        out[j] = vs[j] - lr * (m[j] / bc1) / (Math.sqrt(v[j] / bc2) + eps);
      }

      p.update(new Tensor(new TensorData(out, [...val.shape])));
    }
  }
}

// ════════════════════════════════════════════════════════════════
// Gradient Clipping (global norm)
// ════════════════════════════════════════════════════════════════

function clipGradNorm(parameters, maxNorm) {
  let normSq = 0;
  const grads = [];
  for (const p of parameters) {
    if (!(p.value instanceof Tensor) || !p.value.grad) { grads.push(null); continue; }
    const g = p.value.grad.contiguous();
    grads.push(g);
    const s = g.data.storage;
    for (let i = 0; i < s.length; i++) normSq += s[i] * s[i];
  }
  const norm = Math.sqrt(normSq);
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (let i = 0; i < parameters.length; i++) {
      if (!grads[i]) continue;
      const s = grads[i].data.storage;
      const scaled = new Float64Array(s.length);
      for (let j = 0; j < s.length; j++) scaled[j] = s[j] * scale;
      parameters[i].value.grad = new Tensor(new TensorData(scaled, [...grads[i].shape]));
    }
  }
  return norm;
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
jack and jill went up the hill to fetch a pail of water
jack fell down and broke his crown and jill came tumbling after
up jack got and home did trot as fast as he could caper
humpty dumpty sat on a wall humpty dumpty had a great fall
all the kings horses and all the kings men
could not put humpty together again
mary had a little lamb its fleece was white as snow
and everywhere that mary went the lamb was sure to go
it followed her to school one day which was against the rules
it made the children laugh and play to see a lamb at school
baa baa black sheep have you any wool
yes sir yes sir three bags full
one for the master and one for the dame
and one for the little boy who lives down the lane
roses are red violets are blue sugar is sweet and so are you
the quick brown fox jumps over the lazy dog
the quick brown fox jumps over the lazy dog`.trim();

function loadTrainingText() {
  const dataPath = join(__dirname, 'data', 'input.txt');
  if (existsSync(dataPath)) {
    console.log(`  Loading data from ${dataPath}`);
    return readFileSync(dataPath, 'utf-8');
  }
  console.log('  WARNING: data/input.txt not found — run "node prepare.js" first');
  console.log('  Falling back to built-in 1KB training text\n');
  return FALLBACK_TEXT;
}

// ════════════════════════════════════════════════════════════════
// Model Serialization
// ════════════════════════════════════════════════════════════════

function saveModel(model, tokenizer, path, { step = -1 } = {}) {
  const namedParams = model.namedParameters();
  const params = {};
  for (const [name, param] of namedParams) {
    const t = param.value.contiguous();
    params[name] = {
      shape: [...t.shape],
      data: Array.from(t.data.storage),
    };
  }
  const checkpoint = {
    config: model.config,
    tokenizer: { stoi: tokenizer.stoi, itos: tokenizer.itos, vocabSize: tokenizer.vocabSize },
    parameters: params,
    step,
  };
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
      param.update(new Tensor(new TensorData(new Float64Array(data), shape)));
    }
  }
  console.log(`  Model loaded from ${path}`);
  return { model, tokenizer, config: checkpoint.config, step: checkpoint.step ?? -1 };
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
// Layer Normalization
// ════════════════════════════════════════════════════════════════

class LayerNorm extends Module {
  constructor(dim) {
    super();
    this.gamma = new Parameter(Tensor.ones([dim]));
    this.beta = new Parameter(Tensor.zeros([dim]));
  }
  forward(x) {
    const lastDim = x.dims - 1;
    const mean = x.mean(lastDim);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(lastDim);
    const invStd = variance.add(1e-5).log().mul(-0.5).exp();
    return centered.mul(invStd).mul(this.gamma.value).add(this.beta.value);
  }
}

// ════════════════════════════════════════════════════════════════
// Causal Self-Attention (Multi-Head)
// ════════════════════════════════════════════════════════════════

const _causalMaskCache = new Map();
function getCausalMask(size) {
  if (_causalMaskCache.has(size)) return _causalMaskCache.get(size);
  const storage = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      storage[i * size + j] = j <= i ? 0.0 : -1e9;
    }
  }
  const mask = new Tensor(new TensorData(storage, [1, 1, size, size]));
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

  async forward(x) {
    const [B, S, E] = x.shape;
    const { nHead, headDim, scale } = this;

    let q = await this.queryProj.forward(x);
    let k = await this.keyProj.forward(x);
    let v = await this.valueProj.forward(x);

    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3);
    const kT = k.view(B, S, nHead, headDim).permute(0, 2, 3, 1);
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3);

    let att = (await q.matmul(kT)).mul(scale);
    att = att.add(getCausalMask(S));
    att = softmax(att, 3);
    att = dropout(att, this.dropoutRate, !this.training);

    let out = await att.matmul(v);
    out = out.permute(0, 2, 1, 3).contiguous().view(B, S, E);

    return await this.outProj.forward(out);
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
  async forward(x) {
    let h = await this.fc1.forward(x);
    h = gelu(h);
    h = await this.fc2.forward(h);
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
  async forward(x) {
    x = x.add(await this.attn.forward(this.ln1.forward(x)));
    x = x.add(await this.ffn.forward(this.ln2.forward(x)));
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
    this.headBias = new Parameter(Tensor.zeros([vocabSize]));
  }

  async forward(indices) {
    const batch = indices.length;
    const seqLen = indices[0].length;

    let x = this.tokenEmb.forward(indices);

    const posIndices = [];
    for (let b = 0; b < batch; b++) {
      posIndices.push(Array.from({ length: seqLen }, (_, i) => i));
    }
    x = x.add(this.posEmb.forward(posIndices));

    for (let i = 0; i < this.config.nLayer; i++) {
      x = await this[`block${i}`].forward(x);
    }

    x = this.lnFinal.forward(x);
    // Weight tying: reuse token embedding transposed as output projection
    const wT = this.tokenEmb.weight.value.permute(1, 0);
    return (await x.matmul(wT)).add(this.headBias.value);
  }
}

// ════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════

function oneHot(indices, numClasses) {
  const batch = indices.length;
  const seq = indices[0].length;
  const storage = new Float64Array(batch * seq * numClasses);
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seq; s++) {
      storage[(b * seq + s) * numClasses + indices[b][s]] = 1.0;
    }
  }
  return new Tensor(new TensorData(storage, [batch, seq, numClasses]));
}

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

async function estimateLoss(model, tokenizer, trainData, valData, config) {
  model.eval();
  const losses = { train: 0, val: 0 };
  for (const [name, data] of [['train', trainData], ['val', valData]]) {
    let total = 0;
    for (let i = 0; i < config.evalIters; i++) {
      const { inputs, targets } = getBatch(data, config.blockSize, config.batchSize);
      const logits = await model.forward(inputs);
      const targetOneHot = oneHot(targets, tokenizer.vocabSize);
      const loss = crossEntropyLoss(logits, targetOneHot);
      total += loss.item();
    }
    losses[name] = total / config.evalIters;
  }
  model.train();
  return losses;
}

async function generate(model, tokenizer, prompt, maxTokens, temperature = 0.8) {
  model.eval();
  let context = tokenizer.encode(prompt);
  const blockSize = model.config.blockSize;

  for (let i = 0; i < maxTokens; i++) {
    const ctxWindow = context.slice(-blockSize);
    const logits = await model.forward([ctxWindow]);

    const seqLen = ctxWindow.length;
    const vocabSize = tokenizer.vocabSize;
    const lastLogits = [];
    for (let v = 0; v < vocabSize; v++) {
      lastLogits.push(logits.get([0, seqLen - 1, v]) / temperature);
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
  }

  model.train();
  return tokenizer.decode(context);
}

// ════════════════════════════════════════════════════════════════
// Training
// ════════════════════════════════════════════════════════════════

async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log('  MiniGPT — Training a Tiny Transformer LLM');
  console.log('═══════════════════════════════════════════════════════');
  console.log('');

  let model, tokenizer, startStep = 0;

  const latest = NO_RESUME ? null : findLatestCheckpoint(CONFIG.modelDir);
  if (latest) {
    console.log(`  Resuming from checkpoint at step ${latest.step}`);
    const loaded = loadModel(latest.path);
    model = loaded.model;
    tokenizer = loaded.tokenizer;
    startStep = latest.step;
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
  console.log(`  Optimizer:      Adam (lr=${CONFIG.lr}, β1=${CONFIG.beta1}, β2=${CONFIG.beta2})`);
  console.log(`  Schedule:       ${CONFIG.warmupSteps} warmup → cosine decay to ${CONFIG.minLr}`);
  console.log(`  Grad clipping:  max norm ${CONFIG.maxGradNorm}`);
  console.log(`  Training:       steps ${startStep}→${CONFIG.maxIters}, batch=${CONFIG.batchSize}`);
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

  const startTime = Date.now();
  let bestValLoss = Infinity;

  for (let iter = startStep; iter < CONFIG.maxIters; iter++) {
    optimizer.lr = getLR(iter, CONFIG.warmupSteps, CONFIG.maxIters, CONFIG.lr, CONFIG.minLr);

    const { inputs, targets } = getBatch(trainData, CONFIG.blockSize, CONFIG.batchSize);

    optimizer.zeroGrad();
    const logits = await model.forward(inputs);
    const targetOneHot = oneHot(targets, tokenizer.vocabSize);
    const loss = crossEntropyLoss(logits, targetOneHot);
    await loss.backward();

    const gradNorm = clipGradNorm(params, CONFIG.maxGradNorm);
    optimizer.step();

    if (iter % CONFIG.evalInterval === 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
      const losses = await estimateLoss(model, tokenizer, trainData, valData, CONFIG);
      const marker = losses.val < bestValLoss ? ' *' : '';
      if (losses.val < bestValLoss) bestValLoss = losses.val;
      const etaSeconds = (CONFIG.maxIters - iter) * (Date.now() - startTime) / Math.max(1, iter) / 1000;
      const eta = etaSeconds > 3600
        ? `${(etaSeconds / 3600).toFixed(1)}h`
        : `${(etaSeconds / 60).toFixed(0)}m`;
      console.log(
        `  step ${String(iter).padStart(5)} │` +
        ` train ${losses.train.toFixed(4)} │` +
        ` val ${losses.val.toFixed(4)}${marker} │` +
        ` lr ${optimizer.lr.toExponential(1)} │` +
        ` gnorm ${gradNorm.toFixed(1)} │` +
        ` ${elapsed}s (eta ${eta})`
      );
    }

    if (iter > 0 && iter % CONFIG.generateEvery === 0) {
      const sample = await generate(model, tokenizer, '\n', CONFIG.generateLen);
      console.log('');
      console.log('  ── sample ──────────────────────────────────────────');
      for (const line of sample.split('\n').slice(0, 6)) {
        console.log(`  ${line}`);
      }
      console.log('  ───────────────────────────────────────────────────');
      console.log('');
    }

    if (iter > 0 && iter % CONFIG.checkpointEvery === 0) {
      saveModel(model, tokenizer, join(CONFIG.modelDir, `checkpoint-${iter}.json`), { step: iter });
    }
  }

  console.log('');
  console.log('─────────────────────────────────────────────────────────');
  console.log('  Final Generation (temperature=0.8)');
  console.log('─────────────────────────────────────────────────────────');
  console.log('');
  const finalText = await generate(model, tokenizer, '\n', CONFIG.generateLen * 2);
  for (const line of finalText.split('\n').slice(0, 12)) {
    console.log(`  ${line}`);
  }
  saveModel(model, tokenizer, join(CONFIG.modelDir, 'model-final.json'), { step: CONFIG.maxIters });

  console.log('');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`  Best val loss: ${bestValLoss.toFixed(4)}`);
  console.log(`  Total time:    ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`);
  console.log(`  Model saved:   ${join(CONFIG.modelDir, 'model-final.json')}`);
  console.log('═══════════════════════════════════════════════════════');

  destroyPool();
  destroyDevice();
}

main();

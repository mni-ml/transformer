/**
 * Load a saved MiniGPT checkpoint and generate text.
 *
 * Usage:
 *   node generate.js                          # uses out/model-final.json
 *   node generate.js out/checkpoint-1000.json  # specific checkpoint
 *   node generate.js out/model-final.json "ROMEO:" 300 0.8
 */
import {
  Tensor, native,
  Module, Parameter,
  Linear, Embedding,
  softmax, gelu, dropout, layerNorm,
} from '@mni-ml/framework';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Model architecture (must match train.js) ────────────────

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
    const [B, S, E] = x.shape;
    const { nHead, headDim, scale } = this;
    let q = this.queryProj.forward(x);
    let k = this.keyProj.forward(x);
    let v = this.valueProj.forward(x);
    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3);
    const kT = k.view(B, S, nHead, headDim).permute(0, 2, 3, 1);
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3);
    let att = q.matmul(kT).mul(scale);
    att = att.add(getCausalMask(S));
    att = softmax(att, 3);
    let out = att.matmul(v);
    out = out.permute(0, 2, 1, 3).contiguous().view(B, S, E);
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
}

class MiniGPT extends Module {
  constructor(vocabSize, config) {
    super();
    const { nEmbd, nHead, nLayer, blockSize, dropoutRate = 0 } = config;
    this.config = config;
    this.vocabSize = vocabSize;
    this.tokenEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);
    for (let i = 0; i < nLayer; i++)
      this[`block${i}`] = new TransformerBlock(nEmbd, nHead, dropoutRate);
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

// ── Load + Generate ─────────────────────────────────────────

function loadModel(path) {
  const checkpoint = JSON.parse(readFileSync(path, 'utf-8'));
  const tokenizer = checkpoint.tokenizer;
  tokenizer.encode = (text) => [...text].map(ch => tokenizer.stoi[ch]);
  tokenizer.decode = (indices) => indices.map(i => tokenizer.itos[i]).join('');

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

function generate(model, tokenizer, prompt, maxTokens, temperature = 0.8) {
  model.eval();
  native.noGradStart();
  let context = tokenizer.encode(prompt);
  const blockSize = model.config.blockSize;

  for (let i = 0; i < maxTokens; i++) {
    const ctxWindow = context.slice(-blockSize);
    const logits = model.forward([ctxWindow]);
    const seqLen = ctxWindow.length;
    const vocabSize = tokenizer.vocabSize;
    const data = logits.toFloat32();
    const offset = (seqLen - 1) * vocabSize;
    const lastLogits = [];
    for (let v = 0; v < vocabSize; v++)
      lastLogits.push(data[offset + v] / temperature);

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
  native.noGradEnd();
  return tokenizer.decode(context);
}

// ── Main ────────────────────────────────────────────────────

const args = process.argv.slice(2);
const modelPath = args[0] || join(__dirname, 'out', 'model-final.json');
const prompt = args[1] || '\n';
const numTokens = parseInt(args[2] || '300');
const temperature = parseFloat(args[3] || '0.8');

if (!existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`);
  console.error('Train a model first with: node train.js');
  process.exit(1);
}

console.log(`Loading model from ${modelPath}...`);
const { model, tokenizer, config } = loadModel(modelPath);
console.log(`Model: ${config.nLayer} layers, ${config.nHead} heads, ${config.nEmbd}-dim`);
console.log(`Generating ${numTokens} tokens (temperature=${temperature})...\n`);

const text = generate(model, tokenizer, prompt, numTokens, temperature);
console.log(text);

import {
  Tensor, TensorData,
  Module, Parameter,
  Linear, Embedding,
  SGD,
  softmax, crossEntropyLoss, destroyPool,
} from '@mni-ml/framework';

// ════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════

const CONFIG = {
  nEmbd: 32,
  nHead: 2,
  nLayer: 2,
  blockSize: 16,
  batchSize: 4,
  lr: 0.01,
  maxIters: 200,
  evalInterval: 10,
  generateEvery: 50,
  generateLen: 80,
};

// ════════════════════════════════════════════════════════════════
// Training Data
// ════════════════════════════════════════════════════════════════

const TEXT = `twinkle twinkle little star how i wonder what you are
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
//
// Not built into the framework, so we compose it from primitives:
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// Using the log-exp trick for 1/sqrt: exp(-0.5 * ln(var + eps))
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
//
// Core of the transformer: each token attends to all previous tokens
// (and itself) through multiple parallel attention heads.
//
//   Q, K, V = linear_projections(x)
//   attention = softmax(Q @ K^T / sqrt(d_k) + causal_mask)
//   output = attention @ V
// ════════════════════════════════════════════════════════════════

function createCausalMask(size) {
  const storage = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      storage[i * size + j] = j <= i ? 0.0 : -1e9;
    }
  }
  return new Tensor(new TensorData(storage, [1, 1, size, size]));
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

    // Reshape [B, S, E] → [B, S, nHead, headDim] then rearrange for attention
    q = q.view(B, S, nHead, headDim).permute(0, 2, 1, 3);     // [B, H, S, d]
    const kT = k.view(B, S, nHead, headDim).permute(0, 2, 3, 1); // [B, H, d, S]
    v = v.view(B, S, nHead, headDim).permute(0, 2, 1, 3);     // [B, H, S, d]

    // Scaled dot-product attention with causal mask
    let att = q.matmul(kT).mul(scale);
    att = att.add(createCausalMask(S));
    att = softmax(att, 3);

    // Weighted combination of values, then merge heads back
    let out = att.matmul(v);
    out = out.permute(0, 2, 1, 3).contiguous().view(B, S, E);

    return this.outProj.forward(out);
  }
}

// ════════════════════════════════════════════════════════════════
// Position-wise Feed-Forward Network
//
// Two linear layers with ReLU activation and a 4x expansion in the
// hidden dimension — the "thinking" part of each transformer block.
// ════════════════════════════════════════════════════════════════

class FeedForward extends Module {
  constructor(nEmbd) {
    super();
    this.fc1 = new Linear(nEmbd, 4 * nEmbd);
    this.fc2 = new Linear(4 * nEmbd, nEmbd);
  }
  forward(x) {
    return this.fc2.forward(this.fc1.forward(x).relu());
  }
}

// ════════════════════════════════════════════════════════════════
// Transformer Block
//
// Pre-norm architecture (like GPT-2): LayerNorm before each
// sub-layer, with residual connections around both attention and FFN.
// ════════════════════════════════════════════════════════════════

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

// ════════════════════════════════════════════════════════════════
// MiniGPT — The Full Model
//
// Architecture mirrors GPT-2:
//   1. Token embedding + learned positional embedding
//   2. Stack of N transformer blocks
//   3. Final LayerNorm
//   4. Linear head projecting to vocabulary logits
// ════════════════════════════════════════════════════════════════

class MiniGPT extends Module {
  constructor(vocabSize, config) {
    super();
    const { nEmbd, nHead, nLayer, blockSize } = config;
    this.config = config;
    this.vocabSize = vocabSize;

    this.tokenEmb = new Embedding(vocabSize, nEmbd);
    this.posEmb = new Embedding(blockSize, nEmbd);

    for (let i = 0; i < nLayer; i++) {
      this[`block${i}`] = new TransformerBlock(nEmbd, nHead);
    }

    this.lnFinal = new LayerNorm(nEmbd);
    this.head = new Linear(nEmbd, vocabSize);
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
    return this.head.forward(x);
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

/**
 * Autoregressive text generation: feeds the model its own output
 * token by token, building up a sequence from a prompt.
 */
function generate(model, tokenizer, prompt, maxTokens, temperature = 1.0) {
  let context = tokenizer.encode(prompt);
  const blockSize = model.config.blockSize;

  for (let i = 0; i < maxTokens; i++) {
    const ctxWindow = context.slice(-blockSize);
    const logits = model.forward([ctxWindow]);

    const seqLen = ctxWindow.length;
    const vocabSize = tokenizer.vocabSize;
    const lastLogits = [];
    for (let v = 0; v < vocabSize; v++) {
      lastLogits.push(logits.get([0, seqLen - 1, v]) / temperature);
    }

    // Numerically stable softmax for sampling
    const maxLogit = Math.max(...lastLogits);
    const exps = lastLogits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    const probs = exps.map(e => e / sumExps);

    // Multinomial sampling
    const r = Math.random();
    let cumSum = 0;
    let nextToken = 0;
    for (let v = 0; v < vocabSize; v++) {
      cumSum += probs[v];
      if (r < cumSum) { nextToken = v; break; }
    }

    context.push(nextToken);
  }

  return tokenizer.decode(context);
}

// ════════════════════════════════════════════════════════════════
// Training Loop
// ════════════════════════════════════════════════════════════════

function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════');
  console.log('  MiniGPT — A Tiny Transformer Language Model');
  console.log('═══════════════════════════════════════════════════');
  console.log('');

  const tokenizer = new CharTokenizer(TEXT);
  const data = tokenizer.encode(TEXT);

  console.log(`  Vocabulary:     ${tokenizer.vocabSize} unique characters`);
  console.log(`  Training data:  ${TEXT.length} characters`);
  console.log(`  Architecture:   ${CONFIG.nLayer} blocks, ${CONFIG.nHead} heads, ${CONFIG.nEmbd}-dim embeddings`);
  console.log(`  Context window: ${CONFIG.blockSize} tokens`);

  const model = new MiniGPT(tokenizer.vocabSize, CONFIG);
  const params = model.parameters();
  const totalParams = params.reduce((sum, p) => sum + p.value.size, 0);
  console.log(`  Parameters:     ${totalParams.toLocaleString()}`);
  console.log(`  Optimizer:      SGD (lr=${CONFIG.lr})`);
  console.log('');
  console.log('───────────────────────────────────────────────────');
  console.log('  Training');
  console.log('───────────────────────────────────────────────────');
  console.log('');

  const optimizer = new SGD(params, CONFIG.lr);
  const startTime = Date.now();

  for (let iter = 0; iter < CONFIG.maxIters; iter++) {
    const { inputs, targets } = getBatch(data, CONFIG.blockSize, CONFIG.batchSize);

    optimizer.zeroGrad();
    const logits = model.forward(inputs);
    const targetOneHot = oneHot(targets, tokenizer.vocabSize);
    const loss = crossEntropyLoss(logits, targetOneHot);
    loss.backward();
    optimizer.step();

    if (iter % CONFIG.evalInterval === 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`  step ${String(iter).padStart(4)} │ loss = ${loss.item().toFixed(4)} │ ${elapsed}s`);
    }

    if (iter > 0 && iter % CONFIG.generateEvery === 0) {
      const sample = generate(model, tokenizer, 'the ', CONFIG.generateLen, 0.8);
      console.log(`           │ sample: "${sample}"`);
      console.log('');
    }
  }

  console.log('');
  console.log('───────────────────────────────────────────────────');
  console.log('  Final Generation (temperature=0.8)');
  console.log('───────────────────────────────────────────────────');
  console.log('');
  console.log(generate(model, tokenizer, 'the ', CONFIG.generateLen * 2, 0.8));
  console.log('');
  console.log('═══════════════════════════════════════════════════');
  console.log(`  Done in ${((Date.now() - startTime) / 1000).toFixed(1)}s`);
  console.log('═══════════════════════════════════════════════════');

  destroyPool();
}

main();

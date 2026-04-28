/**
 * KV-cache generation using the local framework KvCache implementation.
 *
 * Usage:
 *   node src/generate_kv.js out/model-final.json "hello" 64 0 data/tokenizer.json fp32
 *   node src/generate_kv.js out/model-final.json "hello" 64 0 data/tokenizer.json int8
 */
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { formatBenchmarkTable, generateWithKvCache, loadModel } from './generation_kv.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const args = process.argv.slice(2);
const modelPath = args[0] || join(ROOT, 'out', 'model-final.json');
const prompt = args[1] || '\n';
const numTokens = parseInt(args[2] || '300', 10);
const temperature = parseFloat(args[3] || '0.8');
const tokenizerPath = args[4] || null;
const cacheMode = args[5] || 'fp32';

if (!existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`);
  console.error('Train a model first with: npm run train');
  process.exit(1);
}

if (cacheMode !== 'fp32' && cacheMode !== 'int8') {
  console.error(`Unsupported cache mode: ${cacheMode}`);
  console.error('Use one of: fp32, int8');
  process.exit(1);
}

console.log(`Loading model from ${modelPath}...`);
const { model, tokenizer, config } = loadModel(modelPath, tokenizerPath);
console.log(`Model: ${config.nLayer} layers, ${config.nHead} heads, ${config.nEmbd}-dim`);
console.log(`Generating ${numTokens} tokens with kv cache (${cacheMode}, temperature=${temperature})...\n`);

const result = generateWithKvCache(model, tokenizer, prompt, numTokens, temperature, cacheMode);
console.log(formatBenchmarkTable([result]));
console.log('');
console.log(result.text);

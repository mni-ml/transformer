/**
 * Compare baseline generation against fp32 and int8 KV-cache generation.
 *
 * Usage:
 *   node src/demo_kv.js out/model-final.json "hello" 64 0 data/tokenizer.json
 */
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  formatBenchmarkTable,
  generateBaseline,
  generateWithKvCache,
  loadModel,
} from './generation_kv.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const args = process.argv.slice(2);
const modelPath = args[0] || join(ROOT, 'out', 'model-final.json');
const prompt = args[1] || '\n';
const numTokens = parseInt(args[2] || '64', 10);
const temperature = parseFloat(args[3] || '0');
const tokenizerPath = args[4] || null;

if (!existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`);
  console.error('Train a model first with: npm run train');
  process.exit(1);
}

console.log(`Loading model from ${modelPath}...`);
const { model, tokenizer, config } = loadModel(modelPath, tokenizerPath);
console.log(`Model: ${config.nLayer} layers, ${config.nHead} heads, ${config.nEmbd}-dim`);
console.log(`Prompt length target: ${numTokens} generated tokens (temperature=${temperature})...\n`);

const results = [
  generateBaseline(model, tokenizer, prompt, numTokens, temperature),
  generateWithKvCache(model, tokenizer, prompt, numTokens, temperature, 'fp32'),
  generateWithKvCache(model, tokenizer, prompt, numTokens, temperature, 'int8'),
];

console.log(formatBenchmarkTable(results));
console.log('');
for (const result of results) {
  console.log(`[${result.mode}]`);
  console.log(result.text);
  console.log('');
}

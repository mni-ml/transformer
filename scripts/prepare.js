import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const DATA_DIR = join(ROOT, 'data');
const OUTPUT = join(DATA_DIR, 'input.txt');
const URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';

async function main() {
  if (existsSync(OUTPUT)) {
    console.log(`Already exists: ${OUTPUT}`);
    return;
  }

  console.log(`Downloading tinyshakespeare from:\n  ${URL}`);
  const res = await fetch(URL);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  const text = await res.text();

  mkdirSync(DATA_DIR, { recursive: true });
  writeFileSync(OUTPUT, text, 'utf-8');

  const chars = new Set(text);
  console.log(`\nSaved ${text.length.toLocaleString()} characters to ${OUTPUT}`);
  console.log(`Vocabulary: ${chars.size} unique characters`);
  console.log(`Lines: ${text.split('\n').length.toLocaleString()}`);
}

main().catch(err => { console.error(err); process.exit(1); });

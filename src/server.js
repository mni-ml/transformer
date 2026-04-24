import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  generateBaseline,
  generateWithKvCache,
  loadModel,
} from './generation_kv.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

const args = process.argv.slice(2);
const modelPath = args[0] || join(ROOT, 'out', 'model-final.json');
const tokenizerPath = args[1] || null;
const port = Number(args[2] || process.env.PORT || 3000);
const weightMode = args[3] || process.env.WEIGHT_MODE || 'fp32';
const maxConcurrent = Number(process.env.MAX_CONCURRENT || 1);

if (!existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`);
  process.exit(1);
}
if (weightMode !== 'fp32' && weightMode !== 'int8') {
  console.error(`Unsupported WEIGHT_MODE=${weightMode}. Use fp32 or int8.`);
  process.exit(1);
}

console.log(`Loading model from ${modelPath} (weights=${weightMode})...`);
const { model, tokenizer, config, quantization } = loadModel(modelPath, tokenizerPath, { weightMode });
console.log(
  `Model ready: ${config.nLayer} layers, ${config.nHead} heads, ${config.nEmbd}-dim, ` +
    `quantized_params=${quantization.quantizedParamCount}`
);

const queue = [];
let active = 0;
let totalRequests = 0;
let totalLatencyMs = 0;

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(payload));
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 2 * 1024 * 1024) {
        reject(new Error('request body too large'));
      }
    });
    req.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch {
        reject(new Error('invalid json body'));
      }
    });
    req.on('error', reject);
  });
}

async function runInference(task) {
  const {
    prompt = '\n',
    maxTokens = 64,
    temperature = 0.8,
    cacheMode = 'int8',
  } = task.body || {};

  const maxNew = Number(maxTokens);
  const temp = Number(temperature);
  if (!Number.isFinite(maxNew) || maxNew < 1 || maxNew > 2048) {
    throw new Error('maxTokens must be in [1, 2048]');
  }
  if (!Number.isFinite(temp) || temp < 0 || temp > 5) {
    throw new Error('temperature must be in [0, 5]');
  }

  const startedAt = performance.now();
  let result;
  if (cacheMode === 'none') {
    result = generateBaseline(model, tokenizer, String(prompt), maxNew, temp);
  } else if (cacheMode === 'fp32' || cacheMode === 'int8') {
    result = generateWithKvCache(model, tokenizer, String(prompt), maxNew, temp, cacheMode);
  } else {
    throw new Error('cacheMode must be one of: none, fp32, int8');
  }
  const requestMs = performance.now() - startedAt;
  totalRequests += 1;
  totalLatencyMs += requestMs;

  return {
    text: result.text,
    metrics: {
      ...result,
      requestMs,
      requestId: task.id,
      queueDepthAtStart: task.queueDepth,
    },
    model: {
      weightMode,
      quantizedParams: quantization.quantizedParamCount,
      packedBytes: quantization.packedBytes,
      config,
    },
  };
}

function drainQueue() {
  while (active < maxConcurrent && queue.length > 0) {
    const task = queue.shift();
    active += 1;
    runInference(task)
      .then((payload) => sendJson(task.res, 200, payload))
      .catch((err) => sendJson(task.res, 400, { error: err.message }))
      .finally(() => {
        active -= 1;
        drainQueue();
      });
  }
}

const server = createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    sendJson(res, 200, {
      ok: true,
      queueDepth: queue.length,
      active,
      model: { weightMode, nLayer: config.nLayer, nHead: config.nHead, nEmbd: config.nEmbd },
    });
    return;
  }

  if (req.method === 'GET' && req.url === '/stats') {
    sendJson(res, 200, {
      totalRequests,
      avgLatencyMs: totalRequests > 0 ? totalLatencyMs / totalRequests : 0,
      queueDepth: queue.length,
      active,
      maxConcurrent,
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/generate') {
    try {
      const body = await readJsonBody(req);
      queue.push({
        id: totalRequests + queue.length + active + 1,
        body,
        res,
        queueDepth: queue.length,
      });
      drainQueue();
    } catch (err) {
      sendJson(res, 400, { error: err.message });
    }
    return;
  }

  sendJson(res, 404, { error: 'not found' });
});

server.listen(port, () => {
  console.log(`Inference server listening on http://localhost:${port}`);
  console.log('POST /generate with {"prompt","maxTokens","temperature","cacheMode"}');
});

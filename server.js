const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Readable } = require('stream');

const PORT = Number(process.env.PORT) || 3000;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
const DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions';
const DEEPSEEK_MODELS_URL = 'https://api.deepseek.com/models';
const OPENCODE_API_KEY = process.env.OPENCODE_API_KEY;
const OPENCODE_CHAT_URL = 'https://opencode.ai/zen/go/v1/chat/completions';
const OPENCODE_MODELS_URL = 'https://opencode.ai/zen/go/v1/models';
const OPENCODE_ZEN_CHAT_URL = 'https://opencode.ai/zen/v1/chat/completions';
const OPENCODE_ZEN_MODELS_URL = 'https://opencode.ai/zen/v1/models';
const OPENCODE_PREFIX = 'opencode/';
const OPENCODE_SESSION_ID = crypto.randomUUID();
const DEFAULT_MODEL = 'deepseek-chat';
const MODEL_NAME_RE = /^[a-z0-9][a-z0-9._-]*$/i;
const MIN_TOP_P = 0.01;
const MIN_TEMPERATURE = 0;
const MAX_TEMPERATURE = 2;

const GO_CHAT_MODELS = new Set([
  'glm-5.3', 'glm-5.3-flash', 'glm-5.2', 'glm-5.1',
  'kimi-k3', 'kimi-k2.7-code', 'kimi-k2.6',
  'longcat-2.0',
  'deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-v4-flash-vision-exp',
  'mimo-v2.5', 'mimo-v2.5-pro',
  'hy4-preview', 'hy3',
  'omen-alpha'
]);

const ZEN_FREE_MODELS = new Set([
  'big-pickle',
  'deepseek-v4-flash-free',
  'mimo-v2.5-free',
  'ling-3.0-flash-fin-free',
  'nemotron-3-ultra-free',
  'nemotron-3.5-lightning-free'
]);

const PUBLIC_DIR = path.join(__dirname, 'public');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon'
};

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body)
  });
  res.end(body);
}

function serveStatic(req, res) {
  let urlPath = decodeURIComponent(req.url.split('?')[0]);
  if (urlPath === '/') urlPath = '/index.html';

  const filePath = path.join(PUBLIC_DIR, path.normalize(urlPath));
  if (!filePath.startsWith(PUBLIC_DIR)) {
    sendJson(res, 403, { error: 'Forbidden' });
    return;
  }

  fs.readFile(filePath, (err, content) => {
    if (err) {
      sendJson(res, 404, { error: 'Not found' });
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
    res.end(content);
  });
}

async function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
      if (data.length > 1e6) reject(new Error('Request too large'));
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

function sanitizeSettings(body) {
  const out = {};
  if (
    typeof body.temperature === 'number' &&
    Number.isFinite(body.temperature) &&
    body.temperature >= MIN_TEMPERATURE &&
    body.temperature <= MAX_TEMPERATURE
  ) {
    out.temperature = body.temperature;
  }
  if (typeof body.top_p === 'number' && Number.isFinite(body.top_p) && body.top_p >= 0 && body.top_p <= 1) {
    out.top_p = Math.max(body.top_p, MIN_TOP_P);
  }
  if (Number.isInteger(body.max_tokens) && body.max_tokens > 0) {
    out.max_tokens = body.max_tokens;
  }
  if (Array.isArray(body.stop)) {
    const stop = body.stop
      .filter((s) => typeof s === 'string' && s.trim())
      .slice(0, 16);
    if (stop.length > 0) out.stop = stop;
  }
  if (
    body.response_format &&
    typeof body.response_format === 'object' &&
    typeof body.response_format.type === 'string' &&
    body.response_format.type
  ) {
    out.response_format = { type: body.response_format.type };
  }
  return out;
}

function upstreamErrorMessage(rawText) {
  if (!rawText) return 'no details';
  try {
    const data = JSON.parse(rawText);
    if (data && typeof data === 'object') {
      const err = data.error;
      if (typeof err === 'string' && err) return err;
      if (err && typeof err.message === 'string' && err.message) return err.message;
    }
  } catch {
    // not JSON — fall through
  }
  const trimmed = rawText.trim();
  return trimmed.length > 200 ? `${trimmed.slice(0, 200)}…` : trimmed;
}

async function handleChat(req, res) {
  let body;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    sendJson(res, 400, { error: 'Invalid JSON body' });
    return;
  }

  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (messages.length === 0) {
    sendJson(res, 400, { error: 'messages must be a non-empty array' });
    return;
  }

  const rawModel = typeof body.model === 'string' ? body.model.trim() : '';
  const isOpenCode = rawModel.startsWith(OPENCODE_PREFIX);
  const provider = isOpenCode ? 'OpenCode' : 'DeepSeek';
  let model = rawModel && MODEL_NAME_RE.test(rawModel) ? rawModel : DEFAULT_MODEL;
  let url = DEEPSEEK_URL;
  let apiKey = DEEPSEEK_API_KEY;

  if (isOpenCode) {
    model = rawModel.slice(OPENCODE_PREFIX.length);
    if (!MODEL_NAME_RE.test(model)) {
      sendJson(res, 400, { error: `Unsupported model: ${model}` });
      return;
    }
    if (GO_CHAT_MODELS.has(model)) {
      url = OPENCODE_CHAT_URL;
    } else if (ZEN_FREE_MODELS.has(model)) {
      url = OPENCODE_ZEN_CHAT_URL;
    } else {
      sendJson(res, 400, { error: `Unsupported model: ${model}` });
      return;
    }
    apiKey = OPENCODE_API_KEY;
  }

  if (!apiKey) {
    sendJson(res, 500, {
      error: isOpenCode ? 'OPENCODE_API_KEY is not set on the server' : 'DEEPSEEK_API_KEY is not set on the server'
    });
    return;
  }

  const upstreamHeaders = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
    'User-Agent': 'pomogator2k/1.0 (https://github.com/vitchuk/ai-challenge)'
  };
  if (isOpenCode) {
    upstreamHeaders['x-opencode-session'] = OPENCODE_SESSION_ID;
  }

  const upstream = await fetch(url, {
    method: 'POST',
    headers: upstreamHeaders,
    body: JSON.stringify({
      model,
      messages,
      stream: true,
      stream_options: { include_usage: true },
      ...sanitizeSettings(body)
    })
  });

  if (!upstream.ok) {
    const errText = await upstream.text();
    console.error(`${provider} API error:`, upstream.status, errText);
    sendJson(res, upstream.status, { error: `${provider} API error: ${upstream.status} — ${upstreamErrorMessage(errText)}` });
    return;
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no'
  });

  const stream = Readable.fromWeb(upstream.body);
  stream.pipe(res);
  stream.on('error', () => res.end());
  req.on('close', () => stream.destroy());
}

async function fetchModelIds(url, apiKey, transform, extraHeaders) {
  const upstream = await fetch(url, {
    headers: { Authorization: `Bearer ${apiKey}`, ...extraHeaders }
  });
  if (!upstream.ok) throw new Error(`HTTP ${upstream.status}`);
  const data = await upstream.json();
  const items = Array.isArray(data.data) ? data.data : [];
  return items
    .map((m) => (m && typeof m.id === 'string' ? transform(m.id) : null))
    .filter(Boolean);
}

async function handleModels(res) {
  if (!DEEPSEEK_API_KEY && !OPENCODE_API_KEY) {
    sendJson(res, 500, { error: 'No API keys configured (DEEPSEEK_API_KEY / OPENCODE_API_KEY)' });
    return;
  }

  const models = [];
  const errors = [];

  if (DEEPSEEK_API_KEY) {
    try {
      const ids = await fetchModelIds(DEEPSEEK_MODELS_URL, DEEPSEEK_API_KEY, (id) => id);
      for (const id of ids) models.push({ id, owned_by: 'deepseek' });
    } catch (err) {
      errors.push(`DeepSeek: ${err.message}`);
    }
  }

  if (OPENCODE_API_KEY) {
    const openCodeHeaders = {
      'x-opencode-session': OPENCODE_SESSION_ID,
      'User-Agent': 'pomogator2k/1.0 (https://github.com/vitchuk/ai-challenge)'
    };

    try {
      const ids = await fetchModelIds(OPENCODE_MODELS_URL, OPENCODE_API_KEY, (id) => id, openCodeHeaders);
      for (const id of ids) {
        if (GO_CHAT_MODELS.has(id)) {
          models.push({ id: `${OPENCODE_PREFIX}${id}`, owned_by: 'opencode' });
        }
      }
    } catch (err) {
      errors.push(`OpenCode Go: ${err.message}`);
    }

    try {
      const ids = await fetchModelIds(OPENCODE_ZEN_MODELS_URL, OPENCODE_API_KEY, (id) => id, openCodeHeaders);
      for (const id of ids) {
        if (ZEN_FREE_MODELS.has(id)) {
          models.push({ id: `${OPENCODE_PREFIX}${id}`, owned_by: 'opencode' });
        }
      }
    } catch (err) {
      errors.push(`OpenCode Zen (free): ${err.message}`);
    }
  }

  if (models.length === 0) {
    sendJson(res, 502, { error: `Failed to load models: ${errors.join('; ')}` });
    return;
  }

  if (errors.length > 0) {
    console.warn('Some model providers failed to load:', errors.join('; '));
  }

  sendJson(res, 200, { object: 'list', data: models });
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/api/models') {
    try {
      await handleModels(res);
    } catch (err) {
      console.error(err);
      sendJson(res, 500, { error: 'Internal server error' });
    }
    return;
  }

  if (req.method === 'POST' && req.url === '/api/chat') {
    try {
      await handleChat(req, res);
    } catch (err) {
      console.error(err);
      sendJson(res, 500, { error: 'Internal server error' });
    }
    return;
  }

  if (req.method === 'GET') {
    serveStatic(req, res);
    return;
  }

  sendJson(res, 405, { error: 'Method not allowed' });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});

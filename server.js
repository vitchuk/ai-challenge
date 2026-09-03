const http = require('http');
const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');

const PORT = Number(process.env.PORT) || 3000;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
const DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions';
const DEEPSEEK_MODELS_URL = 'https://api.deepseek.com/models';
const DEFAULT_MODEL = 'deepseek-chat';
const MODEL_NAME_RE = /^[a-z0-9][a-z0-9._-]*$/i;
const MIN_TOP_P = 0.01;

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
  if (typeof body.temperature === 'number' && Number.isFinite(body.temperature) && body.temperature >= 0 && body.temperature <= 1) {
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

async function handleChat(req, res) {
  if (!DEEPSEEK_API_KEY) {
    sendJson(res, 500, { error: 'DEEPSEEK_API_KEY is not set on the server' });
    return;
  }

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
  const model = rawModel && MODEL_NAME_RE.test(rawModel) ? rawModel : DEFAULT_MODEL;

  const upstream = await fetch(DEEPSEEK_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${DEEPSEEK_API_KEY}`
    },
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
    console.error('DeepSeek API error:', upstream.status, errText);
    sendJson(res, upstream.status, { error: `DeepSeek API error: ${upstream.status}` });
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

async function handleModels(res) {
  if (!DEEPSEEK_API_KEY) {
    sendJson(res, 500, { error: 'DEEPSEEK_API_KEY is not set on the server' });
    return;
  }

  try {
    const upstream = await fetch(DEEPSEEK_MODELS_URL, {
      headers: { Authorization: `Bearer ${DEEPSEEK_API_KEY}` }
    });
    if (!upstream.ok) {
      sendJson(res, upstream.status, { error: `DeepSeek API error: ${upstream.status}` });
      return;
    }
    const data = await upstream.text();
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(data);
  } catch {
    sendJson(res, 502, { error: 'Failed to reach DeepSeek API' });
  }
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

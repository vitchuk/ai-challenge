const messagesEl = document.getElementById('messages');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const tokensTotalEl = document.getElementById('tokens-total');
const modelSelect = document.getElementById('model-select');
const tempRange = document.getElementById('setting-temperature');
const tempValue = document.getElementById('temperature-value');
const topPRange = document.getElementById('setting-top-p');
const topPValue = document.getElementById('top-p-value');
const maxTokensInput = document.getElementById('setting-max-tokens');
const stopInput = document.getElementById('setting-stop');
const modeToggle = document.getElementById('setting-mode');
const modeState = document.getElementById('mode-state');

const history = [];

const FALLBACK_MODELS = ['deepseek-chat', 'deepseek-reasoner'];

const MESSAGE_OVERHEAD_TOKENS = 4;

const JSON_SYSTEM_PROMPT = 'Выдавай ответ строго в формате JSON.';

let tokensBurned = 0;
let prevUsage = null;

function renderTotal() {
  tokensTotalEl.textContent = String(tokensBurned);
}

function currentModel() {
  return modelSelect.value || FALLBACK_MODELS[0];
}

function collectSettings() {
  const settings = {
    temperature: Number(tempRange.value) / 100,
    top_p: Number(topPRange.value) / 100
  };
  const maxTokens = parseInt(maxTokensInput.value, 10);
  if (Number.isInteger(maxTokens) && maxTokens > 0) settings.max_tokens = maxTokens;
  const stop = stopInput.value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (stop.length > 0) settings.stop = stop;
  if (!modeToggle.checked) settings.response_format = { type: 'json_object' };
  return settings;
}

function fillModelOptions(ids) {
  const preferred = ids.includes(FALLBACK_MODELS[0]) ? FALLBACK_MODELS[0] : ids[0];
  modelSelect.innerHTML = '';
  for (const id of ids) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = id;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = preferred;
}

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const ids = (Array.isArray(data.data) ? data.data : [])
      .map((m) => m && m.id)
      .filter((id) => typeof id === 'string');
    if (ids.length === 0) throw new Error('empty model list');
    fillModelOptions(ids);
  } catch {
    fillModelOptions(FALLBACK_MODELS);
  }
}

function addMessage(role, text) {
  const el = document.createElement('div');
  el.classList.add('message', `message--${role}`);
  if (!text) el.classList.add('message--empty');
  const content = document.createElement('div');
  content.className = 'message__content';
  content.textContent = text || '...';
  el.appendChild(content);
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function ensureWelcome() {
  if (messagesEl.children.length === 0) {
    addMessage('assistant', 'Привет! Чем могу помочь?');
  }
}

async function* parseSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      if (data === '[DONE]') return;
      try {
        yield JSON.parse(data);
      } catch {
        // ignore incomplete/empty frames
      }
    }
  }
}

function setBubbleText(el, text) {
  el.classList.remove('message--empty');
  el.querySelector('.message__content').textContent = text;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBubbleTokens(el, n) {
  let hint = el.querySelector('.message__tokens');
  if (!hint) {
    hint = document.createElement('div');
    hint.className = 'message__tokens';
    el.appendChild(hint);
  }
  hint.textContent = `токенов: ${n}`;
}

function createReasoning(el) {
  let node = el.querySelector('.message__reasoning');
  if (node) return node;

  el.classList.remove('message--empty');
  el.querySelector('.message__content').textContent = '';

  node = document.createElement('details');
  node.className = 'message__reasoning';

  const summary = document.createElement('summary');
  const spinner = document.createElement('span');
  spinner.className = 'message__reasoning-spinner';
  const chevron = document.createElement('span');
  chevron.className = 'message__reasoning-chevron';
  chevron.textContent = '\u25B8';
  const label = document.createElement('span');
  label.className = 'message__reasoning-label';
  label.textContent = 'Размышление';
  summary.append(spinner, chevron, label);
  node.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'message__reasoning-body';
  node.appendChild(body);

  el.insertBefore(node, el.querySelector('.message__content'));
  return node;
}

function appendReasoning(el, text) {
  const node = createReasoning(el);
  node.querySelector('.message__reasoning-body').textContent += text;
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function finishReasoning(el) {
  const node = el.querySelector('.message__reasoning');
  if (!node) return;
  const spinner = node.querySelector('.message__reasoning-spinner');
  if (spinner) spinner.remove();
}

function renderJsonEnvelope(el, thinking, response, date, usage) {
  el.classList.remove('message--empty');
  const contentEl = el.querySelector('.message__content');
  let pre = contentEl.querySelector('.message__json');
  if (!pre) {
    contentEl.textContent = '';
    pre = document.createElement('pre');
    pre.className = 'message__json';
    contentEl.appendChild(pre);
  }
  pre.textContent = JSON.stringify(
    {
      thinking,
      response,
      date,
      tokens: usage ? usage.completion_tokens : 0
    },
    null,
    2
  );
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function userMessageTokens(usage) {
  const prevTotal = prevUsage ? prevUsage.prompt_tokens + prevUsage.completion_tokens : 0;
  return Math.max(0, usage.prompt_tokens - prevTotal - MESSAGE_OVERHEAD_TOKENS);
}

async function sendMessage(text) {
  history.push({ role: 'user', content: text });
  const userEl = addMessage('user', text);

  const assistantEl = addMessage('assistant', '');
  const jsonMode = !modeToggle.checked;
  const responseDate = new Date().toISOString();
  let full = '';
  let thinkingText = '';
  let thinking = false;
  let usage = null;

  try {
    const outgoingMessages = jsonMode
      ? [{ role: 'system', content: JSON_SYSTEM_PROMPT }, ...history]
      : history;

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: outgoingMessages,
        model: currentModel(),
        ...collectSettings()
      })
    });

    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }

    for await (const chunk of parseSSE(res)) {
      if (chunk.usage) usage = chunk.usage;
      const delta = chunk.choices?.[0]?.delta;
      if (delta?.reasoning_content) {
        thinking = true;
        thinkingText += delta.reasoning_content;
        appendReasoning(assistantEl, delta.reasoning_content);
        if (jsonMode) renderJsonEnvelope(assistantEl, thinkingText, full, responseDate, usage);
      }
      if (delta?.content) {
        if (thinking) {
          thinking = false;
          finishReasoning(assistantEl);
        }
        full += delta.content;
        if (jsonMode) renderJsonEnvelope(assistantEl, thinkingText, full, responseDate, usage);
        else setBubbleText(assistantEl, full);
      }
    }

    if (thinking) finishReasoning(assistantEl);

    if (jsonMode) {
      renderJsonEnvelope(assistantEl, thinkingText, full, responseDate, usage);
    } else {
      setBubbleText(assistantEl, full || '(пустой ответ)');
    }
    history.push({ role: 'assistant', content: full });

    if (usage) {
      setBubbleTokens(userEl, userMessageTokens(usage));
      setBubbleTokens(assistantEl, usage.completion_tokens);
      tokensBurned += usage.total_tokens;
      prevUsage = {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens
      };
      renderTotal();
      if (jsonMode) renderJsonEnvelope(assistantEl, thinkingText, full, responseDate, usage);
    }
  } catch (err) {
    setBubbleText(assistantEl, `Ошибка: ${err.message}`);
    assistantEl.classList.add('message--error');
    if (history[history.length - 1].role === 'user') history.pop();
  }
}

function autoResize() {
  input.style.height = 'auto';
  const cs = getComputedStyle(input);
  const lineHeight = parseFloat(cs.lineHeight);
  const padding = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
  const border = parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
  const maxTwoLines = lineHeight * 2 + padding + border;
  const contentHeight = input.scrollHeight + border;
  const fits = contentHeight <= maxTwoLines;
  input.style.height = Math.min(contentHeight, maxTwoLines) + 'px';
  input.style.overflowY = fits ? 'hidden' : 'auto';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || sendBtn.disabled) return;

  sendBtn.disabled = true;
  input.value = '';
  autoResize();

  await sendMessage(text);

  sendBtn.disabled = false;
  input.focus();
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener('input', autoResize);
autoResize();
renderTotal();
loadModels();
ensureWelcome();

tempRange.addEventListener('input', () => {
  tempValue.textContent = `${tempRange.value}%`;
});

topPRange.addEventListener('input', () => {
  topPValue.textContent = `${topPRange.value}%`;
});

modeToggle.addEventListener('change', () => {
  modeState.textContent = modeToggle.checked ? 'Обычный' : 'JSON';
});

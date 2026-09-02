const messagesEl = document.getElementById('messages');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const tokensTotalEl = document.getElementById('tokens-total');

const history = [];

const MESSAGE_OVERHEAD_TOKENS = 4;

let tokensBurned = 0;
let prevUsage = null;

function renderTotal() {
  tokensTotalEl.textContent = String(tokensBurned);
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

function userMessageTokens(usage) {
  const prevTotal = prevUsage ? prevUsage.prompt_tokens + prevUsage.completion_tokens : 0;
  return Math.max(0, usage.prompt_tokens - prevTotal - MESSAGE_OVERHEAD_TOKENS);
}

async function sendMessage(text) {
  history.push({ role: 'user', content: text });
  const userEl = addMessage('user', text);

  const assistantEl = addMessage('assistant', '');
  let full = '';
  let usage = null;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: history })
    });

    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }

    for await (const chunk of parseSSE(res)) {
      if (chunk.usage) usage = chunk.usage;
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        full += delta;
        setBubbleText(assistantEl, full);
      }
    }

    setBubbleText(assistantEl, full || '(пустой ответ)');
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
ensureWelcome();

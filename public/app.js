const messagesEl = document.getElementById('messages');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');

const history = [];

function addMessage(role, text) {
  const el = document.createElement('div');
  el.classList.add('message', `message--${role}`);
  if (!text) el.classList.add('message--empty');
  el.textContent = text || '...';
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

async function sendMessage(text) {
  history.push({ role: 'user', content: text });
  addMessage('user', text);

  const assistantEl = addMessage('assistant', '');
  let full = '';

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
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) {
        full += delta;
        assistantEl.textContent = full;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }

    assistantEl.textContent = full || '(пустой ответ)';
    history.push({ role: 'assistant', content: full });
  } catch (err) {
    assistantEl.textContent = `Ошибка: ${err.message}`;
    assistantEl.classList.add('message--error');
    if (history[history.length - 1].role === 'user') history.pop();
  }
}

function autoResize() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
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
ensureWelcome();

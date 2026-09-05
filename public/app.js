const messagesRoot = document.getElementById('messages');
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
const newChatBtn = document.getElementById('new-chat');
const summarizeBtn = document.getElementById('summarize');
const tabsListEl = document.getElementById('tabs-list');

const FALLBACK_MODELS = ['deepseek-chat', 'deepseek-reasoner'];

const MESSAGE_OVERHEAD_TOKENS = 4;

const JSON_SYSTEM_PROMPT = 'Выдавай ответ строго в формате JSON.';

const SUMMARY_CHAT_TITLE = 'Подвести итоги';

const SUMMARY_CONTEXT_PROMPT =
  'У тебя есть доступ к содержимому всех открытых чатов. Используй его при ответе на вопрос пользователя.';

const chats = [];
let activeChatId = null;
let chatCounter = 0;
let tokensBurned = 0;

function renderTotal() {
  tokensTotalEl.textContent = String(tokensBurned);
}

function currentModel() {
  return modelSelect.value || FALLBACK_MODELS[0];
}

function collectSettings() {
  const settings = {};
  settings.temperature = Number(tempRange.value);
  let topP = Number(topPRange.value);
  if (topP === 0) topP = 0.01;
  settings.top_p = topP;
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

function resetGenerationDefaults() {
  if (tempRange) {
    tempRange.value = '1';
    tempValue.textContent = '1';
  }
  if (topPRange) {
    topPRange.value = '1';
    topPValue.textContent = '1';
  }
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

function getActiveChat() {
  return chats.find((c) => c.id === activeChatId) || null;
}

function updateSendButton() {
  const chat = getActiveChat();
  sendBtn.disabled = Boolean(chat && chat.busy);
}

function createChat() {
  const chat = {
    id: `chat-${++chatCounter}`,
    title: 'Новый чат',
    history: [],
    prevUsage: null,
    renamed: false,
    isSummary: false,
    busy: false,
    messagesEl: null
  };

  const messagesEl = document.createElement('div');
  messagesEl.className = 'chat__messages';
  messagesEl.hidden = true;
  messagesRoot.appendChild(messagesEl);
  chat.messagesEl = messagesEl;

  chats.push(chat);
  addMessage(chat, 'assistant', 'Привет! Чем могу помочь?');
  return chat;
}

function renderTabs() {
  tabsListEl.innerHTML = '';
  for (const chat of chats) {
    const li = document.createElement('li');
    li.className = 'tabs__item';
    if (chat.id === activeChatId) li.classList.add('tabs__item--active');
    li.dataset.id = chat.id;

    const title = document.createElement('span');
    title.className = 'tabs__title';
    title.textContent = chat.title;
    title.title = chat.title;

    const close = document.createElement('button');
    close.className = 'tabs__close';
    close.type = 'button';
    close.textContent = '×';
    close.title = 'Закрыть чат';

    li.appendChild(title);
    li.appendChild(close);
    tabsListEl.appendChild(li);
  }
}

function activateChat(chat) {
  activeChatId = chat.id;
  for (const c of chats) {
    c.messagesEl.hidden = c.id !== chat.id;
  }
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
  renderTabs();
  updateSendButton();
}

function closeChat(chat) {
  if (!confirm(`Закрыть чат «${chat.title}»? История будет удалена.`)) return;

  const idx = chats.indexOf(chat);
  chat.messagesEl.remove();
  chats.splice(idx, 1);

  if (chats.length === 0) {
    const fresh = createChat();
    activateChat(fresh);
    return;
  }

  if (chat.id === activeChatId) {
    activateChat(chats[idx - 1] || chats[idx]);
  } else {
    renderTabs();
  }
}

function addMessage(chat, role, text) {
  const el = document.createElement('div');
  el.classList.add('message', `message--${role}`);
  if (!text) el.classList.add('message--empty');
  const content = document.createElement('div');
  content.className = 'message__content';
  content.textContent = text || '...';
  el.appendChild(content);
  chat.messagesEl.appendChild(el);
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
  return el;
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

function setBubbleText(chat, el, text) {
  el.classList.remove('message--empty');
  el.querySelector('.message__content').textContent = text;
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
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

function appendReasoning(chat, el, text) {
  const node = createReasoning(el);
  node.querySelector('.message__reasoning-body').textContent += text;
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

function finishReasoning(el) {
  const node = el.querySelector('.message__reasoning');
  if (!node) return;
  const spinner = node.querySelector('.message__reasoning-spinner');
  if (spinner) spinner.remove();
}

function renderJsonEnvelope(chat, el, thinking, response, date, usage) {
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
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

function emptyResponseText(usage, finishReason) {
  if (finishReason === 'length') {
    const limit = maxTokensInput.value || (usage ? usage.completion_tokens : '');
    return `<LLM уперлась в ограничение по токенам: ${limit}>`;
  }
  return '(пустой ответ)';
}

function userMessageTokens(chat, usage) {
  const prevTotal = chat.prevUsage
    ? chat.prevUsage.prompt_tokens + chat.prevUsage.completion_tokens
    : 0;
  return Math.max(0, usage.prompt_tokens - prevTotal - MESSAGE_OVERHEAD_TOKENS);
}

async function streamAssistant(chat, userEl) {
  chat.busy = true;
  updateSendButton();

  const assistantEl = addMessage(chat, 'assistant', '');
  const jsonMode = !modeToggle.checked;
  const responseDate = new Date().toISOString();
  let full = '';
  let thinkingText = '';
  let thinking = false;
  let usage = null;
  let finishReason = null;

  try {
    let outgoingMessages = [...chat.history];
    if (chat.isSummary) {
      const context = buildGlobalContext();
      outgoingMessages.unshift({
        role: 'system',
        content: context
          ? `${SUMMARY_CONTEXT_PROMPT}\n\n${context}`
          : 'Открытые чаты пусты. Отвечай на вопрос пользователя без дополнительного контекста.'
      });
    }
    if (jsonMode) {
      outgoingMessages.unshift({ role: 'system', content: JSON_SYSTEM_PROMPT });
    }

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
      if (chunk.choices?.[0]?.finish_reason) finishReason = chunk.choices[0].finish_reason;
      const delta = chunk.choices?.[0]?.delta;
      if (delta?.reasoning_content) {
        thinking = true;
        thinkingText += delta.reasoning_content;
        appendReasoning(chat, assistantEl, delta.reasoning_content);
        if (jsonMode) renderJsonEnvelope(chat, assistantEl, thinkingText, full, responseDate, usage);
      }
      if (delta?.content) {
        if (thinking) {
          thinking = false;
          finishReasoning(assistantEl);
        }
        full += delta.content;
        if (jsonMode) renderJsonEnvelope(chat, assistantEl, thinkingText, full, responseDate, usage);
        else setBubbleText(chat, assistantEl, full);
      }
    }

    if (thinking) finishReasoning(assistantEl);

    const finalResponse = full || emptyResponseText(usage, finishReason);
    if (jsonMode) {
      renderJsonEnvelope(chat, assistantEl, thinkingText, finalResponse, responseDate, usage);
    } else {
      setBubbleText(chat, assistantEl, finalResponse);
    }
    chat.history.push({ role: 'assistant', content: full });

    if (usage) {
      if (userEl) setBubbleTokens(userEl, userMessageTokens(chat, usage));
      setBubbleTokens(assistantEl, usage.completion_tokens);
      tokensBurned += usage.total_tokens;
      chat.prevUsage = {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens
      };
      renderTotal();
      if (jsonMode) renderJsonEnvelope(chat, assistantEl, thinkingText, finalResponse, responseDate, usage);
    }
  } catch (err) {
    setBubbleText(chat, assistantEl, `Ошибка: ${err.message}`);
    assistantEl.classList.add('message--error');
    if (chat.history[chat.history.length - 1].role === 'user') chat.history.pop();
  } finally {
    chat.busy = false;
    updateSendButton();
  }
}

async function sendMessage(chat, text) {
  chat.history.push({ role: 'user', content: text });
  const userEl = addMessage(chat, 'user', text);

  if (!chat.renamed) {
    chat.renamed = true;
    chat.title = text.replace(/\s+/g, ' ').trim() || 'Новый чат';
    renderTabs();
  }

  await streamAssistant(chat, userEl);
}

function findSummaryChat() {
  return chats.find((c) => c.isSummary) || null;
}

function createSummaryChat() {
  const chat = {
    id: `chat-${++chatCounter}`,
    title: SUMMARY_CHAT_TITLE,
    history: [],
    prevUsage: null,
    renamed: true,
    isSummary: true,
    busy: false,
    messagesEl: null
  };

  const messagesEl = document.createElement('div');
  messagesEl.className = 'chat__messages';
  messagesEl.hidden = true;
  messagesRoot.appendChild(messagesEl);
  chat.messagesEl = messagesEl;

  chats.push(chat);
  addMessage(chat, 'assistant', 'Задайте вопрос с контекстом всех открытых чатов');
  return chat;
}

function buildGlobalContext() {
  const parts = [];
  for (const chat of chats) {
    if (chat.isSummary) continue;
    if (chat.history.length === 0) continue;
    const lines = [`### Чат «${chat.title}»`];
    for (const m of chat.history) {
      const label = m.role === 'user' ? 'Пользователь' : 'Ассистент';
      lines.push(`${label}: ${m.content}`);
    }
    parts.push(lines.join('\n'));
  }
  return parts.join('\n\n');
}

function openSummaryChat() {
  let chat = findSummaryChat();
  if (!chat) {
    chat = createSummaryChat();
  }
  activateChat(chat);
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
  if (!text) return;

  const chat = getActiveChat();
  if (!chat || chat.busy) return;

  input.value = '';
  autoResize();
  input.focus();

  await sendMessage(chat, text);
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener('input', autoResize);

newChatBtn.addEventListener('click', () => {
  const chat = createChat();
  activateChat(chat);
});

summarizeBtn.addEventListener('click', openSummaryChat);

tabsListEl.addEventListener('click', (e) => {
  const item = e.target.closest('.tabs__item');
  if (!item) return;
  const chat = chats.find((c) => c.id === item.dataset.id);
  if (!chat) return;
  if (e.target.closest('.tabs__close')) {
    closeChat(chat);
  } else {
    activateChat(chat);
  }
});

if (tempRange) {
  tempRange.addEventListener('input', () => {
    tempValue.textContent = tempRange.value;
  });
}

if (topPRange) {
  topPRange.addEventListener('input', () => {
    topPValue.textContent = topPRange.value;
  });
}

modelSelect.addEventListener('change', resetGenerationDefaults);

modeToggle.addEventListener('change', () => {
  modeState.textContent = modeToggle.checked ? 'Обычный' : 'JSON';
});

autoResize();
renderTotal();
loadModels();
const initialChat = createChat();
activateChat(initialChat);

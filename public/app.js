const messagesRoot = document.getElementById('messages');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const tokensTotalEl = document.getElementById('tokens-total');
const modelDropdown = document.getElementById('model-dropdown');
const modelButton = document.getElementById('model-button');
const modelDot = document.getElementById('model-dot');
const modelLabel = document.getElementById('model-label');
const modelMenu = document.getElementById('model-menu');
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

const FALLBACK_MODELS = ['deepseek-v4-flash', 'deepseek-v4-pro'];

const TIER_CHEAP_MAX = 1;
const TIER_MEDIUM_MAX = 4;

const MODEL_PRICES = {
  'deepseek-v4-flash': { in: 0.22, out: 0.66 },
  'deepseek-v4-flash-vision-exp': { in: 0.22, out: 0.66 },
  'deepseek-v4-pro': { in: 0.66, out: 1.98 },
  'deepseek-chat': { in: 0.22, out: 0.66 },
  'deepseek-reasoner': { in: 0.22, out: 0.66 },
  'opencode/glm-5.3-flash': { in: 0.15, out: 0.5 },
  'opencode/glm-5.3': { in: 1.4, out: 4.4 },
  'opencode/glm-5.2': { in: 1.4, out: 4.4 },
  'opencode/glm-5.1': { in: 1.4, out: 4.4 },
  'opencode/kimi-k3': { in: 3, out: 15 },
  'opencode/kimi-k2.7-code': { in: 0.95, out: 4 },
  'opencode/kimi-k2.6': { in: 0.95, out: 4 },
  'opencode/longcat-2.0': { in: 0.3, out: 1.2 },
  'opencode/deepseek-v4-pro': { in: 0.66, out: 1.98 },
  'opencode/deepseek-v4-flash': { in: 0.22, out: 0.66 },
  'opencode/deepseek-v4-flash-vision-exp': { in: 0.22, out: 0.66 },
  'opencode/mimo-v2.5': { in: 0.14, out: 0.28 },
  'opencode/mimo-v2.5-pro': { in: 0.435, out: 0.87 },
  'opencode/hy4-preview': { in: 0.834, out: 2.501 },
  'opencode/hy3': { in: 0.14, out: 0.58 },
  'opencode/omen-alpha': { in: 0.2, out: 0.66 },
  'opencode/big-pickle': { in: 0, out: 0 },
  'opencode/deepseek-v4-flash-free': { in: 0, out: 0 },
  'opencode/mimo-v2.5-free': { in: 0, out: 0 },
  'opencode/ling-3.0-flash-fin-free': { in: 0, out: 0 },
  'opencode/nemotron-3-ultra-free': { in: 0, out: 0 },
  'opencode/nemotron-3.5-lightning-free': { in: 0, out: 0 }
};

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
  return selectedModel || FALLBACK_MODELS[0];
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

let selectedModel = null;

function displayName(id) {
  return id.startsWith('opencode/') ? id.slice('opencode/'.length) : id;
}

function modelGroupLabel(ownedBy) {
  if (ownedBy === 'deepseek') return 'DeepSeek';
  if (ownedBy === 'opencode') return 'OpenCode Go';
  return ownedBy || 'Другие';
}

function modelPrice(id) {
  const p = MODEL_PRICES[id];
  return p && typeof p.out === 'number' ? p.out : -1;
}

function messageCost(id, promptTokens, completionTokens) {
  const p = MODEL_PRICES[id];
  if (!p || typeof p.in !== 'number' || typeof p.out !== 'number') return null;
  return (promptTokens * p.in + completionTokens * p.out) / 1e6;
}

function formatCost(cost) {
  if (typeof cost !== 'number') return '–';
  if (cost === 0) return '0';
  let s = cost.toFixed(7).replace(/0+$/, '');
  s = s.replace(/\.$/, '');
  return s;
}

function priceTier(price) {
  if (typeof price !== 'number') return null;
  if (price <= TIER_CHEAP_MAX) return 'cheap';
  if (price <= TIER_MEDIUM_MAX) return 'medium';
  return 'expensive';
}

function priceLabel(price) {
  if (typeof price !== 'number') return '';
  if (price === 0) return 'free';
  return `$${price.toFixed(2)}/1M`;
}

function setSelectedModel(id) {
  selectedModel = id;
  const out = modelPrice(id);
  const tier = priceTier(out);
  modelDot.className = 'dropdown__dot' + (tier ? ` dropdown__dot--${tier}` : '');
  modelLabel.textContent = displayName(id);
  modelLabel.title = `${displayName(id)}${out > 0 ? ` — output: $${out}/1M токенов` : ''}`;
  resetGenerationDefaults();
}

function renderModelMenu(models) {
  modelMenu.innerHTML = '';

  const groups = [];
  const seen = new Map();
  for (const m of models) {
    const label = modelGroupLabel(m.owned_by);
    if (!seen.has(label)) {
      const header = document.createElement('div');
      header.className = 'dropdown__group';
      header.textContent = label;
      seen.set(label, { header, list: [] });
      groups.push(seen.get(label));
    }
    seen.get(label).list.push(m);
  }

  for (const group of groups) {
    group.list.sort((a, b) => modelPrice(b.id) - modelPrice(a.id));
    modelMenu.appendChild(group.header);
    for (const m of group.list) {
      const out = modelPrice(m.id);
      const tier = priceTier(out);

      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'dropdown__item';
      item.dataset.id = m.id;
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', String(m.id === selectedModel));

      const dot = document.createElement('span');
      dot.className = 'dropdown__dot' + (tier ? ` dropdown__dot--${tier}` : '');
      dot.style.opacity = tier ? '1' : '0.4';

      const name = document.createElement('span');
      name.className = 'dropdown__name';
      name.textContent = displayName(m.id);
      name.title = `${displayName(m.id)}${out > 0 ? ` — output: $${out}/1M токенов` : ''}`;

      const priceEl = document.createElement('span');
      priceEl.className = 'dropdown__price';
      priceEl.textContent = priceLabel(out);

      const check = document.createElement('span');
      check.className = 'dropdown__check';
      check.textContent = '✓';

      item.append(dot, name, priceEl, check);
      item.addEventListener('click', () => {
        setSelectedModel(m.id);
        renderModelMenu(models);
        closeModelMenu();
      });
      modelMenu.appendChild(item);
    }
  }

  const legend = document.createElement('div');
  legend.className = 'dropdown__legend';
  legend.innerHTML =
    '<span><span class="dropdown__legend-dot dropdown__legend-dot--cheap"></span>дешёвый</span>' +
    '<span><span class="dropdown__legend-dot dropdown__legend-dot--medium"></span>средний</span>' +
    '<span><span class="dropdown__legend-dot dropdown__legend-dot--expensive"></span>дорогой</span>';
  modelMenu.appendChild(legend);
}

function openModelMenu() {
  modelMenu.hidden = false;
  modelDropdown.dataset.open = 'true';
  modelButton.setAttribute('aria-expanded', 'true');
}

function closeModelMenu() {
  modelMenu.hidden = true;
  modelDropdown.dataset.open = 'false';
  modelButton.setAttribute('aria-expanded', 'false');
}

function loadModels() {
  fetch('/api/models')
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data) => {
      const models = (Array.isArray(data.data) ? data.data : [])
        .filter((m) => m && typeof m.id === 'string')
        .map((m) => ({ id: m.id, owned_by: typeof m.owned_by === 'string' ? m.owned_by : '' }));
      if (models.length === 0) throw new Error('empty model list');
      renderModelMenu(models);
      const preferred = models.some((m) => m.id === FALLBACK_MODELS[0])
        ? FALLBACK_MODELS[0]
        : models[0].id;
      setSelectedModel(preferred);
    })
    .catch(() => {
      const fallback = FALLBACK_MODELS.map((id) => ({ id, owned_by: 'deepseek' }));
      renderModelMenu(fallback);
      setSelectedModel(fallback[0].id);
    });
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

function createMessageEl(role, text) {
  const el = document.createElement('div');
  el.classList.add('message', `message--${role}`);
  if (!text) el.classList.add('message--empty');
  const content = document.createElement('div');
  content.className = 'message__content';
  content.textContent = text || '...';
  el.appendChild(content);
  return el;
}

function addMessage(chat, role, text) {
  const el = createMessageEl(role, text);
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

function buildQaMeta(model, startTime, usage) {
  const time = (Date.now() - startTime) / 1000;
  const input = usage ? usage.prompt_tokens : null;
  const output = usage ? usage.completion_tokens : null;
  const reasoning = usage && usage.completion_tokens_details ? usage.completion_tokens_details.reasoning_tokens : null;
  const cost = usage ? messageCost(model, usage.prompt_tokens, usage.completion_tokens) : null;
  return {
    time: `${time.toFixed(2)}s`,
    input: input === null ? '–' : String(input),
    output: output === null ? '–' : String(output),
    reasoning: reasoning === null ? '–' : String(reasoning),
    cost: cost === null ? '–' : formatCost(cost),
    model
  };
}

function renderQaStats(qaEl, stats) {
  const el = document.createElement('div');
  el.className = 'qa__stats';
  el.textContent =
    `время: ${stats.time}, вход: ${stats.input}, выход: ${stats.output}, ` +
    `рассужд.: ${stats.reasoning}, цена: ${stats.cost}`;
  qaEl.appendChild(el);
  qaEl.parentElement.scrollTop = qaEl.parentElement.scrollHeight;
}

async function streamAssistant(chat, qaEl) {
  chat.busy = true;
  updateSendButton();

  const assistantEl = createMessageEl('assistant', '');
  qaEl.appendChild(assistantEl);
  const model = currentModel();
  const startTime = Date.now();
  const jsonMode = !modeToggle.checked;
  const responseDate = new Date().toISOString();
  let full = '';
  let thinkingText = '';
  let thinking = false;
  let usage = null;
  let finishReason = null;

  try {
    let outgoingMessages = chat.history.map((m) => ({ role: m.role, content: m.content }));
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
        model,
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

    if (usage) {
      tokensBurned += usage.total_tokens;
      renderTotal();
    }
    const meta = buildQaMeta(model, startTime, usage);
    chat.history.push({ role: 'assistant', content: full, meta });
    renderQaStats(qaEl, meta);
  } catch (err) {
    setBubbleText(chat, assistantEl, `Ошибка: ${err.message}`);
    assistantEl.classList.add('message--error');
    if (chat.history[chat.history.length - 1].role === 'user') chat.history.pop();
    renderQaStats(qaEl, buildQaMeta(model, startTime, null));
  } finally {
    chat.busy = false;
    updateSendButton();
  }
}

async function sendMessage(chat, text) {
  chat.history.push({ role: 'user', content: text });

  const qaEl = document.createElement('div');
  qaEl.className = 'qa';
  chat.messagesEl.appendChild(qaEl);
  qaEl.appendChild(createMessageEl('user', text));
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;

  if (!chat.renamed) {
    chat.renamed = true;
    chat.title = text.replace(/\s+/g, ' ').trim() || 'Новый чат';
    renderTabs();
  }

  await streamAssistant(chat, qaEl);
}

function findSummaryChat() {
  return chats.find((c) => c.isSummary) || null;
}

function createSummaryChat() {
  const chat = {
    id: `chat-${++chatCounter}`,
    title: SUMMARY_CHAT_TITLE,
    history: [],
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
      if (m.role === 'assistant' && m.meta) {
        lines.push(
          `[модель: ${m.meta.model} | время: ${m.meta.time}, вход: ${m.meta.input}, ` +
          `выход: ${m.meta.output}, рассужд.: ${m.meta.reasoning}, цена: ${m.meta.cost}]`
        );
      }
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

modelButton.addEventListener('click', () => {
  if (modelMenu.hidden) openModelMenu();
  else closeModelMenu();
});

modelButton.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    e.preventDefault();
    closeModelMenu();
    modelButton.focus();
  }
});

document.addEventListener('click', (e) => {
  if (!modelDropdown.contains(e.target)) closeModelMenu();
});

modeToggle.addEventListener('change', () => {
  modeState.textContent = modeToggle.checked ? 'Обычный' : 'JSON';
});

autoResize();
renderTotal();
loadModels();
const initialChat = createChat();
activateChat(initialChat);

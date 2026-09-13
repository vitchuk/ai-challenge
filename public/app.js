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
const topKInput = document.getElementById('setting-top-k');
const summaryToggle = document.getElementById('setting-summary');
const keepRecentInput = document.getElementById('setting-keep-recent');
const maxTokensInput = document.getElementById('setting-max-tokens');
const stopInput = document.getElementById('setting-stop');
const modeToggle = document.getElementById('setting-mode');
const modeState = document.getElementById('mode-state');
const newChatBtn = document.getElementById('new-chat');
const summarizeBtn = document.getElementById('summarize');
const tabsListEl = document.getElementById('tabs-list');
const chatInfoEl = document.getElementById('chat-info');
const chartCanvas = document.getElementById('context-chart');
const chartEmpty = document.getElementById('chart-empty');
const chartColorToggle = document.getElementById('chart-color-mode');
const chartLegend = document.querySelector('.chartbar__legend');
const commandMenu = document.getElementById('command-menu');
const commandModal = document.getElementById('command-modal');
const modalBody = document.getElementById('modal-body');
const modalResult = document.getElementById('modal-result');
const modalStats = document.getElementById('modal-stats');
const modalCopy = document.getElementById('modal-copy');
const modalCloseBtn = document.getElementById('modal-close-btn');
const modalClose = document.getElementById('modal-close');

// ── Консольное логирование клиента ─────────────────────────────────────────
// Отображает в консоли браузера всё, что клиент отправляет на сервер
// и получает от него (см. console.log). Флаг можно отключить вручную.
const CLIENT_LOG_ENABLED = true;

function logClient(kind, label, payload) {
  if (!CLIENT_LOG_ENABLED) return;
  const time = new Date().toISOString();
  if (kind === 'send') {
    console.groupCollapsed(`[клиент→сервер] ${time} — ${label}`);
  } else {
    console.groupCollapsed(`[клиент←сервер] ${time} — ${label}`);
  }
  console.log(payload);
  console.groupEnd();
}

// ── Команды (автокомплит) ───────────────────────────────────────────────────
const COMMANDS = [
  {
    name: '/optimize-prompt',
    description: 'Сгенерировать оптимизированный промпт'
  }
];

const OPTIMIZE_SYSTEM_TEMPLATE =
  'Действуй как профессиональный промт-инженер. Создай детальный промт ' +
  'для языковой модели по запросу: USER_TEXT. Промпт должен быть на языке ' +
  'запроса, лаконичен и структурирован.';

// ── Настройки / модели ─────────────────────────────────────────────────────
const FALLBACK_MODELS = [
  { id: 'deepseek-v4-flash', owned_by: 'deepseek' },
  { id: 'deepseek-v4-pro', owned_by: 'deepseek' }
];

const TIER_CHEAP_MAX = 1;
const TIER_MEDIUM_MAX = 4;

// Запасная карта контекстов (на случай, если модель не в списке /api/models
// или её лимит не раскрыт апстримом). Значения подтверждены probe-запросами.
const MODEL_CONTEXT = {
  'deepseek-flash': 1048576,
  'deepseek-v4-flash': 1048576,
  'deepseek-v4-flash-vision-exp': 1048576,
  'deepseek-v4-pro': 1048576,
  'deepseek-chat': 1048576,
  'deepseek-reasoner': 1048576,
  'opencode/deepseek-v4-flash': 1048576,
  'opencode/deepseek-v4-flash-vision-exp': 1048576,
  'opencode/deepseek-v4-pro': 1048576,
  'opencode/kimi-k2.6': 262144,
  'opencode/longcat-2.0': 1048580,
  'opencode/hy4-preview': 1048576,
  'opencode/hy3': 262144
};

const JSON_SYSTEM_PROMPT = 'Выдавай ответ строго в формате JSON.';
const SUMMARY_CHAT_TITLE = 'Подвести итоги';

const CHART_COLOR_MODE_KEY = 'pomogator2k:chart-color-mode';
let chartColorMode = false;
try {
  chartColorMode = localStorage.getItem(CHART_COLOR_MODE_KEY) === '1';
} catch {
  chartColorMode = false;
}

const chats = [];
let activeChatId = null;
let chatCounter = 0;
let tokensBurned = 0;
let selectedModel = null;

// ── Базовые UI-утилиты ─────────────────────────────────────────────────────
function renderTotal() {
  tokensTotalEl.textContent = String(tokensBurned);
}

function computeTokensTotal() {
  let sum = 0;
  for (const chat of chats) {
    for (const r of chat.requests || []) {
      sum += (r.prompt_tokens || 0) + (r.completion_tokens || 0);
    }
  }
  tokensBurned = sum;
  renderTotal();
}

function handleRequestLog(chat, record) {
  // Запись о запросе к LLM (основной ответ или скрытая саммаризация).
  if (!chat || !record) return;
  chat.requests = chat.requests || [];
  chat.requests.push(record);
  const total = (record.prompt_tokens || 0) + (record.completion_tokens || 0);
  if (total > 0) {
    tokensBurned += total;
    renderTotal();
  }
  renderChart();
}

function formatTokens(n) {
  if (typeof n !== 'number') return '—';
  return n.toLocaleString('ru-RU');
}

function currentModel() {
  return selectedModel || FALLBACK_MODELS[0].id;
}

function collectSettings() {
  const settings = {};
  settings.temperature = Number(tempRange.value);
  let topP = Number(topPRange.value);
  if (topP === 0) topP = 0.01;
  settings.top_p = topP;
  const topK = parseInt(topKInput.value, 10);
  if (Number.isInteger(topK) && topK > 0) settings.top_k = topK;
  const maxTokens = parseInt(maxTokensInput.value, 10);
  if (Number.isInteger(maxTokens) && maxTokens > 0) settings.max_tokens = maxTokens;
  const stop = stopInput.value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (stop.length > 0) settings.stop = stop;
  if (!modeToggle.checked) settings.response_format = { type: 'json_object' };
  if (summaryToggle) {
    const requests = parseInt(keepRecentInput.value, 10);
    settings.context_summary = {
      enabled: summaryToggle.checked,
      requests_per_summary: Number.isInteger(requests) ? requests : 5
    };
  }
  return settings;
}

function collectSystemPrompt() {
  if (!modeToggle.checked) return JSON_SYSTEM_PROMPT;
  return null;
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

function resetChatPanel() {
  resetGenerationDefaults();
  if (topKInput) topKInput.value = '';
  if (maxTokensInput) maxTokensInput.value = '';
  if (stopInput) stopInput.value = '';
  if (summaryToggle) summaryToggle.checked = false;
  if (keepRecentInput) keepRecentInput.value = '5';
}

function defaultChatSettings() {
  return {
    temperature: 1,
    top_p: 1,
    top_k: null,
    max_tokens: null,
    stop: [],
    response_format: null,
    context_summary: { enabled: false, requests_per_summary: 5 }
  };
}

function normalizeServerSettings(s) {
  const def = defaultChatSettings();
  if (!s || typeof s !== 'object') return def;
  const cs = s.context_summary;
  return {
    temperature: typeof s.temperature === 'number' ? s.temperature : def.temperature,
    top_p: typeof s.top_p === 'number' ? s.top_p : def.top_p,
    top_k: s.top_k || null,
    max_tokens: s.max_tokens || null,
    stop: Array.isArray(s.stop) ? s.stop : [],
    response_format: s.response_format || null,
    context_summary: {
      enabled: Boolean(cs && cs.enabled),
      requests_per_summary:
        cs && Number.isInteger(cs.requests_per_summary)
          ? Math.min(20, Math.max(1, cs.requests_per_summary))
          : 5
    }
  };
}

function applyChatSettings(chat) {
  const s = chat.settings || defaultChatSettings();
  if (tempRange) {
    const t = typeof s.temperature === 'number' ? s.temperature : 1;
    tempRange.value = String(t);
    tempValue.textContent = String(t);
  }
  if (topPRange) {
    // Ползунок не допускает 0; защищаемся от легаси-значений.
    const p = Math.max(0.01, typeof s.top_p === 'number' ? s.top_p : 1);
    topPRange.value = String(p);
    topPValue.textContent = String(p);
  }
  if (topKInput) topKInput.value = s.top_k ? String(s.top_k) : '';
  if (maxTokensInput) maxTokensInput.value = s.max_tokens ? String(s.max_tokens) : '';
  if (stopInput) stopInput.value = Array.isArray(s.stop) ? s.stop.join(', ') : '';
  const cs = s.context_summary || defaultChatSettings().context_summary;
  if (summaryToggle) summaryToggle.checked = Boolean(cs.enabled);
  if (keepRecentInput) keepRecentInput.value = String(cs.requests_per_summary || 5);
}

function isEstablished(chat) {
  return Boolean(chat && chat.history && chat.history.length > 0);
}

function updateSummaryFieldState(locked) {
  if (!keepRecentInput) return;
  keepRecentInput.disabled = locked || !summaryToggle.checked;
}

function updateSettingsLock() {
  const chat = getActiveChat();
  const locked = isEstablished(chat);
  // Привязываемые первым сообщением контролы: модель, temperature,
  // top_p, top_k, саммаризация (тумблер + N).
  for (const el of [modelButton, tempRange, topPRange, topKInput, summaryToggle]) {
    if (el) el.disabled = locked;
  }
  updateSummaryFieldState(locked);
  if (modelDropdown) modelDropdown.dataset.locked = locked ? 'true' : 'false';
  if (locked) closeModelMenu();
}

function formatContext(n) {
  if (typeof n !== 'number' || n <= 0) return '—';
  if (n >= 1000000) return `${Math.round(n / 1000000)}M`;
  return `${Math.round(n / 1000)}K`;
}

function chatTokens(chat) {
  let sum = 0;
  for (const m of chat && chat.history ? chat.history : []) {
    if (m.meta) {
      sum += (m.meta.prompt_tokens || 0) + (m.meta.completion_tokens || 0);
    }
  }
  return sum;
}

function chatCost(chat) {
  let sum = 0;
  let any = false;
  for (const m of chat && chat.history ? chat.history : []) {
    if (m.meta && typeof m.meta.cost_usd === 'number') {
      sum += m.meta.cost_usd;
      any = true;
    }
  }
  return any ? sum : null;
}

function modelContext(modelId) {
  const m = (cachedModels || []).find((x) => x.id === modelId);
  if (m && typeof m.context === 'number') return m.context;
  return MODEL_CONTEXT[modelId] || null;
}

function lastPromptTokens(chat) {
  if (!chat || !chat.history) return null;
  for (let i = chat.history.length - 1; i >= 0; i--) {
    const m = chat.history[i];
    if (m.meta && typeof m.meta.prompt_tokens === 'number') return m.meta.prompt_tokens;
  }
  return null;
}

function renderChatInfo() {
  if (!chatInfoEl) return;
  const chat = getActiveChat();
  const model = (chat && chat.model) || currentModel();
  const s = (chat && chat.settings) || defaultChatSettings();
  const temp = typeof s.temperature === 'number' ? s.temperature : 1;
  const topP = typeof s.top_p === 'number' ? s.top_p : 1;
  const topK = s.top_k != null ? s.top_k : '—';
  const tokens = chatTokens(chat);
  const ctxValue = modelContext(model);
  const ctx = formatContext(ctxValue);
  chatInfoEl.textContent =
    `${displayName(model)} · temperature ${temp} · top_p ${topP} · top_k ${topK} · токенов: ${tokens} — ${ctx}`;

  // Индикатор использования контекста (по последнему запросу): <80% обычно,
  // ≥80% жёлтый, ≥95% красный.
  const lastPrompt = lastPromptTokens(chat);
  if (lastPrompt != null && typeof ctxValue === 'number' && ctxValue > 0) {
    const pct = Math.round((lastPrompt / ctxValue) * 100);
    const span = document.createElement('span');
    span.className = 'chat-info__ctx';
    if (pct >= 95) span.classList.add('chat-info__ctx--crit');
    else if (pct >= 80) span.classList.add('chat-info__ctx--warn');
    span.textContent = ` · контекст: ${pct}%`;
    chatInfoEl.appendChild(span);
  }

  // Общая стоимость разговора в этом чате (сумма cost_usd по ответам).
  const cost = chatCost(chat);
  const costSpan = document.createElement('span');
  costSpan.className = 'chat-info__cost';
  costSpan.textContent =
    ` · Общая стоимость: ${cost === null ? '—' : '$' + formatCost(cost)}`;
  chatInfoEl.appendChild(costSpan);
}

function displayName(id) {
  return id.startsWith('opencode/') ? id.slice('opencode/'.length) : id;
}

function modelGroupLabel(ownedBy) {
  if (ownedBy === 'deepseek') return 'DeepSeek';
  if (ownedBy === 'opencode') return 'OpenCode Go';
  return ownedBy || 'Другие';
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

function formatCost(cost) {
  if (typeof cost !== 'number') return '–';
  if (cost === 0) return '0';
  let s = cost.toFixed(7).replace(/0+$/, '');
  s = s.replace(/\.$/, '');
  return s;
}

function formatMetaLine(meta) {
  const time = meta && typeof meta.time_s === 'number' ? `${meta.time_s.toFixed(2)}s` : '–';
  const input = meta && meta.prompt_tokens != null ? String(meta.prompt_tokens) : '–';
  const output = meta && meta.completion_tokens != null ? String(meta.completion_tokens) : '–';
  const reasoning =
    meta && meta.reasoning_tokens != null ? String(meta.reasoning_tokens) : '–';
  const cost = meta ? formatCost(meta.cost_usd) : '–';
  const model = meta && meta.model ? meta.model : currentModel();
  return { time, input, output, reasoning, cost, model };
}

function setSelectedModel(id, opts) {
  const resetParams = !opts || opts.resetParams !== false;
  selectedModel = id;
  const out = modelPrice(id);
  const tier = priceTier(out);
  modelDot.className = 'dropdown__dot' + (tier ? ` dropdown__dot--${tier}` : '');
  modelLabel.textContent = displayName(id);
  modelLabel.title = `${displayName(id)}${out > 0 ? ` — output: $${out}/1M токенов` : ''}`;
  if (resetParams) resetGenerationDefaults();
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
    group.list.sort((a, b) => (b.price ?? -1) - (a.price ?? -1));
    modelMenu.appendChild(group.header);
    for (const m of group.list) {
      const out = typeof m.price === 'number' ? m.price : -1;
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
        // запоминаем выбранную модель в активном чате
        const chat = getActiveChat();
        if (chat) chat.model = m.id;
        renderChatInfo();
        renderChart();
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

function modelPrice(id) {
  // Цена берётся из последнего списка моделей, полученного с сервера.
  const m = (cachedModels || []).find((x) => x.id === id);
  return m && typeof m.price === 'number' ? m.price : -1;
}

let cachedModels = null;

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
        .map((m) => ({
          id: m.id,
          owned_by: typeof m.owned_by === 'string' ? m.owned_by : '',
          price: typeof m.price === 'number' ? m.price : -1,
          context: typeof m.context === 'number' ? m.context : null
        }));
      if (models.length === 0) throw new Error('empty model list');
      cachedModels = models;
      renderModelMenu(models);
      if (selectedModel === null) {
        const preferred = models.some((m) => m.id === FALLBACK_MODELS[0].id)
          ? FALLBACK_MODELS[0].id
          : models[0].id;
        setSelectedModel(preferred);
      }
      renderChatInfo();
    })
    .catch(() => {
      cachedModels = FALLBACK_MODELS.map((m) => ({ ...m, price: -1, context: null }));
      renderModelMenu(FALLBACK_MODELS);
      if (selectedModel === null) setSelectedModel(FALLBACK_MODELS[0].id);
      renderChatInfo();
    });
}

// ── HTTP API клиента ────────────────────────────────────────────────────────
async function apiCreateSession(body) {
  logClient('send', 'POST /api/sessions', body);
  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Ошибка ${res.status}`);
  }
  const data = await res.json();
  logClient('receive', 'POST /api/sessions → 201', data);
  return data;
}

async function apiDeleteSession(sid) {
  logClient('send', `DELETE /api/sessions/${sid}`);
  const res = await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 404) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Ошибка ${res.status}`);
  }
  logClient('receive', `DELETE /api/sessions/${sid} → ${res.status}`);
}

async function apiActivateSession(sid) {
  logClient('send', `POST /api/sessions/${sid}/activate`);
  const res = await fetch(`/api/sessions/${sid}/activate`, { method: 'POST' });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Ошибка ${res.status}`);
  }
  logClient('receive', `POST /api/sessions/${sid}/activate → ${res.status}`);
}

// ── Чат-сессии ─────────────────────────────────────────────────────────────
function getActiveChat() {
  return chats.find((c) => c.id === activeChatId) || null;
}

function updateSendButton() {
  const chat = getActiveChat();
  sendBtn.disabled = Boolean(chat && chat.busy);
}

async function createChat(kind = 'chat', title = 'Новый чат') {
  const chat = {
    id: `chat-${++chatCounter}`,
    sid: null,
    title,
    history: [],
    requests: [],
    renamed: false,
    kind,
    busy: false,
    model: null,
    settings: defaultChatSettings(),
    messagesEl: null
  };

  // Новый чат: параметры и модель сбрасываются к дефолту (deepseek-v4-flash).
  setSelectedModel(FALLBACK_MODELS[0].id);
  resetChatPanel();
  if (cachedModels) renderModelMenu(cachedModels);
  updateSettingsLock();
  renderChatInfo();

  const messagesEl = document.createElement('div');
  messagesEl.className = 'chat__messages';
  messagesEl.hidden = true;
  messagesRoot.appendChild(messagesEl);
  chat.messagesEl = messagesEl;

  chats.push(chat);

  // Создаём серверную сессию (для summary-чата — с типом summary).
  try {
    const body = { kind };
    if (kind === 'summary') {
      chat.renamed = true;
    }
    const created = await apiCreateSession(body);
    chat.sid = created.id;
  } catch (err) {
    addMessage(chat, 'assistant', `Ошибка создания чата: ${err.message}`);
  }

  if (kind === 'chat') {
    addMessage(chat, 'assistant', 'Привет! Чем могу помочь?');
  } else {
    addMessage(chat, 'assistant', 'Задайте вопрос с контекстом всех открытых чатов');
  }
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
  // Восстанавливаем модель вкладки (без сброса параметров генерации).
  setSelectedModel(chat.model || FALLBACK_MODELS[0].id, { resetParams: false });
  if (cachedModels) renderModelMenu(cachedModels);
  // Восстанавливаем инкапсулированные настройки чата в панель.
  applyChatSettings(chat);
  updateSettingsLock();
  renderChatInfo();
  renderChart();
  // Сообщаем серверу, какая вкладка открыта (для восстановления после рестарта).
  if (chat.sid) {
    apiActivateSession(chat.sid).catch((err) => {
      logClient('receive', 'POST activate — ошибка', err.message);
    });
  }
}

async function closeChat(chat) {
  if (!confirm(`Закрыть чат «${chat.title}»? История будет удалена.`)) return;

  const idx = chats.indexOf(chat);
  chat.messagesEl.remove();
  chats.splice(idx, 1);

  // Удаляем серверную сессию (освобождаем память сервера).
  if (chat.sid) {
    apiDeleteSession(chat.sid).catch(() => {});
  }

  if (chats.length === 0) {
    const fresh = await createChat();
    activateChat(fresh);
    return;
  }

  if (chat.id === activeChatId) {
    activateChat(chats[idx - 1] || chats[idx]);
  } else {
    renderTabs();
  }
}

// ── Рендер сообщений ───────────────────────────────────────────────────────
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
      if (!data) continue;
      try {
        yield JSON.parse(data);
      } catch {
        // ignore malformed frames
      }
    }
  }
}

function setBubbleText(chat, el, text) {
  el.classList.remove('message--empty');
  el.querySelector('.message__content').textContent = text;
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

function renderContextLimit(chat, el, text) {
  // Спец-блок при достижении лимита контекста: сообщение + действия.
  el.classList.remove('message--empty');
  el.classList.add('message--context-limit');
  const content = el.querySelector('.message__content');
  content.textContent = '';

  const icon = document.createElement('div');
  icon.className = 'message__limit-icon';
  icon.textContent = '⚠';
  const msg = document.createElement('div');
  msg.className = 'message__limit-text';
  msg.textContent = text;
  const actions = document.createElement('div');
  actions.className = 'message__limit-actions';

  const newBtn = document.createElement('button');
  newBtn.type = 'button';
  newBtn.className = 'message__limit-btn';
  newBtn.textContent = 'Новый чат';
  newBtn.addEventListener('click', async () => {
    const fresh = await createChat();
    activateChat(fresh);
  });

  const sumBtn = document.createElement('button');
  sumBtn.type = 'button';
  sumBtn.className = 'message__limit-btn';
  sumBtn.textContent = 'Подвести итоги';
  sumBtn.addEventListener('click', () => openSummaryChat());

  actions.append(newBtn, sumBtn);
  content.append(icon, msg, actions);
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

function showWaiter(containerEl) {
  containerEl.textContent = '';
  if (!containerEl.querySelector('.message__waiter')) {
    const w = document.createElement('span');
    w.className = 'message__waiter';
    containerEl.appendChild(w);
  }
}

function removeWaiter(containerEl) {
  const w = containerEl.querySelector('.message__waiter');
  if (w) w.remove();
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

function setReasoningText(el, text) {
  const node = createReasoning(el);
  node.querySelector('.message__reasoning-body').textContent = text;
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
      tokens: usage && usage.completion_tokens != null ? usage.completion_tokens : 0
    },
    null,
    2
  );
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

function emptyResponseText(meta, finishReason) {
  if (finishReason === 'length') {
    const limit =
      maxTokensInput.value || (meta && meta.completion_tokens != null ? meta.completion_tokens : '');
    return `<LLM уперлась в ограничение по токенам: ${limit}>`;
  }
  return '(пустой ответ)';
}

function renderQaStats(qaEl, line) {
  const el = document.createElement('div');
  el.className = 'qa__stats';
  el.textContent =
    `время: ${line.time}, вход (предыдуший + текущий): ${line.input}, выход: ${line.output}, ` +
    `рассужд.: ${line.reasoning}, цена: $${line.cost}`;
  qaEl.appendChild(el);
  qaEl.parentElement.scrollTop = qaEl.parentElement.scrollHeight;
}

// ── Стрим ответа ассистента (серверные сессии) ─────────────────────────────
async function streamAssistant(chat, qaEl) {
  chat.busy = true;
  updateSendButton();

  const assistantEl = createMessageEl('assistant', '');
  qaEl.appendChild(assistantEl);
  const assistantContent = assistantEl.querySelector('.message__content');
  showWaiter(assistantContent);
  const jsonMode = !modeToggle.checked;
  const responseDate = new Date().toISOString();
  let full = '';
  let thinkingText = '';
  let usage = null;
  let finishReason = null;
  let finalMeta = null;

  const payload = {
    content: chat.history[chat.history.length - 1].content,
    model: currentModel(),
    settings: collectSettings(),
    system_prompt: collectSystemPrompt()
  };

  try {
    logClient('send', `POST /api/sessions/${chat.sid}/messages`, payload);
    const res = await fetch(`/api/sessions/${chat.sid}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }

    for await (const chunk of parseSSE(res)) {
      logClient('receive', `POST /api/sessions/${chat.sid}/messages — событие`, chunk);
      if (chunk.type === 'reasoning_start') {
        createReasoning(assistantEl);
        showWaiter(assistantContent);
      } else if (chunk.type === 'reasoning_end') {
        thinkingText = chunk.content || '';
        if (thinkingText) setReasoningText(assistantEl, thinkingText);
        finishReasoning(assistantEl);
        showWaiter(assistantContent);
      } else if (chunk.type === 'done') {
        finalMeta = chunk.meta || null;
        full = chunk.content || '';
        if (chunk.meta && chunk.meta.completion_tokens != null) {
          usage = { completion_tokens: chunk.meta.completion_tokens };
        }
        if (chunk.meta && chunk.meta.finish_reason) {
          finishReason = chunk.meta.finish_reason;
        }
      } else if (chunk.type === 'request_log') {
        handleRequestLog(chat, chunk.record);
      } else if (chunk.type === 'error') {
        const err = new Error(chunk.error || 'Неизвестная ошибка сервера');
        err.code = chunk.code || null;
        throw err;
      }
    }

    removeWaiter(assistantContent);

    const finalResponse = full || emptyResponseText(finalMeta, finishReason);
    if (jsonMode) {
      renderJsonEnvelope(chat, assistantEl, thinkingText, finalResponse, responseDate, usage);
    } else {
      setBubbleText(chat, assistantEl, finalResponse);
    }

    chat.history.push({
      role: 'assistant',
      content: full,
      meta: finalMeta
    });
    renderQaStats(qaEl, formatMetaLine(finalMeta));
  } catch (err) {
    removeWaiter(assistantContent);
    if (err.code === 'context_length_exceeded') {
      renderContextLimit(chat, assistantEl, err.message);
    } else {
      setBubbleText(chat, assistantEl, `Ошибка: ${err.message}`);
      assistantEl.classList.add('message--error');
    }
    if (chat.history[chat.history.length - 1].role === 'user') chat.history.pop();
    renderQaStats(qaEl, formatMetaLine(null));
  } finally {
    chat.busy = false;
    updateSendButton();
    updateSettingsLock();
    renderChatInfo();
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

// ── «Подвести итоги» ───────────────────────────────────────────────────────
let summaryChatId = null;

async function openSummaryChat() {
  let chat = chats.find((c) => c.kind === 'summary');
  if (!chat) {
    chat = await createChat('summary', SUMMARY_CHAT_TITLE);
  }
  activateChat(chat);
}

// ── Команда /optimize-prompt (модальное окно) ──────────────────────────────
let optimizeSessionId = null;
let optimizeAbortController = null;

function showCommandModal() {
  modalResult.textContent = '';
  modalResult.classList.add('modal__result--empty');
  modalStats.textContent = 'Генерация…';
  commandModal.hidden = false;
}

function hideCommandModal() {
  commandModal.hidden = true;
  // При закрытии удаляем временную сессию (и с сервера, и из клиента).
  if (optimizeSessionId) {
    apiDeleteSession(optimizeSessionId).catch(() => {});
    optimizeSessionId = null;
  }
  if (optimizeAbortController) {
    optimizeAbortController.abort();
    optimizeAbortController = null;
  }
}

async function copyToClipboard(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
  } catch (err) {
    console.error('Не удалось скопировать в буфер обмена:', err);
  }
}

async function runOptimizePrompt(userText) {
  // Изолированный чат: новый сеанс, агент из исходного чата (модель — выбранная
  // в текущем чате), настройки temperature 0.1 / top_p 0.01.
  const systemPrompt = OPTIMIZE_SYSTEM_TEMPLATE.replace('USER_TEXT', userText);
  const created = await apiCreateSession({
    kind: 'ephemeral',
    model: currentModel(),
    settings: { temperature: 0.1, top_p: 0.01 },
    system_prompt: systemPrompt
  });
  optimizeSessionId = created.id;
  showCommandModal();
  modalResult.classList.add('modal__result--empty');
  showWaiter(modalResult);
  modalStats.textContent = '';

  optimizeAbortController = new AbortController();

  const payload = { content: userText };
  try {
    logClient('send', `POST /api/sessions/${optimizeSessionId}/messages`, payload);
    const res = await fetch(`/api/sessions/${optimizeSessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: optimizeAbortController.signal
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }

    let full = '';
    let thinkingText = '';
    let thinking = false;
    let finalMeta = null;

    modalResult.classList.remove('modal__result--empty');
    for await (const chunk of parseSSE(res)) {
      logClient('receive', `POST /api/sessions/${optimizeSessionId}/messages — событие`, chunk);
      if (chunk.type === 'reasoning_start') {
        thinking = true;
      } else if (chunk.type === 'reasoning_end') {
        thinkingText = chunk.content || '';
      } else if (chunk.type === 'request_log') {
        const record = chunk.record || {};
        const total = (record.prompt_tokens || 0) + (record.completion_tokens || 0);
        if (total > 0) {
          tokensBurned += total;
          renderTotal();
        }
      } else if (chunk.type === 'done') {
        finalMeta = chunk.meta || null;
        full = chunk.content || '';
      } else if (chunk.type === 'error') {
        throw new Error(chunk.error || 'Неизвестная ошибка сервера');
      }
    }

    removeWaiter(modalResult);

    if (full) {
      modalResult.textContent = full;
    } else {
      modalResult.textContent = thinking ? thinkingText || '(пустой ответ)' : '(пустой ответ)';
    }

    if (finalMeta) {
      const line = formatMetaLine(finalMeta);
      modalStats.textContent =
        `время: ${line.time}, вход: ${line.input}, выход: ${line.output}, ` +
        `рассужд.: ${line.reasoning}, цена: ${line.cost}`;
    } else {
      modalStats.textContent = '';
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    removeWaiter(modalResult);
    modalResult.textContent = `Ошибка: ${err.message}`;
    modalResult.classList.add('modal__error');
    modalStats.textContent = '';
  }
}

// ── Автокомплит команд ─────────────────────────────────────────────────────
let commandMenuActive = -1;
let commandMenuItems = [];

function currentCommandText() {
  const value = input.value;
  if (!value.startsWith('/')) return '';
  return value.slice(0, value.indexOf(' ') === -1 ? value.length : value.indexOf(' '));
}

function updateCommandMenu() {
  const prefix = currentCommandText();
  if (!prefix) {
    hideCommandMenu();
    return;
  }
  const items = COMMANDS.filter((c) => c.name.startsWith(prefix) || prefix.startsWith(c.name));
  if (items.length === 0) {
    hideCommandMenu();
    return;
  }
  commandMenu.innerHTML = '';
  commandMenuItems = items;
  commandMenuActive = 0;
  for (let i = 0; i < items.length; i++) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'command-menu__item' + (i === 0 ? ' command-menu__item--active' : '');
    const name = document.createElement('span');
    name.className = 'command-menu__name';
    name.textContent = items[i].name;
    const desc = document.createElement('span');
    desc.className = 'command-menu__desc';
    desc.textContent = items[i].description;
    btn.append(name, desc);
    btn.addEventListener('click', () => {
      insertCommand(items[i].name);
    });
    commandMenu.appendChild(btn);
  }
  commandMenu.hidden = false;
}

function hideCommandMenu() {
  commandMenu.hidden = true;
  commandMenuItems = [];
  commandMenuActive = -1;
}

function insertCommand(commandName) {
  // Вставка команды в начало строки поля ввода + пробел.
  const rest = input.value.slice(currentCommandText().length);
  const cleanedRest = rest.startsWith(' ') ? rest.slice(1) : rest;
  input.value = `${commandName} ${cleanedRest}`;
  hideCommandMenu();
  input.focus();
}

// ── Авторесайз поля ввода ─────────────────────────────────────────────────
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

// ── График размера контекста (Chart.js) ────────────────────────────────────
let contextChart = null;

const contextLimitPlugin = {
  id: 'contextLimit',
  afterDatasetsDraw(chart) {
    const limit = chart.$contextLimit;
    if (!limit || limit <= 0) return;
    const area = chart.chartArea;
    const yScale = chart.scales.y;
    let y = yScale.getPixelForValue(limit);
    let offScale = false;
    if (y < area.top) {
      y = area.top + 1;
      offScale = true;
    }
    if (y > area.bottom) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(232, 163, 61, 0.65)';
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(area.left, y);
    ctx.lineTo(area.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(232, 163, 61, 0.9)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    const label = `лимит ${formatContext(limit)}${offScale ? ' (выше)' : ''}`;
    ctx.fillText(label, area.right - 4, y - 2);
    ctx.restore();
  }
};

function chartAvailable() {
  return typeof window.Chart !== 'undefined';
}

function ensureChart() {
  if (!chartAvailable() || !chartCanvas) return null;
  if (contextChart) return contextChart;
  contextChart = new window.Chart(chartCanvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Вход',
          data: [],
          backgroundColor: '#4f8cff',
          borderColor: '#4f8cff',
          borderWidth: 0,
          barPercentage: 1,
          categoryPercentage: 1,
          stack: 'tokens'
        },
        {
          label: 'Выход',
          data: [],
          backgroundColor: '#4caf7d',
          borderColor: '#4caf7d',
          borderWidth: 0,
          barPercentage: 1,
          categoryPercentage: 1,
          stack: 'tokens'
        },
        {
          label: 'Саммаризация',
          data: [],
          backgroundColor: '#e8a33d',
          borderColor: '#e8a33d',
          borderWidth: 0,
          barPercentage: 1,
          categoryPercentage: 1,
          stack: 'tokens'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => item.parsed.y != null,
          callbacks: {
            title: (items) => (items.length ? `Сообщение №${items[0].dataIndex + 1}` : ''),
            label: (item) => {
              const seg = (item.chart.$turns || [])[item.dataIndex];
              if (!seg) return '';
              if (!item.chart.$colorMode) {
                return `всего: ${formatTokens(seg.total)} ток.`;
              }
              if (item.datasetIndex === 0) {
                return `вход: ${formatTokens(seg.input)} ток.`;
              }
              if (item.datasetIndex === 1) {
                const reasoning =
                  seg.reasoning > 0 ? ` (рассуждение: ${formatTokens(seg.reasoning)})` : '';
                return `выход: ${formatTokens(seg.output)} ток.${reasoning}`;
              }
              const reasoning =
                seg.summaryReasoning > 0
                  ? ` (рассуждение: ${formatTokens(seg.summaryReasoning)})`
                  : '';
              return `саммаризация: ${formatTokens(seg.summary)} ток.${reasoning}`;
            },
            footer: (items) => {
              if (!items[0].chart.$colorMode) return '';
              const seg = (items[0].chart.$turns || [])[items[0].dataIndex];
              return seg ? `всего: ${formatTokens(seg.total)} ток.` : '';
            }
          }
        }
      },
      scales: {
        x: {
          stacked: true,
          ticks: { color: '#8a919e', maxTicksLimit: 12, maxRotation: 0, font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { color: '#8a919e', maxTicksLimit: 4, font: { size: 10 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    },
    plugins: [contextLimitPlugin]
  });
  return contextChart;
}

function buildTurns(records) {
  // Группирует записи о запросах в «ходы» — по одному на сообщение
  // пользователя. Саммаризационные запросы прикрепляются к следующему
  // основному; осиротевшая саммаризация (основной запрос упал) образует
  // ход без основного.
  const turns = [];
  let pending = [];
  for (const record of records) {
    if (record.kind === 'summary') {
      pending.push(record);
    } else {
      turns.push({ main: record, summaries: pending });
      pending = [];
    }
  }
  if (pending.length > 0) {
    turns.push({ main: null, summaries: pending });
  }
  return turns;
}

function turnSegments(turn) {
  // Разбивка расхода токенов хода на сегменты столбика.
  const main = turn.main;
  const input = main && typeof main.prompt_tokens === 'number' ? main.prompt_tokens : 0;
  const output = main && typeof main.completion_tokens === 'number' ? main.completion_tokens : 0;
  const reasoning = main && typeof main.reasoning_tokens === 'number' ? main.reasoning_tokens : 0;
  let summary = 0;
  let summaryReasoning = 0;
  for (const r of turn.summaries) {
    summary += (r.prompt_tokens || 0) + (r.completion_tokens || 0);
    summaryReasoning += r.reasoning_tokens || 0;
  }
  return {
    input,
    output,
    reasoning,
    summary,
    summaryReasoning,
    total: input + output + summary
  };
}

function renderChart() {
  const chat = getActiveChat();
  const records = (chat && chat.requests) || [];

  if (!chartAvailable() || !chartCanvas) {
    if (chartEmpty) {
      chartEmpty.hidden = false;
      chartEmpty.textContent = 'График недоступен (нет Chart.js)';
    }
    return;
  }
  if (chartEmpty) chartEmpty.hidden = true;

  const chart = ensureChart();
  if (!chart) return;

  const turns = buildTurns(records);
  const segments = turns.map(turnSegments);
  chart.data.labels = turns.map((_, i) => String(i + 1));
  if (chartColorMode) {
    chart.data.datasets[0].label = 'Вход';
    chart.data.datasets[0].data = segments.map((s) => (s.input > 0 ? s.input : null));
    chart.data.datasets[1].data = segments.map((s) => (s.output > 0 ? s.output : null));
    chart.data.datasets[2].data = segments.map((s) => (s.summary > 0 ? s.summary : null));
  } else {
    // Моно-режим: один столбик — только суммарные токены сообщения.
    chart.data.datasets[0].label = 'Всего';
    chart.data.datasets[0].data = segments.map((s) => (s.total > 0 ? s.total : null));
    chart.data.datasets[1].data = segments.map(() => null);
    chart.data.datasets[2].data = segments.map(() => null);
  }
  if (chartLegend) chartLegend.hidden = !chartColorMode;
  const maxTotal = segments.reduce((m, s) => Math.max(m, s.total), 0);
  chart.options.scales.y.suggestedMax = maxTotal > 0 ? Math.ceil(maxTotal * 1.15) : 10;
  chart.$contextLimit = chat && chat.model ? modelContext(chat.model) : null;
  chart.$turns = segments;
  chart.$colorMode = chartColorMode;
  chart.update();
}

// ── Обработчики событий ────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  const chat = getActiveChat();
  if (!chat || chat.busy) return;

  // Команда /optimize-prompt
  if (text === '/optimize-prompt' || text.startsWith('/optimize-prompt ')) {
    const userText = text.slice('/optimize-prompt'.length).trim();
    input.value = '';
    autoResize();
    if (!userText) {
      addMessage(chat, 'assistant', 'Введите текст запроса после команды /optimize-prompt');
      return;
    }
    try {
      await runOptimizePrompt(userText);
    } catch (err) {
      addMessage(chat, 'assistant', `Ошибка: ${err.message}`);
    }
    return;
  }

  input.value = '';
  autoResize();
  input.focus();

  await sendMessage(chat, text);
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
    return;
  }

  if (!commandMenu.hidden) {
    if (e.key === 'Tab') {
      e.preventDefault();
      const item = commandMenuItems[commandMenuActive];
      if (item) insertCommand(item.name);
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      commandMenuActive = (commandMenuActive + 1) % commandMenuItems.length;
      renderCommandMenuActive();
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      commandMenuActive =
        (commandMenuActive - 1 + commandMenuItems.length) % commandMenuItems.length;
      renderCommandMenuActive();
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      hideCommandMenu();
      return;
    }
  }
});

function renderCommandMenuActive() {
  const items = commandMenu.querySelectorAll('.command-menu__item');
  items.forEach((el, i) => {
    el.classList.toggle('command-menu__item--active', i === commandMenuActive);
  });
}

input.addEventListener('input', () => {
  autoResize();
  updateCommandMenu();
});

input.addEventListener('focus', updateCommandMenu);
input.addEventListener('blur', () => setTimeout(hideCommandMenu, 120));

newChatBtn.addEventListener('click', async () => {
  const chat = await createChat();
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

function syncActiveChatSettings() {
  const chat = getActiveChat();
  if (chat) chat.settings = collectSettings();
  renderChatInfo();
}

if (tempRange) {
  tempRange.addEventListener('input', () => {
    tempValue.textContent = tempRange.value;
    syncActiveChatSettings();
  });
}

if (topPRange) {
  topPRange.addEventListener('input', () => {
    topPValue.textContent = topPRange.value;
    syncActiveChatSettings();
  });
}

if (topKInput) {
  topKInput.addEventListener('input', syncActiveChatSettings);
}

if (maxTokensInput) {
  maxTokensInput.addEventListener('input', syncActiveChatSettings);
}

if (stopInput) {
  stopInput.addEventListener('input', syncActiveChatSettings);
}

if (modeToggle) {
  modeToggle.addEventListener('change', syncActiveChatSettings);
}

if (summaryToggle) {
  summaryToggle.addEventListener('change', () => {
    updateSummaryFieldState(isEstablished(getActiveChat()));
    syncActiveChatSettings();
  });
}

if (keepRecentInput) {
  keepRecentInput.addEventListener('input', syncActiveChatSettings);
}

if (chartColorToggle) {
  chartColorToggle.checked = chartColorMode;
  chartColorToggle.addEventListener('change', () => {
    chartColorMode = chartColorToggle.checked;
    try {
      localStorage.setItem(CHART_COLOR_MODE_KEY, chartColorMode ? '1' : '0');
    } catch {
      /* localStorage может быть недоступен */
    }
    if (chartLegend) chartLegend.hidden = !chartColorMode;
    renderChart();
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

// ── Модальное окно ─────────────────────────────────────────────────────────
modalCopy.addEventListener('click', async () => {
  await copyToClipboard(modalResult.textContent || '');
  const original = modalCopy.textContent;
  modalCopy.textContent = 'Скопировано ✓';
  setTimeout(() => {
    modalCopy.textContent = original;
  }, 1200);
});

function closeModalFromEvent() {
  hideCommandModal();
}

modalClose.addEventListener('click', closeModalFromEvent);
modalCloseBtn.addEventListener('click', closeModalFromEvent);
commandModal.addEventListener('click', (e) => {
  if (e.target === commandModal) closeModalFromEvent();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !commandModal.hidden) {
    hideCommandModal();
  }
});

// ── Инициализация ──────────────────────────────────────────────────────────
autoResize();
renderTotal();
renderChart();
loadModels();

async function restoreSessions() {
  logClient('send', 'GET /api/sessions');
  const res = await fetch('/api/sessions');
  if (!res.ok) {
    logClient('receive', 'GET /api/sessions → ' + res.status);
    return { sessions: [], activeId: null };
  }
  const data = await res.json();
  const sessions = Array.isArray(data.data) ? data.data : [];
  const activeId = typeof data.active_id === 'string' ? data.active_id : null;
  logClient('receive', 'GET /api/sessions → ' + sessions.length + ' сессий', data);
  return { sessions, activeId };
}

function renderRestoredHistory(chat, session) {
  const history = Array.isArray(session.history) ? session.history : [];
  chat.history = history.map((m) => ({
    role: m.role,
    content: m.content,
    meta: m.meta || null
  }));

  if (history.length === 0) {
    addMessage(chat, 'assistant', chat.kind === 'summary'
      ? 'Задайте вопрос с контекстом всех открытых чатов'
      : 'Привет! Чем могу помочь?');
    return;
  }

  let qaEl = null;
  for (const m of history) {
    if (m.role === 'user' || m.role === 'system') {
      // system-запись — первое сообщение чата (системный промпт): рендерим как вопрос
      qaEl = document.createElement('div');
      qaEl.className = 'qa';
      chat.messagesEl.appendChild(qaEl);
      qaEl.appendChild(createMessageEl('user', m.content));
    } else {
      const el = createMessageEl('assistant', m.content);
      (qaEl || chat.messagesEl).appendChild(el);
      if (m.meta) renderQaStats(qaEl || chat.messagesEl, formatMetaLine(m.meta));
      qaEl = null;
    }
  }
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

function sessionTitle(session) {
  if (session.kind === 'summary') return SUMMARY_CHAT_TITLE;
  const firstMsg = (session.history || []).find((m) => m.role === 'user' || m.role === 'system');
  return firstMsg ? firstMsg.content.replace(/\s+/g, ' ').trim() : 'Новый чат';
}

(async () => {
  let restoredSessions = [];
  let activeId = null;
  try {
    const restored = await restoreSessions();
    restoredSessions = restored.sessions;
    activeId = restored.activeId;
  } catch (err) {
    logClient('receive', 'GET /api/sessions — ошибка', err.message);
  }

  if (restoredSessions.length === 0) {
    const initialChat = await createChat();
    activateChat(initialChat);
    return;
  }

  // Восстанавливаем все чаты как вкладки.
  let mostRecent = null;
  let mostRecentTs = -Infinity;
  for (const session of restoredSessions) {
    const chat = {
      id: `chat-${++chatCounter}`,
      sid: session.id,
      title: sessionTitle(session),
      history: [],
      requests: Array.isArray(session.requests)
        ? session.requests.map((r) => ({ ...r }))
        : [],
      renamed: Boolean(session.history && session.history.length > 0),
      kind: session.kind === 'summary' ? 'summary' : 'chat',
      busy: false,
      model: session.model || null,
      settings: normalizeServerSettings(session.settings),
      messagesEl: null
    };
    const messagesEl = document.createElement('div');
    messagesEl.className = 'chat__messages';
    messagesEl.hidden = true;
    messagesRoot.appendChild(messagesEl);
    chat.messagesEl = messagesEl;
    chats.push(chat);
    renderRestoredHistory(chat, session);
    const ts = typeof session.last_active === 'number' ? session.last_active : 0;
    if (ts > mostRecentTs) {
      mostRecentTs = ts;
      mostRecent = chat;
    }
  }
  // Активируем вкладку, открытую у пользователя; иначе — самую свежую.
  computeTokensTotal();
  const activeChat = (activeId && chats.find((c) => c.sid === activeId)) || mostRecent || chats[0];
  activateChat(activeChat);
})();

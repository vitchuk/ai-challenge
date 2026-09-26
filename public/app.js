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
const strategyRadios = Array.from(
  document.querySelectorAll('input[name="context-strategy"]')
);
const strategyParamRow = document.getElementById('strategy-param-row');
const strategyParamInput = document.getElementById('strategy-param');
const strategyParamLabel = document.getElementById('strategy-param-label');
const strategyHint = document.getElementById('strategy-hint');
const maxTokensInput = document.getElementById('setting-max-tokens');
const stopInput = document.getElementById('setting-stop');
const modeToggle = document.getElementById('setting-mode');
const modeState = document.getElementById('mode-state');
const newChatBtn = document.getElementById('new-chat');
const summarizeBtn = document.getElementById('summarize');
const tabsListEl = document.getElementById('tabs-list');
const chatInfoEl = document.getElementById('chat-info');
const chatColumn = document.getElementById('chat-column');
const chatMain = document.getElementById('chat-main');
const viewTabChat = document.getElementById('view-tab-chat');
const viewTabMemory = document.getElementById('view-tab-memory');
const memoryView = document.getElementById('memory-view');
const memoryTabList = document.getElementById('memory-tab-list');
const memoryNewName = document.getElementById('memory-new-name');
const memoryDbToggle = document.getElementById('memory-db-toggle');
const memoryAdd = document.getElementById('memory-add');
const memoryEditor = document.getElementById('memory-editor');
const viewTabProfile = document.getElementById('view-tab-profile');
const viewTabProfileIcon = document.getElementById('view-tab-profile-icon');
const profileView = document.getElementById('profile-view');
const profileTabList = document.getElementById('profile-tab-list');
const profileNewName = document.getElementById('profile-new-name');
const profileAdd = document.getElementById('profile-add');
const profileEditor = document.getElementById('profile-editor');
const viewTabRules = document.getElementById('view-tab-rules');
const rulesView = document.getElementById('rules-view');
const rulesTabList = document.getElementById('rules-tab-list');
const rulesNewName = document.getElementById('rules-new-name');
const rulesAdd = document.getElementById('rules-add');
const rulesEditor = document.getElementById('rules-editor');
const viewTabLogs = document.getElementById('view-tab-logs');
const viewTabLogsIcon = document.getElementById('view-tab-logs-icon');
const logsView = document.getElementById('logs-view');
const logsList = document.getElementById('logs-list');
const logsCount = document.getElementById('logs-count');
const logsRefresh = document.getElementById('logs-refresh');
const logsClear = document.getElementById('logs-clear');
const viewTabMcp = document.getElementById('view-tab-mcp');
const viewTabMcpIcon = document.getElementById('view-tab-mcp-icon');
const mcpView = document.getElementById('mcp-view');
const mcpList = document.getElementById('mcp-list');
const mcpCount = document.getElementById('mcp-count');
const mcpRefresh = document.getElementById('mcp-refresh');
const mcpNewName = document.getElementById('mcp-new-name');
const mcpNewType = document.getElementById('mcp-new-type');
const mcpAdd = document.getElementById('mcp-add');
const viewTabTasks = document.getElementById('view-tab-tasks');
const tasksView = document.getElementById('tasks-view');
const tasksTabList = document.getElementById('tasks-tab-list');
const tasksAdd = document.getElementById('tasks-add');
const taskMessagesArea = document.getElementById('task-messages-area');
const taskRail = document.getElementById('task-rail');
const taskActions = document.getElementById('task-actions');
const factsPanel = document.getElementById('facts-panel');
const factsList = document.getElementById('facts-list');
const factsEmpty = document.getElementById('facts-empty');
const factsSplitter = document.getElementById('facts-splitter');
const branchBtn = document.getElementById('branch-btn');
const chartCanvas = document.getElementById('context-chart');
const chartEmpty = document.getElementById('chart-empty');
const chartColorToggle = document.getElementById('chart-color-mode');
const chartLegend = document.querySelector('.chartbar__legend');
const chartCollapseBtn = document.getElementById('chart-collapse');
const chartbarEl = document.querySelector('.chartbar');
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

const CHART_COLLAPSED_KEY = 'pomogator2k:chart-collapsed';
let chartCollapsed = true; // по умолчанию блок свёрнут
try {
  chartCollapsed = localStorage.getItem(CHART_COLLAPSED_KEY) !== '0';
} catch {
  chartCollapsed = true;
}

const FACTS_PANEL_WIDTH_KEY = 'pomogator2k:facts-panel-width';
// Значения N/K по стратегиям (дефолты: саммаризация 5, sliding 10, facts 10).
let strategyParams = { summarize: 5, sliding: 10, facts: 10 };

// Вид основной области и выбранные внутренние вкладки — переживают перезагрузку.
// Структура: {view: "chat|tasks|memory|profile", inner: {profile, task, memory:{sid: id}}}.
const UI_STATE_KEY = 'pomogator2k:ui-state';
let uiState = { view: 'chat', inner: {} };

function loadUiState() {
  try {
    const raw = localStorage.getItem(UI_STATE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    uiState.view = typeof parsed.view === 'string' ? parsed.view : 'chat';
    uiState.inner =
      parsed.inner && typeof parsed.inner === 'object' ? parsed.inner : {};
    if (!uiState.inner.memory || typeof uiState.inner.memory !== 'object') {
      uiState.inner.memory = {};
    }
  } catch {
    uiState = { view: 'chat', inner: {} };
  }
}

function saveUiState() {
  try {
    localStorage.setItem(UI_STATE_KEY, JSON.stringify(uiState));
  } catch {
    /* localStorage может быть недоступен */
  }
}

function setMemoryActive(chat, storeId) {
  if (!chat) return;
  chat.memoryActiveId = storeId;
  if (chat.sid) {
    uiState.inner.memory = uiState.inner.memory || {};
    uiState.inner.memory[chat.sid] = storeId;
  }
  saveUiState();
}

function setProfileActive(profileId) {
  profileActiveId = profileId;
  uiState.inner.profile = profileId;
  saveUiState();
}

function setRulesActive(storeId) {
  activeRulesId = storeId;
  uiState.inner.rules = storeId;
  saveUiState();
}

function setTaskActive(task) {
  if (!task) return;
  activeTaskId = task.id;
  uiState.inner.task = task.sid || null;
  saveUiState();
}

const STRATEGY_HINTS = {
  none: 'вся история в контексте',
  summarize: 'каждые {N} запросов сжимаются в саммари',
  sliding: 'в контексте только последние {N} реплик пользователя с ответами',
  facts: 'факты + последние {K} реплик пользователя с ответами',
  branching: 'вся история + ветвление чатов'
};

// Общая иконка ветки (git-branch) — для вкладок и кнопки ветвления.
const BRANCH_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/>' +
  '<circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>';

// Иконка базы данных — для кнопки-переключателя и персистентных вкладок памяти.
const DB_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>' +
  '<path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"/></svg>';

// Иконка «человечек» — для кнопки-вида «Профиль».
const PROFILE_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a8 8 0 0 1 16 0v1"/></svg>';

// Зелёная галочка активного профиля.
const CHECK_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M5 13l4 4L19 7"/></svg>';

// Иконка «терминал» — для кнопки-вида «Логи промптов».
const LOGS_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>';

// Иконка «разъём» — для кнопки-вида «MCP-серверы».
const MCP_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M9 2v6M15 2v6"/><path d="M6 8h12v4a6 6 0 0 1-6 6 6 6 0 0 1-6-6V8z"/>' +
  '<path d="M12 18v4"/></svg>';

// Метки статусов MCP-серверов.
const MCP_STATUS_LABELS = {
  connected: 'подключён',
  connecting: 'подключение…',
  error: 'ошибка',
  disabled: 'отключён'
};

// Метки типов запросов в журнале промптов.
const LOG_KIND_LABELS = {
  main: 'запрос',
  summary: 'саммаризация',
  facts: 'факты',
  describe: 'план',
  revise: 'новый план',
  run_all: 'выполнение',
  start_steps: 'шаг',
  confirm_step: 'шаг',
  revise_step: 'правка шага'
};

// Обязательные поля профиля пользователя (ключ — как на сервере).
const PROFILE_FIELDS = [
  { key: 'address', label: 'Как ко мне обращаться', placeholder: 'Например: Иван' },
  { key: 'style', label: 'Стиль общения', placeholder: 'Например: кратко и по делу' },
  { key: 'language', label: 'Язык диалога', placeholder: 'Например: русский' },
  { key: 'format', label: 'Формат ответа', placeholder: 'Например: структурированный текст' },
  { key: 'limit', label: 'Ограничение ответа', placeholder: 'Например: не более 5 предложений' }
];

// Подсказка текущего этапа задачи (протокол «Задачи»).
function taskStageHint(task) {
  switch (task.stage) {
    case 'input':
      return 'Опишите задачу — LLM составит план её выполнения.';
    case 'plan_review':
      return 'План составлен. Подтвердите его или попросите доработку.';
    case 'mode_select':
      return 'План подтверждён. Выберите способ выполнения.';
    case 'step_review':
      return `Шаг ${task.step_results.length} из ${task.steps.length} выполнен. Примите его или доработайте шаг.`;
    case 'review':
      return 'Задача выполнена. Одобрите результат или попросите доработку.';
    case 'done':
      return 'Задача выполнена и одобрена.';
    default:
      return '';
  }
}

const chats = [];
let activeChatId = null;
let chatCounter = 0;
let tokensBurned = 0;
let selectedModel = null;
// Вид основной области: 'chat' | 'memory' | 'profile'; флаг «сохранять в БД»
// для новой вкладки памяти.
let activeView = 'chat';
let memoryDbPersistent = false;
// Профили пользователя (глобальные): список, активный и выбранная вкладка UI.
let profiles = [];
let activeProfileId = null;
let profileActiveId = null;
// Задачи (протокол этапов, глобальные): список и активная вкладка.
let tasks = [];
let activeTaskId = null;
let taskCounter = 0;
// Правила (глобальные ограничения): список вкладок и выбранная вкладка UI.
let rules = [];
let activeRulesId = null;
// Журнал промптов (вкладка «Логи»): записи с сервера (в памяти сервера).
let promptLogs = [];
// MCP-серверы (вкладка «MCP»): конфигурация + live-статус с сервера.
let mcpServers = [];

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
  for (const task of tasks) {
    for (const r of task.requests || []) {
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
  const strategy = currentStrategy();
  settings.context_strategy = {
    strategy,
    n: strategy === 'sliding' ? strategyParams.sliding : strategyParams.summarize,
    k: strategyParams.facts
  };
  return settings;
}

function currentStrategy() {
  const checked = strategyRadios.find((r) => r.checked);
  return checked ? checked.value : 'none';
}

function setStrategyRadio(strategy) {
  for (const radio of strategyRadios) radio.checked = radio.value === strategy;
}

function clampStrategyParam(value, fallback) {
  return Number.isInteger(value) ? Math.min(20, Math.max(1, value)) : fallback;
}

function strategyParamValue(strategy) {
  return strategyParams[strategy] || (strategy === 'summarize' ? 5 : 10);
}

function updateStrategyHint() {
  if (!strategyHint) return;
  const strategy = currentStrategy();
  const value = strategyParamValue(strategy);
  strategyHint.textContent = (STRATEGY_HINTS[strategy] || '')
    .replace('{N}', value)
    .replace('{K}', value);
}

function updateStrategyParamUI() {
  const strategy = currentStrategy();
  const withParam =
    strategy === 'summarize' || strategy === 'sliding' || strategy === 'facts';
  if (strategyParamRow) strategyParamRow.hidden = !withParam;
  if (strategyParamInput) {
    strategyParamInput.value = String(strategyParamValue(strategy));
  }
  if (strategyParamLabel) {
    strategyParamLabel.textContent =
      strategy === 'facts' || strategy === 'sliding' ? 'реплик' : 'запросов';
  }
  updateStrategyHint();
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
  setStrategyRadio('none');
  strategyParams = { summarize: 5, sliding: 10, facts: 10 };
  updateStrategyParamUI();
}

function defaultChatSettings() {
  return {
    temperature: 1,
    top_p: 1,
    top_k: null,
    max_tokens: null,
    stop: [],
    response_format: null,
    context_strategy: { strategy: 'none', n: 5, k: 10 }
  };
}

function normalizeServerSettings(s) {
  const def = defaultChatSettings();
  if (!s || typeof s !== 'object') return def;
  const cs = s.context_strategy;
  const strategies = ['none', 'summarize', 'sliding', 'facts', 'branching'];
  const strategy = cs && strategies.includes(cs.strategy) ? cs.strategy : 'none';
  return {
    temperature: typeof s.temperature === 'number' ? s.temperature : def.temperature,
    top_p: typeof s.top_p === 'number' ? s.top_p : def.top_p,
    top_k: s.top_k || null,
    max_tokens: s.max_tokens || null,
    stop: Array.isArray(s.stop) ? s.stop : [],
    response_format: s.response_format || null,
    context_strategy: {
      strategy,
      n: clampStrategyParam(cs && cs.n, 5),
      k: clampStrategyParam(cs && cs.k, 10)
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
  const cs = s.context_strategy || defaultChatSettings().context_strategy;
  setStrategyRadio(cs.strategy);
  if (cs.strategy === 'sliding') {
    strategyParams.sliding = clampStrategyParam(cs.n, 10);
  } else if (cs.strategy === 'summarize') {
    strategyParams.summarize = clampStrategyParam(cs.n, 5);
  }
  strategyParams.facts = clampStrategyParam(cs.k, 10);
  updateStrategyParamUI();
}

function isEstablished(chat) {
  return Boolean(chat && chat.history && chat.history.length > 0);
}

function updateSettingsLock() {
  const chat = getActiveChat();
  const locked = isEstablished(chat);
  // Привязываемые первым сообщением контролы: модель, temperature,
  // top_p, top_k и стратегия контекста (радио + параметр N/K).
  for (const el of [modelButton, tempRange, topPRange, topKInput]) {
    if (el) el.disabled = locked;
  }
  for (const radio of strategyRadios) radio.disabled = locked;
  if (strategyParamInput) strategyParamInput.disabled = locked;
  if (strategyRadios[0]) {
    const group = strategyRadios[0].closest('.settings__radios');
    if (group) group.classList.toggle('settings__radios--locked', locked);
  }
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
    facts: [],
    memoryStores: [],
    memoryActiveId: null,
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

    if (chat.parentId) li.appendChild(branchIcon());
    li.appendChild(title);
    li.appendChild(close);
    tabsListEl.appendChild(li);
  }
}

function activateChat(chat) {
  activeChatId = chat.id;
  // Задачи — глобальный вид: при выборе чата возвращаемся к «Чат».
  if (activeView === 'tasks') switchView('chat');
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
  updateChatLayout();
  if (activeView === 'memory') renderMemoryView();
  if (activeView === 'profile') renderProfileView();
  if (activeView === 'rules') renderRulesView();
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

// ── Стратегии: раскладка, панель фактов, ветвление ─────────────────────────
function chatStrategy() {
  const chat = getActiveChat();
  const cs =
    chat && chat.kind === 'chat' && chat.settings && chat.settings.context_strategy;
  return cs && cs.strategy ? cs.strategy : 'none';
}

function updateChatLayout() {
  const chat = getActiveChat();
  const strategy = chatStrategy();
  const isChat = Boolean(chat && chat.kind === 'chat');
  // Вкладка «Память» есть только у обычных чатов; иначе принудительно «Чат».
  if (viewTabMemory) viewTabMemory.hidden = !isChat;
  if (!isChat && activeView === 'memory') {
    activeView = 'chat';
    uiState.view = 'chat';
    saveUiState();
    if (viewTabChat) viewTabChat.classList.add('chat__view-tab--active');
    if (viewTabMemory) viewTabMemory.classList.remove('chat__view-tab--active');
    if (chatMain) chatMain.hidden = false;
    if (memoryView) memoryView.hidden = true;
  }
  // Панель фактов — только в виде «Чат» и при стратегии facts.
  const showFacts = strategy === 'facts' && activeView === 'chat';
  if (factsPanel) factsPanel.hidden = !showFacts;
  if (factsSplitter) factsSplitter.hidden = !showFacts;
  if (chatColumn) chatColumn.classList.toggle('chat--facts', showFacts);
  if (showFacts) {
    renderFactsPanel(chat);
    applyFactsWidth();
  }
  if (branchBtn) {
    branchBtn.hidden = strategy !== 'branching' || !isEstablished(chat);
    branchBtn.disabled = Boolean(chat && chat.busy);
  }
}

// ── Виды: Чат / Задачи / Память / Правила / Профиль / Логи ─────────────────
function switchView(view) {
  activeView =
    view === 'memory'
      ? 'memory'
      : view === 'profile'
        ? 'profile'
        : view === 'tasks'
          ? 'tasks'
          : view === 'rules'
            ? 'rules'
            : view === 'logs'
              ? 'logs'
              : view === 'mcp'
                ? 'mcp'
                : 'chat';
  const isMemory = activeView === 'memory';
  const isProfile = activeView === 'profile';
  const isTasks = activeView === 'tasks';
  const isRules = activeView === 'rules';
  const isLogs = activeView === 'logs';
  const isMcp = activeView === 'mcp';
  uiState.view = activeView;
  saveUiState();
  if (viewTabChat) {
    viewTabChat.classList.toggle('chat__view-tab--active', activeView === 'chat');
  }
  if (viewTabTasks) {
    viewTabTasks.classList.toggle('chat__view-tab--active', isTasks);
  }
  if (viewTabMemory) {
    viewTabMemory.classList.toggle('chat__view-tab--active', isMemory);
  }
  if (viewTabRules) {
    viewTabRules.classList.toggle('chat__view-tab--active', isRules);
  }
  if (viewTabProfile) {
    viewTabProfile.classList.toggle('chat__view-tab--active', isProfile);
  }
  if (viewTabLogs) {
    viewTabLogs.classList.toggle('chat__view-tab--active', isLogs);
  }
  if (viewTabMcp) {
    viewTabMcp.classList.toggle('chat__view-tab--active', isMcp);
  }
  if (chatMain) {
    chatMain.hidden = isMemory || isProfile || isTasks || isRules || isLogs || isMcp;
  }
  if (memoryView) memoryView.hidden = !isMemory;
  if (profileView) profileView.hidden = !isProfile;
  if (tasksView) tasksView.hidden = !isTasks;
  if (rulesView) rulesView.hidden = !isRules;
  if (logsView) logsView.hidden = !isLogs;
  if (mcpView) mcpView.hidden = !isMcp;
  updateChatLayout();
  if (isMemory) renderMemoryView();
  if (isProfile) renderProfileView();
  if (isTasks) renderTasksView();
  if (isRules) renderRulesView();
  if (isLogs) loadLogs();
  if (isMcp) loadMcp();
}

function activeMemoryStore(chat) {
  const stores = (chat && chat.memoryStores) || [];
  return stores.find((s) => s.id === chat.memoryActiveId) || stores[0] || null;
}

function renderMemoryView() {
  const chat = getActiveChat();
  if (!memoryView || !memoryTabList || !memoryEditor) return;

  memoryTabList.innerHTML = '';
  const stores = (chat && chat.memoryStores) || [];
  for (const store of stores) {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'memory__tab' + (activeMemoryStore(chat) === store ? ' memory__tab--active' : '');
    tab.title = store.persistent ? `${store.name} (в БД)` : store.name;
    if (store.persistent) {
      const icon = document.createElement('span');
      icon.className = 'memory__tab-icon';
      icon.innerHTML = DB_ICON_SVG;
      tab.appendChild(icon);
    }
    const name = document.createElement('span');
    name.className = 'memory__tab-name';
    name.textContent = store.name;
    const close = document.createElement('span');
    close.className = 'memory__tab-close';
    close.textContent = '×';
    close.title = 'Удалить вкладку';
    close.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteMemoryStore(chat, store.id);
    });
    tab.append(name, close);
    tab.addEventListener('click', () => {
      setMemoryActive(chat, store.id);
      renderMemoryView();
    });
    memoryTabList.appendChild(tab);
  }

  if (memoryDbToggle) {
    memoryDbToggle.classList.toggle('memory__db--active', memoryDbPersistent);
    memoryDbToggle.setAttribute('aria-pressed', memoryDbPersistent ? 'true' : 'false');
  }

  renderMemoryEditor(chat, activeMemoryStore(chat));
}

function renderMemoryEditor(chat, store) {
  memoryEditor.innerHTML = '';
  if (!store) {
    const empty = document.createElement('div');
    empty.className = 'memory__empty';
    empty.textContent = 'Создайте вкладку: укажите название, при необходимости включите сохранение в БД и нажмите «+».';
    memoryEditor.appendChild(empty);
    return;
  }

  const form = document.createElement('div');
  form.className = 'memory__form';
  const keyInput = document.createElement('input');
  keyInput.type = 'text';
  keyInput.id = 'memory-key';
  keyInput.className = 'memory__input';
  keyInput.placeholder = 'Ключ';
  const valueInput = document.createElement('input');
  valueInput.type = 'text';
  valueInput.id = 'memory-value';
  valueInput.className = 'memory__input';
  valueInput.placeholder = 'Значение';
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.className = 'memory__save';
  saveBtn.textContent = 'Сохранить';
  const submit = () => saveMemoryItem(chat, store, keyInput, valueInput);
  saveBtn.addEventListener('click', submit);
  for (const el of [keyInput, valueInput]) {
    el.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit();
      }
    });
  }
  form.append(keyInput, valueInput, saveBtn);
  memoryEditor.appendChild(form);

  const items = document.createElement('div');
  items.className = 'memory__items';
  store.items.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'memory__item';
    const text = document.createElement('span');
    text.className = 'memory__item-text';
    const key = document.createElement('span');
    key.className = 'memory__item-key';
    key.textContent = `${item[0]}:`;
    text.append(key, document.createTextNode(` ${item[1]}`));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'memory__item-del';
    del.textContent = 'удалить';
    del.addEventListener('click', () => deleteMemoryItem(chat, store, index));
    row.append(text, del);
    items.appendChild(row);
  });
  memoryEditor.appendChild(items);
}

function createMemoryStore() {
  const chat = getActiveChat();
  if (!chat || chat.kind !== 'chat') return;
  chat.memoryStores = chat.memoryStores || [];
  const name =
    (memoryNewName ? memoryNewName.value.trim() : '') ||
    `Вкладка ${chat.memoryStores.length + 1}`;
  const store = {
    id: `mem-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name,
    persistent: memoryDbPersistent,
    items: []
  };
  chat.memoryStores.push(store);
  setMemoryActive(chat, store.id);
  if (memoryNewName) memoryNewName.value = '';
  // Кнопка БД сбрасывается — следующая вкладка по умолчанию неперсистентна.
  memoryDbPersistent = false;
  syncMemory(chat);
  renderMemoryView();
  if (memoryNewName) memoryNewName.focus();
}

function deleteMemoryStore(chat, storeId) {
  if (!chat || !Array.isArray(chat.memoryStores)) return;
  const store = chat.memoryStores.find((s) => s.id === storeId);
  if (store && !confirm(`Удалить вкладку памяти «${store.name}»?`)) return;
  chat.memoryStores = chat.memoryStores.filter((s) => s.id !== storeId);
  if (chat.memoryActiveId === storeId) {
    setMemoryActive(chat, chat.memoryStores.length ? chat.memoryStores[0].id : null);
  }
  syncMemory(chat);
  renderMemoryView();
}

function saveMemoryItem(chat, store, keyInput, valueInput) {
  const key = (keyInput.value || '').trim();
  const value = (valueInput.value || '').trim();
  if (!key || !value) return;
  store.items.push([key, value]);
  keyInput.value = '';
  valueInput.value = '';
  syncMemory(chat);
  renderMemoryEditor(chat, store);
  const next = memoryEditor.querySelector('#memory-key');
  if (next) next.focus();
}

function deleteMemoryItem(chat, store, index) {
  store.items.splice(index, 1);
  syncMemory(chat);
  renderMemoryView();
}

async function syncMemory(chat) {
  if (!chat || !chat.sid) return;
  try {
    logClient('send', `PUT /api/sessions/${chat.sid}/memory`, { stores: chat.memoryStores });
    const res = await fetch(`/api/sessions/${chat.sid}/memory`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stores: chat.memoryStores || [] })
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }
    logClient('receive', `PUT /api/sessions/${chat.sid}/memory → ${res.status}`);
  } catch (err) {
    logClient('receive', `PUT /api/sessions/${chat.sid}/memory — ошибка`, err.message);
  }
}

// ── Правила (глобальные ограничения; механика как у «Памяти») ──────────────
function activeRulesStore() {
  return rules.find((s) => s.id === activeRulesId) || rules[0] || null;
}

function renderRulesView() {
  if (!rulesView || !rulesTabList || !rulesEditor) return;
  rulesTabList.innerHTML = '';
  for (const store of rules) {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className =
      'memory__tab' + (activeRulesStore() === store ? ' memory__tab--active' : '');
    tab.title = store.name;
    const name = document.createElement('span');
    name.className = 'memory__tab-name';
    name.textContent = store.name;
    const close = document.createElement('span');
    close.className = 'memory__tab-close';
    close.textContent = '×';
    close.title = 'Удалить вкладку';
    close.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteRulesStore(store.id);
    });
    tab.append(name, close);
    tab.addEventListener('click', () => {
      setRulesActive(store.id);
      renderRulesView();
    });
    rulesTabList.appendChild(tab);
  }
  renderRulesEditor(activeRulesStore());
}

function renderRulesEditor(store) {
  rulesEditor.innerHTML = '';
  if (!store) {
    const empty = document.createElement('div');
    empty.className = 'memory__empty';
    empty.textContent =
      'Создайте вкладку правил: укажите название и нажмите «+». Правила действуют во всех чатах и задачах.';
    rulesEditor.appendChild(empty);
    return;
  }

  const form = document.createElement('div');
  form.className = 'memory__form';
  const keyInput = document.createElement('input');
  keyInput.type = 'text';
  keyInput.id = 'rules-key';
  keyInput.className = 'memory__input';
  keyInput.placeholder = 'Ключ';
  const valueInput = document.createElement('input');
  valueInput.type = 'text';
  valueInput.id = 'rules-value';
  valueInput.className = 'memory__input';
  valueInput.placeholder = 'Значение';
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.className = 'memory__save';
  saveBtn.textContent = 'Сохранить';
  const submit = () => saveRulesItem(store, keyInput, valueInput);
  saveBtn.addEventListener('click', submit);
  for (const el of [keyInput, valueInput]) {
    el.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit();
      }
    });
  }
  form.append(keyInput, valueInput, saveBtn);
  rulesEditor.appendChild(form);

  const items = document.createElement('div');
  items.className = 'memory__items';
  store.items.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'memory__item';
    const text = document.createElement('span');
    text.className = 'memory__item-text';
    const key = document.createElement('span');
    key.className = 'memory__item-key';
    key.textContent = `${item[0]}:`;
    text.append(key, document.createTextNode(` ${item[1]}`));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'memory__item-del';
    del.textContent = 'удалить';
    del.addEventListener('click', () => deleteRulesItem(store, index));
    row.append(text, del);
    items.appendChild(row);
  });
  rulesEditor.appendChild(items);
}

function createRulesStore() {
  const name =
    (rulesNewName ? rulesNewName.value.trim() : '') || `Правила ${rules.length + 1}`;
  const store = {
    id: `rl-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name,
    items: []
  };
  rules.push(store);
  setRulesActive(store.id);
  if (rulesNewName) rulesNewName.value = '';
  syncRules();
  renderRulesView();
  if (rulesNewName) rulesNewName.focus();
}

function deleteRulesStore(storeId) {
  const store = rules.find((s) => s.id === storeId);
  if (store && !confirm(`Удалить вкладку правил «${store.name}»?`)) return;
  rules = rules.filter((s) => s.id !== storeId);
  if (activeRulesId === storeId) {
    setRulesActive(rules.length ? rules[0].id : null);
  }
  syncRules();
  renderRulesView();
}

function saveRulesItem(store, keyInput, valueInput) {
  const key = (keyInput.value || '').trim();
  const value = (valueInput.value || '').trim();
  if (!key || !value) return;
  store.items.push([key, value]);
  keyInput.value = '';
  valueInput.value = '';
  syncRules();
  renderRulesEditor(store);
  const next = rulesEditor.querySelector('#rules-key');
  if (next) next.focus();
}

function deleteRulesItem(store, index) {
  store.items.splice(index, 1);
  syncRules();
  renderRulesView();
}

async function syncRules() {
  try {
    logClient('send', 'PUT /api/rules', { rules });
    const res = await fetch('/api/rules', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules })
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }
    logClient('receive', `PUT /api/rules → ${res.status}`);
  } catch (err) {
    logClient('receive', 'PUT /api/rules — ошибка', err.message);
  }
}

async function loadRules() {
  try {
    logClient('send', 'GET /api/rules');
    const res = await fetch('/api/rules');
    if (!res.ok) throw new Error(`Ошибка ${res.status}`);
    const data = await res.json();
    rules = (Array.isArray(data.rules) ? data.rules : [])
      .filter((s) => s && typeof s.id === 'string')
      .map((s) => ({
        id: s.id,
        name: typeof s.name === 'string' ? s.name : '',
        items: Array.isArray(s.items) ? s.items.map((item) => [...item]) : []
      }));
    logClient('receive', `GET /api/rules → ${rules.length}`, data);
  } catch (err) {
    logClient('receive', 'GET /api/rules — ошибка', err.message);
  }
}

// ── Логи промптов (вкладка «Логи»; журнал в памяти сервера) ────────────────
async function loadLogs() {
  try {
    logClient('send', 'GET /api/logs');
    const res = await fetch('/api/logs');
    if (!res.ok) throw new Error(`Ошибка ${res.status}`);
    const data = await res.json();
    promptLogs = Array.isArray(data.logs) ? data.logs : [];
    logClient('receive', `GET /api/logs → ${promptLogs.length}`, data);
  } catch (err) {
    logClient('receive', 'GET /api/logs — ошибка', err.message);
  }
  if (activeView === 'logs') renderLogsView();
}

async function clearLogs() {
  if (!confirm('Очистить журнал промптов?')) return;
  try {
    logClient('send', 'DELETE /api/logs');
    const res = await fetch('/api/logs', { method: 'DELETE' });
    logClient('receive', `DELETE /api/logs → ${res.status}`);
  } catch (err) {
    logClient('receive', 'DELETE /api/logs — ошибка', err.message);
  }
  promptLogs = [];
  renderLogsView();
}

function formatLogTime(ts) {
  if (typeof ts !== 'number' || ts <= 0) return '';
  try {
    return new Date(ts * 1000).toLocaleTimeString('ru-RU');
  } catch {
    return '';
  }
}

function renderLogEntry(entry) {
  const card = document.createElement('div');
  card.className = 'logs__entry';

  const meta = document.createElement('div');
  meta.className = 'logs__meta';
  const time = document.createElement('span');
  time.className = 'logs__time';
  time.textContent = formatLogTime(entry.time);
  const source = document.createElement('span');
  source.className = 'logs__badge logs__badge--source';
  source.textContent = entry.source || '—';
  const kind = document.createElement('span');
  kind.className = 'logs__badge logs__badge--kind';
  kind.textContent = LOG_KIND_LABELS[entry.kind] || entry.kind || '';
  const model = document.createElement('span');
  model.className = 'logs__model';
  model.textContent = entry.model || '';
  const tokens = document.createElement('span');
  tokens.className = 'logs__tokens';
  tokens.textContent =
    `вход: ${formatTokens(entry.prompt_tokens)} · выход: ${formatTokens(entry.completion_tokens)}`;
  meta.append(time, source, kind, model, tokens);
  if (entry.title) {
    const title = document.createElement('span');
    title.className = 'logs__title-inline';
    title.textContent = entry.title;
    title.title = entry.title;
    meta.appendChild(title);
  }
  card.appendChild(meta);

  const messages = Array.isArray(entry.messages) ? entry.messages : [];
  let userText = '';
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i] && messages[i].role === 'user') {
      userText = messages[i].content || '';
      break;
    }
  }
  if (userText) {
    const block = document.createElement('div');
    block.className = 'logs__user';
    const label = document.createElement('div');
    label.className = 'logs__label';
    label.textContent = 'Запрос пользователя';
    const text = document.createElement('div');
    text.className = 'logs__text';
    text.textContent = userText;
    block.append(label, text);
    card.appendChild(block);
  }

  const details = document.createElement('details');
  details.className = 'logs__prompt';
  const summary = document.createElement('summary');
  summary.textContent = `Итоговый промпт (${messages.length} сообщ.)`;
  details.appendChild(summary);
  for (const message of messages) {
    const row = document.createElement('div');
    row.className = 'logs__msg';
    const role = document.createElement('span');
    role.className = `logs__msg-role logs__msg-role--${message.role || ''}`;
    role.textContent = message.role || '';
    const text = document.createElement('span');
    text.className = 'logs__msg-text';
    text.textContent = message.content || '';
    row.append(role, text);
    details.appendChild(row);
  }
  card.appendChild(details);

  const response = document.createElement('div');
  response.className = 'logs__response';
  const rlabel = document.createElement('div');
  rlabel.className = 'logs__label';
  rlabel.textContent = 'Ответ LLM';
  const rtext = document.createElement('div');
  rtext.className = 'logs__text';
  rtext.textContent = entry.response || '(пустой ответ)';
  response.append(rlabel, rtext);
  card.appendChild(response);
  return card;
}

function renderLogsView() {
  if (!logsView || !logsList) return;
  if (logsCount) logsCount.textContent = promptLogs.length ? String(promptLogs.length) : '';
  logsList.innerHTML = '';
  if (!promptLogs.length) {
    const empty = document.createElement('div');
    empty.className = 'logs__empty';
    empty.textContent = 'Логов пока нет.';
    logsList.appendChild(empty);
    return;
  }
  for (const entry of promptLogs) {
    logsList.appendChild(renderLogEntry(entry));
  }
}

// ── MCP-серверы (вкладка «MCP») ─────────────────────────────────────────────
// Конфигурация хранится на сервере (формат OpenCode: local/remote); клиент
// синхронизирует полное состояние и показывает live-статус и инструменты.

async function loadMcp() {
  try {
    logClient('send', 'GET /api/mcp');
    const res = await fetch('/api/mcp');
    if (!res.ok) {
      logClient('receive', 'GET /api/mcp → ' + res.status);
      return;
    }
    const data = await res.json();
    logClient('receive', 'GET /api/mcp', data);
    mcpServers = Array.isArray(data.servers) ? data.servers : [];
  } catch (err) {
    logClient('receive', 'GET /api/mcp — ошибка', err);
    return;
  }
  if (activeView === 'mcp') renderMcpView();
}

async function syncMcp() {
  try {
    logClient('send', 'PUT /api/mcp', { servers: mcpServers });
    const res = await fetch('/api/mcp', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ servers: mcpServers })
    });
    if (!res.ok) {
      logClient('receive', 'PUT /api/mcp → ' + res.status);
      return;
    }
    const data = await res.json();
    logClient('receive', 'PUT /api/mcp', data);
    mcpServers = Array.isArray(data.servers) ? data.servers : [];
  } catch (err) {
    logClient('receive', 'PUT /api/mcp — ошибка', err);
    return;
  }
  renderMcpView();
}

function replaceMcpServer(server) {
  if (!server) return;
  const index = mcpServers.findIndex((s) => s.id === server.id);
  if (index === -1) mcpServers.push(server);
  else mcpServers[index] = server;
  renderMcpView();
}

function renderMcpView() {
  if (!mcpView || !mcpList) return;
  if (mcpCount) mcpCount.textContent = mcpServers.length ? String(mcpServers.length) : '';
  mcpList.innerHTML = '';
  if (!mcpServers.length) {
    const empty = document.createElement('div');
    empty.className = 'mcp__empty';
    empty.textContent =
      'MCP-серверы не настроены. Добавьте сервер ниже — например, ' +
      'jsonplaceholder (local, stdio).';
    mcpList.appendChild(empty);
    return;
  }
  for (const server of mcpServers) {
    mcpList.appendChild(renderMcpCard(server));
  }
}

function mcpButton(label, variant, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `mcp-server__btn mcp-server__btn--${variant}`;
  button.textContent = label;
  button.addEventListener('click', onClick);
  return button;
}

function mcpField(label, className, value, placeholder, multiline) {
  const field = document.createElement('label');
  field.className = 'mcp-server__field';
  const caption = document.createElement('span');
  caption.className = 'mcp-server__field-label';
  caption.textContent = label;
  field.appendChild(caption);
  const input = document.createElement(multiline ? 'textarea' : 'input');
  if (!multiline) input.type = 'text';
  input.className = `mcp-server__input ${className}`;
  input.value = value || '';
  if (placeholder) input.placeholder = placeholder;
  if (multiline) input.rows = 2;
  field.appendChild(input);
  return field;
}

function renderMcpCard(server) {
  const card = document.createElement('div');
  card.className = 'mcp-server';
  card.dataset.mcpId = server.id;

  const head = document.createElement('div');
  head.className = 'mcp-server__head';

  const status = server.status || 'disabled';
  const dot = document.createElement('span');
  dot.className = `mcp-server__dot mcp-server__dot--${status}`;
  dot.title = MCP_STATUS_LABELS[status] || status;
  head.appendChild(dot);

  const name = document.createElement('span');
  name.className = 'mcp-server__name';
  name.textContent = server.name;
  head.appendChild(name);

  const type = document.createElement('span');
  type.className = 'mcp-server__type';
  type.textContent = server.type === 'remote' ? 'remote · http' : 'local · stdio';
  head.appendChild(type);

  const statusEl = document.createElement('span');
  statusEl.className = `mcp-server__status mcp-server__status--${status}`;
  statusEl.textContent = MCP_STATUS_LABELS[status] || status;
  head.appendChild(statusEl);

  const actions = document.createElement('span');
  actions.className = 'mcp-server__actions';

  const enabledLabel = document.createElement('label');
  enabledLabel.className = 'mcp-server__enabled-wrap';
  enabledLabel.title = 'Подключать при старте и синхронизации';
  const enabled = document.createElement('input');
  enabled.type = 'checkbox';
  enabled.className = 'mcp-server__enabled';
  enabled.checked = server.enabled !== false;
  enabled.addEventListener('change', () => {
    const index = mcpServers.findIndex((s) => s.id === server.id);
    if (index === -1) return;
    mcpServers[index] = { ...mcpServers[index], enabled: enabled.checked };
    syncMcp();
  });
  const enabledText = document.createElement('span');
  enabledText.textContent = 'включён';
  enabledLabel.append(enabled, enabledText);
  actions.appendChild(enabledLabel);

  if (status === 'connected' || status === 'connecting') {
    actions.appendChild(
      mcpButton('Отключить', 'ghost', () => disconnectMcpServer(server.id))
    );
    actions.appendChild(
      mcpButton('Переподключить', 'ghost', () => connectMcpServer(server.id))
    );
  } else {
    actions.appendChild(
      mcpButton('Подключить', 'primary', () => connectMcpServer(server.id))
    );
  }
  actions.appendChild(
    mcpButton('Удалить', 'danger', () => deleteMcpServer(server.id))
  );
  head.appendChild(actions);
  card.appendChild(head);

  if (server.error) {
    const error = document.createElement('div');
    error.className = 'mcp-server__error';
    error.textContent = server.error;
    card.appendChild(error);
  }

  const tools = Array.isArray(server.tools) ? server.tools : [];
  if (tools.length) {
    const details = document.createElement('details');
    details.className = 'mcp-server__tools';
    const summary = document.createElement('summary');
    summary.textContent = `Инструменты (${tools.length})`;
    details.appendChild(summary);
    const list = document.createElement('div');
    list.className = 'mcp-server__tool-list';
    for (const tool of tools) {
      const item = document.createElement('div');
      item.className = 'mcp-server__tool';
      const toolName = document.createElement('span');
      toolName.className = 'mcp-server__tool-name';
      toolName.textContent = tool.name;
      item.appendChild(toolName);
      if (tool.description) {
        const desc = document.createElement('span');
        desc.className = 'mcp-server__tool-desc';
        desc.textContent = tool.description;
        item.appendChild(desc);
      }
      list.appendChild(item);
    }
    details.appendChild(list);
    card.appendChild(details);
  }

  const config = document.createElement('div');
  config.className = 'mcp-server__config';
  if (server.type === 'remote') {
    config.appendChild(
      mcpField('URL', 'mcp-server__url', server.url, 'http://127.0.0.1:8001/mcp')
    );
    config.appendChild(
      mcpField(
        'Заголовки',
        'mcp-server__headers',
        formatKvLines(server.headers),
        'Authorization=Bearer ... (по строке)',
        true
      )
    );
  } else {
    config.appendChild(
      mcpField(
        'Команда',
        'mcp-server__command',
        joinCommand(server.command),
        '.venv/Scripts/python.exe mcp_demo/jsonplaceholder_server.py'
      )
    );
    config.appendChild(
      mcpField(
        'Рабочий каталог',
        'mcp-server__cwd',
        server.cwd,
        'по умолчанию — каталог сервера'
      )
    );
    config.appendChild(
      mcpField(
        'Окружение',
        'mcp-server__env',
        formatKvLines(server.environment),
        'KEY=VALUE (по строке)',
        true
      )
    );
  }
  config.appendChild(
    mcpField('Таймаут, мс', 'mcp-server__timeout', String(server.timeout || 5000), '5000')
  );

  const hint = document.createElement('span');
  hint.className = 'mcp-server__hint';
  config.appendChild(hint);
  config.appendChild(
    mcpButton('Сохранить', 'primary', () => saveMcpServer(server.id, card))
  );
  card.appendChild(config);

  return card;
}

function readMcpCard(card, server) {
  const next = { ...server };
  const command = card.querySelector('.mcp-server__command');
  if (command) next.command = splitCommand(command.value);
  const cwd = card.querySelector('.mcp-server__cwd');
  if (cwd) next.cwd = cwd.value.trim() || null;
  const environment = card.querySelector('.mcp-server__env');
  if (environment) next.environment = parseKvLines(environment.value);
  const url = card.querySelector('.mcp-server__url');
  if (url) next.url = url.value.trim();
  const headers = card.querySelector('.mcp-server__headers');
  if (headers) next.headers = parseKvLines(headers.value);
  const timeout = card.querySelector('.mcp-server__timeout');
  if (timeout) {
    const value = parseInt(timeout.value, 10);
    next.timeout = Number.isFinite(value) && value > 0 ? value : 5000;
  }
  const enabled = card.querySelector('.mcp-server__enabled');
  if (enabled) next.enabled = enabled.checked;
  return next;
}

async function saveMcpServer(id, card) {
  const index = mcpServers.findIndex((s) => s.id === id);
  if (index === -1) return;
  const next = readMcpCard(card, mcpServers[index]);
  const hint = card.querySelector('.mcp-server__hint');
  const invalidUrl =
    next.type === 'remote' && !/^https?:\/\/\S+/.test(next.url || '');
  const invalidCommand =
    next.type !== 'remote' && (!next.command || !next.command.length);
  if (invalidUrl || invalidCommand) {
    if (hint) {
      hint.textContent = invalidUrl
        ? 'Укажите URL вида http://…'
        : 'Укажите команду запуска сервера';
    }
    return;
  }
  if (hint) hint.textContent = '';
  mcpServers[index] = next;
  await syncMcp();
}

function createMcpServer() {
  const name = (mcpNewName.value || '').trim();
  if (!name) {
    mcpNewName.focus();
    return;
  }
  const type = mcpNewType.value === 'remote' ? 'remote' : 'local';
  mcpServers.push({
    id: `mcp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name,
    type,
    enabled: true,
    command: [],
    environment: {},
    cwd: null,
    url: '',
    headers: {},
    timeout: 5000,
    status: 'disabled',
    error: '',
    tools: []
  });
  mcpNewName.value = '';
  renderMcpView();
}

async function deleteMcpServer(id) {
  const server = mcpServers.find((s) => s.id === id);
  if (!server) return;
  if (!confirm(`Удалить MCP-сервер «${server.name}»?`)) return;
  mcpServers = mcpServers.filter((s) => s.id !== id);
  await syncMcp();
}

async function connectMcpServer(id) {
  try {
    logClient('send', `POST /api/mcp/${id}/connect`);
    const res = await fetch(`/api/mcp/${id}/connect`, { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    logClient('receive', `POST /api/mcp/${id}/connect → ${res.status}`, data);
    if (data.server) replaceMcpServer(data.server);
  } catch (err) {
    logClient('receive', `POST /api/mcp/${id}/connect — ошибка`, err);
  }
}

async function disconnectMcpServer(id) {
  try {
    logClient('send', `POST /api/mcp/${id}/disconnect`);
    const res = await fetch(`/api/mcp/${id}/disconnect`, { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    logClient('receive', `POST /api/mcp/${id}/disconnect → ${res.status}`, data);
    if (data.server) replaceMcpServer(data.server);
  } catch (err) {
    logClient('receive', `POST /api/mcp/${id}/disconnect — ошибка`, err);
  }
}

function splitCommand(text) {
  const out = [];
  let current = '';
  let quote = null;
  for (const ch of String(text || '')) {
    if (quote) {
      if (ch === quote) quote = null;
      else current += ch;
    } else if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (/\s/.test(ch)) {
      if (current) {
        out.push(current);
        current = '';
      }
    } else {
      current += ch;
    }
  }
  if (current) out.push(current);
  return out;
}

function joinCommand(argv) {
  return (argv || [])
    .map((part) => (/\s/.test(part) ? `"${part}"` : part))
    .join(' ');
}

function parseKvLines(text) {
  const out = {};
  for (const raw of String(text || '').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const index = line.indexOf('=');
    if (index <= 0) continue;
    const key = line.slice(0, index).trim();
    const value = line.slice(index + 1).trim();
    if (key && value) out[key] = value;
  }
  return out;
}

function formatKvLines(map) {
  if (!map || typeof map !== 'object') return '';
  return Object.entries(map)
    .map(([key, value]) => `${key}=${value}`)
    .join('\n');
}

// ── Профили пользователя (глобальная сущность) ─────────────────────────────
function activeProfile() {
  return profiles.find((p) => p.id === activeProfileId) || null;
}

function normalizeProfileFields(fields) {
  const out = {};
  for (const { key } of PROFILE_FIELDS) {
    out[key] = fields && typeof fields[key] === 'string' ? fields[key] : '';
  }
  return out;
}

function activeProfileTab() {
  return profiles.find((p) => p.id === profileActiveId) || profiles[0] || null;
}

function renderProfileView() {
  if (!profileView || !profileTabList || !profileEditor) return;
  profileTabList.innerHTML = '';
  for (const profile of profiles) {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className =
      'memory__tab' + (activeProfileTab() === profile ? ' memory__tab--active' : '');
    tab.title = profile.id === activeProfileId
      ? `${profile.name} (текущий)`
      : profile.name;
    if (profile.id === activeProfileId) {
      const check = document.createElement('span');
      check.className = 'profile__tab-check';
      check.title = 'Текущий профиль';
      check.innerHTML = CHECK_ICON_SVG;
      tab.appendChild(check);
    }
    const name = document.createElement('span');
    name.className = 'memory__tab-name';
    name.textContent = profile.name;
    const close = document.createElement('span');
    close.className = 'memory__tab-close';
    close.textContent = '×';
    close.title = 'Удалить профиль';
    close.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteProfile(profile.id);
    });
    tab.append(name, close);
    tab.addEventListener('click', () => {
      setProfileActive(profile.id);
      renderProfileView();
    });
    profileTabList.appendChild(tab);
  }
  renderProfileEditor(activeProfileTab());
}

function renderProfileEditor(profile) {
  profileEditor.innerHTML = '';
  if (!profile) {
    const empty = document.createElement('div');
    empty.className = 'memory__empty';
    empty.textContent = 'Создайте профиль: укажите название и нажмите «+».';
    profileEditor.appendChild(empty);
    return;
  }

  const form = document.createElement('div');
  form.className = 'profile__form';

  const inputs = {};
  for (const { key, label, placeholder } of PROFILE_FIELDS) {
    const field = document.createElement('label');
    field.className = 'profile__field';
    const labelEl = document.createElement('span');
    labelEl.className = 'profile__label';
    labelEl.textContent = label;
    const inputEl = document.createElement('input');
    inputEl.type = 'text';
    inputEl.className = 'profile__input';
    inputEl.placeholder = placeholder;
    inputEl.value = profile.fields[key] || '';
    inputs[key] = inputEl;
    field.append(labelEl, inputEl);
    form.appendChild(field);
  }

  const actions = document.createElement('div');
  actions.className = 'profile__actions';
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.className = 'memory__save';
  saveBtn.textContent = 'Сохранить';
  saveBtn.addEventListener('click', () => saveProfile(profile.id, inputs));
  const currentBtn = document.createElement('button');
  currentBtn.type = 'button';
  currentBtn.className =
    'profile__current' + (profile.id === activeProfileId ? ' profile__current--active' : '');
  currentBtn.textContent =
    profile.id === activeProfileId ? 'Текущий профиль' : 'Установить текущим';
  currentBtn.addEventListener('click', () => setCurrentProfile(profile.id, inputs));
  const hint = document.createElement('span');
  hint.className = 'profile__hint';
  actions.append(saveBtn, currentBtn, hint);
  form.appendChild(actions);

  profileEditor.appendChild(form);
}

function readProfileInputs(profile, inputs) {
  const fields = {};
  let complete = true;
  for (const { key } of PROFILE_FIELDS) {
    const value = (inputs[key].value || '').trim();
    fields[key] = value;
    inputs[key].classList.toggle('profile__input--invalid', !value);
    if (!value) complete = false;
  }
  if (!(profile.name || '').trim()) complete = false;
  return { fields, complete };
}

function saveProfile(profileId, inputs) {
  const profile = profiles.find((p) => p.id === profileId);
  if (!profile) return;
  const hint = profileEditor.querySelector('.profile__hint');
  const { fields, complete } = readProfileInputs(profile, inputs);
  if (!complete) {
    if (hint) hint.textContent = 'Заполните все поля профиля.';
    return;
  }
  profile.fields = fields;
  if (hint) hint.textContent = 'Профиль сохранён.';
  syncProfiles();
}

function setCurrentProfile(profileId, inputs) {
  const profile = profiles.find((p) => p.id === profileId);
  if (!profile) return;
  const hint = profileEditor.querySelector('.profile__hint');
  const { fields, complete } = readProfileInputs(profile, inputs);
  if (!complete) {
    if (hint) hint.textContent = 'Заполните все поля профиля.';
    return;
  }
  profile.fields = fields;
  activeProfileId = profileId;
  syncProfiles();
  renderProfileView();
}

function createProfile() {
  const name =
    (profileNewName ? profileNewName.value.trim() : '') ||
    `Профиль ${profiles.length + 1}`;
  const profile = {
    id: `prf-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    name,
    fields: normalizeProfileFields(null)
  };
  profiles.push(profile);
  setProfileActive(profile.id);
  if (profileNewName) profileNewName.value = '';
  syncProfiles();
  renderProfileView();
  if (profileNewName) profileNewName.focus();
}

function deleteProfile(profileId) {
  const profile = profiles.find((p) => p.id === profileId);
  if (profile && !confirm(`Удалить профиль «${profile.name}»?`)) return;
  profiles = profiles.filter((p) => p.id !== profileId);
  if (activeProfileId === profileId) activeProfileId = null;
  if (profileActiveId === profileId) {
    setProfileActive(profiles.length ? profiles[0].id : null);
  }
  syncProfiles();
  renderProfileView();
}

async function syncProfiles() {
  try {
    const payload = { profiles, active_id: activeProfileId };
    logClient('send', 'PUT /api/profiles', payload);
    const res = await fetch('/api/profiles', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }
    logClient('receive', `PUT /api/profiles → ${res.status}`);
  } catch (err) {
    logClient('receive', 'PUT /api/profiles — ошибка', err.message);
  }
}

async function loadProfiles() {
  try {
    logClient('send', 'GET /api/profiles');
    const res = await fetch('/api/profiles');
    if (!res.ok) throw new Error(`Ошибка ${res.status}`);
    const data = await res.json();
    profiles = (Array.isArray(data.profiles) ? data.profiles : [])
      .filter((p) => p && typeof p.id === 'string')
      .map((p) => ({
        id: p.id,
        name: typeof p.name === 'string' ? p.name : '',
        fields: normalizeProfileFields(p.fields)
      }));
    activeProfileId = typeof data.active_id === 'string' ? data.active_id : null;
    logClient('receive', `GET /api/profiles → ${profiles.length}`, data);
  } catch (err) {
    logClient('receive', 'GET /api/profiles — ошибка', err.message);
  }
}

// Уведомление о необходимости профиля (вместо отправки в LLM).
function renderProfileRequired(chat, text) {
  const el = createMessageEl('assistant', '');
  el.classList.remove('message--empty');
  el.classList.add('message--profile-required');
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

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'message__limit-btn';
  btn.textContent = 'Открыть профиль';
  btn.addEventListener('click', () => switchView('profile'));

  actions.appendChild(btn);
  content.append(icon, msg, actions);
  chat.messagesEl.appendChild(el);
  chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
}

// ── Задачи (строгий протокол этапов) ───────────────────────────────────────
function activeTask() {
  return tasks.find((t) => t.id === activeTaskId) || null;
}

function makeTaskMessagesEl() {
  const el = document.createElement('div');
  el.className = 'chat__messages';
  el.hidden = true;
  taskMessagesArea.appendChild(el);
  return el;
}

function renderTaskHistory(task) {
  task.messagesEl.innerHTML = '';
  let qaEl = null;
  for (const m of task.history) {
    if (m.role === 'user' || m.role === 'system') {
      qaEl = document.createElement('div');
      qaEl.className = 'qa';
      task.messagesEl.appendChild(qaEl);
      qaEl.appendChild(createMessageEl('user', m.content));
    } else {
      const el = createMessageEl('assistant', m.content);
      (qaEl || task.messagesEl).appendChild(el);
      if (m.meta) renderQaStats(qaEl || task.messagesEl, formatMetaLine(m.meta));
      qaEl = null;
    }
  }
  task.messagesEl.scrollTop = task.messagesEl.scrollHeight;
}

function addTaskQa(task, text) {
  const qaEl = document.createElement('div');
  qaEl.className = 'qa';
  task.messagesEl.appendChild(qaEl);
  qaEl.appendChild(createMessageEl('user', text));
  task.messagesEl.scrollTop = task.messagesEl.scrollHeight;
  return qaEl;
}

function addTaskNotice(task, text) {
  const el = createMessageEl('assistant', text);
  el.classList.add('message--error');
  task.messagesEl.appendChild(el);
  task.messagesEl.scrollTop = task.messagesEl.scrollHeight;
}

function renderTasksView() {
  if (!tasksView || !tasksTabList) return;
  tasksTabList.innerHTML = '';
  for (const task of tasks) {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className =
      'memory__tab' + (activeTask() === task ? ' memory__tab--active' : '');
    tab.title = task.title;
    const name = document.createElement('span');
    name.className = 'memory__tab-name';
    name.textContent = task.title;
    const close = document.createElement('span');
    close.className = 'memory__tab-close';
    close.textContent = '×';
    close.title = 'Удалить задачу';
    close.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteTask(task);
    });
    tab.append(name, close);
    tab.addEventListener('click', () => {
      setTaskActive(task);
      renderTasksView();
    });
    tasksTabList.appendChild(tab);
  }
  const task = activeTask();
  for (const t of tasks) {
    if (t.messagesEl) t.messagesEl.hidden = t !== task;
  }
  renderTaskRail(task);
  renderTaskActions(task);
}

// ── Рельса этапов/шагов задачи (вертикальная визуализация слева) ───────────
function renderTaskRail(task) {
  if (!taskRail) return;
  taskRail.innerHTML = '';
  if (!task) {
    taskRail.hidden = true;
    return;
  }
  taskRail.hidden = false;

  const stage = task.stage;
  const steps = task.steps || [];
  const results = task.step_results || [];
  const busy = Boolean(task.busy);
  const planning =
    stage === 'input' || stage === 'plan_review' || stage === 'mode_select';
  const inReview = stage === 'review';
  const isDone = stage === 'done';
  const canEdit = !busy && stage !== 'input' && stage !== 'done' && steps.length > 0;

  // Номер выполняемого сейчас шага (спиннер) и шага на подтверждении.
  let runningIndex = -1;
  if (busy && steps.length) {
    runningIndex =
      task.busyAction === 'revise_step'
        ? Math.max(0, results.length - 1)
        : results.length;
  }
  // Шаг «на подтверждении» подсвечиваем только вне стрима: во время
  // выполнения подтверждённые шаги сразу помечаются выполненными.
  const activeIndex =
    !busy && stage === 'step_review' && results.length ? results.length - 1 : -1;
  // Сколько шагов показывать выполненными.
  const doneCount = inReview || isDone
    ? steps.length
    : activeIndex >= 0
      ? activeIndex
      : results.length;

  const planningRunning =
    busy && (task.busyAction === 'describe' || task.busyAction === 'revise');
  taskRail.appendChild(
    railNode('Планирование', 0, {
      state: planningRunning ? 'running' : planning ? 'active' : 'done'
    })
  );

  if (steps.length) {
    steps.forEach((text, index) => {
      let state = 'pending';
      if (index === runningIndex) state = 'running';
      else if (index === activeIndex) state = 'active';
      else if (index < doneCount) state = 'done';
      const node = railNode(text, index + 1, { step: true, state });
      node.title = text;
      if (canEdit) {
        node.classList.add('tasks__node--clickable');
        node.setAttribute('role', 'button');
        node.tabIndex = 0;
        node.addEventListener('click', () => beginStepEdit(task, index));
        node.addEventListener('keydown', (event) => {
          // Только клавиши на самом узле (не из textarea редактора).
          if (event.target !== node) return;
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            beginStepEdit(task, index);
          }
        });
      }
      taskRail.appendChild(node);
    });
  } else if (!planning) {
    // Режим «всё сразу» (run_all): шагов нет — один узел выполнения.
    taskRail.appendChild(
      railNode('Выполнение', 0, {
        state: busy ? 'running' : inReview || isDone ? 'done' : 'pending'
      })
    );
  }

  taskRail.appendChild(
    railNode('Валидация', 0, {
      state: inReview ? 'active' : isDone ? 'done' : 'pending'
    })
  );
}

function railNode(label, number, opts) {
  // Обычный div (а не button): внутри узла-шага может открываться редактор
  // с собственными кнопками — вложенные button в button невалидны и ломают
  // обработку кликов.
  const node = document.createElement('div');
  node.className = 'tasks__node';
  if (opts.step) node.classList.add('tasks__node-step');
  if (opts.state === 'done') node.classList.add('tasks__node--done');
  if (opts.state === 'active' || opts.state === 'running') {
    node.classList.add('tasks__node--active');
  }
  if (opts.state === 'pending') node.classList.add('tasks__node--pending');

  const marker = document.createElement('span');
  marker.className = 'tasks__node-marker';
  if (opts.state === 'running') {
    const spinner = document.createElement('span');
    spinner.className = 'tasks__node-spinner';
    marker.appendChild(spinner);
  } else if (opts.state === 'done') {
    marker.textContent = '✓';
  } else if (number > 0) {
    marker.textContent = String(number);
  }

  const text = document.createElement('span');
  text.className = 'tasks__node-label';
  if (opts.step && number > 0) {
    const num = document.createElement('span');
    num.className = 'tasks__node-num';
    num.textContent = `Шаг ${number}: `;
    text.appendChild(num);
  }
  text.appendChild(document.createTextNode(label));
  node.append(marker, text);
  return node;
}

function beginStepEdit(task, index) {
  if (!taskRail || !task || task.busy) return;
  const nodes = taskRail.querySelectorAll('.tasks__node-step');
  const node = nodes[index];
  if (!node) return;
  // Защита от повторного входа (клик по узлу всплывает из редактора).
  if (node.querySelector('.tasks__node-edit')) return;
  const original = task.steps[index] || '';

  node.innerHTML = '';
  const editor = document.createElement('div');
  editor.className = 'tasks__node-edit';
  // Клики внутри редактора не всплывают к обработчику узла.
  editor.addEventListener('click', (event) => event.stopPropagation());
  editor.addEventListener('keydown', (event) => event.stopPropagation());
  const input = document.createElement('textarea');
  input.className = 'tasks__node-edit-input';
  input.value = original;
  const actions = document.createElement('div');
  actions.className = 'tasks__node-edit-actions';
  const save = document.createElement('button');
  save.type = 'button';
  save.className = 'tasks__node-edit-btn tasks__node-edit-btn--save';
  save.textContent = 'Сохранить';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'tasks__node-edit-btn';
  cancel.textContent = 'Отмена';
  const commit = () => {
    const value = input.value.trim();
    const norm = (s) => s.replace(/\s+/g, ' ').trim();
    // Правка одних пробелов смысл не меняет — ничего не перезапускаем.
    if (!value || norm(value) === norm(original)) {
      renderTaskRail(task);
      return;
    }
    editStepTask(task, index, value);
  };
  save.addEventListener('click', commit);
  cancel.addEventListener('click', () => renderTaskRail(task));
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      commit();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      renderTaskRail(task);
    }
  });
  actions.append(save, cancel);
  editor.append(input, actions);
  node.appendChild(editor);
  input.focus();
}

function taskButton(text, cls, onClick) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = `tasks__btn ${cls}`;
  btn.textContent = text;
  btn.addEventListener('click', onClick);
  return btn;
}

function taskFeedbackRow(task, placeholder, btnText, onSubmit, hidden = false) {
  const row = document.createElement('div');
  row.className = 'tasks__row tasks__feedback';
  row.hidden = hidden;
  const area = document.createElement('textarea');
  area.className = 'tasks__input';
  area.rows = 1;
  area.placeholder = placeholder;
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'tasks__btn tasks__btn--primary';
  btn.textContent = btnText;
  const submit = () => {
    const value = area.value.trim();
    if (!value) return;
    area.value = '';
    onSubmit(value);
  };
  btn.addEventListener('click', submit);
  area.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });
  row.append(area, btn);
  return row;
}

function renderTaskActions(task) {
  if (!taskActions) return;
  taskActions.innerHTML = '';
  if (!task) {
    const empty = document.createElement('div');
    empty.className = 'tasks__hint';
    empty.textContent = 'Создайте задачу кнопкой «+».';
    taskActions.appendChild(empty);
    return;
  }

  const hint = document.createElement('div');
  hint.className = 'tasks__hint';
  hint.textContent = taskStageHint(task);
  taskActions.appendChild(hint);

  // Поле замечаний показывается постоянно (без кнопки-переключателя).
  const remark = (placeholder, action) =>
    taskFeedbackRow(task, placeholder, 'Отправить', (text) =>
      advanceTask(task, action, text)
    );

  if (task.stage === 'input') {
    taskActions.appendChild(remark('Опишите задачу...', 'describe'));
  } else if (task.stage === 'done') {
    const done = document.createElement('div');
    done.className = 'tasks__done';
    const label = document.createElement('span');
    label.textContent = 'Задача выполнена ✓';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'tasks__btn tasks__btn--primary';
    copy.textContent = 'Копировать ответ';
    copy.addEventListener('click', () => copyToClipboard(task.result || ''));
    done.append(label, copy);
    taskActions.appendChild(done);
  } else if (task.stage === 'mode_select') {
    const row = document.createElement('div');
    row.className = 'tasks__row';
    row.append(
      taskButton('Выполнить по шагам', 'tasks__btn--primary', () =>
        advanceTask(task, 'start_steps')
      ),
      taskButton('Выполнить всё сразу', 'tasks__btn--ghost', () =>
        runAllTask(task)
      )
    );
    taskActions.appendChild(row);
  } else {
    const row = document.createElement('div');
    row.className = 'tasks__row';
    if (task.stage === 'plan_review') {
      row.appendChild(
        taskButton('Подтвердить план', 'tasks__btn--primary', () =>
          taskActionJson(task, 'confirm')
        )
      );
      taskActions.append(row, remark('Замечания к плану...', 'revise'));
    } else if (task.stage === 'step_review') {
      row.appendChild(
        taskButton('Принять шаг', 'tasks__btn--primary', () =>
          confirmStepTask(task)
        )
      );
      taskActions.append(row, remark('Что исправить в шаге или куда перейти (например: вернись к шагу 2)...', 'revise_step'));
    } else if (task.stage === 'review') {
      row.appendChild(
        taskButton('Одобрить', 'tasks__btn--primary', () =>
          taskActionJson(task, 'approve')
        )
      );
      taskActions.append(row, remark('Что доработать в результате или какой шаг поправить (например: доработай шаг 2)...', 'revise'));
    }
  }

  if (task.busy) {
    taskActions.querySelectorAll('button, textarea').forEach((el) => {
      el.disabled = true;
    });
  }
}

async function createTask() {
  const task = {
    id: `task-${++taskCounter}`,
    sid: null,
    title: 'Новая задача',
    stage: 'input',
    plan: null,
    result: null,
    steps: [],
    step_results: [],
    history: [],
    requests: [],
    busy: false,
    messagesEl: makeTaskMessagesEl()
  };
  tasks.push(task);
  setTaskActive(task);
  renderTasksView();
  try {
    const created = await apiCreateSession({ kind: 'task' });
    task.sid = created.id;
    setTaskActive(task);
  } catch (err) {
    addTaskNotice(task, `Ошибка создания задачи: ${err.message}`);
  }
  renderTasksView();
}

function makeTaskFromState(state) {
  const task = {
    id: `task-${++taskCounter}`,
    sid: state.id,
    title: state.title || 'Задача',
    stage: state.stage || 'input',
    plan: state.plan || null,
    result: state.result || null,
    steps: Array.isArray(state.steps) ? state.steps.map((s) => String(s)) : [],
    step_results: Array.isArray(state.step_results)
      ? state.step_results.map((s) => String(s))
      : [],
    history: Array.isArray(state.history)
      ? state.history.map((m) => ({ role: m.role, content: m.content, meta: m.meta || null }))
      : [],
    requests: Array.isArray(state.requests) ? state.requests.map((r) => ({ ...r })) : [],
    busy: false,
    messagesEl: makeTaskMessagesEl()
  };
  renderTaskHistory(task);
  return task;
}

async function loadTasks() {
  try {
    logClient('send', 'GET /api/tasks');
    const res = await fetch('/api/tasks');
    if (!res.ok) throw new Error(`Ошибка ${res.status}`);
    const data = await res.json();
    const list = Array.isArray(data.data) ? data.data : [];
    for (const state of list) {
      tasks.push(makeTaskFromState(state));
    }
    if (!activeTaskId && tasks.length) activeTaskId = tasks[tasks.length - 1].id;
    logClient('receive', `GET /api/tasks → ${tasks.length}`, data);
  } catch (err) {
    logClient('receive', 'GET /api/tasks — ошибка', err.message);
  }
}

async function deleteTask(task) {
  if (!task || task.busy) return;
  if (!confirm(`Удалить задачу «${task.title}»?`)) return;
  const idx = tasks.indexOf(task);
  if (task.messagesEl) task.messagesEl.remove();
  tasks.splice(idx, 1);
  if (task.sid) apiDeleteSession(task.sid).catch(() => {});
  if (activeTaskId === task.id || !activeTask()) {
    if (tasks.length) {
      setTaskActive(tasks[tasks.length - 1]);
    } else {
      activeTaskId = null;
      uiState.inner.task = null;
      saveUiState();
    }
  }
  renderTasksView();
}

async function advanceTask(task, action, content) {
  if (!task || task.busy || !task.sid) return;
  if (!activeProfile()) {
    addTaskNotice(task, 'Необходимо создать и установить профиль.');
    return;
  }

  let marker = null;
  if (action === 'describe' || action === 'revise' || action === 'revise_step') {
    if (!content) return;
    marker = content;
  } else if (action === 'start_steps') {
    marker = 'Выполни шаг 1.';
  }
  let qaEl = null;
  if (marker !== null) {
    task.history.push({ role: 'user', content: marker });
    qaEl = addTaskQa(task, marker);
  }
  if (action === 'describe') {
    task.title = content.replace(/\s+/g, ' ').trim().slice(0, 60) || 'Задача';
    renderTasksView();
  }

  task.busy = true;
  task.busyAction = action;
  renderTaskActions(task);
  renderTaskRail(task);
  await streamTaskAdvance(task, action, content, qaEl);
  task.busy = false;
  task.busyAction = null;
  renderTasksView();
}

// «Выполнить всё сразу»: сервер последовательно выполняет каждый шаг плана
// (по одному запросу к LLM на шаг), клиент рисует каждый шаг по мере прихода
// событий step_run/reasoning/done и обновляет рельсу.
async function runAllTask(task) {
  if (!task || task.busy || !task.sid) return;
  if (!activeProfile()) {
    addTaskNotice(task, 'Необходимо создать и установить профиль.');
    return;
  }

  task.busy = true;
  task.busyAction = 'run_all';
  renderTaskActions(task);
  renderTaskRail(task);

  let current = null; // текущий шаг: {qaEl, el, content}
  const closeCurrentError = (message) => {
    if (!current) return;
    removeWaiter(current.content);
    setBubbleText(task, current.el, `Ошибка: ${message}`);
    current.el.classList.add('message--error');
    const last = task.history[task.history.length - 1];
    if (last && last.role === 'user') task.history.pop();
    renderQaStats(current.wrap, formatMetaLine(null));
    current = null;
  };

  try {
    logClient('send', `POST /api/tasks/${task.sid}/advance`, { action: 'run_all' });
    const res = await fetch(`/api/tasks/${task.sid}/advance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'run_all' })
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      const error = new Error(data.error || `Ошибка ${res.status}`);
      error.code = data.code || null;
      throw error;
    }

    for await (const chunk of parseSSE(res)) {
      logClient('receive', `POST /api/tasks/${task.sid}/advance — событие`, chunk);
      if (chunk.type === 'step_run') {
        // Без пользовательских маркеров: шаг выглядит как самостоятельный
        // ответ ассистента с подписью «Шаг N».
        const wrap = document.createElement('div');
        wrap.className = 'tasks__step';
        const label = document.createElement('div');
        label.className = 'tasks__step-label';
        label.textContent = `Шаг ${chunk.index + 1}`;
        const assistantEl = createMessageEl('assistant', '');
        wrap.append(label, assistantEl);
        task.messagesEl.appendChild(wrap);
        const content = assistantEl.querySelector('.message__content');
        showWaiter(content);
        scrollChatToBottom(task);
        current = { wrap, el: assistantEl, content };
        renderTaskRail(task);
      } else if (chunk.type === 'reasoning_start' && current) {
        createReasoning(current.el);
        showWaiter(current.content);
        scrollChatToBottom(task);
      } else if (chunk.type === 'reasoning_end' && current) {
        const thinking = chunk.content || '';
        if (thinking) setReasoningText(current.el, thinking);
        finishReasoning(current.el);
        showWaiter(current.content);
        scrollChatToBottom(task);
      } else if (chunk.type === 'done' && current) {
        const fullText = chunk.content || '';
        const meta = chunk.meta || null;
        removeWaiter(current.content);
        setBubbleText(task, current.el, fullText || emptyResponseText(meta, null));
        task.history.push({ role: 'assistant', content: fullText, meta });
        renderQaStats(current.wrap, formatMetaLine(meta));
        current = null;
      } else if (chunk.type === 'request_log') {
        handleTaskRequestLog(task, chunk.record);
      } else if (chunk.type === 'stage') {
        applyTaskProgress(task, chunk);
        renderTaskRail(task);
      } else if (chunk.type === 'error') {
        const error = new Error(chunk.error || 'Неизвестная ошибка сервера');
        error.code = chunk.code || null;
        throw error;
      }
    }
  } catch (err) {
    if (current) {
      closeCurrentError(err.message);
    } else {
      addTaskNotice(task, `Ошибка: ${err.message}`);
    }
  } finally {
    task.busy = false;
    task.busyAction = null;
    renderTasksView();
  }
}

// Действия протокола без обращения к LLM (JSON-ответ с прогрессом).
async function taskActionJson(task, action, extra) {
  if (!task || task.busy || !task.sid) return;
  const payload = { action };
  if (extra) Object.assign(payload, extra);
  try {
    logClient('send', `POST /api/tasks/${task.sid}/advance`, payload);
    const res = await fetch(`/api/tasks/${task.sid}/advance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Ошибка ${res.status}`);
    }
    const data = await res.json();
    logClient('receive', `POST /api/tasks/${task.sid}/advance → ${res.status}`, data);
    applyTaskProgress(task, data);
  } catch (err) {
    addTaskNotice(task, `Ошибка: ${err.message}`);
  }
  renderTasksView();
}

// Принятие шага: если остались шаги — сразу выполняем следующий (SSE),
// иначе финализируем задачу (JSON-переход в review со склейкой результатов).
async function confirmStepTask(task) {
  if (!task || task.busy || !task.sid) return;
  const remaining = (task.steps || []).length - (task.step_results || []).length;
  if (remaining <= 0) {
    await taskActionJson(task, 'confirm_step');
    return;
  }
  if (!activeProfile()) {
    addTaskNotice(task, 'Необходимо создать и установить профиль.');
    return;
  }
  const index = (task.step_results || []).length;
  const marker = `Выполни шаг ${index + 1}.`;
  task.history.push({ role: 'user', content: marker });
  const qaEl = addTaskQa(task, marker);
  task.busy = true;
  task.busyAction = 'confirm_step';
  renderTaskActions(task);
  renderTaskRail(task);
  await streamTaskAdvance(task, 'confirm_step', null, qaEl);
  task.busy = false;
  task.busyAction = null;
  renderTasksView();
}

// Правка шага плана: если шаг уже выполнен — откат и его повторный запуск (SSE),
// иначе шаг просто обновляется (JSON; стадия и прогресс не меняются).
async function editStepTask(task, index, content) {
  if (!task || task.busy || !task.sid) return;
  const rollback = index < (task.step_results || []).length;
  if (!rollback) {
    await taskActionJson(task, 'edit_step', { index, content });
    return;
  }
  if (!activeProfile()) {
    addTaskNotice(task, 'Необходимо создать и установить профиль.');
    return;
  }
  const marker = `Выполни шаг ${index + 1}.`;
  task.history.push({ role: 'user', content: marker });
  const qaEl = addTaskQa(task, marker);
  // Оптимистичный откат: последующие шаги сразу помечаются «не выполнено»,
  // правленый шаг получает спиннер (вейтер).
  const backup = {
    results: [...(task.step_results || [])],
    result: task.result
  };
  task.step_results = (task.step_results || []).slice(0, index);
  task.result = null;
  task.busy = true;
  task.busyAction = 'edit_step';
  renderTaskActions(task);
  renderTaskRail(task);
  const ok = await streamTaskAdvance(task, 'edit_step', null, qaEl, { index, content });
  if (!ok) {
    // Сбой перезапуска — возвращаем прежний прогресс.
    task.step_results = backup.results;
    task.result = backup.result;
  }
  task.busy = false;
  task.busyAction = null;
  renderTasksView();
}

// Синхронизирует этап и шаговый прогресс из ответа сервера.
function applyTaskProgress(task, data) {
  if (!data) return;
  if (typeof data.stage === 'string') task.stage = data.stage;
  if (Array.isArray(data.steps)) task.steps = data.steps;
  if (Array.isArray(data.step_results)) task.step_results = data.step_results;
  if (data.result !== undefined) task.result = data.result;
}

async function streamTaskAdvance(task, action, content, qaEl, extra) {
  const assistantEl = createMessageEl('assistant', '');
  qaEl.appendChild(assistantEl);
  const assistantContent = assistantEl.querySelector('.message__content');
  showWaiter(assistantContent);
  scrollChatToBottom(task);
  let full = '';
  let thinkingText = '';
  let finalMeta = null;

  const payload = { action };
  if (content != null) payload.content = content;
  if (extra) Object.assign(payload, extra);
  if (action === 'describe') {
    payload.model = currentModel();
    payload.settings = collectSettings();
  }

  try {
    logClient('send', `POST /api/tasks/${task.sid}/advance`, payload);
    const res = await fetch(`/api/tasks/${task.sid}/advance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      const error = new Error(data.error || `Ошибка ${res.status}`);
      error.code = data.code || null;
      throw error;
    }

    for await (const chunk of parseSSE(res)) {
      logClient('receive', `POST /api/tasks/${task.sid}/advance — событие`, chunk);
      if (chunk.type === 'reasoning_start') {
        createReasoning(assistantEl);
        showWaiter(assistantContent);
        scrollChatToBottom(task);
      } else if (chunk.type === 'reasoning_end') {
        thinkingText = chunk.content || '';
        if (thinkingText) setReasoningText(assistantEl, thinkingText);
        finishReasoning(assistantEl);
        showWaiter(assistantContent);
        scrollChatToBottom(task);
      } else if (chunk.type === 'done') {
        finalMeta = chunk.meta || null;
        full = chunk.content || '';
      } else if (chunk.type === 'request_log') {
        handleTaskRequestLog(task, chunk.record);
      } else if (chunk.type === 'stage') {
        applyTaskProgress(task, chunk);
      } else if (chunk.type === 'error') {
        const error = new Error(chunk.error || 'Неизвестная ошибка сервера');
        error.code = chunk.code || null;
        throw error;
      }
    }

    removeWaiter(assistantContent);
    setBubbleText(task, assistantEl, full || emptyResponseText(finalMeta, null));
    task.history.push({ role: 'assistant', content: full, meta: finalMeta });
    if (action === 'describe' || action === 'revise') task.plan = full;
    else if (action === 'run_all') task.result = full;
    renderQaStats(qaEl, formatMetaLine(finalMeta));
    return true;
  } catch (err) {
    removeWaiter(assistantContent);
    setBubbleText(task, assistantEl, `Ошибка: ${err.message}`);
    assistantEl.classList.add('message--error');
    const last = task.history[task.history.length - 1];
    if (last && last.role === 'user') task.history.pop();
    renderQaStats(qaEl, formatMetaLine(null));
    return false;
  }
}

function handleTaskRequestLog(task, record) {
  if (!task || !record) return;
  task.requests.push(record);
  const total = (record.prompt_tokens || 0) + (record.completion_tokens || 0);
  if (total > 0) {
    tokensBurned += total;
    renderTotal();
  }
}

function renderFactsPanel(chat) {
  if (!factsList || !factsEmpty) return;
  const facts = (chat && chat.facts) || [];
  factsList.innerHTML = '';
  for (const fact of facts) {
    const li = document.createElement('li');
    if (Array.isArray(fact) && fact.length === 2) {
      const key = document.createElement('span');
      key.className = 'chat__facts-key';
      key.textContent = `${fact[0]}:`;
      const value = document.createElement('span');
      value.textContent = ` ${fact[1]}`;
      li.append(key, value);
    } else {
      li.textContent = String(fact);
    }
    factsList.appendChild(li);
  }
  factsList.hidden = facts.length === 0;
  factsEmpty.hidden = facts.length > 0;
}

function factsWidthBounds() {
  const total = chatColumn ? chatColumn.clientWidth : 0;
  return { min: 180, max: Math.max(180, total - 260) };
}

function applyFactsWidth() {
  if (!factsPanel) return;
  let width = NaN;
  try {
    width = parseInt(localStorage.getItem(FACTS_PANEL_WIDTH_KEY), 10);
  } catch {
    width = NaN;
  }
  if (!Number.isInteger(width) || width <= 0) return;
  const { min, max } = factsWidthBounds();
  factsPanel.style.width = Math.min(Math.max(width, min), max) + 'px';
}

if (factsSplitter) {
  factsSplitter.addEventListener('mousedown', (event) => {
    event.preventDefault();
    factsSplitter.classList.add('chat__splitter--active');
    const onMove = (moveEvent) => {
      const rect = chatColumn.getBoundingClientRect();
      const { min, max } = factsWidthBounds();
      const width = Math.min(Math.max(rect.right - moveEvent.clientX, min), max);
      factsPanel.style.width = width + 'px';
    };
    const onUp = () => {
      factsSplitter.classList.remove('chat__splitter--active');
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      try {
        localStorage.setItem(
          FACTS_PANEL_WIDTH_KEY,
          String(parseInt(factsPanel.style.width, 10) || 280)
        );
      } catch {
        /* localStorage может быть недоступен */
      }
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

async function apiBranchSession(sid, title) {
  logClient('send', `POST /api/sessions/${sid}/branch`, { title });
  const res = await fetch(`/api/sessions/${sid}/branch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || `Ошибка ${res.status}`);
  }
  const data = await res.json();
  logClient('receive', `POST /api/sessions/${sid}/branch → 201`, data);
  return data;
}

async function apiGetSession(sid) {
  const res = await fetch(`/api/sessions/${sid}`);
  if (!res.ok) throw new Error(`Ошибка ${res.status}`);
  return res.json();
}

function makeChatFromSession(session) {
  const chat = {
    id: `chat-${++chatCounter}`,
    sid: session.id,
    title: session.title || sessionTitle(session),
    parentId: session.parent_id || null,
    history: [],
    requests: Array.isArray(session.requests)
      ? session.requests.map((r) => ({ ...r }))
      : [],
    facts: Array.isArray(session.facts) ? [...session.facts] : [],
    // Память чата: после перезагрузки восстанавливаются только
    // персистентные вкладки (остальные жили до перезагрузки страницы).
    memoryStores: (Array.isArray(session.memory) ? session.memory : [])
      .filter((store) => store && store.persistent)
      .map((store) => ({
        id: store.id,
        name: store.name,
        persistent: true,
        items: Array.isArray(store.items) ? store.items.map((item) => [...item]) : []
      })),
    memoryActiveId: null,
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
  renderRestoredHistory(chat, session);
  return chat;
}

async function branchActiveChat() {
  const chat = getActiveChat();
  if (!chat || !chat.sid || chat.busy) return;
  if (branchBtn) branchBtn.disabled = true;
  try {
    const created = await apiBranchSession(chat.sid, chat.title);
    const session = await apiGetSession(created.id);
    const newChat = makeChatFromSession(session);
    chats.push(newChat);
    renderTabs();
    activateChat(newChat);
  } catch (err) {
    addMessage(chat, 'assistant', `Ошибка ветвления: ${err.message}`);
  } finally {
    updateChatLayout();
  }
}

function branchIcon() {
  const span = document.createElement('span');
  span.className = 'tabs__branch-icon';
  span.innerHTML = BRANCH_ICON_SVG;
  return span;
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

function scrollChatToBottom(chat) {
  // Прокручивает окно чата вниз (к вейтеру/размышлению/новому ответу).
  if (chat && chat.messagesEl) {
    chat.messagesEl.scrollTop = chat.messagesEl.scrollHeight;
  }
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

// Сворачиваемые блоки вызовов MCP-инструментов в бабле ассистента. Блоки
// живут только в live-стриме (в историю чата не персистятся).
function appendToolCall(el, tool, args) {
  el.classList.remove('message--empty');
  const block = document.createElement('details');
  block.className = 'message__tool';
  const summary = document.createElement('summary');
  summary.className = 'message__tool-summary';
  summary.textContent = `Инструмент ${tool}`;
  block.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'message__tool-body';
  const argsEl = document.createElement('pre');
  argsEl.className = 'message__tool-args';
  argsEl.textContent = JSON.stringify(args ?? {}, null, 2);
  body.appendChild(argsEl);
  const resultEl = document.createElement('pre');
  resultEl.className = 'message__tool-result';
  resultEl.hidden = true;
  body.appendChild(resultEl);
  block.appendChild(body);

  const content = el.querySelector('.message__content');
  if (content) el.insertBefore(block, content);
  else el.appendChild(block);

  if (!el._toolBlocks) el._toolBlocks = new Map();
  const stack = el._toolBlocks.get(tool) || [];
  stack.push({ block, resultEl });
  el._toolBlocks.set(tool, stack);
}

function appendToolResult(el, tool, content, isError) {
  const stack = el._toolBlocks ? el._toolBlocks.get(tool) : null;
  const entry = stack && stack.length ? stack[stack.length - 1] : null;
  if (!entry) {
    appendToolCall(el, tool, {});
    appendToolResult(el, tool, content, isError);
    return;
  }
  entry.resultEl.hidden = false;
  entry.resultEl.textContent = content || '(пустой результат)';
  entry.resultEl.classList.toggle('message__tool-result--error', !!isError);
  entry.block.classList.add('message__tool--done');
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
  scrollChatToBottom(chat);
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
      const error = new Error(data.error || `Ошибка ${res.status}`);
      error.code = data.code || null;
      throw error;
    }

    for await (const chunk of parseSSE(res)) {
      logClient('receive', `POST /api/sessions/${chat.sid}/messages — событие`, chunk);
      if (chunk.type === 'reasoning_start') {
        createReasoning(assistantEl);
        showWaiter(assistantContent);
        scrollChatToBottom(chat);
      } else if (chunk.type === 'reasoning_end') {
        thinkingText = chunk.content || '';
        if (thinkingText) setReasoningText(assistantEl, thinkingText);
        finishReasoning(assistantEl);
        showWaiter(assistantContent);
        scrollChatToBottom(chat);
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
      } else if (chunk.type === 'tool_call') {
        appendToolCall(assistantEl, chunk.tool, chunk.args);
        scrollChatToBottom(chat);
      } else if (chunk.type === 'tool_result') {
        appendToolResult(assistantEl, chunk.tool, chunk.content, chunk.is_error);
        scrollChatToBottom(chat);
      } else if (chunk.type === 'facts') {
        chat.facts = Array.isArray(chunk.items) ? chunk.items : [];
        if (chat.id === activeChatId) renderFactsPanel(chat);
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
    } else if (err.code === 'profile_required') {
      qaEl.remove();
      renderProfileRequired(chat, err.message);
    } else {
      setBubbleText(chat, assistantEl, `Ошибка: ${err.message}`);
      assistantEl.classList.add('message--error');
    }
    if (chat.history[chat.history.length - 1].role === 'user') chat.history.pop();
    if (err.code !== 'profile_required') renderQaStats(qaEl, formatMetaLine(null));
  } finally {
    chat.busy = false;
    updateSendButton();
    updateSettingsLock();
    renderChatInfo();
    updateChatLayout();
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
        },
        {
          label: 'Факты',
          data: [],
          backgroundColor: '#a06be8',
          borderColor: '#a06be8',
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
              if (item.datasetIndex === 2) {
                const reasoning =
                  seg.summaryReasoning > 0
                    ? ` (рассуждение: ${formatTokens(seg.summaryReasoning)})`
                    : '';
                return `саммаризация: ${formatTokens(seg.summary)} ток.${reasoning}`;
              }
              const reasoning =
                seg.factsReasoning > 0
                  ? ` (рассуждение: ${formatTokens(seg.factsReasoning)})`
                  : '';
              return `факты: ${formatTokens(seg.facts)} ток.${reasoning}`;
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
  // основному; факты (идут после ответа) — к текущему; осиротевшая
  // саммаризация (основной запрос упал) образует ход без основного.
  const turns = [];
  let pending = [];
  for (const record of records) {
    if (record.kind === 'summary') {
      pending.push(record);
    } else if (record.kind === 'facts') {
      if (turns.length > 0) turns[turns.length - 1].facts.push(record);
      else turns.push({ main: null, summaries: [], facts: [record] });
    } else {
      turns.push({ main: record, summaries: pending, facts: [] });
      pending = [];
    }
  }
  if (pending.length > 0) {
    turns.push({ main: null, summaries: pending, facts: [] });
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
  let facts = 0;
  let factsReasoning = 0;
  for (const r of turn.facts || []) {
    facts += (r.prompt_tokens || 0) + (r.completion_tokens || 0);
    factsReasoning += r.reasoning_tokens || 0;
  }
  return {
    input,
    output,
    reasoning,
    summary,
    summaryReasoning,
    facts,
    factsReasoning,
    total: input + output + summary + facts
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
    chart.data.datasets[3].data = segments.map((s) => (s.facts > 0 ? s.facts : null));
  } else {
    // Моно-режим: один столбик — только суммарные токены сообщения.
    chart.data.datasets[0].label = 'Всего';
    chart.data.datasets[0].data = segments.map((s) => (s.total > 0 ? s.total : null));
    chart.data.datasets[1].data = segments.map(() => null);
    chart.data.datasets[2].data = segments.map(() => null);
    chart.data.datasets[3].data = segments.map(() => null);
  }
  if (chartLegend) chartLegend.hidden = !chartColorMode;
  const maxTotal = segments.reduce((m, s) => Math.max(m, s.total), 0);
  chart.options.scales.y.suggestedMax = maxTotal > 0 ? Math.ceil(maxTotal * 1.15) : 10;
  chart.$contextLimit = chat && chat.model ? modelContext(chat.model) : null;
  chart.$turns = segments;
  chart.$colorMode = chartColorMode;
  chart.update();
}

function applyChartCollapsed() {
  if (chartbarEl) chartbarEl.classList.toggle('chartbar--collapsed', chartCollapsed);
  if (chartCollapseBtn) {
    chartCollapseBtn.setAttribute('aria-expanded', chartCollapsed ? 'false' : 'true');
  }
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

  // Профиль обязателен для обычных чатов — любое сообщение без него
  // блокируется (данные профиля идут в системный промпт).
  if (chat.kind === 'chat' && !activeProfile()) {
    renderProfileRequired(chat, 'Необходимо создать и установить профиль.');
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

for (const radio of strategyRadios) {
  radio.addEventListener('change', () => {
    if (!radio.checked) return;
    updateStrategyParamUI();
    syncActiveChatSettings();
    updateChatLayout();
  });
}

if (strategyParamInput) {
  strategyParamInput.addEventListener('input', () => {
    const strategy = currentStrategy();
    const fallback = strategyParams[strategy] || 5;
    strategyParams[strategy] = clampStrategyParam(
      parseInt(strategyParamInput.value, 10),
      fallback
    );
    syncActiveChatSettings();
    updateStrategyHint();
  });
}

if (branchBtn) {
  branchBtn.innerHTML = BRANCH_ICON_SVG;
  branchBtn.addEventListener('click', branchActiveChat);
}

if (viewTabChat) {
  viewTabChat.addEventListener('click', () => switchView('chat'));
}

if (viewTabMemory) {
  viewTabMemory.addEventListener('click', () => switchView('memory'));
}

if (memoryDbToggle) {
  memoryDbToggle.innerHTML = DB_ICON_SVG;
  memoryDbToggle.addEventListener('click', () => {
    memoryDbPersistent = !memoryDbPersistent;
    memoryDbToggle.classList.toggle('memory__db--active', memoryDbPersistent);
    memoryDbToggle.setAttribute('aria-pressed', memoryDbPersistent ? 'true' : 'false');
  });
}

if (memoryAdd) {
  memoryAdd.addEventListener('click', createMemoryStore);
}

if (memoryNewName) {
  memoryNewName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      createMemoryStore();
    }
  });
}

if (viewTabProfile) {
  viewTabProfile.addEventListener('click', () => switchView('profile'));
}

if (profileAdd) {
  profileAdd.addEventListener('click', createProfile);
}

if (profileNewName) {
  profileNewName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      createProfile();
    }
  });
}

if (viewTabTasks) {
  viewTabTasks.addEventListener('click', () => switchView('tasks'));
}

if (tasksAdd) {
  tasksAdd.addEventListener('click', createTask);
}

if (viewTabRules) {
  viewTabRules.addEventListener('click', () => switchView('rules'));
}

if (rulesAdd) {
  rulesAdd.addEventListener('click', createRulesStore);
}

if (rulesNewName) {
  rulesNewName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      createRulesStore();
    }
  });
}

if (viewTabLogs) {
  viewTabLogs.addEventListener('click', () => switchView('logs'));
}

if (logsRefresh) {
  logsRefresh.addEventListener('click', loadLogs);
}

if (logsClear) {
  logsClear.addEventListener('click', clearLogs);
}

if (viewTabMcp) {
  viewTabMcp.addEventListener('click', () => switchView('mcp'));
}

if (mcpRefresh) {
  mcpRefresh.addEventListener('click', loadMcp);
}

if (mcpAdd) {
  mcpAdd.addEventListener('click', createMcpServer);
}

if (mcpNewName) {
  mcpNewName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      createMcpServer();
    }
  });
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

if (chartCollapseBtn) {
  chartCollapseBtn.addEventListener('click', () => {
    chartCollapsed = !chartCollapsed;
    try {
      localStorage.setItem(CHART_COLLAPSED_KEY, chartCollapsed ? '1' : '0');
    } catch {
      /* localStorage может быть недоступен */
    }
    applyChartCollapsed();
    if (!chartCollapsed) {
      if (contextChart) contextChart.resize();
      renderChart();
    }
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
applyChartCollapsed();
renderChart();
if (viewTabProfileIcon) viewTabProfileIcon.innerHTML = PROFILE_ICON_SVG;
if (viewTabLogsIcon) viewTabLogsIcon.innerHTML = LOGS_ICON_SVG;
if (viewTabMcpIcon) viewTabMcpIcon.innerHTML = MCP_ICON_SVG;
loadUiState();
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
  if (session.title) return session.title;
  const firstMsg = (session.history || []).find((m) => m.role === 'user' || m.role === 'system');
  return firstMsg ? firstMsg.content.replace(/\s+/g, ' ').trim() : 'Новый чат';
}

// Восстанавливает сохранённый вид и выбранные внутренние вкладки.
function applySavedView() {
  if (
    uiState.inner.profile &&
    profiles.some((p) => p.id === uiState.inner.profile)
  ) {
    profileActiveId = uiState.inner.profile;
  }
  const savedTask = tasks.find((t) => t.sid === uiState.inner.task);
  if (savedTask) activeTaskId = savedTask.id;
  if (
    uiState.inner.rules &&
    rules.some((s) => s.id === uiState.inner.rules)
  ) {
    activeRulesId = uiState.inner.rules;
  }

  const chat = getActiveChat();
  const storeId =
    chat && chat.sid && uiState.inner.memory
      ? uiState.inner.memory[chat.sid]
      : null;
  if (chat && storeId && (chat.memoryStores || []).some((s) => s.id === storeId)) {
    chat.memoryActiveId = storeId;
  }

  const known = ['chat', 'tasks', 'memory', 'rules', 'profile', 'logs', 'mcp'];
  switchView(known.includes(uiState.view) ? uiState.view : 'chat');
}

(async () => {
  let restoredSessions = [];
  let activeId = null;
  // Профили пользователя — до восстановления чатов (от них зависит отправка).
  await loadProfiles();
  // Задачи (протокол этапов) — тоже глобальные; восстанавливаем вкладки.
  await loadTasks();
  // Правила — глобальные, нужны для подмешивания в запросы.
  await loadRules();
  // MCP-серверы — глобальные: конфигурация и live-статус для вкладки.
  await loadMcp();
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
    applySavedView();
    return;
  }

  // Восстанавливаем все чаты как вкладки.
  let mostRecent = null;
  let mostRecentTs = -Infinity;
  for (const session of restoredSessions) {
    const chat = makeChatFromSession(session);
    chats.push(chat);
    // После перезагрузки страницы неперсистентные вкладки памяти не
    // восстанавливаются — пересинхронизируем состояние, чтобы устаревшие
    // вкладки исчезли и с сервера (из контекста запросов к LLM).
    if (Array.isArray(session.memory) && session.memory.some((s) => s && !s.persistent)) {
      syncMemory(chat);
    }
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
  applySavedView();
})();

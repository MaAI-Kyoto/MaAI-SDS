import { createMaaiPlot, createNodChart } from '/plot.js?v=ui25';
import { createAudioPlayer } from '/audio-player.js?v=ui1';
import { createStreamingPcm16Resampler } from '/mic-resampler.js?v=ui1';

// PyQt 版と同じ項目を Web に出す。画面定義は /api/ui/manifest（PyQt のウィジェット木）。
// メインは履歴/モジュール/開始終了。それ以外は左上「設定」のタブ。

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const MODULE_KEYS = ['asr', 'llm', 'tts', 'maai'];
const MODULE_LABELS = { asr: 'ASR', llm: 'LLM', tts: 'TTS', maai: 'MaAI' };
const MODULE_STATE_LABELS = {
  stopped: '停止', starting: '起動中', running: '稼働', error: '異常', stopping: '停止中',
};
const SKIP_WINDOW_IDS = new Set(['main', 'engine_manager_window', 'console_window']);

const EXTRA_STACK_MAPS = {
  llm_test_window: { 'LM Studio': 0, OpenAI: 1, Custom: 2 },
  audio_input_settings_window: { mic: 0, wav: 1, tcp: 2 },
};

const COMBO_STACK = {
  asr_engine_combo: 'asr_test_window',
  asr_engine_combo_settings: 'asr_test_window',
  llm_backend_combo: 'llm_test_window',
  audio_input_mode_combo: 'audio_input_settings_window',
};

const TTS_AUTO_STOP = new Set([
  'VOICEVOX', 'VOICEVOX-Core', 'Kokoro-RealtimeTTS', 'Qwen3-TTS-CPP', 'LFM2.5-Audio',
  'Cartesia-Realtime', 'AivisSpeech', 'Supertonic-3', 'Irodori-v4', 'Tsukasa',
]);

const TTS_VOICE_CONFIG_KEYS = {
  VOICEVOX: 'tts_voice_voicevox',
  'VOICEVOX-Core': 'tts_voice_voicevox_core',
  'Kokoro-RealtimeTTS': 'tts_voice_kokoro_realtimetts',
  'Cartesia-Realtime': 'tts_voice_cartesia',
  TEMPLATE: 'tts_voice_template',
  AivisSpeech: 'tts_voice_aivisspeech',
  'Supertonic-3': 'tts_voice_supertonic',
  'Irodori-v4': 'tts_voice_irodori',
};

const MMD_LIVE_CHECK_KEYS = new Set(['mmd_blink_enable']);
const MMD_LIVE_COMBO_KEYS = new Set(['mmd_blink_mode']);

const TTS_VOICE_DEFAULTS = {
  VOICEVOX: '1',
  'VOICEVOX-Core': '8',
  'Kokoro-RealtimeTTS': 'jf_alpha',
  'Qwen3-TTS-CPP': 'Vivian',
  'LFM2.5-Audio': '',
  'Cartesia-Realtime': 'f786b574-daa5-4673-aa0c-cbe3e8534c02',
  AivisSpeech: '888753761',
  'Supertonic-3': 'F1',
  'Irodori-v4': '明るく自然な日本語の声',
  Tsukasa: '',
  TEMPLATE: '1',
};

const TTS_VOICE_LABELS = {
  VOICEVOX: '話者ID（VOICEVOX）',
  'VOICEVOX-Core': '話者ID（VOICEVOX Core）',
  'Kokoro-RealtimeTTS': '音声ID（Kokoro-RealtimeTTS）',
  'Qwen3-TTS-CPP': '話者（Qwen3-TTS C++）',
  'LFM2.5-Audio': '声（LFM2.5-Audio は選べません）',
  'Cartesia-Realtime': '音声ID（Cartesia）',
  AivisSpeech: '話者スタイルID（AivisSpeech）',
  'Supertonic-3': 'プリセット音声（F1～F5／M1～M5）',
  'Irodori-v4': '声質キャプション（参照音声なし時）',
  Tsukasa: '音声（参照音声ファイルで指定）',
};

const OPEN_TAB_BY_ATTR = {
  settings_toggle_btn: 'settings_window',
  tts_test_toggle_btn: 'tts_test_window',
  asr_settings_toggle_btn: 'asr_test_window',
  llm_test_toggle_btn: 'llm_test_window',
  maai_test_toggle_btn: 'maai_test_window',
  obs_settings_toggle_btn: 'obs_settings_window',
  audio_input_settings_toggle_btn: 'audio_input_settings_window',
  audio_output_settings_toggle_btn: 'audio_output_settings_window',
  console_toggle_btn: 'console',
  engine_manager_toggle_btn: 'engine_manager_web',
  llm_s2s_test_open_btn: 'llm_s2s_test_window',
  template_auth_open_btn: 'tts_template_auth_window',
  maai_settings_open_btn: 'maai_settings_window',
  maai_nod_chart_open_btn: 'nod_motion_chart_window',
};

const TTS_BUTTON_ENGINE = {
  voicevox_launch_btn: 'VOICEVOX',
  voicevox_stop_btn: 'VOICEVOX',
  voicevox_core_launch_btn: 'VOICEVOX-Core',
  voicevox_core_stop_btn: 'VOICEVOX-Core',
  kokoro_realtimetts_launch_btn: 'Kokoro-RealtimeTTS',
  kokoro_realtimetts_stop_btn: 'Kokoro-RealtimeTTS',
  qwen_cpp_launch_btn: 'Qwen3-TTS-CPP',
  qwen_cpp_stop_btn: 'Qwen3-TTS-CPP',
  lfm2_audio_launch_btn: 'LFM2.5-Audio',
  lfm2_audio_stop_btn: 'LFM2.5-Audio',
  aivis_launch_btn: 'AivisSpeech',
  aivis_stop_btn: 'AivisSpeech',
  supertonic_launch_btn: 'Supertonic-3',
  supertonic_stop_btn: 'Supertonic-3',
  irodori_launch_btn: 'Irodori-v4',
  irodori_stop_btn: 'Irodori-v4',
  tsukasa_launch_btn: 'Tsukasa',
  tsukasa_stop_btn: 'Tsukasa',
};

const TTS_GROUP_ALWAYS = new Set(['TTS制御', 'TTS直接通信テスト']);

const state = {
  manifest: null,
  config: {},
  dirty: new Map(),
  mdfDirty: new Map(),
  mdfValues: {},
  mdfPath: '',
  mdfReadOnly: false,
  mdfReadOnlyReason: '',
  inputs: new Map(),
  inputNodes: new Map(),
  byAttr: new Map(),
  uiLoading: false,
  lastTtsEngine: '',
  lastLlmBackend: '',
  ttsVoiceByEngine: { ...TTS_VOICE_DEFAULTS },
  activeView: null,
  engines: null,
  modules: Object.fromEntries(MODULE_KEYS.map((k) => [k, { state: 'stopped', error: '' }])),
  latency: { asr: null, llm: null, tts: null },
  streamText: '',
  // 対話サーバー（dialog worker）の状態。running / starting / stopping / stopped / error
  dialogServer: { state: 'starting', error: '' },
  eventsOpen: false,
  chatItems: [],
  chatLastAssistant: '',
  chatPartial: '',
  voiceActive: false,
  llmMessages: [],
  llmTestStreaming: false,
  llmTestBlockStart: -1,
  llmTestFull: '',
  llmTestSegments: [],
  wavPlaying: false,
  runtime: {
    docker: false,
    browser_mic: true,
    mic_source: 'browser',
    local_mic_available: true,
    mmdagent: { launch_available: false, running: false, reason: '' },
  },
  micUsers: new Set(),
  mic: { stream: null, ctx: null, proc: null, source: null, ws: null, starting: false },
  plot: null,
  eventsSock: null,
};

function restoreVoiceMic(stateName) {
  if (!['running', 'starting'].includes(String(stateName || '').toLowerCase())) return;
  acquireBrowserMic('voice').catch((e) => {
    appendConsole(`[マイク] 再接続待ち: ${String(e?.message ?? e)}`);
  });
}

const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    });
    if (!r.ok) throw new Error(`${path}: ${r.status} ${(await r.json().catch(() => ({}))).detail ?? ''}`);
    return r.json();
  },
  async put(path, body) {
    const r = await fetch(path, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    });
    if (!r.ok) throw new Error(`${path}: ${r.status} ${(await r.json().catch(() => ({}))).detail ?? ''}`);
    return r.json();
  },
};

function nodeValue(node) {
  if (!node) return '';
  if (node.type === 'checkbox') return node.checked;
  // エンジンの一覧（1 行＝モデル（実行系））は、選んだ行のエンジンの値を返す。
  if (node.dataset?.engineChoice) return node.selectedOptions[0]?.dataset.engine ?? node.value;
  return node.value;
}

// エンジンの一覧で、engine に当たる行のうち、隠した欄の設定値（subs）も合う行を選ぶ。
function selectEngineChoice(node, engine) {
  const opts = [...node.options].filter((o) => o.dataset.engine === String(engine ?? ''));
  const subsMatch = (o) => JSON.parse(o.dataset.subs || '[]')
    .every((sub) => !sub.config_key || String(state.config[sub.config_key] ?? '') === String(sub.value));
  const pick = opts.find(subsMatch) || opts[0];
  if (pick) {
    node.value = pick.value;
  } else if (engine) {
    const opt = new Option(String(engine), String(engine));
    opt.dataset.engine = String(engine);
    node.append(opt);
    node.value = String(engine);
  }
}

function attrValue(attr, fallback = '') {
  const node = state.byAttr.get(attr);
  if (node) return nodeValue(node);
  return fallback;
}

function setAttrValue(attr, value) {
  const node = state.byAttr.get(attr);
  if (!node) return;
  if (node.type === 'checkbox') node.checked = Boolean(value);
  else if (node.dataset?.engineChoice) selectEngineChoice(node, value);
  else node.value = value == null ? '' : String(value);
}

function registerInput(key, node) {
  if (!key || !node) return;
  state.inputs.set(key, node);
  const list = state.inputNodes.get(key) || [];
  list.push(node);
  state.inputNodes.set(key, list);
}

function syncDuplicateInputs(key, value, source) {
  for (const n of state.inputNodes.get(key) || []) {
    if (n === source) continue;
    if (n.type === 'checkbox') n.checked = Boolean(value);
    else if (n.dataset?.engineChoice) selectEngineChoice(n, value);
    else n.value = value == null ? '' : String(value);
  }
}

function readInput(key) {
  const node = state.inputs.get(key);
  if (node) return String(nodeValue(node) ?? '').trim();
  return String(state.config[key] ?? '').trim();
}

function writeInput(key, value) {
  const text = value == null ? '' : String(value);
  const nodes = state.inputNodes.get(key) || [];
  for (const n of nodes) {
    if (n.type === 'checkbox') n.checked = Boolean(value);
    else n.value = text;
  }
  state.config[key] = nodes[0]?.type === 'checkbox' ? Boolean(value) : text;
}

function markDirty(key, value, node, scope) {
  if (scope === 'mdf') {
    state.mdfDirty.set(key, value);
    state.mdfValues[key] = value;
  } else {
    state.dirty.set(key, value);
  }
  node?.classList.add('field-dirty');
  const n = state.dirty.size + state.mdfDirty.size;
  $('#save-state').textContent = `未保存：${n}件`;
}

function collectAppConfigFromWidgets() {
  const out = { ...state.config };
  for (const [key, node] of state.inputs) {
    if (!node || node.readOnly) continue;
    if (key === 'audio_input_mic_device_index' && selectedWebMicSource() === 'browser') {
      // Browser MediaDeviceInfo ids are opaque strings and must not overwrite
      // the PortAudio integer used by the native worker/Qt controller.
      out.web_browser_mic_device_id = String(nodeValue(node) || '');
      continue;
    }
    out[key] = nodeValue(node);
  }
  for (const [k, v] of state.dirty) out[k] = v;
  return out;
}

async function flushAppConfig() {
  const payload = collectAppConfigFromWidgets();
  await api.post('/api/config', payload);
  state.config = payload;
  for (const [k] of state.dirty) state.inputs.get(k)?.classList.remove('field-dirty');
  state.dirty.clear();
}

async function postCommand(payload) {
  const kind = String(payload?.type || '');
  const skip = kind === 'interrupt' || kind === 'stop_voice_dialog' || kind === 'stop_module'
    || (kind === 'audio_input_send_command' && String(payload.command || '') !== 'wav_play');
  if (kind && !skip) {
    await flushAppConfig();
  }
  return api.post('/api/command', payload);
}

async function saveConfig() {
  const payload = collectAppConfigFromWidgets();
  if (!state.dirty.size) {
    const prev = JSON.stringify(state.config);
    if (prev === JSON.stringify(payload)) return;
  }
  $('#save-state').textContent = '保存中…';
  try {
    await api.post('/api/config', payload);
    state.config = payload;
    for (const [k] of state.dirty) state.inputs.get(k)?.classList.remove('field-dirty');
    state.dirty.clear();
    $('#save-state').textContent = state.mdfDirty.size ? `未保存MDF：${state.mdfDirty.size}件` : '保存しました';
    setTimeout(() => {
      if (!state.dirty.size && !state.mdfDirty.size) $('#save-state').textContent = '';
    }, 2500);
    applyEnginePages();
    state.plot?.setVisibilityFromConfig({ ...state.config, ...Object.fromEntries(state.dirty) });
  } catch (e) {
    $('#save-state').textContent = `保存失敗：${e.message}`;
  }
}

function currentValue(w) {
  if (w.scope === 'mdf' && w.config_key && w.config_key in state.mdfValues) return state.mdfValues[w.config_key];
  if (w.config_key && w.config_key in state.config) return state.config[w.config_key];
  return w.value;
}

function stackMap(winId) {
  return { ...(EXTRA_STACK_MAPS[winId] || {}), ...(state.manifest.stack_maps?.[winId] || {}) };
}

function ttsEnginesForGroup(title) {
  if (!title || TTS_GROUP_ALWAYS.has(title)) return null;
  const rules = [
    [/Qwen3-TTS C\+\+|Qwen3-TTS-CPP/, ['Qwen3-TTS-CPP']],
    [/LFM2\.5-Audio/, ['LFM2.5-Audio']],
    [/VOICEVOX EXE|^VOICEVOX実行ファイル/, ['VOICEVOX']],
    [/VOICEVOX-Core/, ['VOICEVOX-Core']],
    [/^TTS接続/, ['TEMPLATE']],
    [/^モデル設定/, ['Kokoro-RealtimeTTS']],
    [/Kokoro/, ['Kokoro-RealtimeTTS']],
    [/Cartesia/, ['Cartesia-Realtime']],
    [/AivisSpeech/, ['AivisSpeech']],
    [/Supertonic/, ['Supertonic-3']],
    [/Irodori/, ['Irodori-v4']],
    [/Tsukasa/, ['Tsukasa']],
    [/TEMPLATE/, ['TEMPLATE']],
  ];
  for (const [re, engines] of rules) {
    if (re.test(title)) return engines;
  }
  return null;
}

// PySide の json_config_bool と同じ解釈。古い設定の "false" などの文字列を
// Boolean() で読むと true になってしまう。
function configBool(value) {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') return value !== 0;
  if (value === null || value === undefined) return false;
  return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
}

function renderWidget(w) {
  const key = w.config_key;
  const scope = w.scope === 'mdf' ? 'mdf' : 'app';
  let node;
  switch (w.kind) {
    case 'line': {
      node = el('input');
      node.type = w.password ? 'password' : 'text';
      const num = w.numeric;
      if (num && !w.password) {
        // 数値の欄（gui/numeric_fields.py と同じ範囲）。空欄は意味があるので残す。
        node.type = 'number';
        node.min = String(num.min);
        node.max = String(num.max);
        node.step = num.decimals ? 'any' : '1';
        const range = `範囲：${num.min}〜${num.max}${num.note ? `（${num.note}）` : ''}`;
        node.title = w.tooltip ? `${w.tooltip}\n${range}` : range;
      }
      node.value = currentValue(w) ?? '';
      node.addEventListener('change', () => {
        if (num && node.value !== '') {
          let v = Number(node.value);
          if (Number.isFinite(v)) {
            v = Math.min(Math.max(v, num.min), num.max);
            if (!num.decimals) v = Math.round(v);
            if (String(v) !== node.value) node.value = String(v);
          }
        }
        if (key) markDirty(key, node.value, node, scope);
      });
      break;
    }
    case 'combo': {
      if (w.engine_choice) {
        // 1 行＝「モデル（実行系）」（Qt の gui/engine_choices.py）。保存するのはエンジンの値と、
        // 行が決める隠した欄の値（speech.cpp のモデル・Irodori の実行系など）。
        node = el('select');
        node.dataset.engineChoice = '1';
        // 実行系ごとの見出し（optgroup）の下に並べる（Qt は選べない見出し行）。
        let group = null;
        for (const item of w.items ?? []) {
          const opt = new Option(String(item.label ?? item.value), String(item.value));
          opt.dataset.engine = String(item.engine ?? item.value);
          opt.dataset.subs = JSON.stringify(item.subs ?? []);
          if (item.group) {
            if (!group || group.label !== item.group) {
              group = document.createElement('optgroup');
              group.label = item.group;
              node.append(group);
            }
            group.append(opt);
          } else {
            node.append(opt);
          }
        }
        selectEngineChoice(node, currentValue(w));
        node.addEventListener('change', () => {
          const opt = node.selectedOptions[0];
          const engine = opt?.dataset.engine ?? '';
          for (const sub of JSON.parse(opt?.dataset.subs || '[]')) {
            if (!sub.config_key) continue;
            state.config[sub.config_key] = sub.value;
            syncDuplicateInputs(sub.config_key, sub.value);
            markDirty(sub.config_key, sub.value, null, 'app');
          }
          if (key) {
            syncDuplicateInputs(key, engine, node);
            markDirty(key, engine, node, scope);
          }
          if (w.attr === 'tts_engine_combo') onTtsEngineChanged(engine);
          else onAsrEngineChanged(engine);
        });
        break;
      }
      node = el('select');
      // この環境で選ばせない値（CPU 構成での GPU 項目など）。サーバが
      // /api/ui/manifest の応答に足してくる。項目は残して押せなくするだけなので、
      // PyQt 版と同じ見え方になる。
      const blocked = state.manifest?.disabled_choices?.[w.attr] ?? [];
      for (const item of w.items ?? []) {
        const itemValue = typeof item === 'object' && item !== null ? item.value : item;
        const itemLabel = typeof item === 'object' && item !== null ? (item.label ?? itemValue) : item;
        const opt = new Option(String(itemLabel ?? ''), String(itemValue ?? ''));
        if (blocked.includes(itemValue) || blocked.includes(itemLabel)) {
          opt.disabled = true;
          opt.title = state.manifest?.disabled_choices_reason ?? '';
        }
        node.append(opt);
      }
      let v = currentValue(w);
      // 現在値が無効な項目だった場合（GPU 機で書いた設定を CPU 機で開いた等）は、
      // 選べる最初の項目へ寄せる。無効項目が選択されたまま残ると、保存で
      // その値が書き戻ってしまう。
      if (blocked.includes(String(v))) {
        const first = [...node.options].find((o) => !o.disabled);
        if (first) {
          v = first.value;
          if (key) markDirty(key, v, node, scope);
        }
      }
      if (v !== undefined && v !== null && ![...node.options].some((o) => o.value === String(v))) {
        node.append(new Option(String(v), String(v)));
      }
      node.value = String(v ?? '');
      node.addEventListener('change', () => {
        const changedKey = w.attr === 'audio_input_mic_device_combo'
          && selectedWebMicSource() === 'browser'
          ? 'web_browser_mic_device_id'
          : key;
        if (changedKey) {
          if (changedKey === key) syncDuplicateInputs(key, node.value, node);
          markDirty(changedKey, node.value, node, scope);
        }
        if (MMD_LIVE_COMBO_KEYS.has(key) && !state.uiLoading) saveConfig();
        if (w.attr === 'tts_engine_combo') {
          onTtsEngineChanged(node.value);
        } else if (w.attr === 'asr_engine_combo' || w.attr === 'asr_engine_combo_settings') {
          onAsrEngineChanged(node.value);
        } else if (w.attr === 'llm_backend_combo') {
          onLlmBackendChanged(node.value);
        } else if (w.attr === 'audio_input_mode_combo') {
          onAudioInputModeChanged(node.value);
        } else if (w.attr === 'web_mic_source_combo') {
          onWebMicSourceChanged(node.value);
        } else if (w.attr === 'audio_input_mic_device_combo') {
          persistIfUserChange();
        } else if (COMBO_STACK[w.attr]) {
          applyEnginePages();
        }
      });
      break;
    }
    case 'check': {
      const wrap = el('label', 'check-wrap');
      node = el('input');
      node.type = 'checkbox';
      node.checked = configBool(currentValue(w));
      node.addEventListener('change', () => {
        if (key) {
          syncDuplicateInputs(key, node.checked, node);
          markDirty(key, node.checked, node, scope);
          if (MMD_LIVE_CHECK_KEYS.has(key) && !state.uiLoading) saveConfig();
        }
        if (w.attr === 'asr_compare_mode_check') {
          applyAsrCompareVisibility();
          if (!state.uiLoading) saveConfig();
        }
      });
      wrap.append(node);
      if (w.text) wrap.append(el('span', null, ` ${w.text}`));
      if (key) registerInput(key, node);
      if (w.tooltip) wrap.title = w.tooltip;
      if (w.attr) {
        wrap.dataset.attr = w.attr;
        state.byAttr.set(w.attr, node);
      }
      return wrap;
    }
    case 'textarea': {
      node = el('textarea');
      node.rows = w.readonly ? 12 : 4;
      node.value = currentValue(w) ?? '';
      node.readOnly = Boolean(w.readonly);
      if (w.attr) node.dataset.attr = w.attr;
      if (!w.readonly) node.addEventListener('change', () => key && markDirty(key, node.value, node, scope));
      break;
    }
    case 'button': {
      node = el('button', null, w.text || '実行');
      node.type = 'button';
      if (w.tooltip) node.title = w.tooltip;
      if (w.attr) node.dataset.attr = w.attr;
      if (w.browse_target) node.dataset.browseTarget = w.browse_target;
      const ttsEng = TTS_BUTTON_ENGINE[w.attr];
      if (ttsEng) node.dataset.ttsEngine = ttsEng;
      node.addEventListener('click', (ev) => handleButton(w, node, ev));
      if (w.attr) state.byAttr.set(w.attr, node);
      return node;
    }
    default:
      return null;
  }
  if (w.tooltip) node.title = w.tooltip;
  if (w.attr) {
    node.dataset.attr = w.attr;
    state.byAttr.set(w.attr, node);
  }
  if (key) registerInput(key, node);
  return node;
}

function renderGroup(win, group) {
  const box = el('section', 'group');
  const stackIdx = group.stack_index;
  if (stackIdx !== undefined) box.dataset.stackIndex = String(stackIdx);
  if (win.id === 'tts_test_window') {
    const engines = ttsEnginesForGroup(group.title);
    if (engines) box.dataset.ttsEngines = engines.join('|');
  }
  const title = el('h3', null, stackTitle(win, group));
  title.addEventListener('click', () => box.classList.toggle('collapsed'));
  box.append(title);
  const body = el('div', 'group-body');
  // 「詳細設定」の行（row.detail）は、続いているものを 1 つのたためる枠にまとめる（Qt と同じ）。
  let detailBox = null;
  for (const row of group.rows) {
    const line = el('div', 'row');
    line.append(el('label', null, row.label || ''));
    const fields = el('div', 'row-fields');
    for (const w of row.widgets) {
      if (w.hidden) continue; // エンジンの一覧で選ぶようになった欄
      const node = renderWidget(w);
      if (node) fields.append(node);
    }
    if (!fields.children.length && row.widgets.some((w) => w.hidden)) continue;
    line.append(fields);
    if (row.detail) {
      if (!detailBox || detailBox.dataset.title !== row.detail) {
        detailBox = el('details', 'row-detail');
        detailBox.dataset.title = row.detail;
        detailBox.append(el('summary', null, row.detail));
        body.append(detailBox);
      }
      detailBox.append(line);
    } else {
      detailBox = null;
      body.append(line);
    }
  }
  box.append(body);
  return box;
}

function stackTitle(win, group) {
  if (group.stack_index === undefined) return group.title || '設定';
  const map = stackMap(win.id);
  const name = Object.keys(map).find((k) => map[k] === group.stack_index);
  if (name) return `エンジン設定：${name}`;
  return `設定 ${group.stack_index + 1}`;
}

function applyStackVisibility(winId, engineName) {
  const map = stackMap(winId);
  const view = document.getElementById(`view-${winId}`);
  if (!view) return;
  const keys = Object.keys(map);
  if (!keys.length) return;
  const active = map[engineName];
  for (const box of view.querySelectorAll('[data-stack-index]')) {
    box.style.display = Number(box.dataset.stackIndex) === active ? '' : 'none';
  }
}

function liveCombo(attr, key, fallback = '') {
  return String(attrValue(attr, '') || state.inputs.get(key)?.value || state.config[key] || fallback);
}

function applyTtsVisibility() {
  const current = liveCombo('tts_engine_combo', 'tts_engine');
  const view = document.getElementById('view-tts_test_window');
  if (!view) return;
  for (const box of view.querySelectorAll('[data-tts-engines]')) {
    const allowed = box.dataset.ttsEngines.split('|');
    box.style.display = allowed.includes(current) ? '' : 'none';
  }
  for (const node of view.querySelectorAll('[data-tts-engine]')) {
    node.style.display = node.dataset.ttsEngine === current ? '' : 'none';
  }
  for (const row of view.querySelectorAll('.row')) {
    const fields = [...row.querySelectorAll('.row-fields > *')];
    if (!fields.length) continue;
    row.style.display = fields.some((n) => n.style.display !== 'none') ? '' : 'none';
  }
  updateTtsVoiceLabel(current);
}

function applyAsrCompareVisibility() {
  const enabled = Boolean(state.byAttr.get('asr_compare_mode_check')?.checked);
  for (const attr of ['asr_engine_combo', 'asr_engine_combo_settings']) {
    const combo = state.byAttr.get(attr);
    if (combo) combo.disabled = enabled;
  }
  const compareRow = state.byAttr.get('asr_compare_top_engine_combo')?.closest('.row');
  const resultRow = state.byAttr.get('asr_result_edit')?.closest('.row');
  if (compareRow) compareRow.style.display = enabled ? '' : 'none';
  if (resultRow) resultRow.style.display = enabled ? 'none' : '';
}

function applyEnginePages() {
  const asr = liveCombo('asr_engine_combo', 'asr_engine');
  const llm = liveCombo('llm_backend_combo', 'llm_backend');
  const audio = liveCombo('audio_input_mode_combo', 'audio_input_mode');
  applyStackVisibility('asr_test_window', asr);
  applyStackVisibility('llm_test_window', llm);
  applyStackVisibility('audio_input_settings_window', audio);
  applyTtsVisibility();
  applyAsrCompareVisibility();
  renderModules();
}

function persistIfUserChange() {
  if (!state.uiLoading) saveConfig();
}

function initTtsVoiceMap(cfg) {
  state.ttsVoiceByEngine = { ...TTS_VOICE_DEFAULTS };
  for (const [engine, key] of Object.entries(TTS_VOICE_CONFIG_KEYS)) {
    const value = String(cfg?.[key] ?? '').trim();
    if (value) state.ttsVoiceByEngine[engine] = value;
  }
  const current = String(cfg?.tts_engine || '').trim();
  const live = String(cfg?.tts_voice || '').trim();
  if (current && live) state.ttsVoiceByEngine[current] = live;
}

function saveCurrentEngineVoice(engine) {
  if (!engine) return;
  const value = readInput('tts_voice');
  if (!value) return;
  state.ttsVoiceByEngine[engine] = value;
  const key = TTS_VOICE_CONFIG_KEYS[engine];
  if (key) state.config[key] = value;
}

function restoreEngineVoice(engine) {
  if (!engine) return;
  const value = String(state.ttsVoiceByEngine[engine] || '').trim();
  if (value && readInput('tts_voice') !== value) writeInput('tts_voice', value);
}

function updateTtsVoiceLabel(engine) {
  const node = state.byAttr.get('tts_voice_edit');
  const label = node?.closest('.row')?.querySelector('label');
  if (label) label.textContent = TTS_VOICE_LABELS[engine] || '話者／音声';
  // LFM2.5-Audio はモデルの声 1 つだけで選べない（Qt と同じく欄を無効にする）。
  if (node) node.disabled = engine === 'LFM2.5-Audio';
}

function fillIfEmpty(key, value) {
  if (!readInput(key)) writeInput(key, value);
}

function applyEngineDefaults(engine) {
  const host = readInput('tts_host');
  const port = readInput('tts_port');
  const basePath = readInput('tts_base_path');
  const voice = readInput('tts_voice');
  const model = readInput('tts_model');
  const knownPorts = new Set(['14500', '50021', '8880', '8000', '8080']);

  if (engine === 'VOICEVOX') {
    if (state.runtime?.docker && (!host || ['127.0.0.1', 'localhost', '::1'].includes(host.toLowerCase()))) {
      writeInput('tts_host', 'host.docker.internal');
    } else if (!host) {
      writeInput('tts_host', '127.0.0.1');
    }
    if (!port || (knownPorts.has(port) && port !== '50021')) writeInput('tts_port', '50021');
    if (!voice || voice === 'kyoto_shiori') writeInput('tts_voice', '1');
    if (basePath === '/v1') writeInput('tts_base_path', '');
    return;
  }
  if (engine === 'VOICEVOX-Core') {
    if (!voice || voice === 'kyoto_shiori') writeInput('tts_voice', '8');
    return;
  }
  if (engine === 'Kokoro-RealtimeTTS') {
    if (!model) writeInput('tts_model', 'kokoro');
    if (!voice || /^\d+$/.test(voice) || voice === 'kyoto_shiori' || voice === '1') {
      writeInput('tts_voice', 'jf_alpha');
    }
    return;
  }
  if (engine === 'Qwen3-TTS-CPP') {
    fillIfEmpty('qwen3_tts_cpp_host', '127.0.0.1');
    fillIfEmpty('qwen3_tts_cpp_port', '18081');
    fillIfEmpty('qwen3_tts_cpp_model_id', 'qwen3-tts-customvoice');
    fillIfEmpty('qwen3_tts_cpp_speaker', 'Vivian');
    fillIfEmpty('qwen3_tts_cpp_language', 'japanese');
    if (!voice) writeInput('tts_voice', 'Vivian');
    return;
  }
  if (engine === 'LFM2.5-Audio') {
    fillIfEmpty('lfm2_audio_host', '127.0.0.1');
    fillIfEmpty('lfm2_audio_port', '18082');
    fillIfEmpty('lfm2_audio_model_id', 'lfm2.5-audio-jp');
    fillIfEmpty('lfm2_audio_model_path', 'audio.cpp/models/LFM2.5-Audio-1.5B-JP-GGUF');
    fillIfEmpty('lfm2_audio_server_exe', 'audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe');
    return;
  }
  if (engine === 'Cartesia-Realtime') {
    fillIfEmpty('cartesia_tts_model_id', 'sonic-3');
    if (!voice) writeInput('tts_voice', 'f786b574-daa5-4673-aa0c-cbe3e8534c02');
    fillIfEmpty('cartesia_tts_sample_rate', '16000');
    fillIfEmpty('cartesia_tts_api_key_env', 'CARTESIA_API_KEY');
    return;
  }
  if (engine === 'AivisSpeech') {
    if (!/^\d+$/.test(voice)) writeInput('tts_voice', '888753761');
    return;
  }
  if (engine === 'Supertonic-3') {
    if (!['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'].includes(voice.toUpperCase())) {
      writeInput('tts_voice', 'F1');
    }
    return;
  }
  if (engine === 'Irodori-v4') {
    const upper = voice.toUpperCase();
    if (!voice || /^\d+$/.test(voice) || ['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'].includes(upper)) {
      writeInput('tts_voice', '明るく自然な日本語の声');
    }
  }
}

async function autoStopTtsEngine(engine) {
  if (!TTS_AUTO_STOP.has(engine)) return;
  appendConsole(`エンジン切替により ${engine} TTS を停止`);
  const info = state.modules.tts || { state: 'stopped' };
  if (['running', 'starting'].includes(info.state)) {
    state.modules.tts = { ...info, state: 'stopping', error: '' };
    renderModules();
  }
  try {
    await postCommand({ type: 'stop_module', module: 'tts', engine });
  } catch (e) {
    appendConsole(`エンジン切替時の自動停止に失敗: ${engine} / ${e.message || e}`);
  }
}

function onAsrEngineChanged(engine) {
  syncDuplicateInputs('asr_engine', engine);
  state.config.asr_engine = engine;
  applyEnginePages();
  persistIfUserChange();
}

async function onTtsEngineChanged(engine) {
  const previous = state.lastTtsEngine;
  if (previous && previous !== engine) {
    saveCurrentEngineVoice(previous);
    if (!state.uiLoading) await autoStopTtsEngine(previous);
  }
  syncDuplicateInputs('tts_engine', engine);
  state.config.tts_engine = engine;
  restoreEngineVoice(engine);
  applyEngineDefaults(engine);
  saveCurrentEngineVoice(engine);
  state.lastTtsEngine = engine;
  applyEnginePages();
  persistIfUserChange();
}

function onLlmBackendChanged(backend) {
  syncDuplicateInputs('llm_backend', backend);
  state.config.llm_backend = backend;
  state.lastLlmBackend = backend;
  applyEnginePages();
  if (state.uiLoading) return;
  saveConfig().then(() => {
    if (backend === 'LM Studio') refreshLlmModels('lmstudio').catch(() => {});
    else if (backend === 'OpenAI') refreshLlmModels('openai').catch(() => {});
    return postCommand({ type: 'start_llm_warmup', force: true });
  }).catch(() => {});
}

function onAudioInputModeChanged(mode) {
  syncDuplicateInputs('audio_input_mode', mode);
  state.config.audio_input_mode = mode;
  if (String(mode).toLowerCase() !== 'mic') stopBrowserMic();
  else if (state.micUsers.size && selectedWebMicSource() === 'browser') {
    startBrowserMic(true).catch(showError);
  }
  applyEnginePages();
  persistIfUserChange();
}

// この PC のマイクを使えないとき（コンテナの中の Web サーバー、または Docker の対話サーバーを使うとき）。
function localMicUnavailable() {
  return Boolean(state.runtime?.docker || state.runtime?.local_mic_available === false);
}

function onWebMicSourceChanged(source) {
  const normalized = localMicUnavailable() ? 'browser' : (source === 'local' ? 'local' : 'browser');
  state.config.web_mic_source = normalized;
  const combo = state.byAttr.get('web_mic_source_combo');
  if (combo && combo.value !== normalized) combo.value = normalized;
  stopBrowserMic();
  refreshMicDevices().then(() => {
    if (state.micUsers.size && usesBrowserMic()) startBrowserMic(true).catch(showError);
    persistIfUserChange();
  }).catch(() => persistIfUserChange());
}

const wavMonitorAudio = createAudioPlayer();

let wavMonitorWs = null;
let wavMonitorWsTimer = 0;

function wavMonitorEnabled() {
  const node = state.byAttr.get('audio_input_wav_local_playback_check');
  if (node && node.type === 'checkbox') return Boolean(node.checked);
  const v = state.config.audio_input_wav_local_playback;
  if (v === false || v === 'false' || v === 0 || v === '0') return false;
  return true;
}

function connectWavMonitorWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/wav-monitor`;
  if (wavMonitorWs && (wavMonitorWs.readyState === WebSocket.OPEN || wavMonitorWs.readyState === WebSocket.CONNECTING)) {
    return;
  }
  try {
    const sock = new WebSocket(url);
    wavMonitorWs = sock;
    sock.binaryType = 'arraybuffer';
    sock.onmessage = (ev) => {
      if (!wavMonitorEnabled() || !state.wavPlaying) return;
      if (ev.data instanceof ArrayBuffer) {
        wavMonitorAudio.enqueuePcm16(ev.data);
      } else if (ev.data instanceof Blob) {
        ev.data.arrayBuffer().then((buf) => {
          if (wavMonitorEnabled() && state.wavPlaying) wavMonitorAudio.enqueuePcm16(buf);
        });
      }
    };
    sock.onclose = () => {
      if (wavMonitorWs === sock) wavMonitorWs = null;
      window.clearTimeout(wavMonitorWsTimer);
      wavMonitorWsTimer = window.setTimeout(connectWavMonitorWs, 1500);
    };
    sock.onerror = () => {
      try { sock.close(); } catch { /* ignore */ }
    };
  } catch {
    window.clearTimeout(wavMonitorWsTimer);
    wavMonitorWsTimer = window.setTimeout(connectWavMonitorWs, 1500);
  }
}

function resetInputWavMonitor() {
  wavMonitorAudio.reset();
}

function restoreWavPlayback(playing) {
  state.wavPlaying = Boolean(playing);
  setAttrButtonText('audio_input_wav_play_pause_btn', state.wavPlaying ? '一時停止' : '再生');
  resetInputWavMonitor();
  renderModules();
  if (!state.wavPlaying || !wavMonitorEnabled()) return;
  connectWavMonitorWs();
  // 同一オリジンで既に許可された再生はリロード後もここで再開できる。
  // ブラウザがユーザー操作を要求する場合は、既存のpointerdown解除処理が引き継ぐ。
  wavMonitorAudio.ensureContext().catch((e) => {
    appendConsole(`[WAVモニター] 再生許可待ち: ${String(e?.message ?? e)}`);
  });
}

async function armWavMonitorPlayback() {
  connectWavMonitorWs();
  // クリック同一ターンで resume しないとブラウザが無音にする。
  await wavMonitorAudio.ensureContext();
  wavMonitorAudio.reset();
  const sock = wavMonitorWs;
  if (!sock) throw new Error('モニター用の接続を開けませんでした');
  if (sock.readyState === WebSocket.OPEN) return;
  await new Promise((resolve, reject) => {
    const t = window.setTimeout(() => reject(new Error('モニター用の接続がタイムアウトしました')), 2500);
    sock.addEventListener('open', () => {
      window.clearTimeout(t);
      resolve();
    }, { once: true });
    sock.addEventListener('close', () => {
      window.clearTimeout(t);
      reject(new Error('モニター用の接続に失敗しました'));
    }, { once: true });
  });
}

function unlockBrowserAudio() {
  wavMonitorAudio.ensureContext().catch(() => {});
  connectWavMonitorWs();
}

function renderSettingsView(win) {
  const view = el('div', 'view');
  for (const g of win.groups) view.append(renderGroup(win, g));
  if (win.id === 'nod_motion_chart_window') {
    const hint = el('p', 'hint', '最新1回分（音声対話でnod_paraが発火したときに更新されます）');
    const canvas = el('canvas', 'nod-chart');
    canvas.id = 'nod-chart-canvas';
    view.append(hint, canvas);
    requestAnimationFrame(() => { state.nodChart = createNodChart(canvas); });
  } else if (!win.groups.length) {
    view.append(el('p', 'hint', 'この画面には設定項目がありません。'));
  }
  if (win.id === 'obs_settings_window') {
    view.prepend(el('div', 'notice',
      'DockerからホストのOBSへ接続する場合は、ホストにhost.docker.internalを指定してください。'));
  }
  if (win.id === 'llm_test_window') {
    view.prepend(el('div', 'notice',
      '応答は下のチャットと、メイン画面の「LLMストリーミング」に表示されます。DockerからホストのLM Studioへ接続する場合は、ホストをhttp://host.docker.internal:1234にしてください。127.0.0.1はコンテナ自身を指すため接続できません。'));
  }
  if (win.id === 'audio_input_settings_window') {
    view.prepend(el('div', 'notice',
      'Webマイクはブラウザ経由で取り込みます。Docker内のPortAudioからホストのマイクは見えないため、音声対話の開始時にブラウザのマイク許可が必要です。入力モードがwav／tcpのときは使いません。'));
  }
  return view;
}

function renderConsole() {
  const view = el('div', 'view');
  const bar = el('div', 'toolbar');
  const clear = el('button', null, 'クリア');
  bar.append(clear);
  const ta = el('textarea', 'console');
  ta.readOnly = true;
  ta.id = 'console-log';
  clear.addEventListener('click', () => { ta.value = ''; });
  view.append(bar, ta);
  return view;
}

async function renderEngineManager() {
  const view = el('div', 'view');
  const bar = el('div', 'toolbar');
  const applyBtn = el('button', 'primary', '適用');
  bar.append(applyBtn);
  const table = el('table', 'engine-table');
  const help = el('p', 'hint',
    'チェックしたエンジンを「適用」で入れます。チェックを外したエンジンは、Pythonパッケージを外します'
    + '（モデルなどのファイルは残ります）。容量はモデルなどの取得量の目安で、Pythonパッケージは含みません。');
  const summary = el('p', 'hint', '');
  const log = el('textarea', 'console'); log.readOnly = true; log.style.height = '16rem';
  view.append(bar, help, table, summary, log);
  let lastPlan = { install: [], remove: [], download_gb: 0 };

  const checkedIds = () => [...table.querySelectorAll('input[type=checkbox]')]
    .filter((c) => c.checked).map((c) => c.dataset.engine);

  // 各行の状態と合計を、Qt と同じ計算（engine_manager.plan_engine_changes）で出す。
  const updatePlan = async () => {
    lastPlan = await api.post('/api/engines/plan', { ids: checkedIds() });
    const install = new Set(lastPlan.install);
    const remove = new Set(lastPlan.remove);
    for (const e of state.engines.engines) {
      const tag = table.querySelector(`[data-status="${e.id}"]`);
      if (!tag) continue;
      let text = e.installed ? 'インストール済み' : '未インストール';
      let cls = e.installed ? 'tag on' : 'tag';
      if (install.has(e.id)) { text = '適用で入れる'; cls = 'tag add'; }
      else if (remove.has(e.id)) { text = '適用で外す'; cls = 'tag remove'; }
      tag.textContent = text;
      tag.className = cls;
    }
    const parts = [];
    if (install.size) parts.push(`入れる：${install.size} 件（取得 約${lastPlan.download_gb}GB）`);
    if (remove.size) parts.push(`外す：${remove.size} 件`);
    summary.textContent = parts.length ? `「適用」で ${parts.join('、')}` : '変更はありません。';
  };

  const refresh = async () => {
    const data = await api.get('/api/engines');
    state.engines = data;
    table.innerHTML = '';
    for (const kind of ['ASR', 'TTS']) {
      const caption = el('tr');
      const th = el('th', null, kind); th.colSpan = 4;
      caption.append(th);
      table.append(caption);
      for (const e of data.engines.filter((x) => (x.kind || 'TTS') === kind)) {
        const tr = el('tr');
        const cb = el('input'); cb.type = 'checkbox'; cb.dataset.engine = e.id;
        cb.checked = data.enabled.includes(e.id) || e.installed;
        cb.disabled = !e.extra && !e.has_setup;
        if (cb.disabled) cb.title = '外部のサービス・アプリにつなぐだけなので、入れる必要はありません。';
        cb.addEventListener('change', () => updatePlan().catch(showError));
        const c0 = el('td'); c0.append(cb);
        const size = el('td', 'hint', e.download_gb == null ? '' : `約${e.download_gb}GB`);
        const st = el('td'); st.append(el('span', 'tag', ''));
        st.firstChild.dataset.status = e.id;
        tr.append(c0, el('td', null, e.label), size, st);
        table.append(tr);
      }
    }
    await updatePlan();
    if (!data.setup_supported) {
      view.prepend(el('div', 'notice',
        `この環境（${data.platform}）ではセットアップスクリプトを実行できないため「適用」は無効です。`));
      applyBtn.disabled = true;
    }
    // torch と onnxruntime の CUDA ランタイムは cpu / cu128 の排他 extra。
    // setup_env.ps1 を通らず uv だけで入れた場合、ここが唯一の選択場所になる。
    deviceGpu.checked = data.device_extra !== 'cpu';
    deviceCpu.checked = data.device_extra === 'cpu';
    for (const r of [deviceGpu, deviceCpu]) r.disabled = !!data.device_extra_locked;
    deviceNote.textContent = data.device_extra_locked
      ? '起動環境でデバイス構成が固定されているため変更できません。'
      : '「適用」で切り替わります。GPU→CPUは数GBの入れ替えになります。';
  };

  const deviceGpu = el('input'); deviceGpu.type = 'radio'; deviceGpu.name = 'device-extra';
  const deviceCpu = el('input'); deviceCpu.type = 'radio'; deviceCpu.name = 'device-extra';
  const deviceNote = el('div', 'hint', '');
  const deviceRow = el('div', 'row');
  const gpuLabel = el('label'); gpuLabel.append(deviceGpu, document.createTextNode(' GPU（CUDA）'));
  const cpuLabel = el('label'); cpuLabel.append(deviceCpu, document.createTextNode(' CPUのみ'));
  gpuLabel.title = 'CUDA対応のTorchとONNX Runtimeをインストールします。NVIDIA GPUが必要です。';
  cpuLabel.title = 'CPU版のTorchとONNX Runtimeをインストールします。NVIDIA関連パッケージ（約3GB）は不要ですが、推論は遅くなります。';
  deviceRow.append(gpuLabel, cpuLabel);
  const deviceBox = el('fieldset');
  deviceBox.append(el('legend', null, 'デバイス構成'), deviceRow, deviceNote);
  // applyBtn は bar の子なので view.insertBefore の基準にはできない。表を挟んで上に置く。
  view.insertBefore(deviceBox, table);

  applyBtn.addEventListener('click', async () => {
    const ids = checkedIds();
    await updatePlan();
    // 何が入り、何が外れるかを先に見せる（数GBの取得やパッケージの削除が黙って始まらないように）。
    if (lastPlan.install.length || lastPlan.remove.length) {
      const label = new Map(state.engines.engines.map((e) => [e.id, e]));
      const lines = [];
      if (lastPlan.install.length) {
        lines.push(`入れるもの（取得 約${lastPlan.download_gb}GB）：`);
        for (const id of lastPlan.install) {
          const e = label.get(id);
          lines.push(`  ・${e?.label ?? id}${e?.download_gb ? `（約${e.download_gb}GB）` : ''}`);
        }
      }
      if (lastPlan.remove.length) {
        lines.push('外すもの（Pythonパッケージを削除。モデルなどのファイルは残る）：');
        for (const id of lastPlan.remove) lines.push(`  ・${label.get(id)?.label ?? id}`);
      }
      lines.push('', '容量は目安で、Pythonパッケージの分は含みません。適用しますか？');
      if (!window.confirm(lines.join('\n'))) return;
    }
    applyBtn.disabled = true;
    log.value += `\n[apply] ${ids.join(', ')}\n`;
    try {
      const device = deviceCpu.checked ? 'cpu' : 'cu128';
      await api.post('/api/engines/apply', { ids, device });
    } catch (e) {
      log.value += `[error] ${e.message}\n`;
      applyBtn.disabled = false;
    }
  });

  view.dataset.onEngineEvent = '1';
  view._appendLog = (line) => { log.value += line + '\n'; log.scrollTop = log.scrollHeight; };
  view._onDone = async () => { applyBtn.disabled = false; await refresh(); };
  await refresh();
  return view;
}

async function buildViews() {
  const views = $('#views');
  views.innerHTML = '';
  state.uiLoading = true;
  state.byAttr = new Map();
  state.inputs = new Map();
  state.inputNodes = new Map();
  initTtsVoiceMap(state.config);
  const entries = [];

  for (const win of state.manifest.windows) {
    if (SKIP_WINDOW_IDS.has(win.id)) continue;
    const view = renderSettingsView(win);
    view.id = `view-${win.id}`;
    views.append(view);
    entries.push({ id: win.id, title: win.title, view });
  }

  const extras = [
    { id: 'console', title: 'プロセスコンソール', view: renderConsole() },
    { id: 'engine_manager_web', title: 'エンジン管理', view: await renderEngineManager() },
  ];
  for (const e of extras) {
    e.view.id = `view-${e.id}`;
    views.append(e.view);
    entries.push(e);
  }

  state.views = entries;
  buildSettingsNav(entries);
  // ?view=<画面ID> で設定の画面をその画面で開く（リンクの共有・確認用）。
  const requestedView = new URLSearchParams(location.search).get('view');
  if (requestedView && entries.some((e) => e.id === requestedView)) {
    queueMicrotask(() => openSettings(requestedView));
  }
  const initial = entries.find((item) => item.id === 'settings_window') || entries[0];
  activate(initial.id, initial.title);
  state.lastTtsEngine = liveCombo('tts_engine_combo', 'tts_engine');
  state.lastLlmBackend = liveCombo('llm_backend_combo', 'llm_backend');
  applyEnginePages();
  state.uiLoading = false;
}

// 設定の画面の左の一覧。並びは Qt の gui/settings_hub.py の HUB_PAGES と同じ。
const SETTINGS_NAV_GROUPS = [
  ['エンジン', ['asr_test_window', 'llm_test_window', 'tts_test_window', 'maai_settings_window']],
  ['設定', ['voice_dialog_settings', 'settings_window', 'audio_input_settings_window',
    'audio_output_settings_window', 'obs_settings_window', 'tts_template_auth_window']],
  ['テスト', ['llm_s2s_test_window', 'maai_test_window']],
  ['ツール', ['engine_manager_web', 'console', 'nod_motion_chart_window']],
];

function buildSettingsNav(entries) {
  const nav = document.getElementById('settings-nav');
  if (!nav) return;
  nav.replaceChildren();
  const byId = new Map(entries.map((e) => [e.id, e]));
  const listed = new Set();
  for (const [caption, ids] of SETTINGS_NAV_GROUPS) {
    const items = ids.map((id) => byId.get(id)).filter(Boolean);
    if (!items.length) continue;
    nav.append(el('div', 'settings-nav-caption', caption));
    for (const item of items) {
      listed.add(item.id);
      const btn = el('button', null, item.title);
      btn.type = 'button';
      btn.dataset.navId = item.id;
      btn.addEventListener('click', () => activate(item.id, item.title));
      nav.append(btn);
    }
  }
  const rest = entries.filter((e) => !listed.has(e.id));
  for (const item of rest) {
    const btn = el('button', null, item.title);
    btn.type = 'button';
    btn.dataset.navId = item.id;
    btn.addEventListener('click', () => activate(item.id, item.title));
    nav.append(btn);
  }
}

function activateById(id) {
  const item = state.views?.find((v) => v.id === id);
  if (item) activate(item.id, item.title);
}

function activate(id, title) {
  for (const v of state.views) v.view.classList.toggle('active', v.id === id);
  state.activeView = id;
  const heading = document.getElementById('settings-title');
  if (heading) heading.textContent = title || '設定';
  for (const btn of document.querySelectorAll('[data-view-id]')) {
    btn.setAttribute('aria-pressed', btn.dataset.viewId === id ? 'true' : 'false');
  }
  for (const btn of document.querySelectorAll('[data-nav-id]')) {
    btn.setAttribute('aria-selected', btn.dataset.navId === id ? 'true' : 'false');
  }
  applyEnginePages();
  if (id === 'nod_motion_chart_window') state.nodChart?.paint();
}

// モジュール行をクリックしたときに開く画面（Qt の module_status_mixin._MODULE_PAGE_ATTRS と同じ）。
const MODULE_PAGE_IDS = {
  asr: 'asr_test_window',
  llm: 'llm_test_window',
  tts: 'tts_test_window',
  maai: 'maai_settings_window',
};

function openSettings(viewId = '') {
  if (viewId) activateById(viewId);
  $('#settings-drawer').classList.add('open');
  $('#btn-settings').setAttribute('aria-expanded', 'true');
}

function closeSettings() {
  $('#settings-drawer').classList.remove('open');
  $('#btn-settings').setAttribute('aria-expanded', 'false');
}

function toggleSettings() {
  if ($('#settings-drawer').classList.contains('open')) closeSettings();
  else { closePlot(); openSettings(state.activeView || 'settings_window'); }
}

function sendPlotAudioSubscribe(on) {
  const sock = state.eventsSock;
  if (sock && sock.readyState === WebSocket.OPEN) {
    sock.send(JSON.stringify({ type: 'ui_subscribe', plot_audio: Boolean(on) }));
  }
}

function openPlot() {
  closeSettings();
  state.plot?.setVisibilityFromConfig(mergedConfig());
  state.plot?.show();
  sendPlotAudioSubscribe(true);
  $('#btn-plot')?.setAttribute('aria-expanded', 'true');
}

function closePlot() {
  state.plot?.hide();
  sendPlotAudioSubscribe(false);
  $('#btn-plot')?.setAttribute('aria-expanded', 'false');
}

// Qt は音声対話・MaAI・（プロット表示ありの）ASR を起動すると MaAI プロットの窓を開く。Web のプロットは
// 設定の画面を閉じて開くので、設定を開いて作業している間は開かない（作業中の画面を閉じないため）。
// コンソールは Web では設定の画面の 1 ページなので、自動では切り替えない（各モジュールの状態はヘッダーに出ている）。
function autoOpenPlot() {
  if ($('#settings-drawer')?.classList.contains('open')) return;
  if (state.plot?.isOpen()) return;
  openPlot();
}

function togglePlot() {
  if (state.plot?.isOpen()) closePlot();
  else openPlot();
}

function mergedConfig() {
  const out = { ...state.config };
  for (const [k, v] of state.dirty) out[k] = v;
  return out;
}

function appendConsole(line) {
  const ta = document.getElementById('console-log');
  if (!ta) return;
  ta.value += line + '\n';
  if (ta.value.length > 400000) ta.value = ta.value.slice(-200000);
  ta.scrollTop = ta.scrollHeight;
}

function appendAttr(attr, text) {
  const node = state.byAttr.get(attr);
  if (!node || node.tagName !== 'TEXTAREA') return;
  node.value += text;
  node.scrollTop = node.scrollHeight;
}

// ASR テスト窓の結果欄（PySide の _append_asr_result_event / _append_asr_line_to_editor と同じ）。
// 1 発話 1 行で追記し、途中結果（-partial）は同じ行を書き換える。比較モードは上下の欄へ振り分ける。
function appendAsrResult(ev) {
  const text = String(ev.text ?? '').trim();
  if (!text) return;
  const engine = String(ev.engine ?? 'asr').trim() || 'asr';
  const isFinal = ev.is_final === undefined ? true : Boolean(ev.is_final);
  const isStreaming = Boolean(ev.is_streaming);
  const latency = Number(ev.latency_sec) || 0;
  const slot = String(ev.compare_slot ?? '').trim().toLowerCase();
  const attr = slot === 'top'
    ? 'asr_compare_top_result_edit'
    : slot === 'bottom' ? 'asr_compare_bottom_result_edit' : 'asr_result_edit';
  const tag = isFinal ? `[${engine}]` : `[${engine}-partial]`;
  const line = `${tag} ${text} (latency=${latency.toFixed(3)}s)`;
  for (const node of document.querySelectorAll(`textarea[data-attr="${attr}"]`)) {
    const lines = node.value ? node.value.split('\n') : [];
    const lastTag = /^\[([^\]]+)\]/.exec(lines.at(-1) ?? '')?.[1]?.trim() ?? '';
    const replaceLast = lastTag === `${engine}-partial` && ((isStreaming && !isFinal) || isFinal);
    if (replaceLast) lines[lines.length - 1] = line;
    else lines.push(line);
    node.value = lines.join('\n');
    node.scrollTop = node.scrollHeight;
  }
}

// LLM テスト窓の応答ブロック（PySide の _redraw_llm_test_assistant_block と同じ形）。
// 「assistant> 」の直後から末尾までを、TTS へ送った区切りごとの行 + 未送出の残りで描き直す。
function ensureLlmTestBlock() {
  if (state.llmTestStreaming && state.llmTestBlockStart >= 0) return;
  const node = state.byAttr.get('llm_chat_view');
  if (!node || node.tagName !== 'TEXTAREA') return;
  if (!node.value.endsWith('アシスタント：')) node.value += 'アシスタント：';
  state.llmTestStreaming = true;
  state.llmTestBlockStart = node.value.length;
}

function redrawLlmTestBlock() {
  const node = state.byAttr.get('llm_chat_view');
  if (!node || node.tagName !== 'TEXTAREA' || state.llmTestBlockStart < 0) return;
  const full = state.llmTestFull;
  const segments = state.llmTestSegments;
  let body = full;
  if (segments.length) {
    const confirmed = segments.join('');
    const tail = full.startsWith(confirmed) ? full.slice(confirmed.length) : '';
    body = [...segments, ...(tail ? [tail] : [])].join('\n');
  }
  node.value = node.value.slice(0, state.llmTestBlockStart) + body;
  node.scrollTop = node.scrollHeight;
}

function setTextByAttr(attr, text) {
  for (const ta of document.querySelectorAll(`textarea[data-attr="${attr}"], input[data-attr="${attr}"]`)) {
    ta.value = text;
    if (ta.tagName === 'TEXTAREA') ta.scrollTop = ta.scrollHeight;
  }
}

function fmtLatency(sec) {
  if (sec == null || Number.isNaN(Number(sec))) return '-';
  return `${(Number(sec) * 1000).toFixed(0)}ms`;
}

function renderLatency() {
  $('#latency').textContent =
    `遅延　ASR：${fmtLatency(state.latency.asr)} / LLM：${fmtLatency(state.latency.llm)} / TTS：${fmtLatency(state.latency.tts)}`;
}

// 会話の欄（吹き出し）。Qt の gui/chat_view.py と同じ見え方。
const CHAT_ROLE_NAMES = {
  user: 'あなた',
  user_partial: 'あなた（認識中）',
  assistant: 'アシスタント',
  assistant_streaming: 'アシスタント（生成中）',
  system: 'システム',
};

function renderStream() {
  renderLive();
}

function renderHistory(history) {
  const snap = history && typeof history === 'object' && !Array.isArray(history)
    ? history
    : { messages: Array.isArray(history) ? history : [] };
  const items = [];
  let lastAssistant = '';
  // 会話の欄は確定した発話と応答だけ。system（プロンプト）は出さない。
  for (const msg of snap.messages ?? []) {
    const role = String(msg.role ?? '').trim().toLowerCase();
    const text = String(msg.content ?? msg.text ?? '').trim();
    if (role === 'user') items.push(['user', text || '（空）']);
    else if (role === 'assistant' && text) { items.push(['assistant', text]); lastAssistant = text; }
  }
  for (const pending of snap.pending_texts ?? []) {
    const t = String(pending).trim();
    if (t) items.push(['user', t]);
  }
  state.chatItems = items;
  state.chatLastAssistant = lastAssistant;
  state.chatPartial = String(snap.current_partial ?? '').trim();
  renderChat();
}

// 進行中（認識中・生成中）は会話の欄の下の小さな欄に出す。
function renderLive() {
  const streaming = String(state.streamText || '').trim();
  const pairs = [
    ['#live-user', state.chatPartial || ''],
    ['#live-assistant', streaming && streaming !== state.chatLastAssistant ? streaming : ''],
  ];
  for (const [sel, text] of pairs) {
    const node = $(sel);
    if (!node) continue;
    node.textContent = text || '—';
    node.classList.toggle('empty', !text);
  }
}

function renderChat() {
  renderLive();
  const node = $('#dialog-history');
  if (!node) return;
  const items = state.chatItems ?? [];
  const follow = node.scrollTop + node.clientHeight >= node.scrollHeight - 40;
  node.replaceChildren();
  if (!items.length) {
    node.append(el('div', 'chat-empty', '音声対話を始めると、ここに会話が表示されます。'));
    return;
  }
  for (const [role, text] of items) {
    const row = el('div', `chat-row chat-row-${role.startsWith('user') ? 'user' : role === 'system' ? 'system' : 'assistant'}`);
    const bubble = el('div', `chat-bubble chat-bubble-${role}`);
    if (role !== 'system') bubble.append(el('div', 'chat-name', CHAT_ROLE_NAMES[role] || role));
    bubble.append(el('div', 'chat-text', text));
    row.append(bubble);
    node.append(row);
  }
  if (follow) node.scrollTop = node.scrollHeight;
}

function setVoiceControls(stateName) {
  const running = stateName === 'running';
  const starting = stateName === 'starting';
  const stopping = stateName === 'stopping';
  state.voiceActive = running || starting;
  if (!running && !starting && !stopping) {
    releaseBrowserMic('voice');
    resetInputWavMonitor();
  }
  $('#btn-start').disabled = running || starting;
  $('#btn-stop').disabled = !(running || starting || stopping);
  renderModules();
  const badge = $('#voice-state');
  const label = {
    running: '音声対話：稼働',
    starting: '音声対話：起動中',
    stopping: '音声対話：停止中',
    stopped: '音声対話：停止',
    error: '音声対話：エラー',
  }[stateName] || (stateName || '停止');
  badge.textContent = label;
  setChip('voice', running ? 'ok' : (starting || stopping) ? 'warn' : stateName === 'error' ? 'err' : 'idle',
    label.replace('稼働', '実行中'));
  badge.className = `badge ${stateName === 'running' ? 'running' : stateName === 'error' ? 'error' : (starting || stopping) ? 'starting' : ''}`;
}

function handleEvent(ev) {
  state.plot?.handleEvent(ev);
  switch (ev.type) {
    case 'runtime_snapshot': {
      if (ev.dialog_server && typeof ev.dialog_server === 'object') {
        state.dialogServer = {
          state: String(ev.dialog_server.state || 'stopped'),
          error: String(ev.dialog_server.error || ''),
        };
        renderServerChip();
      }
      applyModuleEvent(ev);
      const voiceState = String(ev.voice_state ?? 'stopped').toLowerCase();
      setVoiceControls(voiceState);
      restoreWavPlayback(ev.wav_playing);
      renderNodThresholdDefaults(ev.maai_nod_threshold_defaults);
      if (ev.history && typeof ev.history === 'object') renderHistory(ev.history);
      if (ev.mmd_remote && typeof ev.mmd_remote === 'object') {
        applyMmdagentStatus({
          ...(state.runtime?.mmdagent || {}),
          remote_connected: Boolean(ev.mmd_remote.connected),
          remote_mode: String(ev.mmd_remote.mode || ''),
        });
      }
      restoreVoiceMic(voiceState);
      break;
    }
    case 'status': {
      const st = String(ev.state ?? '').toLowerCase();
      if (st) {
        setVoiceControls(st);
        restoreVoiceMic(st);
      }
      else if (ev.message) $('#voice-state').textContent = ev.message;
      break;
    }
    case 'log': {
      const msg = String(ev.message ?? '');
      appendConsole(msg);
      if (/LLM warmup完了/.test(msg) && !state.llmTestStreaming) {
        state.streamText = msg.replace(/^\[LLMWorker\]\s*/, '');
        renderStream();
      }
      break;
    }
    case 'fatal': {
      const source = String(ev.module ?? ev._src ?? ev.worker ?? '').trim().toLowerCase();
      const aliases = { ttsworker: 'tts', maaiworker: 'maai' };
      const module = aliases[source] || source;
      const reason = String(ev.message ?? ev.error ?? 'ワーカーで致命的なエラーが発生しました');
      if (MODULE_KEYS.includes(module)) {
        state.modules[module] = { state: 'error', error: reason };
        renderModules();
      }
      appendConsole(`[${module || 'worker'}][FATAL] ${reason}`);
      showError(new Error(`${MODULE_LABELS[module] || module || 'ワーカー'}：${reason}`));
      break;
    }
    case 'history_snapshot':
      renderHistory(ev.history);
      break;
    case 'llm_chunk':
      state.streamText = String(ev.assistant_text ?? '');
      renderStream();
      break;
    case 'llm_done':
      if (ev.assistant_text) {
        state.streamText = String(ev.assistant_text);
        renderStream();
      }
      break;
    case 'asr_result':
      appendAsrResult(ev);
      break;
    case 'module_state':
    case 'module_started':
    case 'module_stopped':
    case 'module_start_failed':
      applyModuleEvent(ev);
      if (ev.type === 'module_stopped' && String(ev.module || '').toLowerCase() === 'maai') {
        renderNodThresholdDefaults(null);
      }
      break;
    case 'maai_nod_threshold_defaults':
      renderNodThresholdDefaults(ev.defaults);
      break;
    case 'latency': {
      const kind = String(ev.kind ?? '').toLowerCase();
      if (kind in state.latency) {
        state.latency[kind] = ev.value == null ? null : Number(ev.value);
        renderLatency();
      }
      break;
    }
    case 'engine_apply_output': {
      const v = document.getElementById('view-engine_manager_web');
      v?._appendLog?.(String(ev.line ?? ''));
      break;
    }
    case 'engine_apply_done': {
      const v = document.getElementById('view-engine_manager_web');
      v?._appendLog?.(`[done] ok=${ev.ok} ${ev.message ?? ''}`);
      v?._onDone?.();
      break;
    }
    case 'mmdagent_status':
      applyMmdagentStatus(ev);
      break;
    case 'remote_status':
      applyMmdagentStatus({
        ...(state.runtime?.mmdagent || {}),
        remote_connected: Boolean(ev.connected),
        remote_mode: String(ev.mode || ''),
      });
      appendConsole(
        `[MMDAgent Remote] ${ev.connected ? '接続' : '切断'}${ev.mode ? ` mode=${ev.mode}` : ''}`,
      );
      break;
    case 'connection': {
      // 起動時にワーカーが上がらなかったときは、理由がここに来る。
      if (ev.state === 'disconnected' && ev.message && state.dialogServer.state !== 'stopped') {
        state.dialogServer = { state: 'error', error: String(ev.message) };
        renderModules();
      }
      renderServerChip();
      break;
    }
    case 'dialog_server': {
      state.dialogServer = { state: String(ev.state || 'stopped'), error: String(ev.error || '') };
      if (ev.error) appendConsole(`[対話サーバー] ${ev.error}`);
      renderServerChip();
      renderModules();
      break;
    }
    case 'llm_test_chat_chunk': {
      const delta = String(ev.text ?? '');
      const full = String(ev.assistant_text ?? ev.delta ?? '');
      state.streamText = full || state.streamText + delta;
      renderStream();
      ensureLlmTestBlock();
      state.llmTestFull += delta;
      redrawLlmTestBlock();
      break;
    }
    case 'llm_test_tts_segment': {
      // PySide の _on_llm_tts_stream_push_ui: TTS へ送った区切りごとに行を分けて表示する。
      const cumulative = String(ev.cumulative ?? '');
      const pushed = String(ev.pushed_segment ?? '');
      if (!cumulative) break;
      const shown = cumulative.length <= 200 ? cumulative : `${cumulative.slice(0, 197)}...`;
      appendConsole(`[LLM→TTS push][llm_test] ここまで: ${shown}`);
      ensureLlmTestBlock();
      if (pushed) state.llmTestSegments.push(pushed);
      redrawLlmTestBlock();
      break;
    }
    case 'llm_test_finished': {
      const text = String(ev.assistant_text ?? '');
      if (text) {
        state.streamText = text;
        renderStream();
      }
      ensureLlmTestBlock();
      if (ev.ok && text) state.llmTestFull = text;
      redrawLlmTestBlock();
      state.llmTestStreaming = false;
      state.llmTestBlockStart = -1;
      state.llmTestSegments = [];
      state.llmTestFull = '';
      if (ev.ok && text && !state.llmMessages.some((m) => m.role === 'assistant' && m.content === text)) {
        state.llmMessages.push({ role: 'assistant', content: text });
      }
      if (!ev.ok) {
        const err = String(ev.error ?? '不明なエラー');
        // PySide と同じく、失敗したターンの user 発話は履歴から外す。
        if (state.llmMessages.at(-1)?.role === 'user') state.llmMessages.pop();
        appendAttr('llm_chat_view', `\n[error] ${err}\n`);
        appendConsole(`LLMテスト失敗: ${err}`);
        showError(new Error(err));
      } else {
        const sec = (v, d = 0) => (Number.isFinite(Number(v)) ? Number(v) : d).toFixed(3);
        const total = ev.total_sec ?? 0;
        appendAttr(
          'llm_chat_view',
          `\n(ttft=${sec(ev.ttft_sec)}s total=${sec(total)}s overall=${sec(ev.total_overall_sec ?? total)}s)\n`,
        );
        appendConsole(`[LLMテスト完了] ${ev.backend ?? ''} ${ev.model ?? ''}`);
      }
      break;
    }
    case 'tts_test_finished':
      appendConsole(ev.ok ? `[TTSテスト完了] ${ev.engine ?? ''}` : `[TTSテスト失敗] ${ev.error ?? ''}`);
      break;
    case 'wav_monitor_audio':
      break;
    case 'wav_playback_finished':
      state.wavPlaying = false;
      setAttrButtonText('audio_input_wav_play_pause_btn', '再生');
      resetInputWavMonitor();
      renderModules();
      break;
    case 'wav_command_done': {
      const cmd = String(ev.command || '');
      if (cmd === 'wav_play') {
        state.wavPlaying = true;
        setAttrButtonText('audio_input_wav_play_pause_btn', '一時停止');
      } else {
        state.wavPlaying = false;
        setAttrButtonText('audio_input_wav_play_pause_btn', '再生');
        resetInputWavMonitor();
      }
      renderModules();
      break;
    }
    case 'processing_stopped':
      resetInputWavMonitor();
      break;
    case 'nod_motion_chart':
      state.nodChart?.setChartData(ev);
      break;
    case 'maai_nod_result':
    case 'maai_bc_result':
      appendAttr('maai_test_log_edit', `${JSON.stringify(ev)}\n`);
      break;
    default:
      break;
  }
}

function setAttrButtonText(attr, text) {
  const node = state.byAttr.get(attr);
  if (node && node.tagName === 'BUTTON') node.textContent = text;
}

// MaAI テストの頷きのしきい値の横に、ワーカーが ready で返した既定値を出す（Qt の「既定: …」と同じ）。
const NOD_THRESHOLD_DEFAULT_FIELDS = [
  ['maai_test_nod_count_t0_edit', (d) => d?.count?.t0],
  ['maai_test_nod_count_t1_edit', (d) => d?.count?.t1],
  ['maai_test_nod_count_t2_edit', (d) => d?.count?.t2],
  ['maai_test_nod_swing_threshold_edit', (d) => d?.swing],
];

function renderNodThresholdDefaults(defaults) {
  const d = defaults && typeof defaults === 'object' ? defaults : null;
  for (const [attr, pick] of NOD_THRESHOLD_DEFAULT_FIELDS) {
    const input = document.querySelector(`input[data-attr="${attr}"]`);
    if (!input) continue;
    let hint = input.parentElement?.querySelector(`span[data-default-for="${attr}"]`);
    if (!hint) {
      hint = el('span');
      hint.dataset.defaultFor = attr;
      hint.style.marginLeft = '8px';
      hint.style.color = 'var(--muted)';
      input.insertAdjacentElement('afterend', hint);
    }
    const v = Number(d ? pick(d) : NaN);
    hint.textContent = Number.isFinite(v) ? `既定: ${v}` : '既定: —';
  }
}

function applyModuleEvent(ev) {
  // PySide の _set_module_panel_state と同じく、比較 ASR（asr_compare）は ASR 行に集約する。
  const panelName = (raw) => {
    const n = String(raw ?? '').toLowerCase();
    return n === 'asr_compare' ? 'asr' : n;
  };
  if (ev.modules && typeof ev.modules === 'object') {
    for (const [rawName, info] of Object.entries(ev.modules)) {
      const st = typeof info === 'string' ? info : info?.state ?? '';
      const name = panelName(rawName);
      // スナップショットに asr と asr_compare が両方あるときは、動いている方を出す。
      if (name !== rawName && ['stopped', ''].includes(st) && state.modules[name]) continue;
      state.modules[name] = { state: st, error: info?.error ?? '' };
    }
  }
  const name = panelName(ev.module);
  if (name) {
    let st = String(ev.state ?? '').toLowerCase();
    if (ev.type === 'module_started') st = st || 'running';
    if (ev.type === 'module_stopped') st = st || 'stopped';
    if (ev.type === 'module_start_failed') st = 'error';
    const prev = state.modules[name]?.state ?? '';
    state.modules[name] = { state: st || 'stopped', error: String(ev.error ?? '') };
    if (st === 'running' && prev !== 'running') {
      const asrPlot = ['1', 'true', 'yes', 'on'].includes(String(mergedConfig().asr_maai_gui_plot ?? '').toLowerCase());
      if (name === 'maai' || (name === 'asr' && asrPlot)) autoOpenPlot();
    }
    if (['asr', 'maai'].includes(name) && ['stopped', 'error'].includes(st || 'stopped')) {
      releaseBrowserMic(name);
    }
  }
  renderModules();
}

function moduleSummary(key) {
  if (key === 'asr') return liveCombo('asr_engine_combo', 'asr_engine');
  if (key === 'llm') return liveCombo('llm_backend_combo', 'llm_backend');
  if (key === 'tts') return liveCombo('tts_engine_combo', 'tts_engine');
  return '';
}

// 対話サーバー（ASR・LLM・TTS・MaAI を動かす裏のプロセス）の起動・停止。Qt の「対話サーバー」行と同じ。
async function dialogServerAction(action) {
  const cur = state.dialogServer.state;
  if (action === 'start' && !['stopped', 'error'].includes(cur)) return;
  if (action === 'stop' && !['running', 'error'].includes(cur)) return;
  state.dialogServer = { state: action === 'start' ? 'starting' : 'stopping', error: '' };
  renderModules();
  renderServerChip();
  try {
    const res = await api.post(`/api/dialog-server/${action}`);
    state.dialogServer = { state: String(res.state || (action === 'start' ? 'running' : 'stopped')), error: '' };
  } catch (e) {
    state.dialogServer = { state: 'error', error: String(e.message ?? e) };
    throw e;
  } finally {
    renderModules();
    renderServerChip();
  }
}

// ローカル / Docker の切り替え。サーバーが開き直すので、つながり直したらページを読み込み直す。
async function switchDialogServerMode(mode) {
  const label = mode === 'docker' ? 'Docker' : 'ローカル';
  const ok = window.confirm(
    `対話サーバーを「${label}」に切り替えます。\n`
    + 'Web サーバーを開き直します（会話の履歴は消え、動いているモジュールと音声対話は止まります）。\n'
    + 'よろしいですか？',
  );
  if (!ok) {
    renderModules();
    return;
  }
  const res = await api.post('/api/dialog-server-mode', { mode });
  if (!res.restarting) return;
  state.switchingDialogServer = true;
  state.dialogServer = { state: 'stopping', error: '' };
  renderModules();
  setConn('warn', `「${label}」に切り替え中…`);
  // 古いサーバーが止まるとイベントの接続が切れる。切れなくても 30 秒後には読み込み直しを試す。
  window.setTimeout(() => reloadWhenServerBack(), 30000);
}

async function reloadWhenServerBack() {
  if (state.reloadingAfterSwitch) return;
  state.reloadingAfterSwitch = true;
  for (let i = 0; i < 180; i += 1) {
    await new Promise((resolve) => { window.setTimeout(resolve, 1000); });
    try {
      const r = await fetch('/api/runtime', { cache: 'no-store' });
      if (r.ok) {
        location.reload();
        return;
      }
    } catch { /* まだ起動中 */ }
  }
  state.reloadingAfterSwitch = false;
}

// モジュールの起動や音声対話の開始の前に、止まっていれば対話サーバーを起動する。
async function ensureDialogServer() {
  if (state.dialogServer.state === 'running') return;
  if (['stopped', 'error'].includes(state.dialogServer.state)) await dialogServerAction('start');
}

const DIALOG_SERVER_CHIP = {
  running: ['ok', '接続済み'],
  starting: ['warn', '起動中…'],
  stopping: ['warn', '停止中…'],
  stopped: ['idle', '停止'],
  error: ['err', 'エラー'],
};

function renderServerChip() {
  if (state.switchingDialogServer) return;
  if (!state.eventsOpen) {
    setConn('err', 'Webサーバーと切断');
    return;
  }
  const [kind, text] = DIALOG_SERVER_CHIP[state.dialogServer.state] || DIALOG_SERVER_CHIP.error;
  setConn(kind, text);
  const node = $('#conn-text');
  if (node) node.title = state.dialogServer.error || '';
}

async function sendModuleCommand(module, action) {
  const info = state.modules[module] || { state: 'stopped' };
  const st = info.state || 'stopped';
  if (action === 'start' && !['stopped', 'error'].includes(st)) return;
  if (action === 'stop' && !['running', 'starting'].includes(st)) return;
  if (action === 'start') {
    try { await ensureDialogServer(); } catch (e) { showError(e); return; }
  }
  if (action === 'start' && (module === 'asr' || module === 'maai')) {
    try { await acquireBrowserMic(module); } catch (e) { showError(e); return; }
  }
  if (action === 'stop' && (module === 'asr' || module === 'maai')) releaseBrowserMic(module);
  state.modules[module] = { ...info, state: action === 'start' ? 'starting' : 'stopping', error: '' };
  renderModules();
  return postCommand({ type: action === 'start' ? 'start_module' : 'stop_module', module })
    .catch((e) => {
      if (action === 'start' && (module === 'asr' || module === 'maai')) releaseBrowserMic(module);
      state.modules[module] = { state: 'error', error: String(e.message ?? e) };
      renderModules();
      $('#save-state').textContent = String(e.message ?? e);
    });
}

async function startAllModules() {
  const button = document.getElementById('btn-start-all');
  if (!button || button.dataset.busy === '1') return;
  button.dataset.busy = '1';
  button.disabled = true;
  button.textContent = '起動要求中…';
  try {
    for (const module of MODULE_KEYS) {
      const moduleState = String(state.modules[module]?.state || 'stopped');
      if (!['stopped', 'error'].includes(moduleState)) continue;
      await sendModuleCommand(module, 'start');
    }
  } finally {
    delete button.dataset.busy;
    button.textContent = '一括起動';
    renderModules();
  }
}

// 動いているモジュールをすべて止める。音声対話中なら先に音声対話を終える（Qt の一括停止と同じ）。
async function stopAllModules() {
  const button = document.getElementById('btn-stop-all');
  if (!button || button.dataset.busy === '1') return;
  button.dataset.busy = '1';
  button.disabled = true;
  button.textContent = '停止中…';
  try {
    if (state.voiceActive) await postCommand({ type: 'stop_voice_dialog' });
    for (const module of MODULE_KEYS) {
      const moduleState = String(state.modules[module]?.state || 'stopped');
      if (!['running', 'starting'].includes(moduleState)) continue;
      await sendModuleCommand(module, 'stop');
    }
  } finally {
    delete button.dataset.busy;
    button.textContent = '一括停止';
    renderModules();
  }
}

function isWavInputMode() {
  const v = String(attrValue('audio_input_mode_combo', state.config.audio_input_mode || '')).trim().toLowerCase();
  return v === 'wav';
}

function wavFileSummary() {
  const path = String(attrValue('audio_input_wav_path_edit', state.config.audio_input_wav_path || '')).trim();
  if (!path) return '';
  const parts = path.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1] || path;
}

function renderModules() {
  const host = $('#modules');
  if (!host) return;
  host.innerHTML = '';
  {
    const st = state.dialogServer.state;
    const box = el('div', `module server ${st}`);
    const start = el('button', 'module-btn', '▶');
    start.type = 'button';
    start.title = '対話サーバー起動';
    start.disabled = !['stopped', 'error'].includes(st);
    start.addEventListener('click', () => dialogServerAction('start').catch(showError));
    const stop = el('button', 'module-btn', '■');
    stop.type = 'button';
    stop.title = '対話サーバー停止（動いているモジュールと音声対話も止まります）';
    stop.disabled = !['running', 'error'].includes(st);
    stop.addEventListener('click', () => dialogServerAction('stop').catch(showError));
    const where = String(state.runtime?.dialog_server_where || (state.runtime?.docker ? 'container' : 'local'));
    box.append(
      el('span', 'lamp'),
      el('span', 'm-name', '対話サーバー'),
      el('span', 'm-state', MODULE_STATE_LABELS[st] || st),
      start,
      stop,
    );
    const detail = el('span', 'm-detail server-detail');
    if (where === 'container') {
      detail.append(el('span', 'm-summary', 'Docker（Webサーバーと同じコンテナ）'));
    } else {
      // ローカル / Docker の選択。選ぶと Web サーバーを開き直す（Qt の「対話サーバー」行と同じ）。
      const select = el('select', 'server-mode');
      for (const [value, label] of [['local', 'ローカル'], ['docker', 'Docker']]) {
        const option = el('option', null, label);
        option.value = value;
        if (value === 'docker' && state.runtime?.dialog_server_docker_unavailable) {
          option.disabled = true;
          option.title = state.runtime.dialog_server_docker_unavailable;
        }
        select.append(option);
      }
      select.value = where;
      select.disabled = Boolean(state.switchingDialogServer);
      select.title = where === 'docker'
        ? 'Docker の対話サーバー（compose の orchestrator）を使っています。切り替えると Web サーバーを開き直します。'
        : 'この Web サーバーの子プロセスとして動いています。切り替えると Web サーバーを開き直します。';
      select.addEventListener('change', () => switchDialogServerMode(select.value).catch((e) => {
        state.switchingDialogServer = false;
        renderModules();
        showError(e);
      }));
      detail.append(select);
    }
    if (state.dialogServer.error) detail.append(el('span', 'm-error', state.dialogServer.error));
    box.append(detail);
    box.title = state.dialogServer.error || 'ASR・LLM・TTS・MaAI を動かして音声対話を進める裏のプロセスです。';
    host.append(box);
  }
  for (const key of MODULE_KEYS) {
    const info = state.modules[key] || { state: 'stopped' };
    const st = info.state || 'stopped';
    const box = el('div', `module has-settings ${st}`);
    const start = el('button', 'module-btn', '▶');
    start.type = 'button';
    start.title = `${MODULE_LABELS[key]}起動`;
    start.disabled = !['stopped', 'error'].includes(st);
    start.addEventListener('click', () => sendModuleCommand(key, 'start'));
    const stop = el('button', 'module-btn', '■');
    stop.type = 'button';
    stop.title = `${MODULE_LABELS[key]}停止`;
    stop.disabled = !['running', 'starting'].includes(st);
    stop.addEventListener('click', () => sendModuleCommand(key, 'stop'));
    const summary = moduleSummary(key);
    // そのエンジンの画面を開くボタン（名前と要約のクリックだけでは押せると分からない）。
    const settingsBtn = el('button', 'module-btn module-settings', '設定');
    settingsBtn.type = 'button';
    settingsBtn.title = `${MODULE_LABELS[key]}の設定を開く`;
    settingsBtn.addEventListener('click', () => openSettings(MODULE_PAGE_IDS[key]));
    // 名前と要約をクリックすると、そのエンジンの画面（設定とテスト）を開く（Qt と同じ）。
    const pageId = MODULE_PAGE_IDS[key];
    const name = el('span', 'm-name', MODULE_LABELS[key] || key);
    if (pageId) {
      name.classList.add('m-link');
      name.addEventListener('click', () => openSettings(pageId));
    }
    box.append(
      el('span', 'lamp'),
      name,
      el('span', 'm-state', MODULE_STATE_LABELS[st] || st),
      start,
      stop,
      settingsBtn,
    );
    const detail = el('span', 'm-detail');
    if (summary) {
      const sum = el('span', 'm-summary', summary);
      if (pageId) {
        sum.classList.add('m-link');
        sum.addEventListener('click', () => openSettings(pageId));
      }
      detail.append(sum);
    }
    if (info.error) detail.append(el('span', 'm-error', info.error));
    box.append(detail);
    box.title = info.error || (summary ? `${summary}\n（クリックで設定を開く）` : '');
    host.append(box);
  }
  if (isWavInputMode()) {
    const playing = Boolean(state.wavPlaying);
    const box = el('div', `module ${playing ? 'running' : 'stopped'}`);
    const start = el('button', 'module-btn', '▶');
    start.type = 'button';
    start.title = 'WAV再生';
    start.disabled = playing;
    start.addEventListener('click', () => playWavFromHud().catch(showError));
    const stop = el('button', 'module-btn', '■');
    stop.type = 'button';
    stop.title = 'WAV停止';
    stop.disabled = !playing;
    stop.addEventListener('click', () => stopWavFromHud().catch(showError));
    const summary = wavFileSummary();
    box.append(
      el('span', 'lamp'),
      el('span', 'm-name', 'WAV'),
      el('span', 'm-state', playing ? '再生中' : '停止'),
      start,
      stop,
    );
    const detail = el('span', 'm-detail');
    if (summary) detail.append(el('span', 'm-summary', summary));
    box.append(detail);
    box.title = summary || '';
    host.append(box);
  }
  const startAll = document.getElementById('btn-start-all');
  if (startAll && startAll.dataset.busy !== '1') {
    startAll.disabled = !MODULE_KEYS.some((key) => {
      const moduleState = String(state.modules[key]?.state || 'stopped');
      return ['stopped', 'error'].includes(moduleState);
    });
  }
  const stopAll = document.getElementById('btn-stop-all');
  if (stopAll && stopAll.dataset.busy !== '1') {
    stopAll.disabled = !state.voiceActive && !MODULE_KEYS.some((key) => {
      const moduleState = String(state.modules[key]?.state || 'stopped');
      return ['running', 'starting'].includes(moduleState);
    });
  }
}

async function toggleWavPlayPause() {
  if (state.wavPlaying) {
    state.wavPlaying = false;
    setAttrButtonText('audio_input_wav_play_pause_btn', '再生');
    resetInputWavMonitor();
    renderModules();
    await postCommand({ type: 'audio_input_send_command', command: 'wav_stop' });
    return;
  }
  if (wavMonitorEnabled()) {
    await armWavMonitorPlayback();
  }
  state.wavPlaying = true;
  setAttrButtonText('audio_input_wav_play_pause_btn', '一時停止');
  renderModules();
  await postCommand({ type: 'audio_input_send_command', command: 'wav_play' });
}

async function endWavPlayback() {
  state.wavPlaying = false;
  setAttrButtonText('audio_input_wav_play_pause_btn', '再生');
  resetInputWavMonitor();
  renderModules();
  await postCommand({ type: 'audio_input_send_command', command: 'wav_end' });
}

async function playWavFromHud() {
  if (state.wavPlaying) return;
  await toggleWavPlayPause();
}

async function stopWavFromHud() {
  if (!state.wavPlaying) return;
  await toggleWavPlayPause();
}

function setConn(kind, text) {
  $('#conn-dot').className = `dot ${kind}`;
  $('#conn-text').textContent = `対話サーバー：${text}`;
}

// 上の帯の状態（Qt の _refresh_header_status と同じ 3 つ）。kind: ok / warn / idle / err
function setChip(name, kind, text) {
  const dot = document.getElementById(`chip-${name}-dot`);
  const label = document.getElementById(`chip-${name}-text`);
  if (dot) dot.className = `dot ${kind}`;
  if (label) label.textContent = text;
}

function showError(e) {
  const message = String(e?.message ?? e);
  $('#save-state').textContent = message;
  // 設定の引き出しを閉じていても見えるよう、メイン画面にもしばらく出す。
  console.error(e);
  const hud = document.getElementById('hud-error');
  if (hud) {
    hud.textContent = message;
    hud.classList.remove('hidden');
    window.clearTimeout(showError.timer);
    showError.timer = window.setTimeout(() => hud.classList.add('hidden'), 8000);
  }
}

function setMdfReadOnly(readOnly, reason = '') {
  state.mdfReadOnly = Boolean(readOnly);
  state.mdfReadOnlyReason = state.mdfReadOnly
    ? String(reason || 'このMDFは読み取り専用です。')
    : '';
  for (const [attr, node] of state.byAttr) {
    if (!attr.startsWith('mdf:')) continue;
    node.disabled = state.mdfReadOnly;
    if (state.mdfReadOnly) node.title = state.mdfReadOnlyReason;
  }
  for (const attr of ['save_mdf_btn', 'save_as_mdf_btn', 'reset_defaults_btn']) {
    const button = state.byAttr.get(attr);
    if (!button) continue;
    button.disabled = state.mdfReadOnly;
    if (state.mdfReadOnly) button.title = state.mdfReadOnlyReason;
  }
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = el('textarea');
  area.value = text;
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.append(area);
  area.select();
  document.execCommand('copy');
  area.remove();
}

function renderMmdagentHostGuide(status) {
  const launch = state.byAttr.get('mmdagent_launch_btn');
  const body = launch?.closest('.group')?.querySelector('.group-body');
  if (!body) return;
  let panel = document.getElementById('mmdagent-host-guide');
  if (!panel) {
    panel = el('div', 'mmdagent-host-guide');
    panel.id = 'mmdagent-host-guide';
    body.append(panel);
  }
  const docker = Boolean(status.docker ?? state.runtime?.docker);
  panel.hidden = !docker;
  if (!docker) return;
  panel.replaceChildren();

  panel.append(el('div', 'mmdagent-guide-title', 'WindowsホストのMMDAgent-EX'));
  panel.append(el(
    'div',
    `mmdagent-remote-state ${status.remote_connected ? 'connected' : 'disconnected'}`,
    status.remote_connected
      ? `Remote接続済み${status.remote_mode ? `（${status.remote_mode}）` : ''}`
      : 'Remote未接続',
  ));

  const addRow = (label, value, className = '') => {
    const row = el('div', `mmdagent-guide-row ${className}`.trim());
    row.append(el('span', 'mmdagent-guide-label', label));
    row.append(el('span', 'mmdagent-guide-value', String(value || '—')));
    panel.append(row);
  };
  const transport = status.transport || {};
  addRow('MDF', `${status.host_mdf_name || '—'}（コンテナ：${status.mdf_path || '—'}）`);
  addRow('通信方式', transport.label || transport.mode || '—');
  addRow('MDF設定', transport.mdf_settings || '—', 'mono');
  addRow('接続先', transport.instruction || transport.endpoint || '—', 'mono');
  if (transport.both_enabled) {
    panel.append(el(
      'div',
      'mmdagent-guide-warning',
      'EnableServerとEnableClientが両方trueのため、EnableServer（TCPクライアント）が優先されます。',
    ));
  }

  const command = String(status.host_command || '');
  panel.append(el('div', 'mmdagent-guide-label', 'ホストPowerShell（Releaseフォルダーで実行）'));
  const commandRow = el('div', 'mmdagent-command-row');
  commandRow.append(el('code', null, command || '—'));
  const copy = el('button', null, 'コピー');
  copy.type = 'button';
  copy.disabled = !command;
  copy.addEventListener('click', async () => {
    try {
      await copyText(command);
      copy.textContent = 'コピー済み';
      window.setTimeout(() => { copy.textContent = 'コピー'; }, 1200);
    } catch (e) {
      showError(e);
    }
  });
  commandRow.append(copy);
  panel.append(commandRow);
  if (status.mdf_read_only_reason) {
    panel.append(el('div', 'mmdagent-guide-readonly', status.mdf_read_only_reason));
  }
}

function applyMmdagentStatus(status) {
  if (!status || typeof status !== 'object') return;
  state.runtime.mmdagent = status;
  const launch = state.byAttr.get('mmdagent_launch_btn');
  const stop = state.byAttr.get('mmdagent_stop_btn');
  const available = Boolean(status.launch_available);
  const running = Boolean(status.running);
  if (launch) {
    launch.disabled = running || !available;
    launch.textContent = running
      ? `MMDAgent-EX起動中${status.pid ? `（PID：${status.pid}）` : ''}`
      : (available ? 'MMDAgent-EXを起動' : 'MMDAgent-EXはホスト側で起動');
    launch.title = available
      ? `実行ファイル：${status.exec_path || ''}\nMDF：${status.mdf_path || ''}`
      : String(status.reason || 'この環境からは起動できません');
  }
  if (stop) {
    stop.disabled = !available || !running;
    stop.title = available ? 'このWebサーバーが起動したMMDAgent-EXを停止します' : String(status.reason || '');
  }
  if (status.mdf_read_only) {
    setMdfReadOnly(true, status.mdf_read_only_reason);
  }
  renderMmdagentHostGuide(status);
  renderMmdCard(status);
}

// メイン画面の MMDAgent-EX カード。アバターは別ウィンドウの MMDAgent-EX に出るので、
// ここでは接続状態と、起動（native）またはホストでの起動方法（Docker）だけを見せる。
function renderMmdCard(status) {
  const card = document.getElementById('mmd-card');
  if (!card) return;
  const docker = Boolean(status.docker ?? state.runtime?.docker);
  const available = Boolean(status.launch_available);
  const running = Boolean(status.running);
  const connected = Boolean(status.remote_connected);
  const transport = status.transport || {};
  card.replaceChildren();

  setChip('mmd', connected ? 'ok' : running ? 'warn' : 'idle',
    connected ? `MMDAgent-EX：接続済み${status.remote_mode ? `（${status.remote_mode}）` : ''}`
      : running ? 'MMDAgent-EX：起動中（未接続）' : 'MMDAgent-EX：未接続');
  // 右の列をスクロールなしで収めるため、見出し行は置かず、操作の行の右端に接続の印を出す。
  const top = el('div', 'mmd-card-actions');
  card.append(top);
  const pill = el(
    'span',
    `mmd-card-pill ${connected ? 'ok' : 'off'}`,
    connected ? `接続済み${status.remote_mode ? `（${status.remote_mode}）` : ''}` : '未接続',
  );

  // いつも見るもの（状態・接続先・操作）だけを並べ、方式や MDF の値は「詳細」に畳む。
  // 音声対話の欄が画面外へ押し出されないよう、Qt のメイン画面と同じく短く保つ。
  const row = (parent, label, value) => {
    if (!value) return;
    const r = el('div', 'mmd-card-row');
    r.append(el('span', 'mmd-card-label', label));
    r.append(el('span', 'mmd-card-value', String(value)));
    parent.append(r);
  };
  const details = el('details', 'mmd-card-details');
  details.append(el('summary', null, '詳細'));
  if (!docker) {
    // プロセスの状態は操作の行に短く出す（PID は「詳細」へ）。
    top.append(el('span', 'mmd-card-proc', `プロセス：${running ? '起動中' : '停止'}`));
    row(details, 'PID', status.pid);
    row(details, 'EXE', status.exec_path);
    row(details, 'MDF', status.mdf_path);
  }
  row(docker ? card : details, '接続先', transport.endpoint || transport.instruction);
  row(details, '通信方式', transport.label || transport.mode);

  if (docker) {
    row(details, 'MDF', status.host_mdf_name || status.mdf_path);
    row(details, 'MDF設定', transport.mdf_settings);
    if (status.mdf_read_only_reason) {
      details.append(el('div', 'mmd-card-note', String(status.mdf_read_only_reason)));
    }
    card.append(el('div', 'mmd-card-label', 'ホストのReleaseフォルダーで実行'));
    const command = String(status.host_command || '');
    const cmdRow = el('div', 'mmd-card-cmd');
    cmdRow.append(el('code', null, command || '—'));
    const copy = el('button', null, 'コピー');
    copy.type = 'button';
    copy.disabled = !command;
    copy.addEventListener('click', async () => {
      try {
        await copyText(command);
        copy.textContent = 'コピー済み';
        window.setTimeout(() => { copy.textContent = 'コピー'; }, 1200);
      } catch (e) {
        showError(e);
      }
    });
    cmdRow.append(copy);
    card.append(cmdRow);
  } else if (available) {
    const actions = top;
    const start = el('button', 'primary', running ? '起動中' : '起動');
    start.type = 'button';
    start.disabled = running;
    start.addEventListener('click', () => startMmdagent().catch(showError));
    const stop = el('button', null, '停止');
    stop.type = 'button';
    stop.disabled = !running;
    stop.addEventListener('click', () => stopMmdagent().catch(showError));
    actions.prepend(start, stop);
  } else if (status.reason) {
    card.append(el('div', 'mmd-card-note', String(status.reason)));
  }
  top.append(pill);
  details.append(el('div', 'mmd-card-hint', 'アバターはMMDAgent-EXのウィンドウに表示されます。'));
  // 開閉状態は 3 秒ごとの再描画でも保つ。
  if (state.mmdDetailsOpen) details.open = true;
  details.addEventListener('toggle', () => { state.mmdDetailsOpen = details.open; });
  card.append(details);
}

async function refreshMmdagentStatus() {
  const status = await api.get('/api/mmdagent');
  applyMmdagentStatus(status);
  return status;
}

async function startMmdagent() {
  await flushAppConfig();
  if (state.mdfDirty.size) {
    await saveMdf(false);
    if (state.mdfDirty.size) throw new Error('MDFを保存できなかったためMMDAgent-EXを起動しません。');
  }
  const status = await api.post('/api/mmdagent/start', {});
  applyMmdagentStatus(status);
  appendConsole(`[MMDAgent] ${status.message || '起動要求を送信しました'}`);
  if (status.fst_message) appendConsole(`[MMDAgent] ${status.fst_message}`);
  if (status.bridge_error) appendConsole(`[MMDAgent][WARN] ${status.bridge_error}`);
  $('#save-state').textContent = status.message || 'MMDAgent-EXを起動しました';
}

async function stopMmdagent() {
  const status = await api.post('/api/mmdagent/stop', {});
  applyMmdagentStatus(status);
  appendConsole(`[MMDAgent] ${status.message || '停止要求を送信しました'}`);
  $('#save-state').textContent = status.message || 'MMDAgent-EXを停止しました';
}

async function pickHostInputWav() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'audio/wav,audio/x-wav,.wav';
  const file = await new Promise((resolve) => {
    input.addEventListener('change', () => resolve(input.files?.[0] || null), { once: true });
    input.click();
  });
  if (!file) return;
  $('#save-state').textContent = 'WAVをアップロード中…';
  const r = await fetch(`/api/input-wav?name=${encodeURIComponent(file.name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  });
  if (!r.ok) {
    let detail = await r.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail ?? detail;
    } catch {
      /* 本文はそのまま */
    }
    throw new Error(String(detail));
  }
  const data = await r.json();
  const path = data.path;
  setAttrValue('audio_input_wav_path_edit', path);
  const keyNode = state.byAttr.get('audio_input_wav_path_edit');
  markDirty('audio_input_wav_path', path, keyNode, 'app');
  await saveConfig();
  renderModules();
}

const DOCKER_HOST_UPLOAD_BTNS = {
  maai_test_bc_local_model_browse_btn: '.pt,.pth,application/octet-stream',
  maai_test_turn_vap_local_model_browse_btn: '.pt,.pth,application/octet-stream',
  maai_mimi_onnx_path_browse_btn: '.onnx,application/octet-stream',
  maai_mimi_onnx_meta_path_browse_btn: '.json,application/json',
  maai_backchannel_dictionary_path_browse_btn: '.json,application/json',
};

async function pickHostUpload(w) {
  const targetAttr = w.browse_target || inferBrowseTarget(w.attr);
  const accept = DOCKER_HOST_UPLOAD_BTNS[w.attr] || '';
  const input = document.createElement('input');
  input.type = 'file';
  if (accept) input.accept = accept;
  const file = await new Promise((resolve) => {
    input.addEventListener('change', () => resolve(input.files?.[0] || null), { once: true });
    input.click();
  });
  if (!file) return;
  $('#save-state').textContent = `${file.name}をアップロード中…`;
  const r = await fetch(`/api/host-upload?name=${encodeURIComponent(file.name)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  });
  if (!r.ok) {
    let detail = await r.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail ?? detail;
    } catch {
      /* 本文はそのまま */
    }
    throw new Error(String(detail));
  }
  const data = await r.json();
  const path = data.path;
  setAttrValue(targetAttr, path);
  const keyNode = state.byAttr.get(targetAttr);
  const wkey = [...state.inputs.entries()].find(([, n]) => n === keyNode)?.[0]
    || (targetAttr.endsWith('_edit') ? targetAttr.slice(0, -5) : '');
  if (wkey) markDirty(wkey, path, keyNode, 'app');
  await saveConfig();
  $('#save-state').textContent = `保存済み：${file.name}`;
}

function applyDockerWavBrowseUi() {
  if (!state.runtime?.docker) return;
  const btn = state.byAttr.get('audio_input_wav_browse_btn');
  if (btn) {
    btn.textContent = 'PCから選ぶ…';
    btn.title = 'このPCのWAVを選んでコンテナにコピーします。Shiftキーを押しながらクリックすると、コンテナ内の参照ダイアログを開きます。';
  }
  const monitor = state.byAttr.get('audio_input_wav_local_playback_check');
  if (monitor) {
    monitor.title = 'WAVをパイプラインへ送ると同時に、このブラウザでモニター再生します。';
    const span = monitor.parentElement?.querySelector('span');
    if (span) span.textContent = ' ブラウザで再生';
  }
  for (const attr of Object.keys(DOCKER_HOST_UPLOAD_BTNS)) {
    const b = state.byAttr.get(attr);
    if (!b) continue;
    b.textContent = 'PCから選ぶ…';
    b.title = 'このPCのファイルを選んでコンテナにコピーします。Shiftキーを押しながらクリックすると、コンテナ内の参照ダイアログを開きます。';
  }
}

function applyRuntimeCapabilities() {
  const source = state.byAttr.get('web_mic_source_combo');
  if (source && source.tagName === 'SELECT') {
    const local = [...source.options].find((option) => option.value === 'local');
    if (local) {
      local.disabled = localMicUnavailable();
      local.title = localMicUnavailable()
        ? 'Dockerの対話サーバーはホストのPortAudioデバイスを直接使えないため、ブラウザマイク固定です'
        : '';
    }
    if (localMicUnavailable()) source.value = 'browser';
  }
  applyMmdagentStatus(state.runtime?.mmdagent || {});
}

async function handleButton(w, node, ev) {
  const attr = w.attr || '';
  try {
    if (attr === 'asr_maai_plot_btn' || w.text === 'MaAIプロット') {
      openPlot();
      return;
    }
    if (OPEN_TAB_BY_ATTR[attr]) {
      openSettings(OPEN_TAB_BY_ATTR[attr]);
      return;
    }
    if (attr === 'mmdagent_launch_btn') {
      await startMmdagent();
      return;
    }
    if (attr === 'mmdagent_stop_btn') {
      await stopMmdagent();
      return;
    }
    if (attr === 'audio_input_wav_browse_btn' && state.runtime?.docker && !ev?.shiftKey) {
      await pickHostInputWav();
      return;
    }
    if (DOCKER_HOST_UPLOAD_BTNS[attr] && state.runtime?.docker && !ev?.shiftKey) {
      await pickHostUpload(w);
      return;
    }
    if (w.browse_target || /browse/i.test(attr) || String(w.text || '').includes('参照')) {
      await openBrowse(w.browse_target || inferBrowseTarget(attr), w);
      return;
    }
    if (attr.endsWith('_module_start_btn')) {
      sendModuleCommand(attr.replace('_module_start_btn', ''), 'start');
      return;
    }
    if (attr.endsWith('_module_stop_btn')) {
      sendModuleCommand(attr.replace('_module_stop_btn', ''), 'stop');
      return;
    }
    if (attr === 'voice_dialog_start_btn') {
      await startVoiceDialog();
      return;
    }
    if (attr === 'voice_dialog_stop_btn') {
      await stopVoiceDialog();
      return;
    }
    if (attr === 'asr_launch_btn') {
      sendModuleCommand('asr', 'start');
      return;
    }
    if (attr === 'asr_stop_btn') {
      sendModuleCommand('asr', 'stop');
      return;
    }
    if (attr === 'asr_clear_btn') {
      setTextByAttr('asr_result_edit', '');
      return;
    }
    if (attr === 'maai_test_launch_btn') {
      sendModuleCommand('maai', 'start');
      return;
    }
    if (attr === 'maai_test_stop_btn') {
      sendModuleCommand('maai', 'stop');
      return;
    }
    if (attr === 'maai_test_clear_btn') {
      setTextByAttr('maai_test_log_edit', '');
      return;
    }
    if (attr in TTS_BUTTON_ENGINE) {
      const engine = TTS_BUTTON_ENGINE[attr];
      const stop = attr.endsWith('_stop_btn');
      await postCommand({
        type: stop ? 'stop_module' : 'start_module',
        module: 'tts',
        engine,
      });
      return;
    }
    if (attr === 'tts_test_send_btn') {
      unlockBrowserAudio();
      await postCommand({
        type: 'start_tts_gui_test',
        utterance: attrValue('tts_test_text_edit', state.config.tts_test_text || ''),
        engine: attrValue('tts_engine_combo', state.config.tts_engine || ''),
      });
      return;
    }
    if (attr === 'llm_chat_send_btn') {
      await sendLlmTest();
      return;
    }
    if (attr === 'llm_chat_reset_btn') {
      state.llmMessages = [];
      setTextByAttr('llm_chat_view', '');
      return;
    }
    if (attr === 'llm_warmup_btn') {
      await postCommand({ type: 'start_llm_warmup', force: true });
      return;
    }
    if (attr === 'llm_model_lmstudio_refresh_btn') {
      await refreshLlmModels('lmstudio');
      return;
    }
    if (attr === 'llm_model_openai_refresh_btn') {
      await refreshLlmModels('openai');
      return;
    }
    if (attr === 'llm_model_unload_btn') {
      const res = await api.post('/api/llm-admin', {
        op: 'lmstudio_bulk_unload',
        host: attrValue('llm_host_edit', state.config.llm_host || ''),
        current_model: attrValue('llm_model_lmstudio_combo', state.config.llm_model_lmstudio || ''),
      });
      appendConsole(`[LM Studio unload] attempted=${res.attempted} ok=${res.ok} ${res.error || ''}`);
      return;
    }
    if (attr === 'llm_s2s_test_run_btn') {
      const res = await api.post('/api/s2s-preview', {
        text: attrValue('llm_s2s_test_input_edit', ''),
        chunk_size: Number(attrValue('llm_s2s_test_chunk_size_edit', '2')) || 2,
        enabled: Boolean(state.byAttr.get('llm_test_tts_input_streaming_check')?.checked
          ?? state.config.llm_test_tts_input_streaming_enabled),
      });
      setAttrValue('llm_s2s_test_output_edit', res.text || '');
      return;
    }
    if (attr === 'audio_input_wav_play_pause_btn') {
      await toggleWavPlayPause();
      return;
    }
    if (attr === 'audio_input_wav_end_btn') {
      await endWavPlayback();
      return;
    }
    if (attr === 'reload_mdf_btn') {
      await loadMdf();
      return;
    }
    if (attr === 'save_mdf_btn') {
      await saveMdf(false);
      return;
    }
    if (attr === 'save_as_mdf_btn') {
      await saveMdf(true);
      return;
    }
    if (attr === 'reset_defaults_btn') {
      const data = await api.get('/api/mdf/defaults');
      applyMdfValues(data.values || {});
      for (const k of Object.keys(data.values || {})) state.mdfDirty.set(k, data.values[k]);
      $('#save-state').textContent = 'MDFを既定値に戻しました（未保存）';
      return;
    }
    if (attr === 'obs_remote_test_btn' || attr === 'obs_manual_start_btn' || attr === 'obs_manual_stop_btn') {
      const action = attr === 'obs_remote_test_btn' ? 'test' : attr === 'obs_manual_start_btn' ? 'start' : 'stop';
      const res = await api.post(`/api/obs/${action}`, obsPayload());
      appendConsole(`[OBS ${action}] ok=${res.ok} ${res.error || ''} ${res.hint || ''}`);
      const actionLabel = { test: '接続テスト', start: '録画開始', stop: '録画停止' }[action] || action;
      $('#save-state').textContent = res.ok ? `OBS${actionLabel}成功` : (res.error || res.hint || 'OBS操作失敗');
      return;
    }
    if (attr === 'template_load_btn') {
      await openBrowse(null, { ...w, fileOnly: true, onFile: async (path) => {
        const file = await api.get(`/api/file?path=${encodeURIComponent(path)}`);
        setAttrValue('tts_template_edit', file.content || '');
        const key = 'tts_template';
        markDirty(key, file.content || '', state.inputs.get(key), 'app');
      } });
      return;
    }
    if (attr === 'template_save_btn') {
      await openBrowse(null, { ...w, saveAs: true, onFile: async (path) => {
        await api.put('/api/file', { path, content: attrValue('tts_template_edit', '') });
        $('#save-state').textContent = `テンプレートを保存：${path}`;
      } });
      return;
    }
    if (attr === 'engine_install_apply_btn') {
      activateById('engine_manager_web');
      return;
    }
    node.disabled = true;
    node.title = 'この操作はまだWeb版に対応していません';
  } catch (e) {
    showError(e);
  }
}

function inferBrowseTarget(attr) {
  if (attr && attr.endsWith('_browse_btn')) return `${attr.slice(0, -'_browse_btn'.length)}_edit`;
  return '';
}

function obsPayload() {
  return {
    obs_remote_host: attrValue('obs_remote_host_edit', state.config.obs_remote_host),
    obs_remote_port: attrValue('obs_remote_port_edit', state.config.obs_remote_port),
    obs_remote_password: attrValue('obs_remote_password_edit', state.config.obs_remote_password),
  };
}

function collectMdfValues() {
  const values = { ...state.mdfValues };
  for (const [k, v] of state.mdfDirty) values[k] = v;
  return values;
}

function applyMdfValues(values) {
  state.mdfValues = { ...values };
  for (const [attr, node] of state.byAttr) {
    if (!attr.startsWith('mdf:')) continue;
    const key = attr.slice(4);
    if (!(key in values)) continue;
    const v = values[key];
    if (node.type === 'checkbox') node.checked = String(v).toLowerCase() === 'true' || v === true;
    else node.value = v == null ? '' : String(v);
    node.classList.remove('field-dirty');
  }
  state.mdfDirty.clear();
}

async function loadMdf(path) {
  const q = path ? `?path=${encodeURIComponent(path)}` : '';
  const data = await api.get(`/api/mdf${q}`);
  state.mdfPath = data.path || '';
  applyMdfValues(data.values || {});
  setMdfReadOnly(
    Boolean(state.runtime?.docker) && data.writable === false,
    data.read_only_reason || 'このMDFは読み取り専用です。',
  );
  if (data.path) {
    setAttrValue('mdf_edit', data.path);
    markDirty('mdf_path', data.path, state.inputs.get('mdf_path'), 'app');
  }
  $('#save-state').textContent = data.writable === false
    ? 'MDFを読み込みました（読み取り専用）'
    : 'MDFを読み込みました';
}

async function saveMdf(saveAs) {
  if (state.mdfReadOnly) {
    $('#save-state').textContent = state.mdfReadOnlyReason;
    return;
  }
  const values = collectMdfValues();
  if (saveAs) {
    await openBrowse(null, {
      saveAs: true,
      onFile: async (path) => {
        const res = await api.post('/api/mdf/save-as', { path, values });
        state.mdfPath = res.path;
        applyMdfValues(res.values || values);
        setAttrValue('mdf_edit', res.path);
        markDirty('mdf_path', res.path, state.inputs.get('mdf_path'), 'app');
        $('#save-state').textContent = `MDFを新規保存：${res.path}`;
      },
    });
    return;
  }
  try {
    // 編集したキーだけを touched として送る。values は全キー（既定値込み）なので、
    // これを送らないとサーバが全キーを「触った」扱いにして、既定値まで明示行として
    // 書き出す（3 行の .mdf が 165 行に膨れた）。PyQt 版と同じ loaded/touched の分離。
    const touched = Array.from(state.mdfDirty.keys());
    const res = await api.post('/api/mdf', { path: state.mdfPath || attrValue('mdf_edit', ''), values, touched });
    applyMdfValues(res.values || values);
    $('#save-state').textContent = 'MDFを保存しました';
  } catch (e) {
    $('#save-state').textContent = `MDF保存失敗：${e.message}（新規保存を使ってください）`;
  }
}

async function sendLlmTest() {
  const user = String(attrValue('llm_user_input_edit', '')).trim();
  if (!user) return;
  const backend = String(attrValue('llm_backend_combo', state.config.llm_backend || 'LM Studio'));
  const host = backend === 'Custom'
    ? String(attrValue('llm_custom_base_url_edit', state.config.llm_custom_base_url || ''))
    : backend === 'OpenAI'
      ? ''
      : String(attrValue('llm_host_edit', state.config.llm_host || ''));
  const model = backend === 'OpenAI'
    ? String(attrValue('llm_model_openai_combo', state.config.llm_model_openai || ''))
    : backend === 'Custom'
      ? String(attrValue('llm_model_custom_combo', state.config.llm_model_custom || ''))
      : String(attrValue('llm_model_lmstudio_combo', state.config.llm_model_lmstudio || ''));
  const system = String(attrValue('llm_system_prompt_edit', state.config.llm_system_prompt || '')).trim();
  if (!state.llmMessages.length && system) state.llmMessages.push({ role: 'system', content: system });
  state.llmMessages.push({ role: 'user', content: user });
  state.llmTestStreaming = false;
  state.llmTestBlockStart = -1;
  state.llmTestFull = '';
  state.llmTestSegments = [];
  setAttrValue('llm_user_input_edit', '');
  {
    // PySide と同じ表記（"you> " の後に "assistant> " を続け、応答ブロックを始める）。
    const view = state.byAttr.get('llm_chat_view');
    const prefix = view && view.value ? '\n' : '';
    appendAttr('llm_chat_view', `${prefix}ユーザー：${user}\n`);
    ensureLlmTestBlock();
  }
  state.streamText = '';
  renderStream();
  await postCommand({
    type: 'start_llm_gui_test',
    messages: state.llmMessages.map((m) => ({ ...m })),
    backend,
    host,
    model,
  });
}

async function refreshLlmModels(kind) {
  const op = kind === 'openai' ? 'openai_fetch_models' : 'lmstudio_fetch_models';
  const host = String(attrValue('llm_host_edit', state.config.llm_host || ''));
  const res = await api.post('/api/llm-admin', kind === 'openai' ? { op } : { op, host });
  if (res.error) {
    appendConsole(`[モデル取得失敗] ${res.error}`);
    showError(new Error(res.error));
    return;
  }
  if (res.host && kind !== 'openai') {
    setAttrValue('llm_host_edit', res.host);
    if (state.config.llm_host !== res.host) {
      markDirty('llm_host', res.host, state.byAttr.get('llm_host_edit'), 'app');
    }
  }
  if (res.hint) appendConsole(`[モデル取得] ${res.hint}`);
  const attr = kind === 'openai' ? 'llm_model_openai_combo' : 'llm_model_lmstudio_combo';
  const sel = state.byAttr.get(attr);
  if (sel && sel.tagName === 'SELECT') {
    const cur = sel.value;
    sel.innerHTML = '';
    for (const name of res.models || []) sel.append(new Option(name, name));
    if (cur && ![...sel.options].some((o) => o.value === cur)) sel.append(new Option(cur, cur));
    sel.value = cur || (res.models || [])[0] || '';
  }
  appendConsole(`[モデル取得] ${kind} ${(res.models || []).length} 件`);
}

function selectedWebMicSource() {
  if (state.runtime?.docker) return 'browser';
  const source = String(attrValue('web_mic_source_combo', state.config.web_mic_source || 'browser')).toLowerCase();
  return source === 'local' ? 'local' : 'browser';
}

function usesBrowserMic() {
  const mode = String(attrValue('audio_input_mode_combo', state.config.audio_input_mode || 'mic')).toLowerCase();
  return selectedWebMicSource() === 'browser' && mode === 'mic';
}

function setMicUi(text, rms = 0) {
  const box = $('#mic-status');
  const label = $('#mic-label');
  const bar = $('#mic-meter i');
  if (!box || !label) return;
  box.classList.toggle('hidden', !usesBrowserMic());
  label.textContent = text;
  if (bar) bar.style.width = `${Math.min(100, Math.round(rms * 160))}%`;
}

function stopBrowserMic() {
  const m = state.mic;
  try { m.ws?.close(); } catch { /* ignore */ }
  try { m.proc?.disconnect(); } catch { /* ignore */ }
  try { m.source?.disconnect(); } catch { /* ignore */ }
  try { m.ctx?.close(); } catch { /* ignore */ }
  m.stream?.getTracks().forEach((t) => t.stop());
  state.mic = { stream: null, ctx: null, proc: null, source: null, ws: null, starting: false };
  setMicUi('マイク：停止', 0);
}

async function startBrowserMic(restart = false) {
  if (!usesBrowserMic()) return;
  if (state.mic.starting) return;
  if (state.mic.ws && state.mic.ws.readyState === WebSocket.OPEN && !restart) return;
  state.mic.starting = true;
  try {
    if (restart || state.mic.stream) {
      try { state.mic.ws?.close(); } catch { /* ignore */ }
      try { state.mic.proc?.disconnect(); } catch { /* ignore */ }
      try { state.mic.source?.disconnect(); } catch { /* ignore */ }
      try { state.mic.ctx?.close(); } catch { /* ignore */ }
      state.mic.stream?.getTracks().forEach((t) => t.stop());
      state.mic = { stream: null, ctx: null, proc: null, source: null, ws: null, starting: true };
    }
    const deviceId = String(attrValue('audio_input_mic_device_combo', '') || '');
    const audio = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    };
    if (deviceId && deviceId !== 'default' && !/^\d+:/.test(deviceId)) {
      audio.deviceId = { exact: deviceId };
    }
    setMicUi('マイク：許可待ち');
    const stream = await navigator.mediaDevices.getUserMedia({ audio });
    const ctx = new AudioContext();
    await ctx.resume();
    const resampler = createStreamingPcm16Resampler(ctx.sampleRate, 16000);
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(512, 1, 1);
    let micPcmCarry = new Int16Array(0);
    const mute = ctx.createGain();
    mute.gain.value = 0;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/mic`);
    ws.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', () => reject(new Error('マイク送信用WebSocketを開けません')), { once: true });
    });
    proc.onaudioprocess = (ev) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const input = ev.inputBuffer.getChannelData(0);
      let sum = 0;
      for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      setMicUi('マイク：取り込み中', Math.sqrt(sum / Math.max(1, input.length)));
      const pcm = resampler.process(input);
      if (!pcm.length) return;
      const joined = new Int16Array(micPcmCarry.length + pcm.length);
      joined.set(micPcmCarry, 0);
      joined.set(pcm, micPcmCarry.length);
      let offset = 0;
      // AudioInputWorker と同じ 20ms (320 samples @ 16kHz) で送る。
      // ScriptProcessor ごとの約170 samplesをそのまま送ると、Web版だけ
      // user/system 合計約188 JSON events/secになりプロット配信が詰まる。
      while (joined.length - offset >= 320 && ws.readyState === WebSocket.OPEN) {
        const frame = joined.slice(offset, offset + 320);
        ws.send(frame.buffer);
        offset += 320;
      }
      micPcmCarry = joined.slice(offset);
    };
    source.connect(proc);
    proc.connect(mute);
    mute.connect(ctx.destination);
    state.mic = { stream, ctx, proc, source, ws, starting: false };
    await fillBrowserMics();
    appendConsole(`[マイク] ブラウザ取り込み開始 (${Math.round(ctx.sampleRate)}Hz → 16kHz)`);
  } catch (e) {
    state.mic.starting = false;
    stopBrowserMic();
    setMicUi(`マイク：失敗`);
    throw e;
  }
}

async function acquireBrowserMic(who) {
  state.micUsers.add(who);
  if (!usesBrowserMic()) return;
  await startBrowserMic();
}

function releaseBrowserMic(who) {
  state.micUsers.delete(who);
  if (!state.micUsers.size) stopBrowserMic();
}

async function fillBrowserMics() {
  const sel = state.byAttr.get('audio_input_mic_device_combo');
  if (!sel || sel.tagName !== 'SELECT') return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter((d) => d.kind === 'audioinput');
    const cur = String(
      state.dirty.get('web_browser_mic_device_id')
      ?? state.config.web_browser_mic_device_id
      ?? '',
    );
    sel.innerHTML = '';
    sel.append(new Option('default', 'default'));
    for (const d of mics) sel.append(new Option(d.label || d.deviceId, d.deviceId));
    if (cur && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
    if (!sel.dataset.browserMicBound) {
      sel.dataset.browserMicBound = '1';
      sel.addEventListener('change', () => {
        if (state.micUsers.size) startBrowserMic(true).catch(showError);
      });
    }
  } catch {
    /* 許可前はラベルが空でもよい */
  }
}

async function startVoiceDialog() {
  try {
    unlockBrowserAudio();
    await ensureDialogServer();
    await acquireBrowserMic('voice');
    await postCommand({ type: 'start_voice_dialog' });
    autoOpenPlot();
  } catch (e) {
    releaseBrowserMic('voice');
    showError(e);
  }
}

async function stopVoiceDialog() {
  releaseBrowserMic('voice');
  await postCommand({ type: 'stop_voice_dialog' }).catch(showError);
}

async function refreshMicDevices() {
  if (selectedWebMicSource() === 'browser') {
    await fillBrowserMics();
    return;
  }
  try {
    const data = await api.get('/api/audio-devices');
    const sel = state.byAttr.get('audio_input_mic_device_combo');
    if (!sel || sel.tagName !== 'SELECT') return;
    const cur = String(
      state.dirty.get('audio_input_mic_device_index')
      ?? state.config.audio_input_mic_device_index
      ?? '-1',
    );
    sel.innerHTML = '';
    for (const d of data.devices || []) sel.append(new Option(d.label, String(d.id)));
    if (cur && ![...sel.options].some((o) => o.value === cur)) sel.append(new Option(cur, cur));
    sel.value = cur || sel.options[0]?.value || '';
  } catch {
    /* マイク列挙はコンテナでは空でもよい */
  }
}

function ensureBrowseModal() {
  let modal = document.getElementById('browse-modal');
  if (modal) return modal;
  modal = el('div', 'browse-modal');
  modal.id = 'browse-modal';
  modal.innerHTML = `
    <div class="browse-card">
      <header>
        <h3>パスを選択</h3>
        <button type="button" data-browse-close>閉じる</button>
      </header>
      <div class="browse-path"></div>
      <div class="browse-roots"></div>
      <div class="browse-list"></div>
      <footer>
        <input type="text" class="browse-file" placeholder="ファイル名（新規保存時）">
        <button type="button" class="primary" data-browse-ok>選択</button>
      </footer>
    </div>`;
  document.body.append(modal);
  return modal;
}

async function openBrowse(targetAttr, opts = {}) {
  const modal = ensureBrowseModal();
  const list = modal.querySelector('.browse-list');
  const pathEl = modal.querySelector('.browse-path');
  const rootsEl = modal.querySelector('.browse-roots');
  const fileEl = modal.querySelector('.browse-file');
  fileEl.style.display = opts.saveAs ? '' : 'none';
  modal.classList.add('open');
  let current = '';
  const start = targetAttr ? attrValue(targetAttr, '') : '';
  if (start) current = start.replace(/[/\\][^/\\]*$/, '') || start;

  const close = () => modal.classList.remove('open');
  modal.querySelector('[data-browse-close]').onclick = close;

  const load = async (path) => {
    // 保存ダイアログでは書き込み可能なルートだけを出す。読み取り用の広いルートを
    // 出すと、選べるのに保存だけ 400 で弾かれる場所が見えてしまう。
    const w = opts.saveAs ? '&write=1' : '';
    // Docker の対話サーバーを使うときは、コンテナから読めるフォルダーをコンテナのパスで見せる。
    // MMDAgent-EX の実行ファイルはこの PC で動かすので、この PC のパスのまま選ぶ。
    const h = targetAttr === 'exec_edit' ? '&host=1' : '';
    const data = await api.get(`/api/browse?path=${encodeURIComponent(path || '')}${w}${h}`);
    current = data.path;
    pathEl.textContent = data.path;
    rootsEl.innerHTML = '';
    for (const root of data.roots || []) {
      const b = el('button', null, root);
      b.type = 'button';
      b.addEventListener('click', () => load(root).catch(showError));
      rootsEl.append(b);
    }
    list.innerHTML = '';
    const up = el('button', null, '..');
    up.type = 'button';
    up.addEventListener('click', () => {
      const parent = current.replace(/[/\\][^/\\]+$/, '');
      load(parent || current).catch(showError);
    });
    list.append(up);
    for (const ent of data.entries || []) {
      const b = el('button', ent.is_dir ? 'is-dir' : 'is-file', (ent.is_dir ? '📁 ' : '') + ent.name);
      b.type = 'button';
      b.addEventListener('click', () => {
        const next = `${current.replace(/[/\\]$/, '')}/${ent.name}`;
        if (ent.is_dir) load(next).catch(showError);
        else fileEl.value = ent.name;
      });
      b.addEventListener('dblclick', () => {
        const next = `${current.replace(/[/\\]$/, '')}/${ent.name}`;
        if (!ent.is_dir) choose(next);
      });
      list.append(b);
    }
  };

  const choose = async (path) => {
    close();
    if (opts.onFile) {
      await opts.onFile(path);
      return;
    }
    if (!targetAttr) return;
    setAttrValue(targetAttr, path);
    const keyNode = state.byAttr.get(targetAttr);
    const wkey = [...state.inputs.entries()].find(([, n]) => n === keyNode)?.[0];
    if (wkey) markDirty(wkey, path, keyNode, targetAttr.startsWith('mdf:') ? 'mdf' : 'app');
    else if (targetAttr === 'mdf_edit') markDirty('mdf_path', path, keyNode, 'app');
    else if (targetAttr === 'exec_edit') markDirty('exec_path', path, keyNode, 'app');
    if (targetAttr === 'audio_input_wav_path_edit') renderModules();
  };

  modal.querySelector('[data-browse-ok]').onclick = () => {
    const name = fileEl.value.trim();
    const path = name ? `${current.replace(/[/\\]$/, '')}/${name}` : current;
    choose(path).catch(showError);
  };
  await load(current).catch(async () => {
    await load('').catch(showError);
  });
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const sock = new WebSocket(`${proto}://${location.host}/ws/events`);
  state.eventsSock = sock;
  sock.addEventListener('open', () => {
    state.eventsOpen = true;
    renderServerChip();
    sendPlotAudioSubscribe(Boolean(state.plot?.isOpen()));
  });
  sock.addEventListener('close', () => {
    state.eventsOpen = false;
    if (state.switchingDialogServer) {
      reloadWhenServerBack();  // 切り替えで開き直したサーバーが立ち上がったら読み込み直す
      return;
    }
    renderServerChip();
    setTimeout(connect, 3000);
  });
  sock.addEventListener('message', (m) => {
    try { handleEvent(JSON.parse(m.data)); } catch { /* 不正なフレームは無視 */ }
  });
}

// ---- 会話の欄を別ウィンドウに出す（Qt の gui/chat_window.py と同じ） -------------------------
// メイン画面は ?chat=1 の窓を開いて自分の会話の欄を隠す。会話だけの窓は /ws/events から会話に関わる
// イベントだけを受けて描く。handleEvent は使わない（音声対話の最中に開くとブラウザのマイクを取り直したり、
// WAV の確認音を鳴らしたりして、音声が二重になるため）。
const CHAT_WINDOW_NAME = 'maai-sds-chat';

function isChatOnlyWindow() {
  return new URLSearchParams(location.search).get('chat') === '1';
}

function setChatDetached(detached) {
  state.chatDetached = Boolean(detached);
  document.body.classList.toggle('chat-detached', state.chatDetached);
  window.clearInterval(state.chatWatch);
  if (state.chatDetached) {
    // 会話の窓が閉じられたら（「メイン画面に戻す」や窓の × で）、メイン画面に戻す
    state.chatWatch = window.setInterval(() => {
      if (!state.chatWin || state.chatWin.closed) setChatDetached(false);
    }, 700);
  }
}

function toggleChatPopout() {
  if (state.chatDetached) {
    try { state.chatWin?.close(); } catch { /* 閉じられなくても戻す */ }
    setChatDetached(false);
    return;
  }
  const win = window.open(`${location.pathname}?chat=1`, CHAT_WINDOW_NAME, 'popup,width=560,height=760');
  if (!win) {
    showError(new Error('別ウィンドウを開けませんでした。ブラウザのポップアップの許可を確認してください。'));
    return;
  }
  state.chatWin = win;
  win.focus();
  setChatDetached(true);
}

function handleChatEvent(ev) {
  switch (ev.type) {
    case 'runtime_snapshot':
      if (ev.history && typeof ev.history === 'object') renderHistory(ev.history);
      break;
    case 'history_snapshot':
      renderHistory(ev.history);
      break;
    case 'llm_chunk':
    case 'llm_done':
    case 'llm_test_finished':
      if (ev.assistant_text || ev.type === 'llm_chunk') {
        state.streamText = String(ev.assistant_text ?? '');
        renderStream();
      }
      break;
    case 'llm_test_chat_chunk': {
      const full = String(ev.assistant_text ?? ev.delta ?? '');
      state.streamText = full || state.streamText + String(ev.text ?? '');
      renderStream();
      break;
    }
    default:
      break;
  }
}

function connectChatEvents() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const sock = new WebSocket(`${proto}://${location.host}/ws/events`);
  sock.addEventListener('close', () => setTimeout(connectChatEvents, 3000));
  sock.addEventListener('message', (m) => {
    try { handleChatEvent(JSON.parse(m.data)); } catch { /* 不正なフレームは無視 */ }
  });
}

function chatMain() {
  document.body.classList.add('chat-only');
  document.title = '会話 - MaAI-SDS';
  $('#btn-chat-back')?.addEventListener('click', () => window.close());
  renderChat();
  connectChatEvents();
}

async function main() {
  if (isChatOnlyWindow()) {
    chatMain();
    return;
  }
  connectWavMonitorWs();
  renderModules();
  renderStream();
  renderLatency();
  setVoiceControls('stopped');

  $('#btn-settings').addEventListener('click', toggleSettings);
  $('#btn-chat-popout').addEventListener('click', toggleChatPopout);
  // メイン画面を閉じる・読み込み直すときは、開いた会話の窓も閉じる（取り残さない）
  window.addEventListener('pagehide', () => { try { state.chatWin?.close(); } catch { /* 無視 */ } });
  $('#btn-settings-close').addEventListener('click', closeSettings);
  $('#btn-plot').addEventListener('click', togglePlot);
  for (const button of document.querySelectorAll('[data-view-id]')) {
    button.addEventListener('click', () => {
      closePlot();
      openSettings(button.dataset.viewId || 'settings_window');
    });
  }
  state.plot = createMaaiPlot($('#plot-overlay'));
  $('#btn-start-all').addEventListener('click', () => startAllModules().catch(showError));
  $('#btn-stop-all').addEventListener('click', () => stopAllModules().catch(showError));
  $('#btn-start').addEventListener('click', () => startVoiceDialog());
  $('#btn-stop').addEventListener('click', () => stopVoiceDialog());
  document.addEventListener('pointerdown', () => {
    unlockBrowserAudio();
    if (state.voiceActive) restoreVoiceMic('running');
  }, { once: true });
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      saveConfig();
      if (state.mdfDirty.size) saveMdf(false);
    }
    if (e.key === 'Escape') {
      const modal = document.getElementById('browse-modal');
      if (modal?.classList.contains('open')) modal.classList.remove('open');
      else if (state.plot?.isOpen()) closePlot();
      else closeSettings();
    }
  });
  document.addEventListener('focusout', () => { if (state.dirty.size) saveConfig(); });

  try {
    [state.manifest, state.config, state.runtime] = await Promise.all([
      api.get('/api/ui/manifest'), api.get('/api/config'), api.get('/api/runtime').catch(() => state.runtime),
    ]);
  } catch (e) {
    document.body.innerHTML = `<pre style="padding:2rem">起動に失敗しました\n${e.message}</pre>`;
    return;
  }
  await buildViews();
  if (state.runtime?.dialog_server_notice) {
    appendConsole(`[対話サーバー] ${state.runtime.dialog_server_notice}`);
    window.setTimeout(() => window.alert(state.runtime.dialog_server_notice), 0);
  }
  applyRuntimeCapabilities();
  applyDockerWavBrowseUi();
  renderModules();
  state.plot?.setVisibilityFromConfig(state.config);
  await loadMdf().catch(() => {});
  await refreshMicDevices();
  setMicUi('マイク：停止');
  setInterval(() => refreshMmdagentStatus().catch(() => {}), 3000);
  connect();
}

main();

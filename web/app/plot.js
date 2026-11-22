// PyQt `MaaiPlotWindow` の Web 版。/ws/events の maai_result / audio_frame / asr_result を描く。

const WINDOW_SEC = 10;
const WAVE_HZ = 80;

function firstFloat(v, fallback = 0) {
  if (v == null) return fallback;
  if (typeof v === 'number') return Number.isFinite(v) ? v : fallback;
  if (Array.isArray(v)) {
    for (let i = v.length - 1; i >= 0; i--) {
      const x = firstFloat(v[i], NaN);
      if (Number.isFinite(x)) return x;
    }
    return fallback;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function pushPoint(buf, t, v) {
  buf.push(t, v);
  const cut = t - WINDOW_SEC - 0.5;
  let i = 0;
  while (i < buf.length && buf[i] < cut) i += 2;
  if (i) buf.splice(0, i);
}

function downsampleWave(samples, tsStart, tsEnd, out, fallbackNow) {
  const n = samples.length;
  if (!n) return;
  const t0 = Number(tsStart);
  const t1 = Number(tsEnd);
  const dur = Number.isFinite(t0) && Number.isFinite(t1) && t1 > t0 ? (t1 - t0) : n / 16000;
  const bins = Math.max(1, Math.round(dur * WAVE_HZ));
  const step = n / bins;
  for (let b = 0; b < bins; b++) {
    const a = Math.floor(b * step);
    const z = Math.min(n, Math.floor((b + 1) * step));
    let peak = 0;
    for (let i = a; i < z; i++) {
      const x = Math.abs(Number(samples[i]) || 0);
      if (x > peak) peak = x;
    }
    const t = (Number.isFinite(t0) ? t0 : fallbackNow) + (b / bins) * dur;
    pushPoint(out, t, peak);
  }
}

export function createAudioViewportClock(options = {}) {
  const wallNow = options.wallNow || (() => Date.now() / 1000);
  const monotonicNow = options.monotonicNow || (() => performance.now() / 1000);
  let sourceTs = null;
  let sourceMonotonic = null;
  return {
    note(rawTs) {
      const ts = Number(rawTs);
      if (!Number.isFinite(ts) || (sourceTs !== null && ts <= sourceTs)) return;
      sourceTs = ts;
      sourceMonotonic = monotonicNow();
    },
    now() {
      if (sourceTs === null || sourceMonotonic === null) return wallNow();
      return sourceTs + Math.max(0, monotonicNow() - sourceMonotonic);
    },
  };
}

const ROWS = [
  { id: 'x1', label: '入力波形1', kind: 'wave', ymin: -1, ymax: 1, color: '#dcc828', col: 'left' },
  { id: 'asr', label: 'ASRセグメント', kind: 'asr', ymin: 0, ymax: 1, col: 'left' },
  { id: 'voice_latency', label: '音声対話レイテンシ（色：ASR／LLM／TTS）＋割り込み', kind: 'latency', ymin: 0, ymax: 1, col: 'left' },
  { id: 'x2', label: '入力波形2', kind: 'wave', ymin: -1, ymax: 1, color: '#50a0f0', col: 'left' },
  { id: 'p_now', label: 'p_now（緑：ターン取得／赤：割り込み）', kind: 'prob', ymin: 0, ymax: 1, invert: true, col: 'left' },
  { id: 'p_future', label: '出力p_future（長期ターン予測）', kind: 'prob', ymin: 0, ymax: 1, invert: true, col: 'left' },
  { id: 'p_bins', label: 'p_bins（現在時刻の区間別確率）', kind: 'bins', ymin: 0, ymax: 2, col: 'left', hiddenByDefault: true },
  { id: 'vad', label: '音声活動検出（VAD）', kind: 'vad', ymin: -1, ymax: 1, col: 'left' },
  { id: 'silero_vad_score', label: 'Silero VAD（緑：開始／青：終了／赤：割り込み）', kind: 'line', ymin: 0, ymax: 1, color: '#ec7063', fill: true, col: 'left', hiddenByDefault: true },
  { id: 'p_bc', label: 'p_bc', kind: 'line', ymin: 0, ymax: 1, color: '#50c878', fill: true, col: 'right' },
  { id: 'p_nod', label: 'p_nod', kind: 'line', ymin: 0, ymax: 1, color: '#50c878', fill: true, col: 'right' },
  { id: 'nod_range', label: 'nod_range（0.035～0.15）', kind: 'line', ymin: 0.035, ymax: 0.15, color: '#b48cdc', ticks: [0.035, 0.093, 0.15], col: 'right', hiddenByDefault: true },
  { id: 'nod_speed', label: 'nod_speed（0.08～0.25）', kind: 'line', ymin: 0.08, ymax: 0.25, color: '#64c8dc', ticks: [0.08, 0.165, 0.25], col: 'right', hiddenByDefault: true },
  { id: 'nod_count', label: 'nod_count（クラス1／2／3以上）', kind: 'line', ymin: -0.15, ymax: 2.15, color: '#ffaa32', ticks: [0, 1, 2], tickLabels: ['1', '2', '3+'], col: 'right', hiddenByDefault: true },
  { id: 'nod_swing_up', label: 'nod_swing_up（確率）', kind: 'line', ymin: 0, ymax: 1, color: '#f064a0', fill: true, col: 'right', hiddenByDefault: true },
];

export function createMaaiPlot(root, options = {}) {
  const viewportClock = createAudioViewportClock();
  const series = {};
  for (const row of ROWS) {
    series[row.id] = [];
    if (row.kind === 'vad') {
      series.vad_1 = [];
      series.vad_2 = [];
    }
  }
  series.x1 = [];
  series.x2 = [];
  const asrSegs = [];
  // Qt の _open_stream_segments と同じ: ストリーミング途中結果は (slot, engine) ごとに 1 本を更新する。
  const openAsrStreams = new Map();
  // 壁時計 − 音声サンプルクロック。ブラウザマイクでは音声時刻が壁時計から遅れていくため、
  // 壁時計で刻まれた遅延帯・割り込み・ターンロックをこの差で音声時刻（表示の now）へ寄せる。
  let wallLag = null;
  const latencySpans = [];
  const nodRegions = [];
  const bcRegions = [];
  const turnMarks = [];
  const interruptMarks = [];
  const turnLockEdges = [];
  let pBins = { arr: null, times: [0.2, 0.4, 0.6, 0.8] };
  let thresholds = {
    interruptVad: 0.7,
    interruptPnow: 0.5,
    turnPnow: 0.5,
    vadOn: 0.1,
    vadOff: 0.2,
    pBc: 0.5,
    nodSwing: 0.5,
  };
  let asrArrivalAlign = false;
  // 設定に値が無いパネルは hiddenByDefault に従う（maai/plot_window.py の MAAI_PLOT_HIDDEN_BY_DEFAULT と揃える）
  const vis = Object.fromEntries(ROWS.map((r) => [r.id, !r.hiddenByDefault]));
  const canvases = new Map();
  let open = false;
  let raf = 0;
  let lastDraw = 0;

  const head = document.createElement('header');
  head.className = 'drawer-head';
  head.innerHTML = '<h2>MaAIプロット</h2>';
  const connection = document.createElement('span');
  connection.className = 'plot-connection';
  connection.dataset.plotConnection = '';
  if (options.standalone) head.append(connection);
  if (!options.standalone) {
    const popout = document.createElement('a');
    popout.className = 'plot-popout';
    popout.href = '/plot-view';
    popout.target = 'maai-plot-view';
    popout.textContent = '別タブで開く';
    head.append(popout);
  }
  const close = document.createElement('button');
  close.type = 'button';
  close.textContent = options.standalone ? 'タブを閉じる' : '閉じる';
  head.append(close);
  const body = document.createElement('div');
  body.className = 'plot-body';
  const left = document.createElement('div');
  left.className = 'plot-col';
  const right = document.createElement('div');
  right.className = 'plot-col';
  body.append(left, right);
  root.append(head, body);

  for (const row of ROWS) {
    const wrap = document.createElement('div');
    wrap.className = 'plot-row';
    wrap.dataset.plotId = row.id;
    const lab = document.createElement('div');
    lab.className = 'plot-label';
    lab.textContent = row.label;
    const cv = document.createElement('canvas');
    wrap.append(lab, cv);
    (row.col === 'right' ? right : left).append(wrap);
    canvases.set(row.id, { canvas: cv, wrap, row });
  }

  function nowSec() {
    return viewportClock.now();
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const { canvas } of canvases.values()) {
      const w = Math.max(40, canvas.clientWidth);
      const h = Math.max(24, canvas.clientHeight);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
  }

  function plotRect(w, h) {
    return { x: 43, y: 2, w: Math.max(1, w - 48), h: Math.max(1, h - 17) };
  }

  function xOf(t, now, plot) {
    return plot.x + ((t - (now - WINDOW_SEC)) / WINDOW_SEC) * plot.w;
  }

  function yOf(v, ymin, ymax, plot) {
    return plot.y + plot.h - ((v - ymin) / (ymax - ymin)) * plot.h;
  }

  function withPlotClip(ctx, plot, drawFn) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();
    drawFn();
    ctx.restore();
  }

  function tickText(row, value, index) {
    if (row.tickLabels?.[index] != null) return row.tickLabels[index];
    if (row.invert) return (1 - value).toFixed(1);
    const span = (row.ymax ?? 1) - (row.ymin ?? 0);
    if (span < 0.25) return Number(value).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
    return Number(value).toFixed(span <= 2 ? 1 : 0);
  }

  function drawGrid(ctx, row, plot) {
    ctx.fillStyle = '#000';
    ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
    ctx.strokeStyle = 'rgba(170,170,170,.18)';
    ctx.lineWidth = 1;
    ctx.font = '9px system-ui, sans-serif';
    ctx.fillStyle = '#aaa';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const ymin = row.ymin ?? 0;
    const ymax = row.ymax ?? 1;
    const ticks = row.ticks || [ymin, (ymin + ymax) / 2, ymax];
    ticks.forEach((value, index) => {
      const y = yOf(value, ymin, ymax, plot);
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      if (!['asr', 'latency'].includes(row.kind)) {
        ctx.fillText(tickText(row, value, index), plot.x - 4, y);
      }
    });
    if (row.kind !== 'bins') {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let i = 0; i <= 5; i++) {
        const x = plot.x + (i / 5) * plot.w;
        ctx.beginPath();
        ctx.moveTo(x, plot.y);
        ctx.lineTo(x, plot.y + plot.h);
        ctx.stroke();
        ctx.fillText(String(-WINDOW_SEC + (WINDOW_SEC * i) / 5), x, plot.y + plot.h + 2);
      }
    }
    ctx.strokeStyle = '#555';
    ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
  }

  function drawLine(ctx, buf, now, plot, ymin, ymax, stroke, fill = false, baseline = ymin) {
    if (buf.length < 4) return;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < buf.length; i += 2) {
      const x = xOf(buf[i], now, plot);
      const y = yOf(buf[i + 1], ymin, ymax, plot);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    if (fill) {
      const lastX = xOf(buf[buf.length - 2], now, plot);
      const firstX = xOf(buf[0], now, plot);
      const baseY = yOf(baseline, ymin, ymax, plot);
      ctx.lineTo(lastX, baseY);
      ctx.lineTo(firstX, baseY);
      ctx.closePath();
      ctx.save();
      ctx.globalAlpha = 0.32;
      ctx.fillStyle = stroke;
      ctx.fill();
      ctx.restore();
      ctx.beginPath();
      for (let i = 0; i < buf.length; i += 2) {
        const x = xOf(buf[i], now, plot);
        const y = yOf(buf[i + 1], ymin, ymax, plot);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function drawProbability(ctx, buf, now, plot) {
    const midY = yOf(0.5, 0, 1, plot);
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, Math.max(0, midY - plot.y));
    ctx.clip();
    drawLine(ctx, buf, now, plot, 0, 1, '#f5bd00', true, 0.5);
    ctx.restore();
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, midY, plot.w, Math.max(0, plot.y + plot.h - midY));
    ctx.clip();
    drawLine(ctx, buf, now, plot, 0, 1, '#50a0f0', true, 0.5);
    ctx.restore();
  }

  function drawThresh(ctx, y, plot, color, dotted = false) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.setLineDash(dotted ? [2, 3] : [5, 4]);
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.restore();
  }

  function drawRegions(ctx, list, now, plot, color, label) {
    const cut = now - WINDOW_SEC;
    for (const r of list) {
      if (r.end < cut) continue;
      const x0 = Math.max(plot.x, xOf(r.start, now, plot));
      const x1 = Math.min(plot.x + plot.w, xOf(r.end, now, plot));
      ctx.fillStyle = color;
      ctx.fillRect(x0, plot.y, Math.max(2, x1 - x0), plot.h);
      ctx.strokeStyle = 'rgba(180,140,20,.7)';
      ctx.strokeRect(x0, plot.y, Math.max(2, x1 - x0), plot.h);
      if (Number.isFinite(r.trigger)) {
        const tx = xOf(r.trigger, now, plot);
        ctx.strokeStyle = '#dc3c3c';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(tx, plot.y);
        ctx.lineTo(tx, plot.y + plot.h);
        ctx.stroke();
        ctx.fillStyle = '#dc3c3c';
        ctx.font = '9px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(label === 'NOD' && r.count ? `NOD×${r.count}` : label, tx, plot.y + 1);
      }
    }
  }

  function drawBins(ctx, plot) {
    const arr = pBins.arr;
    if (!arr || !arr.length) return;
    const times = pBins.times.length === arr[0].length ? pBins.times : arr[0].map(() => 1);
    const total = times.reduce((a, b) => a + b, 0) || 1;
    let x = plot.x;
    for (let b = 0; b < arr[0].length; b++) {
      const bw = (times[b] / total) * plot.w;
      for (let s = 0; s < 2; s++) {
        const v = Math.max(0, Math.min(1, Number(arr[s]?.[b]) || 0));
        const y0 = plot.y + (s === 0 ? 0 : plot.h / 2);
        const base = s === 0 ? [70, 140, 230] : [250, 150, 40];
        ctx.fillStyle = `rgb(${Math.round(base[0] * (0.35 + 0.65 * v))},${Math.round(base[1] * (0.35 + 0.65 * v))},${Math.round(base[2] * (0.35 + 0.65 * v))})`;
        ctx.fillRect(x + 1, y0 + 1, Math.max(2, bw - 2), plot.h / 2 - 2);
        ctx.fillStyle = v >= 0.55 ? '#fff' : '#111';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(v.toFixed(2), x + bw / 2, y0 + plot.h / 4);
      }
      x += bw;
    }
    ctx.fillStyle = '#aaa';
    ctx.font = '9px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    let elapsed = 0;
    ctx.fillText('0', plot.x, plot.y + plot.h + 2);
    for (const seconds of times) {
      elapsed += seconds;
      ctx.fillText(elapsed.toFixed(1), plot.x + (elapsed / total) * plot.w, plot.y + plot.h + 2);
    }
  }

  function drawWave(ctx, buf, now, plot, color) {
    if (buf.length < 4) return;
    ctx.beginPath();
    const mid = plot.y + plot.h / 2;
    ctx.moveTo(xOf(buf[0], now, plot), mid);
    for (let i = 0; i < buf.length; i += 2) {
      const x = xOf(buf[i], now, plot);
      const amp = Math.min(1, buf[i + 1]) * (plot.h * 0.48);
      ctx.lineTo(x, mid - amp);
    }
    for (let i = buf.length - 2; i >= 0; i -= 2) {
      const x = xOf(buf[i], now, plot);
      const amp = Math.min(1, buf[i + 1]) * (plot.h * 0.48);
      ctx.lineTo(x, mid + amp);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.55;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  function draw() {
    if (!open) return;
    const now = nowSec();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const { canvas, row } of canvases.values()) {
      if (!vis[row.id]) continue;
      const ctx = canvas.getContext('2d');
      const w = Math.max(40, canvas.clientWidth);
      const h = Math.max(24, canvas.clientHeight);
      const plot = plotRect(w, h);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, w, h);
      drawGrid(ctx, row, plot);
      if (row.kind === 'wave') withPlotClip(ctx, plot, () => drawWave(ctx, series[row.id], now, plot, row.color));
      else if (row.kind === 'bins') drawBins(ctx, plot);
      else if (row.kind === 'prob') {
        withPlotClip(ctx, plot, () => drawProbability(ctx, series[row.id], now, plot));
        drawThresh(ctx, yOf(0.5, 0, 1, plot), plot, '#787878');
        if (row.id === 'p_now') {
          drawThresh(ctx, yOf(1 - thresholds.interruptPnow, 0, 1, plot), plot, '#e65a5a');
          drawThresh(ctx, yOf(1 - thresholds.turnPnow, 0, 1, plot), plot, '#50c878');
        }
      } else if (row.kind === 'vad') {
        withPlotClip(ctx, plot, () => {
          drawLine(ctx, series.vad_1, now, plot, -1, 1, '#f5bd00', true, 0);
          drawLine(ctx, series.vad_2, now, plot, -1, 1, '#50a0f0', true, 0);
        });
      } else if (row.kind === 'asr') {
        ctx.font = '10px system-ui';
        withPlotClip(ctx, plot, () => {
        for (const seg of asrSegs) {
          if (seg.end < now - WINDOW_SEC) continue;
          const x0 = Math.max(plot.x, xOf(seg.start, now, plot));
          const x1 = Math.min(plot.x + plot.w, xOf(seg.end, now, plot));
          ctx.globalAlpha = seg.final ? 0.72 : 0.46;
          ctx.fillStyle = seg.color;
          ctx.fillRect(x0, plot.y + plot.h * 0.15, Math.max(2, x1 - x0), plot.h * 0.7);
          ctx.globalAlpha = 1;
          if (seg.text && x1 - x0 > 24) {
            ctx.fillStyle = '#141414';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(seg.text.slice(0, 36), x0 + 3, plot.y + plot.h * 0.5);
          }
        }
        });
      } else if (row.kind === 'latency') {
        withPlotClip(ctx, plot, () => {
        if (turnLockEdges.length) {
          const boundaries = [now - WINDOW_SEC];
          for (const edge of turnLockEdges) {
            if (edge.t > now - WINDOW_SEC && edge.t <= now) boundaries.push(edge.t);
          }
          boundaries.push(now);
          boundaries.sort((a, b) => a - b);
          const lockedAt = (t) => {
            let locked = false;
            for (const edge of turnLockEdges) {
              if (edge.t > t) break;
              locked = edge.locked;
            }
            return locked;
          };
          for (let i = 0; i + 1 < boundaries.length; i++) {
            const start = boundaries[i];
            const end = boundaries[i + 1];
            if (!lockedAt((start + end) / 2)) continue;
            const x0 = xOf(start, now, plot);
            const x1 = xOf(end, now, plot);
            ctx.fillStyle = 'rgba(220,90,70,.39)';
            ctx.fillRect(x0, yOf(0.085, 0, 1, plot), Math.max(1, x1 - x0), plot.h * 0.085);
          }
        }
        for (const span of latencySpans) {
          if (span.end < now - WINDOW_SEC) continue;
          const x0 = Math.max(plot.x, xOf(span.start, now, plot));
          const x1 = Math.min(plot.x + plot.w, xOf(span.end, now, plot));
          ctx.fillStyle = span.kind === 'asr' ? 'rgba(245,189,0,.51)'
            : span.kind === 'llm' ? 'rgba(120,220,120,.51)' : 'rgba(230,110,220,.51)';
          const bandTop = yOf(0.97, 0, 1, plot);
          const bandBottom = yOf(0.09, 0, 1, plot);
          ctx.fillRect(x0, bandTop, Math.max(2, x1 - x0), bandBottom - bandTop);
          ctx.strokeStyle = '#282828';
          ctx.strokeRect(x0, bandTop, Math.max(2, x1 - x0), bandBottom - bandTop);
          if (x1 - x0 > 30) {
            ctx.fillStyle = '#141414';
            ctx.font = '9px system-ui';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${Math.round((span.end - span.start) * 1000)}ms`, x0 + 2, plot.y + plot.h / 2);
          }
        }
        for (const t of turnMarks) {
          if (t < now - WINDOW_SEC) continue;
          const x = xOf(t, now, plot);
          ctx.strokeStyle = '#fff';
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          ctx.moveTo(x, plot.y);
          ctx.lineTo(x, plot.y + plot.h);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = '#ebebeb';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText('TURN', x, plot.y + 1);
        }
        for (const t of interruptMarks) {
          if (t < now - WINDOW_SEC) continue;
          const x = xOf(t, now, plot);
          ctx.strokeStyle = '#ff4141';
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          ctx.moveTo(x, plot.y);
          ctx.lineTo(x, plot.y + plot.h);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = '#ff968c';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText('INT', x, plot.y + 8);
        }
        });
      } else {
        if (row.id === 'silero_vad_score') {
          drawThresh(ctx, yOf(thresholds.vadOn, 0, 1, plot), plot, '#3cb45a', true);
          drawThresh(ctx, yOf(thresholds.vadOff, 0, 1, plot), plot, '#5a96f0', true);
          drawThresh(ctx, yOf(thresholds.interruptVad, 0, 1, plot), plot, '#e65a5a');
        }
        withPlotClip(ctx, plot, () => {
          if (['p_nod', 'nod_range', 'nod_speed', 'nod_count', 'nod_swing_up'].includes(row.id)) {
            drawRegions(ctx, nodRegions, now, plot, '#ffdc502c', 'NOD');
          }
          if (row.id === 'p_bc') drawRegions(ctx, bcRegions, now, plot, '#ffdc502c', 'BC');
          drawLine(ctx, series[row.id], now, plot, row.ymin ?? 0, row.ymax ?? 1, row.color || '#c9d6ea', row.fill);
        });
        if (['p_bc', 'p_nod', 'nod_swing_up'].includes(row.id)) {
          const threshold = row.id === 'p_bc' ? thresholds.pBc
            : row.id === 'nod_swing_up' ? thresholds.nodSwing : 0.5;
          drawThresh(ctx, yOf(threshold, 0, 1, plot), plot, row.id === 'p_nod' ? '#787878' : '#e65a5a');
        }
      }
    }
  }

  function loop(ts) {
    if (!open) return;
    if (ts - lastDraw > 66) {
      lastDraw = ts;
      draw();
    }
    raf = requestAnimationFrame(loop);
  }

  function applyVisibility() {
    const visibleCount = ROWS.reduce((n, row) => n + (vis[row.id] === false ? 0 : 1), 0);
    const twoColumns = visibleCount >= 9;
    body.classList.toggle('one-column', !twoColumns);
    for (const row of ROWS) {
      const { wrap } = canvases.get(row.id);
      const target = twoColumns && row.col === 'right' ? right : left;
      target.append(wrap);
      wrap.style.display = vis[row.id] === false ? 'none' : '';
    }
    resize();
  }

  function ingestResult(result) {
    if (!result || typeof result !== 'object') return;
    const sourceTs = firstFloat(result.source_ts_end, NaN);
    if (Number.isFinite(sourceTs)) viewportClock.note(sourceTs);
    const t = Number.isFinite(sourceTs) ? sourceTs : firstFloat(result.t, NaN);
    const ev = Number.isFinite(t) ? t : nowSec();
    const set = (id, raw, map) => {
      let v = firstFloat(raw, NaN);
      if (!Number.isFinite(v)) return;
      if (map) v = map(v);
      pushPoint(series[id], ev, v);
    };
    if ('p_now' in result) set('p_now', result.p_now, (v) => 1 - Math.max(0, Math.min(1, v)));
    if ('p_future' in result) set('p_future', result.p_future, (v) => 1 - Math.max(0, Math.min(1, v)));
    if ('silero_vad_score' in result) set('silero_vad_score', result.silero_vad_score, (v) => Math.max(0, v));
    if ('p_bc' in result) set('p_bc', result.p_bc, (v) => Math.max(0, Math.min(1, v)));
    if ('p_bc_threshold' in result) {
      thresholds.pBc = Math.max(0, Math.min(1, firstFloat(result.p_bc_threshold, thresholds.pBc)));
    }
    if ('p_nod' in result) set('p_nod', result.p_nod);
    if ('nod_range' in result) set('nod_range', result.nod_range, (v) => Math.max(0.035, Math.min(0.15, v)));
    if ('nod_speed' in result) set('nod_speed', result.nod_speed, (v) => Math.max(0.08, Math.min(0.25, v)));
    if ('nod_count' in result) set('nod_count', result.nod_count, (v) => Math.max(0, Math.min(2, Math.round(v))));
    if ('nod_swing_up' in result || 'nod_swing_up_prob' in result) {
      set('nod_swing_up', result.nod_swing_up ?? result.nod_swing_up_prob, (v) => Math.max(0, Math.min(1, v)));
    }
    if ('nod_swing_up_threshold' in result) {
      thresholds.nodSwing = Math.max(0, Math.min(1, firstFloat(result.nod_swing_up_threshold, thresholds.nodSwing)));
    }
    if ('vad' in result) {
      const arr = Array.isArray(result.vad) ? result.vad : [result.vad];
      pushPoint(series.vad_1, ev, firstFloat(arr[0], 0));
      pushPoint(series.vad_2, ev, -Math.abs(firstFloat(arr[1], 0)));
    }
    if (result.p_bins || result.bin_times) {
      const raw = result.p_bins;
      if (Array.isArray(raw) && raw.length) {
        pBins.arr = raw;
        if (Array.isArray(result.bin_times) && result.bin_times.length) {
          pBins.times = result.bin_times.map((x) => Number(x) || 1);
        }
      }
    }
  }

  function noteWallLag(ev) {
    const wall = Number(ev.wall_ts ?? ev.emitted_at);
    const audioEnd = Number(ev.ts_end);
    if (!Number.isFinite(wall) || !Number.isFinite(audioEnd)) return;
    const lag = wall - audioEnd;
    // 配送の揺れで大きく出た値に引っ張られないよう、下がる方向へは即追従し、上がる方向へは緩やかに。
    if (wallLag === null || lag < wallLag) wallLag = lag;
    else wallLag += (lag - wallLag) * 0.02;
  }

  // 壁時計（time.time()）で刻まれた時刻を音声時刻へ。音声がまだ無い間はそのまま。
  function wallToAudio(t) {
    const v = Number(t);
    if (!Number.isFinite(v)) return NaN;
    return wallLag === null ? v : v - wallLag;
  }

  function ingestAudio(ev) {
    const audio = Array.isArray(ev.audio_peaks) ? ev.audio_peaks : ev.audio;
    if (!Array.isArray(audio) || !audio.length) return;
    noteWallLag(ev);
    viewportClock.note(ev.ts_end);
    const sys = ev.type === 'system_audio_frame' || String(ev.channel || '').toLowerCase() === 'system';
    downsampleWave(audio, ev.ts_start, ev.ts_end, series[sys ? 'x2' : 'x1'], nowSec());
  }

  function engineColor(engine) {
    const fixed = {
      'qwen3-asr': '#1f77b4',
      'external-asr-tcp': '#d62728',
      'chirp3-asr': '#2ca02c',
      'faster-whisper': '#ff7f0e',
      'sherpa-onnx': '#9467bd',
    };
    const key = String(engine || 'asr').trim().toLowerCase();
    if (fixed[key]) return fixed[key];
    const palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
    let hash = 0;
    for (let i = 0; i < key.length; i++) hash = ((hash * 31) + key.charCodeAt(i)) >>> 0;
    return palette[hash % palette.length];
  }

  function ingestAsr(ev) {
    const start = firstFloat(ev.segment_start_ts, NaN);
    const end = firstFloat(ev.segment_end_ts, NaN);
    if (!Number.isFinite(start) && !Number.isFinite(end)) return;
    let t0 = Number.isFinite(start) ? start : end;
    let t1 = Number.isFinite(end) ? end : start;
    if (t1 < t0) t1 = t0;
    const isFinal = ev.is_final !== false;
    if (asrArrivalAlign && isFinal) {
      const arriveWall = wallToAudio(ev.final_arrival_ts ?? ev.ts);
      const arrive = Number.isFinite(arriveWall) ? arriveWall : nowSec();
      const dur = Math.max(0.05, t1 - t0);
      t1 = arrive;
      t0 = arrive - dur;
    }
    const text = String(ev.text ?? '');
    const color = engineColor(ev.engine);
    // 途中結果のたびに帯を足すと同じ発話が何本も重なる。Qt の update_asr_segment と同じく、
    // 途中結果の間は開いている 1 本を伸ばし、確定でその帯を閉じる（疑似ストリーミングの
    // 途中結果も is_streaming が付かないだけで同じ発話なので、同じ扱いにする）。
    const key = `${String(ev.compare_slot ?? '').trim().toLowerCase()}|${String(ev.engine ?? 'asr').trim()}`;
    const open = openAsrStreams.get(key);
    const alive = open && asrSegs.includes(open);
    if (!isFinal) {
      if (alive) {
        Object.assign(open, { end: t1, text, final: false });
      } else {
        const seg = { start: t0, end: t1, text, final: false, color };
        asrSegs.push(seg);
        openAsrStreams.set(key, seg);
      }
    } else {
      openAsrStreams.delete(key);
      if (alive) Object.assign(open, { end: t1, text, final: isFinal });
      else asrSegs.push({ start: t0, end: t1, text, final: isFinal, color });
    }
    const cut = nowSec() - WINDOW_SEC - 2;
    while (asrSegs.length && asrSegs[0].end < cut) asrSegs.shift();
  }

  function ingestLatency(ev) {
    const kind = String(ev.kind ?? '').toLowerCase();
    if (kind === 'interrupt') {
      const mark = wallToAudio(ev.end_ts ?? ev.event_ts);
      interruptMarks.push(Number.isFinite(mark) ? mark : nowSec());
      while (interruptMarks.length > 300) interruptMarks.shift();
      return;
    }
    if (!['asr', 'llm', 'tts'].includes(kind)) return;
    const value = Number(ev.value);
    const hasValue = ev.value != null && Number.isFinite(value);
    // 遅延帯の時刻はランタイムの壁時計（time.time()）。表示の now は音声時刻なので換算する。
    let start = wallToAudio(ev.start_ts);
    let end = wallToAudio(ev.end_ts);
    if (!Number.isFinite(start) && Number.isFinite(end) && hasValue) start = end - Math.max(0, value);
    if (!Number.isFinite(end) && Number.isFinite(start) && hasValue) end = start + Math.max(0, value);
    if (!Number.isFinite(start) && !Number.isFinite(end) && hasValue) {
      const at = wallToAudio(ev.event_ts);
      end = Number.isFinite(at) ? at : nowSec();
      start = end - Math.max(0, value);
    }
    // Windows版同様、value=nullかつ時刻なしのリセット通知は帯にしない。
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;
    if (end < start) end = start;
    const duplicate = latencySpans.some((s) => s.kind === kind
      && Math.abs(s.start - start) < 0.002 && Math.abs(s.end - end) < 0.002);
    if (!duplicate) {
      const sameStart = [...latencySpans].reverse().find((s) => s.kind === kind && Math.abs(s.start - start) < 0.05);
      if (sameStart) sameStart.end = end;
      else latencySpans.push({ kind, start, end });
    }
    const cut = nowSec() - WINDOW_SEC - 2;
    while (latencySpans.length && latencySpans[0].end < cut) latencySpans.shift();
    while (latencySpans.length > 300) latencySpans.shift();
  }

  function ingestHistory(ev) {
    const history = ev.history;
    if (!history || typeof history !== 'object') return;
    const raw = history.turn_blocked_until_user_speech ?? history.turn_locked_until_tts_done;
    if (raw == null) return;
    const locked = Boolean(raw);
    if (turnLockEdges.length && turnLockEdges[turnLockEdges.length - 1].locked === locked) return;
    const at = wallToAudio(history.ts);
    turnLockEdges.push({ t: Number.isFinite(at) ? at : nowSec(), locked });
    const cut = nowSec() - WINDOW_SEC - 2;
    while (turnLockEdges.length > 1 && turnLockEdges[1].t < cut) turnLockEdges.shift();
    while (turnLockEdges.length > 300) turnLockEdges.shift();
  }

  function pruneRegions(list, now) {
    const cut = now - WINDOW_SEC - 1;
    while (list.length && list[0].end < cut) list.shift();
  }

  function handleEvent(ev) {
    if (!ev || typeof ev !== 'object') return;
    if (['maai_result', 'maai_bc_result', 'maai_nod_result'].includes(ev.type)) ingestResult(ev.result);
    else if (ev.type === 'raw_audio' || ev.type === 'audio_frame' || ev.type === 'system_audio_frame') ingestAudio(ev);
    else if (ev.type === 'asr_result') ingestAsr(ev);
    else if (ev.type === 'latency') ingestLatency(ev);
    else if (ev.type === 'history_snapshot') ingestHistory(ev);
    else if (ev.type === 'runtime_snapshot' && ev.history) ingestHistory({ history: ev.history });
    else if (ev.type === 'nod_trigger_visual') {
      const t0 = firstFloat(ev.used_start_ts ?? ev.start_ts ?? ev.ts, nowSec());
      const t1 = firstFloat(ev.used_end_ts ?? ev.end_ts, t0 + 0.4);
      nodRegions.push({
        start: t0,
        end: t1,
        trigger: firstFloat(ev.trigger_ts, t1),
        count: Number(ev.count || 0),
      });
      pruneRegions(nodRegions, nowSec());
    } else if (ev.type === 'bc_trigger_visual') {
      const t0 = firstFloat(ev.used_start_ts ?? ev.start_ts ?? ev.ts, nowSec());
      const t1 = firstFloat(ev.used_end_ts ?? ev.end_ts, t0 + 0.4);
      bcRegions.push({ start: t0, end: t1, trigger: firstFloat(ev.trigger_ts, t1) });
      pruneRegions(bcRegions, nowSec());
    } else if (ev.type === 'turn' && !ev.from_orchestrator) {
      turnMarks.push(firstFloat(ev.ts, nowSec()));
      const cut = nowSec() - WINDOW_SEC - 1;
      while (turnMarks.length && turnMarks[0] < cut) turnMarks.shift();
      while (turnMarks.length > 300) turnMarks.shift();
    }
  }

  // PySide の json_config_bool と同じ解釈（"false" 文字列を真にしない）。
  function configBool(value) {
    if (typeof value === 'boolean') return value;
    if (typeof value === 'number') return value !== 0;
    if (value === null || value === undefined) return false;
    return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
  }

  function setVisibilityFromConfig(cfg) {
    for (const row of ROWS) {
      const key = `maai_plot_show_${row.id}`;
      vis[row.id] = cfg[key] === undefined ? !row.hiddenByDefault : configBool(cfg[key]);
    }
    const n = (k, d) => {
      const v = Number(cfg[k]);
      return Number.isFinite(v) ? v : d;
    };
    thresholds = {
      interruptVad: n('voice_dialog_interrupt_vad_threshold', 0.7),
      interruptPnow: n('voice_dialog_interrupt_p_now_threshold', 0.5),
      turnPnow: n('asr_maai_turn_threshold', 0.5),
      vadOn: n('asr_vad_on_threshold', 0.1),
      vadOff: n('asr_vad_off_threshold', 0.2),
      pBc: n('maai_backchannel_prob_threshold', thresholds.pBc),
      nodSwing: n('maai_test_nod_swing_threshold', 0.5),
    };
    asrArrivalAlign = configBool(cfg.asr_segment_arrival_align);
    applyVisibility();
  }

  function show() {
    open = true;
    root.classList.add('open');
    resize();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(loop);
  }

  function hide() {
    open = false;
    root.classList.remove('open');
    cancelAnimationFrame(raf);
  }

  close.addEventListener('click', () => {
    if (options.standalone) window.close();
    else hide();
  });
  window.addEventListener('resize', () => { if (open) resize(); });

  return {
    show,
    hide,
    toggle() { if (open) hide(); else show(); },
    isOpen: () => open,
    handleEvent,
    setVisibilityFromConfig,
    setConnectionState(text, ok = false) {
      connection.textContent = text;
      connection.classList.toggle('ok', Boolean(ok));
    },
  };
}

export function createNodChart(canvas) {
  const data = {
    motion: [],
    time: [],
    title: '頷きモーション',
    info: [],
  };

  function paint() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(40, canvas.clientWidth);
    const h = Math.max(180, canvas.clientHeight);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    const ml = 56;
    const mr = 16;
    const mt = 36;
    const mb = 36;
    const pw = w - ml - mr;
    const ph = h - mt - mb;
    if (!data.motion.length || data.motion.length !== data.time.length) {
      ctx.fillStyle = '#808080';
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('No motion data', w / 2, h / 2);
      return;
    }
    const tMin = 0;
    const tMax = 3;
    const yMin = -0.1;
    const yMax = 0.1;
    ctx.strokeStyle = '#d3d3d3';
    ctx.fillStyle = '#696969';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    for (let i = 0; i <= 5; i++) {
      const yv = yMin + (yMax - yMin) * i / 5;
      const py = mt + ph - (yv - yMin) / (yMax - yMin) * ph;
      ctx.beginPath();
      ctx.moveTo(ml, py);
      ctx.lineTo(ml + pw, py);
      ctx.stroke();
      ctx.fillText(yv.toFixed(3), 4, py + 4);
    }
    for (let i = 0; i <= 5; i++) {
      const tv = tMin + (tMax - tMin) * i / 5;
      const px = ml + (tv - tMin) / (tMax - tMin) * pw;
      ctx.beginPath();
      ctx.moveTo(px, mt);
      ctx.lineTo(px, mt + ph);
      ctx.stroke();
      ctx.fillText(tv.toFixed(2), px - 12, mt + ph + 14);
    }
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#808080';
    const zeroY = mt + ph - (0 - yMin) / (yMax - yMin) * ph;
    ctx.beginPath();
    ctx.moveTo(ml, zeroY);
    ctx.lineTo(ml + pw, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = '#008000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.motion.length; i++) {
      const px = ml + (data.time[i] - tMin) / (tMax - tMin) * pw;
      const py = mt + ph - (data.motion[i] - yMin) / (yMax - yMin) * ph;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = '#000';
    ctx.font = '13px system-ui';
    ctx.fillText(data.title, ml, mt - 12);
    ctx.fillStyle = '#0000b4';
    ctx.font = '11px system-ui';
    data.info.forEach((line, i) => ctx.fillText(line, ml + pw - 210, mt + 16 + i * 15));
  }

  function setChartData(payload) {
    const motion = payload.motion || payload.series || [];
    const time = payload.time_axis || payload.time || [];
    data.motion = Array.from(motion, (x) => Number(x) || 0);
    data.time = Array.from(time, (x) => Number(x) || 0);
    const n = payload.nod_count ?? 0;
    const interp = payload.interpolation || '';
    const pNod = Number(payload.p_nod || 0);
    data.title = `頷きモーション（n=${n}、${interp}、p_nod=${pNod.toFixed(3)}）`;
    data.info = [
      'VAP予測パラメーター：',
      `range=${Number(payload.raw_range || payload.range_rad || 0).toFixed(4)}`,
      `speed=${Number(payload.raw_speed || payload.velocity || 0).toFixed(4)}`,
      `swing_up=${Number(payload.swing_up_prob || 0).toFixed(3)}`,
    ];
    paint();
  }

  window.addEventListener('resize', paint);
  paint();
  return { setChartData, paint };
}

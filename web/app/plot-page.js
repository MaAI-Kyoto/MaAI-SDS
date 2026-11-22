import { createMaaiPlot } from '/plot.js?v=ui25';

const root = document.getElementById('plot-standalone');
const plot = createMaaiPlot(root, { standalone: true });
let socket = null;
let reconnectTimer = 0;
let stopped = false;

async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    plot.setVisibilityFromConfig(await response.json());
  } catch {
    plot.setVisibilityFromConfig({});
  }
  plot.show();
}

function connect() {
  if (stopped) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${location.host}/ws/events`);
  plot.setConnectionState('接続中…');
  socket.addEventListener('open', () => {
    plot.setConnectionState('接続済み', true);
    socket.send(JSON.stringify({ type: 'ui_subscribe', plot_audio: true }));
  });
  socket.addEventListener('message', (event) => {
    try { plot.handleEvent(JSON.parse(event.data)); } catch { /* ignore malformed frames */ }
  });
  socket.addEventListener('close', () => {
    plot.setConnectionState('切断・再接続中');
    if (!stopped) reconnectTimer = window.setTimeout(connect, 3000);
  });
}

window.addEventListener('beforeunload', () => {
  stopped = true;
  window.clearTimeout(reconnectTimer);
  socket?.close();
});

await loadConfig();
connect();

"""Non-Qt transport bridge for MMDAgent communication.

Runs inside the dialog/worker process.
Supports TCP client, TCP server, and WebSocket server modes.
Thread-safe send operations; background threads handle connections.
"""
from __future__ import annotations

import asyncio
import json
import select
import socket
import threading
from typing import Callable

from .avatar_keys import blink_bind_command, head_bind_commands


def _remote_plugin_settings_digest(settings: dict) -> str:
    """接続モード・待受/接続先のみ比較（同一なら stop/再起動しない）。"""
    keys = (
        "tcp_enable_server",
        "tcp_enable_client",
        "tcp_listen_port",
        "tcp_port",
        "tcp_client_host",
        "tcp_server_host",
        "ws_host",
        "ws_port",
        "ws_directory",
    )
    d = {k: settings.get(k) for k in keys}
    return json.dumps(d, sort_keys=True, default=str)


class MmdRemoteBridge:
    MODE_NONE = "none"
    MODE_WS = "websocket"
    MODE_TCP_CLIENT = "tcp_client"
    MODE_TCP_SERVER = "tcp_server"

    def __init__(
        self,
        *,
        log_fn: Callable[[str], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_disconnected: Callable[[], None] | None = None,
    ) -> None:
        self._log = log_fn or (lambda _: None)
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._mode = self.MODE_NONE
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()

        self._tcp_host = ""
        self._tcp_port = 0
        self._tcp_sock: socket.socket | None = None
        self._tcp_thread: threading.Thread | None = None

        self._tcp_server_sock: socket.socket | None = None
        self._tcp_peer: socket.socket | None = None
        self._tcp_accept_thread: threading.Thread | None = None

        self._ws_host = ""
        self._ws_port = 0
        self._ws_directory = "/"
        self._ws_peer: object | None = None
        self._ws_peers: list[object] = []
        self._ws_lock = threading.Lock()
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_server_obj: object | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_ready = threading.Event()
        # 初回 SND 直前と同じ __AV_START 列。Plugin_Remote は __AV_START / __AV_SETMODEL で
        # アバター制御を有効にしモデルを割り当てるので、接続相手が替わるたびに送り直す。
        self._av_snd_segment_preamble_pending = True
        # この接続で一度でも __AV_START を送ったか。SNDBRKS では戻さない。
        self._av_started_on_connection = False
        self._last_remote_settings_digest: str | None = None
        # `__AV_SETMODEL` で指定するモデルの別名（FST の MODEL_ADD。設定の model_alias）。
        # Plugin_Remote は割り当てたモデルにだけ音声を渡して口パクし、割り当てが無いと音声を捨てる。
        self._model_alias = "0"
        # 頷き・まばたき用の結び付け（MODEL_BINDBONE / MODEL_BINDFACE）。avatar_keys を参照。
        # 接続時と MODEL_EVENT_ADD|<別名> のたびに送り直す（モデルの読み込み前に送ったものは無視される）。
        self._binding_lock = threading.Lock()
        self._head_binding: tuple[str, float] | None = None
        self._blink_morph: str | None = None
        self._inbound_buf = ""

    @property
    def mode(self) -> str:
        return self._mode

    def ensure_snd_segment_preamble(self) -> bool:
        """__AV_START / __AV_SETMODEL / SNDFILE（初回 SND フレームと同一）。接続直後・SNDBRKS 後に必要。

        既に満たしている場合は True、未接続または送信失敗で未完了の場合は False。
        """
        if not self.is_connected():
            return False
        if not self._av_snd_segment_preamble_pending:
            return True
        if self._av_started_on_connection and self._mode != self.MODE_WS:
            # MMDAgent-EX の __AV_SETMODEL（assignModel）はアバター制御を作り直す。SNDBRKS は音声の
            # 区切りだけで有効化もモデル割り当ても解かないので、接続ごとに 1 回送れば足りる。
            self._av_snd_segment_preamble_pending = False
            return True
        ok = self._send_av_start()
        if ok:
            self._av_snd_segment_preamble_pending = False
        return ok

    def ensure_av_started(self) -> bool:
        """この接続でまだ __AV_START を送っていなければ送る（まばたき等の常時送信用）。

        MMDAgent-EX は __AV_SETMODEL のたびに口パクを含む顔の目標値を初期化するので、
        SND 区切りごとの前置きは ensure_snd_segment_preamble に任せ、ここでは送り直さない。
        """
        if not self.is_connected():
            return False
        if self._av_started_on_connection:
            return True
        return self._send_av_start()

    def _send_av_start(self) -> bool:
        ok = bool(self.send_text("__AV_START\n"))
        ok = bool(self.send_text(f"__AV_SETMODEL,{self._model_alias}\n")) and ok
        if ok:
            self._av_started_on_connection = True
        return ok

    # -- 頷き・まばたきの結び付け -------------------------------------------------

    def set_head_binding(self, bone: str, eye_scale: float) -> None:
        """頭（と目）のボーンを頷きの KeyValue に結び付ける。変わったときだけ送り直す。"""
        binding = (str(bone or "").strip() or "頭", round(float(eye_scale), 4))
        with self._binding_lock:
            if binding == self._head_binding:
                return
            self._head_binding = binding
        if self.is_connected():
            self._send_bindings()

    def set_blink_binding(self, morph: str) -> None:
        """まばたきの表情モーフを KeyValue に結び付ける。変わったときだけ送り直す。"""
        morph = str(morph or "").strip()
        with self._binding_lock:
            if not morph or morph == self._blink_morph:
                return
            self._blink_morph = morph
        if self.is_connected():
            self._send_bindings()

    def _binding_commands(self) -> list[str]:
        with self._binding_lock:
            head, morph, alias = self._head_binding, self._blink_morph, self._model_alias
        commands: list[str] = []
        if head is not None:
            commands.extend(head_bind_commands(alias, head[0], head[1]))
        if morph:
            commands.append(blink_bind_command(alias, morph))
        return commands

    def _send_bindings(self) -> None:
        for command in self._binding_commands():
            if not self.send_text(command):
                return

    def _send_bindings_async(self) -> None:
        # 受信スレッド（WebSocket ではイベントループ）から送ると詰まるので、別スレッドで送る
        if self._head_binding is None and self._blink_morph is None:
            return
        threading.Thread(target=self._send_bindings, daemon=True, name="mmd-bridge-bind").start()

    def _on_inbound(self, data: object) -> None:
        """MMDAgent-EX から届いたメッセージ。モデルの読み込み完了で結び付けを送り直す。"""
        if isinstance(data, (bytes, bytearray)):
            text = bytes(data).decode("utf-8", errors="replace")
        else:
            text = str(data)
        text = self._inbound_buf + text
        lines = text.split("\n")
        self._inbound_buf = lines.pop()[-4096:]
        for line in lines:
            line = line.strip()
            if line == f"MODEL_EVENT_ADD|{self._model_alias}":
                self._send_bindings_async()

    def _on_peer_connected(self) -> None:
        self._inbound_buf = ""
        if self._on_connected:
            self._on_connected()
        self._send_bindings_async()

    def mark_snd_segment_ended(self) -> None:
        """SNDBRKS 相当の区切りのあと、次の SND の前にプリアンブルを再度送れるようにする。"""
        self._av_snd_segment_preamble_pending = True

    # -- Public API --

    def configure(self, settings: dict) -> None:
        """Stop current connection and start new one based on settings."""
        alias = str(settings.get("model_alias") or "0").strip() or "0"
        if alias != self._model_alias:
            # 別のモデルに替わったら、次の音声の前に __AV_START / __AV_SETMODEL を送り直す
            print(f"Remote bridge: 音声と口パクを渡すモデル = {alias!r}（FST の MODEL_ADD）", flush=True)
            self._model_alias = alias
            self._av_started_on_connection = False
            self._av_snd_segment_preamble_pending = True
            self._send_bindings_async()
        digest = _remote_plugin_settings_digest(settings)
        if (
            digest == self._last_remote_settings_digest
            and self._mode != self.MODE_NONE
        ):
            return

        self.stop()

        enable_server = bool(settings.get("tcp_enable_server", False))
        enable_client = bool(settings.get("tcp_enable_client", False))

        if enable_server:
            try:
                port = int(settings.get("tcp_listen_port", 0))
            except (TypeError, ValueError):
                port = 0
            if 1 <= port <= 65535:
                host = str(settings.get("tcp_client_host", "127.0.0.1")).strip() or "127.0.0.1"
                self._start_tcp_client(host, port)
                self._last_remote_settings_digest = digest
            else:
                self._log("Remote bridge: TCP ListenPort が不正です")
            return

        if enable_client:
            try:
                port = int(settings.get("tcp_port", 0))
            except (TypeError, ValueError):
                port = 0
            if 1 <= port <= 65535:
                # 既定は localhost のみ。LAN 公開時のみ tcp_server_host を明示設定する。
                bind_host = str(settings.get("tcp_server_host", "127.0.0.1")).strip() or "127.0.0.1"
                self._start_tcp_server(port, bind_host)
                self._last_remote_settings_digest = digest
            else:
                self._log("Remote bridge: TCP Port が不正です")
            return

        # 既定は localhost のみ。LAN 公開が必要な場合のみ ws_host を明示設定する
        # （認証が無いため 0.0.0.0 既定は LAN 上の任意クライアントから制御可能になる）。
        ws_host = str(settings.get("ws_host", "127.0.0.1")).strip() or "127.0.0.1"
        ws_dir = str(settings.get("ws_directory", "/chat"))
        if not ws_dir.startswith("/"):
            ws_dir = f"/{ws_dir}"
        try:
            ws_port = int(settings.get("ws_port", 0))
        except (TypeError, ValueError):
            ws_port = 0
        if 1 <= ws_port <= 65535:
            self._start_websocket(ws_host, ws_port, ws_dir)
            self._last_remote_settings_digest = digest
        else:
            self._log("Remote bridge: WebSocket ポートが不正です")

    def stop(self) -> None:
        """Close all connections and background threads."""
        self._stop_event.set()

        ws_loop = self._ws_loop
        if ws_loop is not None:
            try:
                if ws_loop.is_running():
                    ws_loop.call_soon_threadsafe(ws_loop.stop)
            except Exception:
                pass

        sock = self._tcp_sock
        self._tcp_sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

        peer = self._tcp_peer
        self._tcp_peer = None
        if peer is not None:
            try:
                peer.close()
            except Exception:
                pass

        srv = self._tcp_server_sock
        self._tcp_server_sock = None
        if srv is not None:
            try:
                srv.close()
            except Exception:
                pass

        for t in (self._tcp_thread, self._tcp_accept_thread, self._ws_thread):
            if t is not None and t.is_alive():
                t.join(timeout=4.0)
        self._tcp_thread = None
        self._tcp_accept_thread = None
        self._ws_thread = None
        with self._ws_lock:
            self._ws_peers.clear()
        self._ws_peer = None
        self._ws_loop = None
        self._ws_server_obj = None
        self._ws_ready.clear()
        self._mode = self.MODE_NONE
        self._stop_event.clear()
        self._av_snd_segment_preamble_pending = True
        self._av_started_on_connection = False
        self._last_remote_settings_digest = None

    def send_text(self, text: str) -> bool:
        """Send a text command to MMDAgent.

        For WebSocket: sent as a text frame.
        For TCP: sent as UTF-8 bytes.
        """
        if self._mode == self.MODE_WS:
            return self._ws_send(text)
        return self._tcp_send(text.encode("utf-8"))

    def send_binary(self, data: bytes) -> bool:
        """Send binary data (SND audio frames) to MMDAgent.

        For WebSocket: sent as a binary frame.
        For TCP: sent as raw bytes.
        """
        if self._mode == self.MODE_WS:
            return self._ws_send(data)
        return self._tcp_send(data)

    def is_connected(self) -> bool:
        if self._mode == self.MODE_TCP_CLIENT:
            return self._tcp_sock is not None
        if self._mode == self.MODE_TCP_SERVER:
            return self._tcp_peer is not None
        if self._mode == self.MODE_WS:
            with self._ws_lock:
                return bool(self._ws_peers)
        return False

    # -- TCP Client --

    def _start_tcp_client(self, host: str, port: int) -> None:
        self._mode = self.MODE_TCP_CLIENT
        self._tcp_host = host
        self._tcp_port = port
        self._stop_event.clear()
        self._log(f"Remote bridge: TCP client 開始 → {host}:{port}")
        t = threading.Thread(
            target=self._tcp_client_loop,
            daemon=True,
            name="mmd-bridge-tcp-client",
        )
        self._tcp_thread = t
        t.start()

    def _tcp_client_loop(self) -> None:
        while not self._stop_event.is_set():
            sock = self._tcp_sock
            if sock is not None:
                try:
                    ready, _, _ = select.select([sock], [], [], 1.0)
                    if ready:
                        data = sock.recv(4096)
                        if data:
                            self._on_inbound(data)
                        if not data:
                            self._tcp_sock = None
                            try:
                                sock.close()
                            except Exception:
                                pass
                            self._log("Remote bridge: TCP client 切断")
                            if self._on_disconnected:
                                self._on_disconnected()
                except Exception:
                    old = self._tcp_sock
                    self._tcp_sock = None
                    if old is not None:
                        try:
                            old.close()
                        except Exception:
                            pass
                        self._log("Remote bridge: TCP client 切断(エラー)")
                        if self._on_disconnected:
                            self._on_disconnected()
                continue

            try:
                new_sock = socket.create_connection(
                    (self._tcp_host, self._tcp_port), timeout=2.0
                )
                self._tcp_sock = new_sock
                self._av_snd_segment_preamble_pending = True
                self._av_started_on_connection = False
                self._log("Remote bridge: TCP client 接続済み")
                self._on_peer_connected()
            except Exception:
                if self._stop_event.wait(timeout=1.5):
                    return

    # -- TCP Server --

    def _start_tcp_server(self, port: int, bind_host: str = "127.0.0.1") -> None:
        self._mode = self.MODE_TCP_SERVER
        self._tcp_port = port
        self._stop_event.clear()
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((bind_host, port))
            srv.listen(1)
            srv.settimeout(1.0)
            self._tcp_server_sock = srv
        except Exception as ex:
            self._log(f"Remote bridge: TCP server 待受失敗: {ex}")
            self._mode = self.MODE_NONE
            return
        self._log(f"Remote bridge: TCP server 待受開始 {bind_host}:{port}")
        t = threading.Thread(
            target=self._tcp_server_accept_loop,
            daemon=True,
            name="mmd-bridge-tcp-server",
        )
        self._tcp_accept_thread = t
        t.start()

    def _tcp_server_accept_loop(self) -> None:
        srv = self._tcp_server_sock
        while not self._stop_event.is_set() and srv is not None:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            old_peer = self._tcp_peer
            if old_peer is not None:
                try:
                    old_peer.close()
                except Exception:
                    pass
            self._tcp_peer = conn
            self._av_snd_segment_preamble_pending = True
            self._av_started_on_connection = False
            self._log(f"Remote bridge: TCP server クライアント接続 {addr}")
            self._on_peer_connected()

            try:
                while not self._stop_event.is_set():
                    ready, _, _ = select.select([conn], [], [], 1.0)
                    if ready:
                        data = conn.recv(4096)
                        if not data:
                            break
                        self._on_inbound(data)
            except Exception:
                pass

            if self._tcp_peer is conn:
                self._tcp_peer = None
            try:
                conn.close()
            except Exception:
                pass
            self._log("Remote bridge: TCP server クライアント切断")
            if self._on_disconnected:
                self._on_disconnected()

    # -- WebSocket Server --

    def _start_websocket(self, host: str, port: int, directory: str) -> None:
        self._mode = self.MODE_WS
        self._ws_host = host
        self._ws_port = port
        self._ws_directory = directory
        self._stop_event.clear()
        self._ws_ready.clear()
        self._log(f"Remote bridge: WebSocket server 開始 {host}:{port}{directory}")
        t = threading.Thread(
            target=self._ws_server_thread,
            daemon=True,
            name="mmd-bridge-ws-server",
        )
        self._ws_thread = t
        t.start()
        self._ws_ready.wait(timeout=5.0)

    def _ws_server_thread(self) -> None:
        try:
            import websockets  # noqa: F811
        except ImportError:
            self._log("Remote bridge: websockets ライブラリが見つかりません")
            self._ws_ready.set()
            return

        # websockets 15+: `run_until_complete(serve(...))` だけでは内部で get_running_loop() が
        # 失敗する。スレッド専用ループを set_event_loop し、serve → serve_forever を一段挟む。
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._ws_loop = loop
        bridge = self
        server: object | None = None

        async def handler(connection):  # websockets.asyncio.server.ServerConnection
            req = connection.request
            path = (req.path if req is not None else "") or "/"
            path = path.split("?", 1)[0]
            if path != bridge._ws_directory:
                bridge._log(f"Remote bridge: WS パス不一致: {path}")
                return

            with bridge._ws_lock:
                first = not bridge._ws_peers
                if connection not in bridge._ws_peers:
                    bridge._ws_peers.append(connection)
                n = len(bridge._ws_peers)
                bridge._ws_peer = connection
            bridge._av_snd_segment_preamble_pending = True
            bridge._av_started_on_connection = False
            bridge._log(f"Remote bridge: WebSocket クライアント接続 (n={n})")
            if first:
                bridge._on_peer_connected()
            try:
                async for message in connection:
                    bridge._on_inbound(message)
            except Exception:
                pass
            finally:
                with bridge._ws_lock:
                    if connection in bridge._ws_peers:
                        bridge._ws_peers.remove(connection)
                    empty = not bridge._ws_peers
                    bridge._ws_peer = bridge._ws_peers[0] if bridge._ws_peers else None
                    left = len(bridge._ws_peers)
                if empty:
                    bridge._log("Remote bridge: WebSocket クライアント切断")
                    if bridge._on_disconnected:
                        bridge._on_disconnected()
                else:
                    bridge._log(f"Remote bridge: WebSocket クライアント切断 (残 {left})")

        async def runner():
            nonlocal server
            server = await websockets.serve(handler, bridge._ws_host, bridge._ws_port)
            bridge._ws_server_obj = server
            bridge._ws_ready.set()
            await server.serve_forever()

        try:
            loop.run_until_complete(runner())
        except Exception as ex:
            msg = str(ex).lower()
            if "event loop stopped" not in msg and "non-thread-safe" not in msg:
                self._log(f"Remote bridge: WS server エラー: {ex}")
            self._ws_ready.set()
        finally:
            if server is not None:
                try:
                    server.close()  # type: ignore[union-attr]
                    loop.run_until_complete(server.wait_closed())  # type: ignore[union-attr]
                except Exception:
                    pass
            with self._ws_lock:
                peers = list(self._ws_peers)
                self._ws_peers.clear()
                self._ws_peer = None
            for ws in peers:
                try:
                    loop.run_until_complete(ws.close())
                except Exception:
                    pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            self._ws_loop = None
            self._ws_server_obj = None

    # -- Send helpers --

    def _tcp_send(self, data: bytes) -> bool:
        notify_disconnected = False
        ok = False
        with self._send_lock:
            if self._mode == self.MODE_TCP_CLIENT:
                sock = self._tcp_sock
                if sock is None:
                    return False
                try:
                    sock.sendall(data)
                    ok = True
                except Exception:
                    self._tcp_sock = None
                    try:
                        sock.close()
                    except Exception:
                        pass
                    notify_disconnected = True
            elif self._mode == self.MODE_TCP_SERVER:
                peer = self._tcp_peer
                if peer is None:
                    return False
                try:
                    peer.sendall(data)
                    ok = True
                except Exception:
                    self._tcp_peer = None
                    try:
                        peer.close()
                    except Exception:
                        pass
                    notify_disconnected = True

        # コールバックは _send_lock を解放してから呼ぶ。保持したまま呼ぶと、pacer スレッド等が
        # 送信で _send_lock を取得しようとして join タイムアウトまでデッドロックしうる。
        if notify_disconnected and self._on_disconnected:
            self._on_disconnected()
        return ok

    def _ws_send(self, data: str | bytes) -> bool:
        loop = self._ws_loop
        if loop is None or not loop.is_running():
            return False
        with self._ws_lock:
            peers = list(self._ws_peers)
        if not peers:
            return False

        async def _broadcast() -> bool:
            ok = False
            dead: list[object] = []
            for ws in peers:
                try:
                    await ws.send(data)  # type: ignore[union-attr]
                    ok = True
                except Exception:
                    dead.append(ws)
            if dead:
                with self._ws_lock:
                    for ws in dead:
                        if ws in self._ws_peers:
                            self._ws_peers.remove(ws)
                    if not self._ws_peers:
                        self._ws_peer = None
                    elif self._ws_peer in dead:
                        self._ws_peer = self._ws_peers[0]
            return ok

        try:
            future = asyncio.run_coroutine_threadsafe(_broadcast(), loop)
            return bool(future.result(timeout=5.0))
        except Exception:
            return False

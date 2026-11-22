
from __future__ import annotations

import json
import socket
import threading
from typing import Callable


JsonDict = dict[str, object]


def _safe_emit(callback: Callable[[JsonDict], None], payload: JsonDict) -> None:
    try:
        callback(payload)
    except Exception:
        pass


class JsonLineDuplexServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, int(port)))
        self._server.listen(1)
        self._server.settimeout(0.5)
        self._conn: socket.socket | None = None
        self._reader = None
        self._writer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._send_lock = threading.Lock()

    @property
    def host(self) -> str:
        return str(self._server.getsockname()[0])

    @property
    def port(self) -> int:
        return int(self._server.getsockname()[1])

    def start(
        self,
        on_event: Callable[[JsonDict], None],
        on_disconnected: Callable[[], None] | None = None,
        on_connected: Callable[[], None] | None = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._connected_event.clear()

        def _loop() -> None:
            try:
                while not self._stop_event.is_set():
                    try:
                        conn, _addr = self._server.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        # Windows: stop() が別スレッドで listen ソケットを閉じた直後に
                        # WinError 10038 (WSAENOTSOCK) になり得る。
                        break
                    # Accept loop timeout should not leak into connection I/O.
                    conn.settimeout(None)
                    self._conn = conn
                    self._reader = conn.makefile("r", encoding="utf-8", errors="replace")
                    self._writer = conn.makefile("w", encoding="utf-8", errors="replace")
                    self._connected_event.set()
                    if on_connected is not None:
                        try:
                            on_connected()
                        except Exception:
                            pass
                    try:
                        while not self._stop_event.is_set():
                            line = self._reader.readline()
                            if not line:
                                break
                            text = line.strip()
                            if not text:
                                continue
                            try:
                                payload = json.loads(text)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(payload, dict):
                                _safe_emit(on_event, payload)
                    except OSError:
                        pass
                    finally:
                        self._connected_event.clear()
                        try:
                            if self._reader is not None:
                                self._reader.close()
                        except Exception:
                            pass
                        # writer の close/None 化は send() と同じ _send_lock 下で行う。
                        # こうしないと send() が write 中に writer を閉じてしまい競合する。
                        with self._send_lock:
                            try:
                                if self._writer is not None:
                                    self._writer.close()
                            except Exception:
                                pass
                            self._writer = None
                        try:
                            conn.close()
                        except Exception:
                            pass
                        self._conn = None
                        self._reader = None
                        # stop() が先に _stop_event を立ててからソケットを閉じた場合は
                        # 「相手切断」ではなく意図的シャットダウン。on_disconnected は不要。
                        if on_disconnected is not None and not self._stop_event.is_set():
                            try:
                                on_disconnected()
                            except Exception:
                                pass
                        # 再接続を許可する: ここで break すると外側の accept ループまで抜け、
                        # listen ソケットが閉じて二度目以降のクライアントを受け付けられなくなる。
            finally:
                try:
                    self._server.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_loop, daemon=True, name="jsonline-ipc-server")
        self._thread.start()

    def wait_connected(self, timeout_sec: float) -> bool:
        return self._connected_event.wait(timeout=max(0.0, float(timeout_sec)))

    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    def send(self, payload: JsonDict) -> None:
        wire = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._send_lock:
            w = self._writer
            if w is None:
                raise RuntimeError("IPC writer is not connected.")
            w.write(wire)
            w.flush()

    def stop(self) -> None:
        self._stop_event.set()
        self._connected_event.clear()
        try:
            if self._conn is not None:
                try:
                    self._conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._conn.close()
        except Exception:
            pass
        try:
            self._server.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._conn = None
        self._reader = None
        with self._send_lock:
            self._writer = None


class JsonLineDuplexClient:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._sock: socket.socket | None = None
        self._reader = None
        self._writer = None
        self._send_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def connect(self, attempts: int = 100, sleep_sec: float = 0.05) -> None:
        import time

        last_error: Exception | None = None
        for _ in range(max(1, int(attempts))):
            try:
                sock = socket.create_connection((self._host, self._port), timeout=2.0)
                # Keep connect timeout only, then switch to blocking.
                sock.settimeout(None)
                self._sock = sock
                self._reader = sock.makefile("r", encoding="utf-8", errors="replace")
                self._writer = sock.makefile("w", encoding="utf-8", errors="replace")
                return
            except OSError as ex:
                last_error = ex
                time.sleep(float(sleep_sec))
        if last_error is None:
            raise RuntimeError("IPC接続に失敗しました。")
        raise RuntimeError(f"IPC接続に失敗しました: {last_error}") from last_error

    def start_reader(
        self,
        on_event: Callable[[JsonDict], None],
        on_disconnected: Callable[[], None] | None = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            try:
                while not self._stop_event.is_set():
                    if self._reader is None:
                        break
                    line = self._reader.readline()
                    if not line:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        _safe_emit(on_event, payload)
            except OSError:
                pass
            finally:
                # close() が先に _stop_event を立ててから reader を閉じた場合は
                # 意図的シャットダウン。ここで on_disconnected を呼ぶと Controller 側が
                # 「IPC切断→再接続」を二重にスケジュールしループする。
                if on_disconnected is not None and not self._stop_event.is_set():
                    try:
                        on_disconnected()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=_loop, daemon=True, name="jsonline-ipc-client-reader")
        self._thread.start()

    def reader_alive(self) -> bool:
        """start_reader の読み取りが続いているか（相手に切られたら False）。"""
        return self._thread is not None and self._thread.is_alive()

    def send(self, payload: JsonDict) -> None:
        wire = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._send_lock:
            w = self._writer
            if w is None:
                raise RuntimeError("IPC writer is not connected.")
            w.write(wire)
            w.flush()

    def close(self) -> None:
        self._stop_event.set()
        # A makefile().readline() in the reader thread may remain blocked if
        # the file object is closed first.  Shut down the socket before
        # closing either wrapper so reconnect and application shutdown cannot
        # hang waiting for peer traffic.
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            # makefile() の読み書きが残っていると sock.close() は OS のハンドルを閉じない。
            # Windows では shutdown だけでは別スレッドの recv が戻らない（相手が黙ってつないだまま
            # だと reader.close() が読み取り中のロック待ちで止まる）。ハンドルを直接閉じて recv を
            # 終わらせる。
            try:
                fd = sock.detach()
                if fd is not None and fd >= 0:
                    socket.close(fd)
            except Exception:
                try:
                    sock.close()
                except Exception:
                    pass
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        self._reader = None
        # writer の close/None 化は send() と同じ _send_lock 下で行い、
        # send() が write 中に writer を閉じる競合を防ぐ。
        with self._send_lock:
            try:
                if self._writer is not None:
                    self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None


class JsonLineEventServer(JsonLineDuplexServer):
    """音声・ASR などイベントストリーム向けの DuplexServer（着信側）。"""


class JsonLineEventClient(JsonLineDuplexClient):
    """イベントストリーム送信側の DuplexClient。"""


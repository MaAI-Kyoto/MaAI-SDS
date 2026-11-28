"""ブラウザ版 MMDAgent-EX（MMDAgent-EX Web）を手元の HTTP サーバーで配り、シーンを指定して開く。

アバターの表示先がブラウザのとき、メイン画面の「MMDAgent-EX起動」（Web UI も同じ）で次を行う。

1. この PC の HTTP サーバー（127.0.0.1 だけで待ち受ける）で、MMDAgent-EX Web のページ（`/`）と、
   MDF の入ったコンテンツのフォルダー（`/content/`）を配る。コンテンツのファイルの一覧は
   `/content/files.json`（MMDAgent-EX Web の `src/launch.js` が読む形）。
2. `index.html?content=/content/&mdf=<MDF>&ws=<対話サーバーの WebSocket>&connect=1` をブラウザで開く。
   ページはフォルダーを選ばずにシーンを始め、対話サーバーへつなぐ。

コンテンツのフォルダーは MDF のフォルダーの 1 つ上（README の置き方では `example/`。MDF の FST は `../gene/...` のように
サンプル集のアバターをたどる）。MDF は、隣に `<名前>_web.mdf` があればそれを使う（`maai_sds.mdf` → `maai_sds_web.mdf`）。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_HTTP_PORT = 8010
CONTENT_PREFIX = "/content/"
FILE_LIST_NAME = "files.json"
# URL でシーンを開けるのは、この機能（src/launch.js）がある版から
_REQUIRED_WEB_FILES = ("index.html", "src/app.js", "src/launch.js")
# Windows では拡張子の種類をレジストリから読むので、.js が text/plain になることがある
# （モジュールのスクリプトは種類が違うと読まれない）。ページに要るものは固定する。
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class WebViewerError(RuntimeError):
    """MMDAgent-EX Web を開けない理由（画面にそのまま出す）。"""


def default_web_dir(platform: str | None = None) -> Path:
    """MMDAgent-EX Web を探す既定の場所（README の置き方）。Windows は C:\\mmdagent-ex-web、ほかはホームの下。"""
    if (platform or sys.platform) == "win32":
        return Path(r"C:\mmdagent-ex-web")
    return Path.home() / "mmdagent-ex-web"


def resolve_web_dir(raw: str | os.PathLike[str] | None) -> Path:
    """設定の MMDAgent-EX Web のフォルダー（空なら既定の場所）。使えなければ WebViewerError。"""
    text = str(raw or "").strip()
    web_dir = Path(text).expanduser() if text else default_web_dir()
    if not (web_dir / "index.html").is_file():
        raise WebViewerError(
            f"MMDAgent-EX Web が見つかりません：{web_dir}\n\n"
            "git clone https://github.com/kazu44ttaka/mmdagent-ex-web.git でこの場所に置くか、"
            "「MMDAgent-EX起動設定」で置いた場所を指定してください。"
        )
    missing = [name for name in _REQUIRED_WEB_FILES if not (web_dir / name).is_file()]
    if missing:
        raise WebViewerError(
            f"MMDAgent-EX Web が古いため、URL でシーンを開けません：{web_dir}\n\n"
            "そのフォルダーで git pull してから、もう一度起動してください。"
        )
    return web_dir


def browser_mdf_for(mdf_path: str | os.PathLike[str] | None) -> Path:
    """ブラウザで開く MDF。隣に `<名前>_web.mdf` があればそれ（TCP ではなく WebSocket でつなぐ設定）。"""
    text = str(mdf_path or "").strip()
    if not text:
        raise WebViewerError(
            "MDF ファイルが設定されていません。「MMDAgent-EX起動設定」の「ファイル設定」で、"
            "サンプル集の maai_sds/maai_sds_web.mdf を指定してください。"
        )
    mdf = Path(text).expanduser()
    if not mdf.stem.lower().endswith("_web"):
        sibling = mdf.with_name(f"{mdf.stem}_web{mdf.suffix}")
        if sibling.is_file():
            mdf = sibling
    if not mdf.is_file():
        raise WebViewerError(f"MDF ファイルが見つかりません：{mdf}")
    return mdf


def content_root_for(mdf: Path) -> Path:
    """ページに配るコンテンツのフォルダー（MDF のフォルダーの 1 つ上）。"""
    folder = mdf.resolve().parent
    return folder.parent if folder.parent != folder else folder


def list_content_files(root: Path) -> list[str]:
    """コンテンツのフォルダーの中のファイル（`/` 区切りの相対パス）。`.git` などの隠しフォルダーは除く。"""
    out: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        base = Path(current)
        for name in sorted(files):
            if name.startswith("."):
                continue
            out.append((base / name).relative_to(root).as_posix())
    return out


def launch_url(port: int, mdf_rel: str, ws_url: str, *, connect: bool = True) -> str:
    """シーンと接続先を指定したページの URL（MMDAgent-EX Web の src/launch.js の形）。"""
    query = urllib.parse.urlencode(
        {"content": CONTENT_PREFIX, "mdf": mdf_rel, "ws": ws_url, **({"connect": "1"} if connect else {})},
        safe="/:",
    )
    return f"http://localhost:{int(port)}/index.html?{query}"


class _Server(ThreadingHTTPServer):
    # Windows の SO_REUSEADDR は、ほかのプログラムが使っているポートにも重ねて bind できてしまう
    # （どちらに届くか分からなくなる）。POSIX では閉じた直後のポートを使い直すために残す。
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


def _make_handler(web_dir: Path, content_dir: Path) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            route = urllib.parse.urlsplit(path).path
            if route.startswith(CONTENT_PREFIX):
                self.directory = str(content_dir)
                path = "/" + route[len(CONTENT_PREFIX):]
            else:
                self.directory = str(web_dir)
            return super().translate_path(path)

        def guess_type(self, path: str) -> str:  # type: ignore[override]
            known = _CONTENT_TYPES.get(Path(str(path)).suffix.lower())
            return known or super().guess_type(path)

        def end_headers(self) -> None:
            # ページを更新（git pull）したら、すぐ新しいものを読ませる
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def do_GET(self) -> None:  # noqa: N802
            route = urllib.parse.urlsplit(self.path).path
            if route == CONTENT_PREFIX + FILE_LIST_NAME:
                body = json.dumps({"files": list_content_files(content_dir)}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # 1 ファイルごとの行はコンソールに出さない

    return Handler


class WebViewerServer:
    """MMDAgent-EX Web のページとコンテンツを配る HTTP サーバー（別スレッド）。"""

    def __init__(self, web_dir: Path, content_dir: Path, port: int = DEFAULT_HTTP_PORT, host: str = "127.0.0.1") -> None:
        self.web_dir = Path(web_dir)
        self.content_dir = Path(content_dir)
        self.port = int(port)
        self.host = host
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            server = _Server((self.host, self.port), _make_handler(self.web_dir, self.content_dir))
        except OSError as ex:
            raise WebViewerError(
                f"MMDAgent-EX Web を配るポート {self.port} を開けません（{ex}）。"
                "ほかのプログラムが使っているときは、「MMDAgent-EX起動設定」でページのポートを変えてください。"
            ) from ex
        self._server = server
        self.port = int(server.server_address[1])  # 0 を渡したとき（テスト）の実際のポート
        self._thread = threading.Thread(target=server.serve_forever, name="mmdagent-ex-web-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        self._thread = None

    def is_running(self) -> bool:
        return self._server is not None

    def serves(self, web_dir: Path, content_dir: Path, port: int) -> bool:
        return (
            self.is_running()
            and self.web_dir == Path(web_dir)
            and self.content_dir == Path(content_dir)
            and (int(port) in (0, self.port))
        )


class WebViewerLauncher:
    """設定から MMDAgent-EX Web を配って URL を作る（GUI と Web UI で 1 つずつ持つ）。"""

    def __init__(self) -> None:
        self._server: WebViewerServer | None = None
        self._lock = threading.Lock()

    def launch(self, *, web_dir: str | None, mdf_path: str | None, http_port: int, ws_port: int) -> str:
        """サーバーを用意して（設定が同じなら使い回す）、ブラウザで開く URL を返す。"""
        web = resolve_web_dir(web_dir)
        mdf = browser_mdf_for(mdf_path)
        content = content_root_for(mdf)
        with self._lock:
            if self._server is None or not self._server.serves(web, content, http_port):
                if self._server is not None:
                    self._server.stop()
                server = WebViewerServer(web, content, http_port)
                server.start()
                self._server = server
            port = self._server.port
        mdf_rel = mdf.resolve().relative_to(content).as_posix()
        return launch_url(port, mdf_rel, f"ws://localhost:{int(ws_port)}/chat")

    def stop(self) -> bool:
        """サーバーを止める。動いていたら True。"""
        with self._lock:
            server, self._server = self._server, None
        if server is None:
            return False
        server.stop()
        return True

    def is_running(self) -> bool:
        return self._server is not None and self._server.is_running()

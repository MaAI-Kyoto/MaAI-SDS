import argparse
import os
import signal
import sys
import threading
import traceback

# RecursionError 発生時、標準のトレースバック整形が再帰制限に掛かり
# 「object type name: RecursionError」「lost sys.stderr」だけになることがある。
# 表示前に一時的に再帰上限を上げ、__stderr__ へ直接出す。
_REAL_STDERR = getattr(sys, "__stderr__", None) or sys.stderr


def _bump_recursion_limit_for_traceback() -> int:
    old = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(old + 2000, 8000))
    except Exception:
        pass
    return old


def _install_early_exception_hooks() -> None:
    def _excepthook(exc_type, exc, tb) -> None:
        old = _bump_recursion_limit_for_traceback()
        try:
            traceback.print_exception(exc_type, exc, tb, file=_REAL_STDERR, chain=False)
        except BaseException:
            try:
                _REAL_STDERR.write(
                    f"{getattr(exc_type, '__name__', str(exc_type))}: (traceback printing failed)\n"
                )
            except Exception:
                pass
        finally:
            try:
                sys.setrecursionlimit(old)
            except Exception:
                pass

    sys.excepthook = _excepthook

    def _thread_exc(args: object) -> None:
        old = _bump_recursion_limit_for_traceback()
        try:
            thread = getattr(args, "thread", None)
            tname = getattr(thread, "name", "?") if thread is not None else "?"
            _REAL_STDERR.write(f"Exception in thread {tname!r}:\n")
            traceback.print_exception(
                getattr(args, "exc_type", None),
                getattr(args, "exc_value", None),
                getattr(args, "exc_traceback", None),
                file=_REAL_STDERR,
                chain=False,
            )
        except BaseException:
            try:
                _REAL_STDERR.write("threading: (traceback printing failed)\n")
            except Exception:
                pass
        finally:
            try:
                sys.setrecursionlimit(old)
            except Exception:
                pass

    threading.excepthook = _thread_exc


_install_early_exception_hooks()

try:
    from PySide6.QtCore import QTimer, QtMsgType, qInstallMessageHandler
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:
    # `gui` は optional extra（ヘッドレスの Docker イメージには意図的に入れていない
    # ——Dockerfile 参照）なので、素の `uv sync` / `uv run` では PySide6 が入らない。
    # ここがこのファイル最初の重い import なので、案内を出さないと新規クローンでの
    # `uv run python main.py` が生の ModuleNotFoundError で死ぬ。エンジン管理の GUI は
    # PySide6 が無いと開けないので、この経路でしか伝えられない。
    #
    # 案内するのは **起動までの 1 コマンドだけ**。ASR/TTS エンジンごとの依存は
    # 起動後にエンジン管理から入れる（torch / onnxruntime-gpu は maai / kokoro-onnx が
    # 引くので、この 1 コマンドで既に入っている）。
    _REAL_STDERR.write(
        "PySide6 がインストールされていません。GUI の起動には 'gui' extra が必要です。\n"
        "\n"
        "  uv sync --extra gui\n"
        "  .venv\\Scripts\\python main.py\n"
        "\n"
        "**`uv run` は使わないこと。** 既定で extra 無しの exact sync をするため、\n"
        "`uv run python main.py` は PySide6 を削除して GUI を起動不能にし、torch も\n"
        "素の版へ戻す。どうしても uv 経由で起動するなら `uv run --no-sync` を使う。\n"
        "\n"
        "上の 2 行だけで CPU 環境が揃う（torch は PyPI 版が入り、この環境では\n"
        "CPU 専用）。GPU を使う場合と、ASR/TTS エンジンごとの依存・モデルは、\n"
        "起動後に GUI の「エンジン管理」から入れてください。\n"
    )
    sys.exit(1)

from .gui.controller import MMDAgentController
from .gui.theme import apply_dark_theme


def _qt_message_filter(msg_type, _context, message: str) -> None:
    """Reimplements Qt's default message handling (print to stderr, abort on fatal), but
    drops one specific known-benign warning.

    Qt6 can print `QThreadStorage: entry N destroyed before
    end of thread 0x...` at app exit. This app runs many fire-and-forget `daemon=True`
    background threads (subprocess stdout/stderr pumps, IPC sender loops, etc. — see
    gui/controller.py) that call into Qt (signal `.emit()`) from off the GUI thread; some of
    those can still be alive when the last window closes and Qt tears down its internal
    per-thread QThreadStorage bookkeeping as part of `app.exec()` returning. That race is
    exactly why `main()` already calls `os._exit()` below instead of a normal Python exit
    (see its comment) — auditing/joining every one of those threads at the precise right
    moment to make Qt6 never print this would be a much larger, riskier change for a message
    that has shown no functional impact (no crash, no hang, no leaked data). Filter just this
    one known message; everything else still reaches stderr (and still aborts on Fatal, like
    Qt's own default handler would).
    """
    if "QThreadStorage" in message and "destroyed before end of thread" in message:
        return
    try:
        sys.stderr.write(message + "\n")
    except Exception:
        pass
    if msg_type == QtMsgType.QtFatalMsg:
        import os as _os

        _os.abort()


def _parse_launch_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--remote-orchestrator",
        metavar="HOST:PORT",
        help="connect Qt to an existing dialog orchestrator instead of spawning one",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="controller config file (defaults to controller_config.json)",
    )
    parser.add_argument(
        "--dialog-server",
        choices=("local", "docker"),
        help="dialog server to use (defaults to the one last chosen in the GUI)",
    )
    # GUI の「対話サーバー」切り替えで開き直すとき、前の GUI が閉じるのを待つ。
    parser.add_argument("--wait-pid", type=int, default=0, help=argparse.SUPPRESS)
    args, qt_args = parser.parse_known_args(argv[1:])
    return args, [argv[0], *qt_args]


def main() -> None:
    qInstallMessageHandler(_qt_message_filter)
    launch_args, qt_argv = _parse_launch_args(list(sys.argv))
    from . import paths as _paths
    from . import dialog_server_mode

    if launch_args.wait_pid:
        dialog_server_mode.wait_for_pid_exit(launch_args.wait_pid)
    from .gui import app_icon

    if not launch_args.config and _paths.ensure_config_file():
        print("[maai-sds] controller_config.json が無いので controller_config.example.json から作りました", flush=True)
    app_icon.set_windows_app_id()  # タスクバーで Python と別に並べる（窓を作る前）
    app = QApplication(qt_argv)
    app.setWindowIcon(app_icon.app_icon())
    apply_dark_theme(app)
    # ローカル / Docker のどちらの対話サーバーを使うか（GUI で選んだものを覚えている）。
    _mode, _env, mode_notice = dialog_server_mode.resolve_launch(
        launch_args.dialog_server,
        launch_args.remote_orchestrator,
        dict(os.environ),
        _paths.REPO_ROOT,
        dialog_server_mode.load_saved_mode(),
    )
    dialog_server_mode.apply_env(_env)
    win = MMDAgentController(
        config_path=launch_args.config,
        remote_orchestrator=launch_args.remote_orchestrator,
    )
    from .gui import window_state

    window_state.restore_main(win)  # 前に閉じたときの位置・大きさ（表示する前に戻す）
    win.show()
    QTimer.singleShot(0, lambda: window_state.restore_children(win))  # 子の窓（設定・コンソールなど）の位置・大きさ（開きはしない）
    if mode_notice:
        QTimer.singleShot(0, lambda: QMessageBox.information(win, "対話サーバー", mode_notice))

    # シグナルハンドラからは Qt API を呼ばない（別スレッド扱いになり
    # 「Timers cannot be stopped from another thread」等が出る）。
    shutdown_flag = {"requested": False}
    shutdown_initiated = {"done": False}

    def _request_shutdown() -> None:
        if shutdown_initiated["done"]:
            return
        shutdown_initiated["done"] = True
        win.close()
        app.quit()

    def _handle_sigint(_sig, _frame) -> None:
        if shutdown_flag["requested"]:
            return
        shutdown_flag["requested"] = True
        # 1回目の Ctrl+C で即時に GUI スレッド上の終了処理をキックする。
        QTimer.singleShot(0, _request_shutdown)

    signal.signal(signal.SIGINT, _handle_sigint)

    # イベントループを定期的に起こして SIGINT を Python に届けつつ、
    # 終了要求はメイン（GUI）スレッドで処理する。
    heartbeat = QTimer(app)
    heartbeat.setInterval(200)

    def _heartbeat_tick() -> None:
        if shutdown_flag["requested"]:
            heartbeat.stop()
            _request_shutdown()

    heartbeat.timeout.connect(_heartbeat_tick)
    heartbeat.start()

    exit_code = app.exec()
    # closeEvent がすべてのワーカーを停止済み。Python 通常終了によるデーモンスレッドの
    # RecursionError / "lost sys.stderr" を防ぐため os._exit で即終了する。
    import os as _os
    _os._exit(exit_code)


if __name__ == "__main__":
    main()

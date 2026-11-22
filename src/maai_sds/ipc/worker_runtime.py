from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Any, Callable


def parse_json_object(raw_json: str, *, worker_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as ex:
        raise ValueError(f"[{worker_name}][FATAL] invalid config json: {ex}") from ex
    if not isinstance(payload, dict):
        raise ValueError(f"[{worker_name}][FATAL] config json must be object.")
    return payload


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    """IPC 用に JSON 直列化可能な構造へ再帰変換する。

    - NumPy 配列は tolist、スカラーは Python ネイティブへ（float32 等が str 化しないようにする）。
    - float の nan / inf は JSON 互換のため None にする（json.dumps が失敗しないようにする）。
    """
    if _depth > 128:
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, _depth + 1) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return to_jsonable(value.tolist(), _depth + 1)
        except Exception:
            pass
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return to_jsonable(value.item(), _depth + 1)
    except Exception:
        pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_emit(send_func: Callable[[dict[str, object]], None], event_type: str, **data: object) -> None:
    payload: dict[str, object] = {"type": event_type}
    payload.update(data)
    try:
        send_func(payload)
    except Exception:
        pass


# IPC 接続のリトライ。200 回 × 0.05 秒 = 最大 10 秒待つ。
# 親（Orchestrator / Controller）がサーバを立てるより先に子プロセスが起動することが
# あるため、すぐ諦めずに待つ。**この数値は挙動に直結する**ので変えないこと。
CONNECT_ATTEMPTS = 200
CONNECT_SLEEP_SEC = 0.05


# 起動側が自分の PID を入れる。トランポリンを挟んでも論理的な親を追える。
PARENT_PID_ENV = "MAAI_SDS_PARENT_PID"


def _parent_exit_probe() -> "Callable[[], bool] | None":
    """「親プロセスが終了したか」を返す関数を作る。作れなければ None。

    Windows では親の PID を持っていても、親が死んだあと PID が別プロセスに
    再利用されうる。そこで起動時に SYNCHRONIZE ハンドルを開いて持ち続け、
    それを待つ。ハンドルを握っている間はそのハンドルが指す対象が入れ替わらないので、
    「この親が終わった」を取り違えない。

    POSIX では親が死ぬと getppid() が変わる（init/reaper に付け替わる）ので、
    起動時の値と比べるだけでよい。
    """
    # os.getppid() では足りない。Windows の venv の python.exe はトランポリンで、
    # 実際の親はそれになる（実測: Popen が返す pid と子自身の pid が別）。
    # トランポリンは論理的な親（GUI / orchestrator）が死んでも生き残るので、
    # getppid() を見ていると親の死を拾えない。
    # そこで起動側が MAAI_SDS_PARENT_PID に自分の PID を入れ、それを見る。
    raw = str(os.environ.get(PARENT_PID_ENV, "") or "").strip()
    try:
        ppid = int(raw) if raw else os.getppid()
    except ValueError:
        ppid = os.getppid()
    if ppid <= 1 or ppid == os.getpid():
        return None  # 親を特定できない（単体実行やテスト）
    if os.name != "nt":
        return lambda: os.getppid() != ppid

    import ctypes  # noqa: PLC0415

    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0x00000000
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(SYNCHRONIZE, False, ppid)
    if not handle:
        return None
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

    def _exited() -> bool:
        return k32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == WAIT_OBJECT_0

    return _exited


def start_parent_death_watchdog(
    on_parent_exit: Callable[[], None],
    *,
    worker_name: str = "Worker",
    poll_sec: float = 2.0,
    hard_exit_after_sec: float = 10.0,
) -> bool:
    """親プロセスが消えたら `on_parent_exit` を呼ぶ監視を始める。

    stdin の EOF に頼ると、非対話起動（`< /dev/null` や CI）で起動直後に EOF が
    来て監視を自分で無効化してしまう。実際にそれで、GUI がクラッシュした後に
    orchestrator と audio_output が生き残り続けた（1 回の起動ごとに 1 組ずつ
    溜まり、18 組まで積み上がって CPU とメモリを食っていた）。
    PID の生存を直接見れば、綺麗に終了しなかった場合にも効く。

    親を特定できないときは何もせず False を返す（起動を妨げない）。
    """
    try:
        probe = _parent_exit_probe()
    except Exception:  # noqa: BLE001
        probe = None
    if probe is None:
        return False

    def _loop() -> None:
        while True:
            time.sleep(max(0.2, float(poll_sec)))
            try:
                if not probe():
                    continue
            except Exception:  # noqa: BLE001
                return
            # 親が死ぬと stdout/stderr のパイプの読み手も消えるので、ここでの
            # print は例外を投げうる。素で書くとこのスレッドがコールバック前に
            # 死に、ワーカーが残る（実測でそうなった）。必ず包む。
            try:
                print(f"[{worker_name}] 親プロセスが終了したため停止します。",
                      file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001
                pass
            try:
                on_parent_exit()
            except Exception:  # noqa: BLE001
                pass
            # 綺麗な停止が終わらない場合の最後の砦。孤児を残さないことが目的なので、
            # 猶予を過ぎたら落とす（親は既に居ないので誰も結果を待っていない）。
            deadline = time.time() + float(hard_exit_after_sec)
            while time.time() < deadline:
                time.sleep(0.25)
            os._exit(0)

    threading.Thread(target=_loop, daemon=True, name="parent-death-watchdog").start()
    return True


def connect_worker_client(client: Any, *, worker_name: str) -> bool:
    """ワーカーの IPC 接続。失敗時はメッセージを出して False を返す。

    呼び出し側は `if not connect_worker_client(...): return 2` と書く
    （終了コード 2 = 起動時の致命的失敗。従来どおり）。
    7 ワーカーに同じ 5 行が並んでいたのを 1 本化したもので、
    ログ文言・終了コード・リトライ回数は変えていない。
    """
    try:
        client.connect(attempts=CONNECT_ATTEMPTS, sleep_sec=CONNECT_SLEEP_SEC)
    except Exception as ex:
        print(f"[{worker_name}][FATAL] IPC connect failed: {ex}", file=sys.stderr, flush=True)
        return False
    # ここが 7 ワーカー共通の唯一の通り道なので、親死監視もここで始める。
    # ワーカーは子を持たないので、親が消えたら即終了してよい（IPC の相手も居ない）。
    start_parent_death_watchdog(
        lambda: os._exit(0), worker_name=worker_name,
    )
    return True


def finish_worker(client: Any, on_stopped: Callable[[], None]) -> None:
    """ワーカーの終了処理。**戻ってこない**（`os._exit(0)` する）。

    5 ワーカーに同じ 10 行が並んでいたのを 1 本化した。挙動は変えていない:
      - `stopped` の通知と `client.close()` はどちらも失敗しても無視する
        （終了処理なので、ここで例外を上げると後片付けが止まる）
      - 最後は `os._exit(0)`。デーモンスレッドや ONNX Runtime の
        バックグラウンドが残っていても確実に落とすための即時終了で、
        `sys.exit` に変えると終わらないことがある
    """
    try:
        on_stopped()
    except BaseException:
        pass
    try:
        client.close()
    except BaseException:
        pass
    import os as _os

    _os._exit(0)

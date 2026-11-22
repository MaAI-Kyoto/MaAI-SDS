import json
import os
from pathlib import Path


class ConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self, data: dict) -> None:
        self._write(data, raise_on_error=False)

    def save_raising(self, data: dict) -> None:
        self._write(data, raise_on_error=True)

    def _write(self, data: dict, *, raise_on_error: bool) -> None:
        # アトミック書き込み: 同一ディレクトリの temp に書いてから os.replace で置換する。
        # controller_config.json は Orchestrator プロセスも読むため、直接上書きすると
        # 書込中のクラッシュや並行読込で半端な（壊れた）JSON を掴ませる恐れがある。
        try:
            payload = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as ex:
            if raise_on_error:
                raise ValueError(f"configuration is not JSON-serializable: {ex}") from ex
            return
        tmp_path = self.path.with_name(self.path.name + f".tmp.{os.getpid()}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp_path, self.path)
            except OSError:
                # Docker Desktop の Windows バインドマウントは rename が EBUSY になる。
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        except Exception as ex:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if raise_on_error:
                raise OSError(f"failed to write {self.path}: {ex}") from ex


import json
import os
from pathlib import Path


def example_path_for(path: Path) -> Path:
    """設定ファイルの見本の場所（同じフォルダーの `<名前>.example<拡張子>`）。

    controller_config.json → controller_config.example.json、
    controller_config.docker.json → controller_config.docker.example.json。
    """
    return path.with_name(f"{path.stem}.example{path.suffix}")


def _read_json_object(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


class ConfigStore:
    """設定ファイルの読み書き。

    読むときは、同じフォルダーに見本（example_path_for）があれば下に敷き、設定ファイルに無い項目を
    見本の値で埋める。既定値の置き場を見本 1 か所にするため（以前は画面の部品・読み込み時の既定・
    対話サーバー側の既定がそれぞれ別の値を持ち、見本ともずれていた）。見本の無い場所（テストの
    一時ファイルなど）では、設定ファイルの中身だけを返す。
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict:
        example = example_path_for(self.path)
        merged = _read_json_object(example) if example.is_file() and example != self.path else {}
        if self.path.exists():
            merged.update(_read_json_object(self.path))
        return merged

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


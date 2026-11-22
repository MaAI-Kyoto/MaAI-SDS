"""Docker の対話サーバー（compose の orchestrator サービス）を GUI から起動・停止する。

`scripts/start_qt_docker.ps1` が、自分で使った compose ファイルを環境変数で GUI に渡す。

- MAAI_SDS_QT_COMPOSE_FILES   … compose ファイル（os.pathsep 区切り。Windows は `;`）
- MAAI_SDS_QT_COMPOSE_DIR     … docker compose を実行するフォルダー（省略時は 1 つ目のファイルのフォルダー）
- MAAI_SDS_QT_COMPOSE_PROJECT … compose のプロジェクト名（省略可）

渡されていない（手で compose を起動した）ときは、GUI からは起動・停止できない。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

SERVICE = "orchestrator"
PROFILE = "qt"
ENABLED_ENGINES = "enabled_engines.json"  # docker-compose.yml がファイルとしてマウントする（Git 管理外）


@dataclass(frozen=True)
class DockerComposeTarget:
    files: tuple[str, ...]
    workdir: str
    project: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "DockerComposeTarget | None":
        source = os.environ if env is None else env
        raw = str(source.get("MAAI_SDS_QT_COMPOSE_FILES", "") or "").strip()
        files = tuple(part.strip() for part in raw.split(os.pathsep) if part.strip())
        if not files:
            return None
        workdir = str(source.get("MAAI_SDS_QT_COMPOSE_DIR", "") or "").strip() or str(Path(files[0]).parent)
        project = str(source.get("MAAI_SDS_QT_COMPOSE_PROJECT", "") or "").strip()
        return cls(files=files, workdir=workdir, project=project)

    def prepare_mounts(self) -> str:
        """compose がファイルとしてマウントする enabled_engines.json を用意する。問題があれば理由を返す。

        clone した直後には無い（Git 管理外）。無いまま compose を起動すると、Docker が同じ名前の
        フォルダーを作ってしまい、エンジン管理の保存がホストでもコンテナでも失敗する。
        """
        path = Path(self.files[0]).parent / ENABLED_ENGINES
        if path.is_dir():
            return (f"{path} がフォルダーになっています（ファイルが無いまま compose を起動したため）。"
                    "フォルダーを消してから、もう一度起動してください。")
        if not path.exists():
            try:
                path.write_text("[]\n", encoding="utf-8")
            except OSError as ex:
                return f"{path} を作れませんでした: {ex}"
        return ""

    def base_command(self) -> list[str]:
        """`docker compose -f ... [-p ...]`（サービスの指定の前まで）。"""
        args = ["docker", "compose"]
        for path in self.files:
            args += ["-f", path]
        if self.project:
            args += ["-p", self.project]
        return args

    def command(self, action: str, *, service: str = SERVICE, profile: str = PROFILE) -> list[str]:
        """action は "start"（up -d）か "stop"。既定は対話サーバー（orchestrator）。

        同じ compose の別のサービス（Qwen3-ASR の qwen3-asr・プロファイル qwen3 など）にも使う。
        """
        args = self.base_command() + ["--profile", profile]
        if action == "start":
            args += ["up", "-d", service]
        elif action == "stop":
            args += ["stop", service]
        else:
            raise ValueError(f"unknown action: {action}")
        return args

    def run(self, action: str, *, timeout_sec: float = 180.0, service: str = SERVICE, profile: str = PROFILE,
            env: dict[str, str] | None = None) -> tuple[bool, str]:
        """コマンドを実行して (成功したか, 出力の末尾) を返す。GUI スレッドからは呼ばない。

        env は compose の変数（${QWEN_ASR_MODEL} など）として足す。
        """
        if action == "start":
            problem = self.prepare_mounts()
            if problem:
                return False, problem
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                self.command(action, service=service, profile=profile),
                env={**os.environ, **env} if env else None,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                creationflags=creationflags,
            )
        except FileNotFoundError:
            return False, "docker コマンドが見つかりません。Docker Desktop が入っているか確認してください。"
        except subprocess.TimeoutExpired:
            return False, f"docker compose が {int(timeout_sec)} 秒で終わりませんでした。"
        output = "\n".join(s for s in (proc.stdout.strip(), proc.stderr.strip()) if s)
        return proc.returncode == 0, output[-2000:]

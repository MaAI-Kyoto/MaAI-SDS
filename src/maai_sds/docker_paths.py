"""Translate paths across the bind mounts used by ``docker-compose.yml``.

The native Qt process sees Windows paths while the remote orchestrator sees the
same files below Linux mount points.  Keeping the mapping here prevents GUI
code from growing one-off ``/host_repo`` substitutions.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Mapping


_ENV_KEYS = ("INPUT_WAV_HOST", "MMDAGENT_ASSETS_HOST")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def load_dotenv_values(path: Path) -> dict[str, str]:
    """Read the small KEY=VALUE subset used by Compose's project ``.env``."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _expand_refs(value: str, env: Mapping[str, str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2) or ""
        return str(env.get(key, match.group(0)))

    return os.path.expandvars(_ENV_REF_RE.sub(_replace, value))


def _absolute_host_path(path: str | os.PathLike[str], *, base: Path) -> Path:
    text = os.path.expanduser(os.path.expandvars(os.fspath(path)))
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    # abspath removes '..' without following junctions/symlinks.  Compose also
    # receives the lexical host path, which is what file pickers return.
    return Path(os.path.abspath(os.fspath(candidate)))


def _norm_host(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True)
class MountMapping:
    host: Path
    container: PurePosixPath
    is_file: bool = False


class DockerPathMapper:
    """Bidirectional mapping for the bind mounts shared by Web and Qt Docker."""

    def __init__(self, repo_root: Path, mounts: list[MountMapping]) -> None:
        self.repo_root = _absolute_host_path(repo_root, base=Path.cwd())
        normalized: list[MountMapping] = []
        for mount in mounts:
            normalized.append(
                MountMapping(
                    host=_absolute_host_path(mount.host, base=self.repo_root),
                    container=PurePosixPath(posixpath.normpath(str(mount.container))),
                    is_file=bool(mount.is_file),
                )
            )
        # Prefer the most specific mount when paths overlap (for example
        # repo/mmdagent_assets is also reachable through /host_repo).
        self.mounts = tuple(
            sorted(
                normalized,
                key=lambda item: (len(_norm_host(item.host)), len(item.container.parts)),
                reverse=True,
            )
        )

    @classmethod
    def from_compose_project(
        cls,
        repo_root: Path,
        *,
        env: Mapping[str, str] | None = None,
        env_file: Path | None = None,
    ) -> "DockerPathMapper":
        """Build mappings from the mounts declared in ``docker-compose.yml``.

        Compose project variables override values from ``.env``.  Supplying
        ``env`` makes that overlay explicit and keeps tests independent of the
        caller's process environment.
        """
        root = _absolute_host_path(repo_root, base=Path.cwd())
        values = load_dotenv_values(env_file or (root / ".env"))
        overlay: Mapping[str, str] = os.environ if env is None else env
        for key in _ENV_KEYS:
            if key in overlay:
                values[key] = str(overlay[key])

        def _host_value(key: str, default: str) -> Path:
            raw = str(values.get(key, "") or default)
            raw = _expand_refs(raw, {**values, **dict(overlay)})
            return _absolute_host_path(raw, base=root)

        input_wav = _host_value("INPUT_WAV_HOST", "./input_wav")
        mmdagent_assets = _host_value("MMDAGENT_ASSETS_HOST", "./mmdagent_assets")
        mounts = [
            MountMapping(root / "controller_config.docker.json", PurePosixPath("/app/controller_config.json"), True),
            MountMapping(root / "enabled_engines.json", PurePosixPath("/app/enabled_engines.json"), True),
            MountMapping(mmdagent_assets, PurePosixPath("/app/mmdagent_assets")),
            MountMapping(input_wav, PurePosixPath("/app/input_wav")),
            MountMapping(root / "voicevox_runtime_linux", PurePosixPath("/app/voicevox_runtime")),
            MountMapping(root / "maai_assets", PurePosixPath("/app/maai_assets")),
            MountMapping(root / "scripts", PurePosixPath("/app/scripts")),
            MountMapping(root / "logs", PurePosixPath("/app/logs")),
            MountMapping(root / "web", PurePosixPath("/app/web")),
            MountMapping(root / "src", PurePosixPath("/app/src")),
            MountMapping(root, PurePosixPath("/host_repo")),
        ]
        return cls(root, mounts)

    def host_to_container(self, path: str | os.PathLike[str]) -> str | None:
        candidate = _absolute_host_path(path, base=self.repo_root)
        candidate_norm = _norm_host(candidate)
        for mount in self.mounts:
            root_norm = _norm_host(mount.host)
            if mount.is_file:
                if candidate_norm == root_norm:
                    return mount.container.as_posix()
                continue
            try:
                common = os.path.commonpath((candidate_norm, root_norm))
            except ValueError:
                continue
            if common != root_norm:
                continue
            relative = os.path.relpath(candidate, mount.host)
            if relative == ".":
                return mount.container.as_posix()
            parts = [part for part in relative.replace("\\", "/").split("/") if part]
            return mount.container.joinpath(*parts).as_posix()
        return None

    def container_to_host(self, path: str | os.PathLike[str]) -> Path | None:
        text = str(path).strip().replace("\\", "/")
        if not text:
            return None
        # Worker paths are commonly relative to the image WORKDIR (/app), for
        # example voicevox_runtime/models/vvms/0.vvm.
        candidate_text = text if text.startswith("/") else f"/app/{text}"
        candidate = PurePosixPath(posixpath.normpath(candidate_text))
        mounts = sorted(self.mounts, key=lambda item: len(item.container.parts), reverse=True)
        for mount in mounts:
            if mount.is_file:
                if candidate == mount.container:
                    return mount.host
                continue
            try:
                relative = candidate.relative_to(mount.container)
            except ValueError:
                continue
            return mount.host.joinpath(*relative.parts)
        return None

    def host_equivalent(self, path: str | os.PathLike[str]) -> Path | None:
        """Return a host path for either path namespace, or ``None`` if unmapped."""
        text = str(path).strip()
        if not text:
            return None
        mapped = self.container_to_host(text)
        if mapped is not None:
            return mapped
        candidate = _absolute_host_path(text, base=self.repo_root)
        return candidate if self.host_to_container(candidate) is not None else None

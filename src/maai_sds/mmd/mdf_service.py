import re
from pathlib import Path

from ..utils import to_bool


class MdfService:
    def __init__(
        self,
        bool_keys: set[str],
        default_values: dict[str, str],
        removed_legacy_keys: set[str],
        default_fst_content: str,
    ) -> None:
        self.bool_keys = bool_keys
        self.default_values = default_values
        self.removed_legacy_keys = removed_legacy_keys
        self.default_fst_content = default_fst_content

    @staticmethod
    def parse_mdf_key_values(lines: list[str]) -> dict[str, str]:
        parsed: dict[str, str] = {}
        pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if m:
                parsed[m.group(1)] = m.group(2)
        return parsed

    @staticmethod
    def set_mdf_value(lines: list[str], key: str, value: str) -> list[str]:
        pattern = re.compile(rf"^(\s*){re.escape(key)}\s*=.*$")
        replaced = False
        updated: list[str] = []
        for line in lines:
            match = pattern.match(line)
            if match and not line.lstrip().startswith("#"):
                updated.append(f"{match.group(1)}{key}={value}")
                replaced = True
            else:
                updated.append(line)
        if not replaced:
            if updated and updated[-1] != "":
                updated.append("")
            updated.append(f"{key}={value}")
        return updated

    @staticmethod
    def remove_mdf_key(lines: list[str], key: str) -> list[str]:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$")
        return [line for line in lines if not (pattern.match(line) and not line.lstrip().startswith("#"))]

    def is_default_value(self, key: str, value: str) -> bool:
        default = self.default_values.get(key)
        if default is None:
            return False
        if key in self.bool_keys:
            return to_bool(value) == to_bool(default)
        return value.strip() == default.strip()

    def read_mdf(self, mdf_path: Path) -> tuple[list[str], str]:
        try:
            text = mdf_path.read_text(encoding="utf-8")
            return text.splitlines(), "utf-8"
        except UnicodeDecodeError:
            text = mdf_path.read_text(encoding="cp932", errors="replace")
            return text.splitlines(), "cp932"

    def write_values(
        self,
        mdf_path: Path,
        field_keys: list[str],
        widget_to_value,
        loaded_keys: set[str],
        touched_keys: set[str],
    ) -> tuple[bool, set[str]]:
        lines, encoding = self.read_mdf(mdf_path)
        original_lines = list(lines)
        for legacy_key in self.removed_legacy_keys:
            lines = self.remove_mdf_key(lines, legacy_key)

        for key in field_keys:
            value = widget_to_value(key)
            should_write = (key in loaded_keys) or (key in touched_keys)
            if not should_write:
                continue
            if value == "":
                lines = self.remove_mdf_key(lines, key)
                continue
            if self.is_default_value(key, value):
                if key in loaded_keys:
                    lines = self.set_mdf_value(lines, key, value)
                else:
                    lines = self.remove_mdf_key(lines, key)
                continue
            lines = self.set_mdf_value(lines, key, value)

        # 値が変わらないなら書かない。MMDAgent-EX の起動ボタンは起動前に毎回保存するので、
        # 書くと中身が同じでも更新日時が変わる（Linux から書くと CRLF の MDF が LF にもなる）。
        # 同じ MDF を別の環境（clone や dev）と共用していても、触らずに済む。
        if lines == original_lines:
            return True, set(self.parse_mdf_key_values(lines).keys())

        # 読み込み時に cp932 で errors='replace' した U+FFFD は cp932 で書き戻すと
        # UnicodeEncodeError になるため、書き込み側も errors='replace' で保護する。
        mdf_path.write_text("\n".join(lines) + "\n", encoding=encoding, errors="replace")
        return True, set(self.parse_mdf_key_values(lines).keys())

    def ensure_fst_for_mdf(self, mdf_path: Path) -> tuple[bool, str]:
        fst_path = mdf_path.with_suffix(".fst")
        if fst_path.exists():
            return True, ""
        try:
            fst_path.write_text(self.default_fst_content, encoding="utf-8")
            return True, f"状態：同名FSTがないため生成（{fst_path.name}）"
        except Exception as ex:
            return False, str(ex)


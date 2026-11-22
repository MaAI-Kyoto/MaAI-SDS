"""LLM ストリーミング文分割のプレビュー用（GUI からのロジックを分離）。"""

from __future__ import annotations

from typing import Any

from .worker import _split_sentences_custom_delimiters


def build_preview_lines(
    text: str,
    *,
    enabled: bool,
    chunk_size: int,
    settings: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"enabled={enabled}",
        f"tokenizer=custom",
        f"chunk_size={chunk_size}",
        "",
    ]
    if not enabled:
        lines.append("[input_streaming_enabled=False] 完了時に全文1回送信")
        lines.append(text)
        return "\n".join(lines)

    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    assistant_text = ""
    pushed_complete_count = 0
    sent_text = ""
    lines.append("=== streaming ===")
    for idx, chunk in enumerate(chunks, start=1):
        assistant_text += chunk
        segs = _split_sentences_custom_delimiters(assistant_text, settings)
        complete_count = max(0, len(segs) - 1)
        if complete_count > pushed_complete_count:
            for j in range(pushed_complete_count, complete_count):
                seg = str(segs[j])
                if not seg:
                    continue
                lines.append(f"[{idx:03d}] push(confirmed): {seg}")
                sent_text += seg
            pushed_complete_count = complete_count
    final_text = assistant_text[len(sent_text) :] if assistant_text.startswith(sent_text) else assistant_text
    if final_text:
        lines.append(f"[end] push(final): {final_text}")
    lines.append("")
    lines.append("=== full text ===")
    lines.append(assistant_text)
    return "\n".join(lines)

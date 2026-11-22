from __future__ import annotations

import queue
from typing import Any

QUEUE_POLICY_DROP_OLDEST = "drop_oldest"
QUEUE_POLICY_DROP_NEWEST = "drop_newest"
QUEUE_POLICY_BLOCK = "block"


def queue_put(
    target_queue: queue.Queue[Any],
    item: Any,
    policy: str = QUEUE_POLICY_DROP_OLDEST,
    *,
    block_timeout_sec: float = 0.05,
) -> bool:
    if policy == QUEUE_POLICY_BLOCK:
        try:
            target_queue.put(item, timeout=max(0.0, float(block_timeout_sec)))
            return True
        except queue.Full:
            return False
    try:
        target_queue.put_nowait(item)
        return True
    except queue.Full:
        if policy == QUEUE_POLICY_DROP_NEWEST:
            return False
        if policy != QUEUE_POLICY_DROP_OLDEST:
            return False
        try:
            _ = target_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            target_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

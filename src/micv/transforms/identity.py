from __future__ import annotations

import hashlib
from typing import Any


def stable_int_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def record_seed_identity(record: Any) -> str:
    metadata = getattr(record, "metadata", {})
    if isinstance(metadata, dict):
        for key in ("md5", "group_id", "manifest_path"):
            value = metadata.get(key)
            if value not in {None, ""}:
                return f"{key}:{value}"
    return f"path:{getattr(record, 'path')}"
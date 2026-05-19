"""Persistent Cache-Augmented Generation store."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CAGCache:
    """Disk-backed cache for reused OSINT answers.

    Values are stored as strings because the OSINT agent already serializes its
    structured result to JSON. The cache honors a TTL and survives app restarts,
    which makes repeated-query behavior visible during the course demo.
    """

    _global_stores: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.ttl_seconds = int(self.config.get("ttl_seconds", 3600))
        persist_dir = Path(self.config.get("persist_dir", "data/cache"))
        filename = self.config.get("filename", "cag_cache.json")
        self.cache_file = persist_dir / filename
        persist_dir.mkdir(parents=True, exist_ok=True)

        store_key = str(self.cache_file.resolve())
        if store_key not in self._global_stores:
            self._global_stores[store_key] = self._load_store()
        self._store = self._global_stores[store_key]

    def _load_store(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_file.exists():
            return {}
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): value
            for key, value in data.items()
            if isinstance(value, dict) and "value" in value
        }

    def _flush(self) -> None:
        tmp = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        tmp.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cache_file)

    def _is_expired(self, created_at: Any) -> bool:
        if self.ttl_seconds <= 0:
            return False
        try:
            age = time.time() - float(created_at)
        except (TypeError, ValueError):
            return True
        return age > self.ttl_seconds

    def get(self, key: str) -> Optional[str]:
        """Return cached value if present and not expired."""
        record = self._store.get(str(key))
        if not record:
            return None
        if self._is_expired(record.get("created_at")):
            self._store.pop(str(key), None)
            self._flush()
            return None
        value = record.get("value")
        return str(value) if value is not None else None

    def set(self, key: str, value: str) -> None:
        """Store a value under the given key."""
        if not key:
            return
        self._store[str(key)] = {"value": str(value), "created_at": time.time()}
        self._flush()

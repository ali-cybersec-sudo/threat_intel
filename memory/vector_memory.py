"""Long‑term vector memory – placeholder until real implementation."""

from typing import Any, Dict, List


class VectorMemory:
    """Persistent memory backed by a vector store."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._store: Dict[str, str] = {}

    def store(self, key: str, value: str) -> None:
        """Persist a piece of knowledge."""
        self._store[key] = value

    def retrieve(self, query: str) -> List[str]:
        """Placeholder – returns empty list for now."""
        return []
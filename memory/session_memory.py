"""Short‑term session memory – placeholder until real implementation."""

from typing import Any, Dict, List


class SessionMemory:
    """In‑memory conversation history for a single session."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._history: List[str] = []

    def add(self, message: str) -> None:
        """Append a message to the session."""
        self._history.append(message)

    def get_recent(self, n: int = 5) -> List[str]:
        """Return the last *n* messages."""
        return self._history[-n:]
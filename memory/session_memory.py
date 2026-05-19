"""Short-term in-memory conversation history."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SessionMemory:
    """In-memory conversation history for a single session."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_turns: int = int(self.config.get("max_turns", 100))
        self._history: List[Any] = []

    def add(self, message: Any) -> None:
        """Append a message to the session."""
        self._history.append(message)
        if self.max_turns > 0 and len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns :]

    def get_recent(self, n: int = 5) -> List[Any]:
        """Return the last *n* messages."""
        return self._history[-n:]

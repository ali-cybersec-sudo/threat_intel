"""Web‑search tool – placeholder until real implementation."""

from typing import Any, Dict


class WebSearchTool:
    """Concrete implementation TBD – inherits from BaseAgent via composition."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        # place holder for future: DuckDuckGo, Tavily, etc.

    def search(self, query: str) -> str:
        """Placeholder – returns a mock result."""
        return f"[mock search result for: {query}]"
"""input_guard.py
==================

Security gate that validates and sanitises user input before it reaches
any agent.  Checks for prompt-injection patterns, excessively long
payloads, forbidden characters and known malicious patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Patterns that suggest prompt-injection or abuse ──────────────────────

_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:prior|above)", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"\{\{.*?\}\}"),  # template injection
    re.compile(r"\$\{.*?\}"),    # expression injection
]

# Characters that should never appear in a CTI query
_FORBIDDEN_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class InputGuard:
    """Validate and sanitise incoming user queries.

    Parameters
    ----------
    config : dict
        Guard configuration.  Recognised keys:

        * ``max_length`` – maximum allowed query length (default 5000).
        * ``block_injections`` – whether to reject injection patterns
          (default ``True``).
        * ``extra_blocked_patterns`` – optional list of additional regex
          strings to block.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_length: int = int(self.config.get("max_length", 5000))
        self.block_injections: bool = bool(self.config.get("block_injections", True))

        # Compile any extra patterns supplied via config
        self._extra_patterns: List[re.Pattern[str]] = []
        for pat_str in self.config.get("extra_blocked_patterns", []):
            try:
                self._extra_patterns.append(re.compile(pat_str, re.IGNORECASE))
            except re.error as exc:
                logger.warning("Skipping invalid extra pattern '%s': %s", pat_str, exc)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def validate(self, text: str) -> bool:
        """Return ``True`` if *text* passes all checks, ``False`` otherwise.

        When a check fails the method logs a warning with the reason.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("InputGuard: empty or non-string input rejected.")
            return False

        if len(text) > self.max_length:
            logger.warning(
                "InputGuard: input too long (%d chars, max %d).",
                len(text), self.max_length,
            )
            return False

        if _FORBIDDEN_CHARS.search(text):
            logger.warning("InputGuard: forbidden control characters detected.")
            return False

        if self.block_injections:
            for pattern in _INJECTION_PATTERNS + self._extra_patterns:
                if pattern.search(text):
                    logger.warning(
                        "InputGuard: potential injection detected (pattern: %s).",
                        pattern.pattern,
                    )
                    return False

        return True

    # Alias kept for backwards compatibility with BaseAgent._validate_input
    validate_input = validate

    def sanitise(self, text: str) -> str:
        """Return a cleaned version of *text*.

        * Strips leading / trailing whitespace.
        * Removes forbidden control characters.
        * Truncates to ``max_length``.
        """
        cleaned = _FORBIDDEN_CHARS.sub("", text).strip()
        return cleaned[: self.max_length]

    def __repr__(self) -> str:
        return f"InputGuard(max_length={self.max_length}, block_injections={self.block_injections})"

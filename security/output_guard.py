"""output_guard.py
==================

Security gate that sanitises agent output before it is returned to the
user.  Redacts sensitive data patterns (API keys, internal paths, emails)
and enforces output-length limits.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Patterns to redact from output ───────────────────────────────────────

_REDACTION_RULES: List[Dict[str, Any]] = [
    {
        "name": "api_key",
        "pattern": re.compile(
            r"(?:api[_-]?key|token|secret|password|bearer)\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
        "replacement": "[REDACTED_CREDENTIAL]",
    },
    {
        "name": "email",
        "pattern": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "replacement": "[REDACTED_EMAIL]",
    },
    {
        "name": "internal_path",
        "pattern": re.compile(r"(?:/home/\w+|C:\\Users\\\w+)[^\s]*"),
        "replacement": "[REDACTED_PATH]",
    },
    {
        "name": "private_ip",
        "pattern": re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3})\b"
        ),
        "replacement": "[REDACTED_INTERNAL_IP]",
    },
]


class OutputGuard:
    """Sanitise agent output before returning to the user.

    Parameters
    ----------
    config : dict | None
        Guard configuration.  Recognised keys:

        * ``max_output_length`` – truncate output beyond this limit
          (default 50 000 chars).
        * ``redact_credentials`` – enable credential redaction
          (default ``True``).
        * ``redact_pii`` – enable PII (email) redaction (default ``True``).
        * ``extra_redact_patterns`` – list of dicts with ``pattern`` and
          ``replacement`` keys for custom redaction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.max_output_length: int = int(self.config.get("max_output_length", 50_000))
        self.redact_credentials: bool = bool(self.config.get("redact_credentials", True))
        self.redact_pii: bool = bool(self.config.get("redact_pii", True))

        # Build the active rule set
        self._rules: List[Dict[str, Any]] = []
        for rule in _REDACTION_RULES:
            name = rule["name"]
            if name == "api_key" and not self.redact_credentials:
                continue
            if name == "email" and not self.redact_pii:
                continue
            self._rules.append(rule)

        # Extra rules from config
        for extra in self.config.get("extra_redact_patterns", []):
            try:
                compiled = re.compile(extra["pattern"], re.IGNORECASE)
                self._rules.append({
                    "name": "custom",
                    "pattern": compiled,
                    "replacement": extra.get("replacement", "[REDACTED]"),
                })
            except (re.error, KeyError) as exc:
                logger.warning("Skipping invalid extra redaction rule: %s", exc)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def sanitise(self, text: str) -> str:
        """Apply all redaction rules and length limits to *text*."""
        if not isinstance(text, str):
            return str(text)

        result = text
        for rule in self._rules:
            result = rule["pattern"].sub(rule["replacement"], result)

        if len(result) > self.max_output_length:
            logger.warning(
                "OutputGuard: output truncated from %d to %d chars.",
                len(result), self.max_output_length,
            )
            result = result[: self.max_output_length] + "\n\n[OUTPUT TRUNCATED]"

        return result

    def sanitise_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitise all string values in a dictionary."""
        sanitised: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitised[key] = self.sanitise(value)
            elif isinstance(value, dict):
                sanitised[key] = self.sanitise_dict(value)
            elif isinstance(value, list):
                sanitised[key] = [
                    self.sanitise(v) if isinstance(v, str)
                    else self.sanitise_dict(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                sanitised[key] = value
        return sanitised

    def __repr__(self) -> str:
        return (
            f"OutputGuard(max_output_length={self.max_output_length}, "
            f"rules={len(self._rules)})"
        )

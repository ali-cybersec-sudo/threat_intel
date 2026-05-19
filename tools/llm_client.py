"""Shared LLM client with provider fallback support."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import requests

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM provider is misconfigured or unreachable."""
    pass


class LLMClient:
    """Call configured LLM providers with automatic fallback.

    Default behavior: primary provider then optional fallback provider on
    transport errors, 429/5xx responses, or empty output.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.loader = ConfigLoader.instance()
        self.cfg = config or dict(self.loader.settings)
        self.default_provider = self.cfg.get("default_llm", "openrouter")
        self.llm_cfg = self.cfg.get("llm", {})
        self.fallback_provider = self.cfg.get("llm_fallback_provider", "groq")

    def validate_connection(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Validate that the LLM provider is configured and reachable.

        Performs a lightweight 1-token completion to confirm:
        1. The API key environment variable is set.
        2. The endpoint is reachable.
        3. The API key is valid (not 401/403).

        Returns a dict with status info on success.
        Raises LLMConfigurationError with actionable instructions on failure.
        """
        target = provider or self.default_provider
        env_var = f"{target.upper()}_API_KEY"
        api_key = os.getenv(env_var, "")

        # Step 1: Check env var exists
        if not api_key or api_key.startswith("your_") or api_key == "sk-or-...":
            raise LLMConfigurationError(
                f"\n{'=' * 60}\n"
                f"  LLM CONFIGURATION ERROR\n"
                f"{'=' * 60}\n"
                f"  {env_var} is not set or contains a placeholder value.\n\n"
                f"  To fix:\n"
                f"  1. Get an API key from https://openrouter.ai/keys\n"
                f"  2. Copy .env.example to .env\n"
                f"  3. Set {env_var}=sk-or-v1-your-key-here\n"
                f"  4. Restart the application\n"
                f"{'=' * 60}"
            )

        # Step 2: Check provider config exists
        try:
            cfg = self._provider_cfg(target)
        except RuntimeError as exc:
            raise LLMConfigurationError(
                f"LLM provider '{target}' not configured in settings.yaml: {exc}"
            ) from exc

        endpoint = cfg.get("endpoint", "")
        if not endpoint:
            raise LLMConfigurationError(
                f"No endpoint configured for provider '{target}' in settings.yaml"
            )

        # Step 3: Lightweight ping — 1-token completion
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.get("model", "openrouter/auto"),
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }

        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        except requests.ConnectionError as exc:
            raise LLMConfigurationError(
                f"Cannot reach LLM endpoint '{endpoint}': {exc}\n"
                f"Check your network connection and firewall settings."
            ) from exc
        except requests.Timeout:
            raise LLMConfigurationError(
                f"LLM endpoint '{endpoint}' timed out (10s).\n"
                f"The service may be temporarily unavailable."
            )

        if resp.status_code == 401:
            raise LLMConfigurationError(
                f"\n{'=' * 60}\n"
                f"  INVALID API KEY\n"
                f"{'=' * 60}\n"
                f"  {env_var} was rejected by {endpoint} (HTTP 401).\n\n"
                f"  To fix:\n"
                f"  1. Go to https://openrouter.ai/keys\n"
                f"  2. Generate a new API key\n"
                f"  3. Update {env_var} in your .env file\n"
                f"{'=' * 60}"
            )

        if resp.status_code == 402:
            raise LLMConfigurationError(
                f"\n{'=' * 60}\n"
                f"  INSUFFICIENT CREDITS\n"
                f"{'=' * 60}\n"
                f"  Your OpenRouter account has no remaining credits (HTTP 402).\n\n"
                f"  To fix:\n"
                f"  1. Go to https://openrouter.ai/credits\n"
                f"  2. Add credits to your account\n"
                f"  3. Restart the application\n"
                f"{'=' * 60}"
            )

        if resp.status_code == 403:
            raise LLMConfigurationError(
                f"API key forbidden for provider '{target}' (HTTP 403).\n"
                f"Check your API key permissions at https://openrouter.ai/keys"
            )

        if resp.status_code >= 500:
            logger.warning(
                "LLM provider '%s' returned %d during validation — "
                "service may be degraded but key is valid.",
                target, resp.status_code,
            )

        result = {
            "provider": target,
            "status": "ok",
            "status_code": resp.status_code,
            "endpoint": endpoint,
            "model": cfg.get("model"),
        }
        logger.info("LLM connection validated: provider=%s, status=%d", target, resp.status_code)
        return result

    def _api_key_for(self, provider: str) -> str:
        return self.loader.get_api_key(provider)

    def _provider_cfg(self, provider: str) -> Dict[str, Any]:
        cfg = self.llm_cfg.get(provider)
        if not cfg:
            raise RuntimeError(f"LLM config missing for provider '{provider}'")
        return dict(cfg)

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    return msg.get("content", "")
                txt = first.get("text")
                if txt:
                    return txt
        out = data.get("output") or data.get("result")
        if isinstance(out, str):
            return out
        if isinstance(out, list):
            parts = []
            for item in out:
                if isinstance(item, dict):
                    parts.append(item.get("content", str(item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return ""

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        cfg = self._provider_cfg(provider)
        key = self._api_key_for(provider)
        endpoint = cfg.get("endpoint")
        if not endpoint:
            raise RuntimeError(f"Endpoint missing for provider '{provider}'")

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": cfg.get("model"),
            "temperature": temperature if temperature is not None else cfg.get("temperature", 0.2),
            "max_tokens": max_tokens if max_tokens is not None else cfg.get("max_tokens", 1024),
        }

        if "chat" in endpoint.lower() or cfg.get("mode", "chat") == "chat":
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout or cfg.get("timeout", 30))
        status = resp.status_code
        if status >= 400:
            raise RuntimeError(f"HTTP {status}: {resp.text[:800]}")

        data = resp.json()
        text = self._extract_text(data)
        meta = {
            "provider": provider,
            "status_code": status,
            "model": cfg.get("model"),
        }
        return text, meta

    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        allow_fallback: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        primary = provider or self.default_provider
        fallback = self.fallback_provider if allow_fallback else None

        fallback_triggered = False
        primary_error: Optional[str] = None

        try:
            text, meta = self._call_provider(
                primary,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if text and text.strip():
                meta["fallback_triggered"] = False
                return text, meta
            raise RuntimeError("Empty model output")
        except Exception as exc:
            primary_error = str(exc)
            logger.warning("Primary LLM provider '%s' failed: %s", primary, exc)

        if fallback and fallback != primary:
            fallback_triggered = True
            text, meta = self._call_provider(
                fallback,
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if not text or not text.strip():
                raise RuntimeError("Fallback provider returned empty output")
            meta["fallback_triggered"] = True
            meta["primary_error"] = primary_error
            return text, meta

        raise RuntimeError(primary_error or "LLM generation failed")

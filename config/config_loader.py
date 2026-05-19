"""config_loader.py
===================

Utility module that loads environment variables and YAML configuration files
for the CTI multi‑agent system.

The class follows the **Singleton** pattern – only one instance exists per
process.  It reads:

* ``.env`` (via ``python‑dotenv``) – holds all secret keys.
* ``settings.yaml`` – system‑wide configuration (LLM providers, agents,
  memory, tools, security, UI, etc.).
* ``prompts.yaml`` – prompt templates for each agent.

The loader exposes typed accessor methods that raise clear ``RuntimeError``
exceptions when required configuration is missing.  All methods are fully
type‑annotated and documented using the Google docstring style.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Singleton configuration loader.

    The loader is instantiated lazily – the first call to ``instance()``
    creates the object and reads all configuration files.  Subsequent calls
    return the same object, ensuring a single source of truth throughout the
    application.

    Example
    -------
    >>> from config.config_loader import ConfigLoader
    >>> cfg = ConfigLoader.instance()
    >>> api_key = cfg.get_api_key("openai")
    >>> llm_cfg = cfg.get_llm_config("openai")
    """

    _instance: Optional["ConfigLoader"] = None

    @classmethod
    def instance(cls) -> "ConfigLoader":
        """Return the global ``ConfigLoader`` instance.

        The method creates the instance on first use.  It is thread‑safe for the
        typical single‑threaded usage pattern of this project.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---------------------------------------------------------------------
    # Construction – performed once by ``instance()``
    # ---------------------------------------------------------------------
    def __init__(self) -> None:
        # Load environment variables – ``.env`` is expected at the project root
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path)

        # Load YAML configuration files
        config_dir = Path(__file__).resolve().parent
        self._settings: Dict[str, Any] = self._load_yaml(config_dir / "settings.yaml")
        self._prompts: Dict[str, Any] = self._load_yaml(config_dir / "prompts.yaml")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Read a YAML file and return its contents.

        Parameters
        ----------
        path: Path
            Absolute path to the YAML file.

        Returns
        -------
        dict
            Parsed YAML content.

        Raises
        ------
        RuntimeError
            If the file cannot be read or is not valid YAML.
        """
        if not path.is_file():
            raise RuntimeError(f"Configuration file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Failed to parse YAML file {path}: {exc}") from exc

    # ---------------------------------------------------------------------
    # Public accessors – API keys
    # ---------------------------------------------------------------------
    def get_api_key(self, service_name: str) -> str:
        """Return the secret API key for ``service_name``.

        The method maps a logical service name (e.g. ``"openai"``) to the
        corresponding environment variable.  A ``RuntimeError`` is raised if the
        variable is missing or empty.
        """
        env_var = f"{service_name.upper()}_API_KEY"
        value = os.getenv(env_var)
        if not value:
            raise RuntimeError(
                f"API key for service '{service_name}' not found. Set the '{env_var}' environment variable."
            )
        return value

    # ---------------------------------------------------------------------
    # Public accessors – LLM configuration
    # ---------------------------------------------------------------------
    def get_llm_config(self, provider: Optional[str] = None) -> Mapping[str, Any]:
        """Return configuration for a given LLM provider.

        Parameters
        ----------
        provider: str | None
            Provider name (``"openai"``, ``"groq"``, ``"gemini"`` or ``"claude"``).
            If ``None`` the default provider from ``settings.yaml`` is used.
        """
        default = self._settings.get("default_llm", "openai")
        provider_key = provider or default
        llm_cfg = self._settings.get("llm", {}).get(provider_key)
        if llm_cfg is None:
            raise RuntimeError(f"LLM configuration for provider '{provider_key}' not found.")
        return llm_cfg

    # ---------------------------------------------------------------------
    # Public accessors – Agent configuration
    # ---------------------------------------------------------------------
    def get_agent_config(self, agent_name: str) -> Mapping[str, Any]:
        """Return the configuration dictionary for ``agent_name``.

        The ``agents`` section of ``settings.yaml`` must contain a mapping for each
        known agent.  Missing configuration raises ``RuntimeError``.
        """
        agents_cfg = self._settings.get("agents", {})
        cfg = agents_cfg.get(agent_name)
        if cfg is None:
            raise RuntimeError(f"Configuration for agent '{agent_name}' not found in settings.yaml.")
        return cfg

    # ---------------------------------------------------------------------
    # Public accessors – Tool configuration
    # ---------------------------------------------------------------------
    def get_tool_config(self, tool_name: str) -> Mapping[str, Any]:
        """Return configuration for a tool (web_search, rag, cache, …)."""
        tools_cfg = self._settings.get("tools", {})
        cfg = tools_cfg.get(tool_name)
        if cfg is None:
            raise RuntimeError(f"Configuration for tool '{tool_name}' not found in settings.yaml.")
        return cfg

    # ---------------------------------------------------------------------
    # Public accessors – Memory configuration
    # ---------------------------------------------------------------------
    def get_memory_config(self, memory_type: str) -> Mapping[str, Any]:
        """Return configuration for a memory type (session, vector)."""
        mem_cfg = self._settings.get("memory", {})
        cfg = mem_cfg.get(memory_type)
        if cfg is None:
            raise RuntimeError(f"Memory configuration for '{memory_type}' not found in settings.yaml.")
        return cfg

    # ---------------------------------------------------------------------
    # Public accessors – Prompt templates
    # ---------------------------------------------------------------------
    def get_prompt_template(self, agent_name: str, template_key: str) -> str:
        """Return a specific prompt template for an agent.

        Parameters
        ----------
        agent_name: str
            Key under which the agent's prompts are stored in ``prompts.yaml``.
        template_key: str
            Specific template name (e.g. ``"system_prompt"`` or ``"search_template"``).
        """
        agent_prompts = self._prompts.get(agent_name, {})
        tmpl = agent_prompts.get(template_key)
        if tmpl is None:
            raise RuntimeError(
                f"Prompt template '{template_key}' for agent '{agent_name}' not found in prompts.yaml."
            )
        return tmpl

    # ---------------------------------------------------------------------
    # Helper – internal dict access (used by other components)
    # ---------------------------------------------------------------------
    @property
    def settings(self) -> Mapping[str, Any]:
        """Expose the raw ``settings.yaml`` content (read‑only)."""
        return self._settings

    @property
    def prompts(self) -> Mapping[str, Any]:
        """Expose the raw ``prompts.yaml`` content (read‑only)."""
        return self._prompts

    # ---------------------------------------------------------------------
    # Debug / representation helpers
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"ConfigLoader(settings_path={Path(__file__).resolve().parent / 'settings.yaml'})"

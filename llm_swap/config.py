"""Configuration models and YAML loader for llm_swap.

This module defines Pydantic v2 models that represent the full configuration
schema for llm_swap, including server settings, logging, health checks,
provider definitions, and routing rules. It also provides a YAML loader
that reads a config file, performs environment-variable substitution,
validates the contents, and returns a typed ``Config`` object.

Typical usage::

    from llm_swap.config import load_config

    cfg = load_config("config.yaml")
    print(cfg.server.port)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    host: str = Field(default="127.0.0.1", description="Interface to bind on.")
    port: int = Field(default=8000, ge=1, le=65535, description="TCP port.")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", description="Uvicorn log level."
    )
    request_timeout: int = Field(
        default=120, ge=1, description="Seconds before abandoning an upstream request."
    )


# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Structured request/response logging configuration."""

    enabled: bool = Field(default=True, description="Enable structured logging.")
    log_file: Optional[str] = Field(
        default=None, description="Path to log file; None logs to stdout only."
    )
    log_format: Literal["json", "text"] = Field(
        default="json", description="Output format for log records."
    )
    log_request_body: bool = Field(
        default=False, description="Include full prompt text in log records."
    )
    log_response_body: bool = Field(
        default=False, description="Include full completion text in log records."
    )


# ---------------------------------------------------------------------------
# Health-check settings
# ---------------------------------------------------------------------------


class HealthCheckConfig(BaseModel):
    """Provider health-check configuration."""

    enabled: bool = Field(default=True, description="Enable background health checks.")
    interval_seconds: int = Field(
        default=30, ge=1, description="Seconds between health-check polls."
    )
    timeout_seconds: int = Field(
        default=5, ge=1, description="Timeout for a single health-check request."
    )
    unhealthy_threshold: int = Field(
        default=2,
        ge=1,
        description="Consecutive failures before marking a provider degraded.",
    )
    healthy_threshold: int = Field(
        default=1,
        ge=1,
        description="Consecutive successes before restoring a degraded provider.",
    )


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

ProviderType = Literal["openai", "anthropic", "mistral", "ollama"]


class ProviderConfig(BaseModel):
    """Configuration for a single backend LLM provider."""

    name: str = Field(description="Unique name for this provider, used in routing rules.")
    type: ProviderType = Field(
        description="Provider type. One of: openai, anthropic, mistral, ollama."
    )
    api_key: Optional[str] = Field(
        default=None, description="API key; may be None for local providers like Ollama."
    )
    base_url: str = Field(description="Base URL for the provider's API.")
    timeout: int = Field(default=60, ge=1, description="Per-request timeout in seconds.")
    max_retries: int = Field(
        default=2, ge=0, description="Maximum number of retry attempts on transient errors."
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers forwarded to the provider.",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Ensure the provider name is non-empty and contains no whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("Provider name must not be empty.")
        if re.search(r"\s", v):
            raise ValueError("Provider name must not contain whitespace.")
        return v

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, v: str) -> str:
        """Ensure the base URL starts with http:// or https://."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {v!r}")
        # Strip trailing slash for uniformity
        return v.rstrip("/")


# ---------------------------------------------------------------------------
# Routing rule sub-models
# ---------------------------------------------------------------------------


class BackendEntry(BaseModel):
    """A single backend entry in an alias or model-route definition."""

    provider: str = Field(description="Name of the provider (must match a defined provider).")
    model: str = Field(description="Model name to pass to the provider.")
    priority: int = Field(
        default=1,
        ge=1,
        description="Priority level; lower numbers are tried first.",
    )


class AliasRoute(BaseModel):
    """Maps a logical model alias to an ordered list of backend entries."""

    alias: str = Field(description="Logical alias name clients use as the model parameter.")
    backends: List[BackendEntry] = Field(
        min_length=1, description="Ordered list of backend entries."
    )

    @field_validator("alias")
    @classmethod
    def alias_must_not_be_empty(cls, v: str) -> str:
        """Ensure the alias is non-empty."""
        v = v.strip()
        if not v:
            raise ValueError("Alias must not be empty.")
        return v


class ModelRoute(BaseModel):
    """Maps a specific model name to an ordered list of backend entries."""

    model: str = Field(description="Exact model name to match in incoming requests.")
    backends: List[BackendEntry] = Field(
        min_length=1, description="Ordered list of backend entries."
    )

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, v: str) -> str:
        """Ensure the model name is non-empty."""
        v = v.strip()
        if not v:
            raise ValueError("Model name must not be empty.")
        return v


RoutingStrategy = Literal["priority", "round_robin"]


class RoutingConfig(BaseModel):
    """Full routing configuration including strategy, aliases, and model routes."""

    default_provider: str = Field(
        description="Provider name used when no alias or model route matches."
    )
    strategy: RoutingStrategy = Field(
        default="priority",
        description="How to select among providers at the same priority level.",
    )
    aliases: List[AliasRoute] = Field(
        default_factory=list,
        description="Alias-to-backend mappings.",
    )
    model_routes: List[ModelRoute] = Field(
        default_factory=list,
        description="Exact model-name-to-backend mappings.",
    )

    @model_validator(mode="after")
    def aliases_must_be_unique(self) -> "RoutingConfig":
        """Validate that alias names are unique."""
        seen: set[str] = set()
        for ar in self.aliases:
            if ar.alias in seen:
                raise ValueError(f"Duplicate alias name: {ar.alias!r}")
            seen.add(ar.alias)
        return self

    @model_validator(mode="after")
    def model_routes_must_be_unique(self) -> "RoutingConfig":
        """Validate that model route names are unique."""
        seen: set[str] = set()
        for mr in self.model_routes:
            if mr.model in seen:
                raise ValueError(f"Duplicate model route: {mr.model!r}")
            seen.add(mr.model)
        return self


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Root configuration object for llm_swap.

    All fields have sensible defaults so that a minimal config.yaml only needs
    to specify ``providers`` and ``routing``.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    providers: List[ProviderConfig] = Field(
        min_length=1, description="List of backend provider definitions."
    )
    routing: RoutingConfig = Field(description="Routing rules and aliases.")

    @model_validator(mode="after")
    def validate_provider_references(self) -> "Config":
        """Ensure that all provider names referenced in routing rules exist."""
        defined_names = {p.name for p in self.providers}

        # Check default provider
        if self.routing.default_provider not in defined_names:
            raise ValueError(
                f"routing.default_provider {self.routing.default_provider!r} "
                f"is not a defined provider. Defined: {sorted(defined_names)}"
            )

        # Check alias backends
        for ar in self.routing.aliases:
            for be in ar.backends:
                if be.provider not in defined_names:
                    raise ValueError(
                        f"Alias {ar.alias!r} references unknown provider "
                        f"{be.provider!r}. Defined: {sorted(defined_names)}"
                    )

        # Check model route backends
        for mr in self.routing.model_routes:
            for be in mr.backends:
                if be.provider not in defined_names:
                    raise ValueError(
                        f"Model route {mr.model!r} references unknown provider "
                        f"{be.provider!r}. Defined: {sorted(defined_names)}"
                    )

        return self

    @model_validator(mode="after")
    def provider_names_must_be_unique(self) -> "Config":
        """Ensure that provider names are unique."""
        seen: set[str] = set()
        for p in self.providers:
            if p.name in seen:
                raise ValueError(f"Duplicate provider name: {p.name!r}")
            seen.add(p.name)
        return self

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Return the :class:`ProviderConfig` with the given name, or ``None``.

        Args:
            name: The unique provider name to look up.

        Returns:
            The matching :class:`ProviderConfig` or ``None`` if not found.
        """
        for p in self.providers:
            if p.name == name:
                return p
        return None


# ---------------------------------------------------------------------------
# Environment-variable substitution
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively replace ``${VAR_NAME}`` placeholders with environment values.

    Unset variables are replaced with an empty string and a warning is printed
    to stderr so that config parsing still succeeds; callers can detect missing
    keys via Pydantic validation (e.g. an empty api_key).

    Args:
        obj: A Python object produced by ``yaml.safe_load``.

    Returns:
        The same structure with all string leaf values interpolated.
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    if isinstance(obj, str):
        def _replace(match: re.Match) -> str:  # type: ignore[type-arg]
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                import sys
                print(
                    f"[llm_swap] WARNING: environment variable ${{{var_name}}} "
                    "is not set; substituting empty string.",
                    file=sys.stderr,
                )
                return ""
            return value

        return _ENV_VAR_RE.sub(_replace, obj)
    return obj


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when the configuration file cannot be loaded or validated."""


def load_config(path: str | Path) -> Config:
    """Load, validate, and return a :class:`Config` from a YAML file.

    The function performs the following steps:

    1. Read and parse the YAML file.
    2. Substitute ``${ENV_VAR}`` placeholders from the environment.
    3. Validate the data against the Pydantic :class:`Config` model.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A fully validated :class:`Config` object.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or fails
            Pydantic validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file {config_path}: {exc}") from exc

    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if raw_data is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration file must contain a YAML mapping at the top level, "
            f"got {type(raw_data).__name__}: {config_path}"
        )

    interpolated = _substitute_env_vars(raw_data)

    try:
        return Config.model_validate(interpolated)
    except Exception as exc:  # pydantic.ValidationError is the expected type
        raise ConfigError(f"Configuration validation failed for {config_path}: {exc}") from exc


def load_config_from_dict(data: Dict[str, Any]) -> Config:
    """Validate and return a :class:`Config` from a plain Python dictionary.

    Useful in tests and programmatic construction where no file is involved.

    Args:
        data: A dictionary matching the configuration schema.

    Returns:
        A fully validated :class:`Config` object.

    Raises:
        ConfigError: If the dictionary fails Pydantic validation.
    """
    try:
        return Config.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc

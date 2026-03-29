"""Unit tests for llm_swap.config — configuration loading, validation, and error cases.

Tests use in-memory dictionaries (via ``load_config_from_dict``) and temporary
YAML files (via ``tmp_path``) to exercise all code paths without touching the
real filesystem or requiring live provider credentials.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

from llm_swap.config import (
    AliasRoute,
    BackendEntry,
    Config,
    ConfigError,
    HealthCheckConfig,
    LoggingConfig,
    ModelRoute,
    ProviderConfig,
    RoutingConfig,
    ServerConfig,
    _substitute_env_vars,
    load_config,
    load_config_from_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dict(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal valid configuration dictionary."""
    cfg: Dict[str, Any] = {
        "providers": [
            {
                "name": "openai",
                "type": "openai",
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
            }
        ],
        "routing": {
            "default_provider": "openai",
        },
    }
    cfg.update(overrides)
    return cfg


def _write_yaml(tmp_path: Path, content: str, filename: str = "config.yaml") -> Path:
    """Write *content* to a temporary YAML file and return the path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# ServerConfig
# ---------------------------------------------------------------------------


class TestServerConfig:
    """Tests for ServerConfig defaults and validation."""

    def test_defaults(self) -> None:
        sc = ServerConfig()
        assert sc.host == "127.0.0.1"
        assert sc.port == 8000
        assert sc.log_level == "info"
        assert sc.request_timeout == 120

    def test_custom_values(self) -> None:
        sc = ServerConfig(host="0.0.0.0", port=9090, log_level="debug", request_timeout=30)
        assert sc.host == "0.0.0.0"
        assert sc.port == 9090
        assert sc.log_level == "debug"
        assert sc.request_timeout == 30

    def test_invalid_port_too_low(self) -> None:
        with pytest.raises(Exception):
            ServerConfig(port=0)

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(Exception):
            ServerConfig(port=99999)

    def test_invalid_log_level(self) -> None:
        with pytest.raises(Exception):
            ServerConfig(log_level="verbose")  # type: ignore[arg-type]

    def test_invalid_request_timeout(self) -> None:
        with pytest.raises(Exception):
            ServerConfig(request_timeout=0)


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    """Tests for LoggingConfig defaults and validation."""

    def test_defaults(self) -> None:
        lc = LoggingConfig()
        assert lc.enabled is True
        assert lc.log_file is None
        assert lc.log_format == "json"
        assert lc.log_request_body is False
        assert lc.log_response_body is False

    def test_custom_values(self) -> None:
        lc = LoggingConfig(
            enabled=False,
            log_file="/tmp/out.log",
            log_format="text",
            log_request_body=True,
            log_response_body=True,
        )
        assert lc.enabled is False
        assert lc.log_file == "/tmp/out.log"
        assert lc.log_format == "text"
        assert lc.log_request_body is True
        assert lc.log_response_body is True

    def test_invalid_log_format(self) -> None:
        with pytest.raises(Exception):
            LoggingConfig(log_format="xml")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HealthCheckConfig
# ---------------------------------------------------------------------------


class TestHealthCheckConfig:
    """Tests for HealthCheckConfig defaults and validation."""

    def test_defaults(self) -> None:
        hc = HealthCheckConfig()
        assert hc.enabled is True
        assert hc.interval_seconds == 30
        assert hc.timeout_seconds == 5
        assert hc.unhealthy_threshold == 2
        assert hc.healthy_threshold == 1

    def test_interval_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            HealthCheckConfig(interval_seconds=0)

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            HealthCheckConfig(timeout_seconds=0)

    def test_unhealthy_threshold_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            HealthCheckConfig(unhealthy_threshold=0)


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    """Tests for ProviderConfig field validation."""

    def test_valid_openai_provider(self) -> None:
        p = ProviderConfig(
            name="openai",
            type="openai",
            api_key="sk-abc",
            base_url="https://api.openai.com/v1",
        )
        assert p.name == "openai"
        assert p.type == "openai"
        assert p.api_key == "sk-abc"
        # Trailing slash should be stripped
        assert not p.base_url.endswith("/")

    def test_base_url_trailing_slash_stripped(self) -> None:
        p = ProviderConfig(
            name="openai",
            type="openai",
            base_url="https://api.openai.com/v1/",
        )
        assert p.base_url == "https://api.openai.com/v1"

    def test_ollama_no_api_key(self) -> None:
        p = ProviderConfig(
            name="ollama",
            type="ollama",
            api_key=None,
            base_url="http://localhost:11434",
        )
        assert p.api_key is None

    def test_name_empty_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(
                name="  ",
                type="openai",
                base_url="https://api.openai.com/v1",
            )

    def test_name_with_whitespace_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(
                name="open ai",
                type="openai",
                base_url="https://api.openai.com/v1",
            )

    def test_base_url_invalid_scheme_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(
                name="bad",
                type="openai",
                base_url="ftp://example.com",
            )

    def test_invalid_provider_type_raises(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(
                name="bad",
                type="unknown_provider",  # type: ignore[arg-type]
                base_url="https://example.com",
            )

    def test_default_headers_empty(self) -> None:
        p = ProviderConfig(
            name="openai", type="openai", base_url="https://api.openai.com/v1"
        )
        assert p.headers == {}

    def test_custom_headers(self) -> None:
        p = ProviderConfig(
            name="openai",
            type="openai",
            base_url="https://api.openai.com/v1",
            headers={"X-Custom": "value"},
        )
        assert p.headers == {"X-Custom": "value"}

    def test_max_retries_cannot_be_negative(self) -> None:
        with pytest.raises(Exception):
            ProviderConfig(
                name="openai",
                type="openai",
                base_url="https://api.openai.com/v1",
                max_retries=-1,
            )


# ---------------------------------------------------------------------------
# BackendEntry
# ---------------------------------------------------------------------------


class TestBackendEntry:
    """Tests for BackendEntry validation."""

    def test_valid_entry(self) -> None:
        be = BackendEntry(provider="openai", model="gpt-4o", priority=1)
        assert be.provider == "openai"
        assert be.model == "gpt-4o"
        assert be.priority == 1

    def test_default_priority(self) -> None:
        be = BackendEntry(provider="openai", model="gpt-4o")
        assert be.priority == 1

    def test_priority_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            BackendEntry(provider="openai", model="gpt-4o", priority=0)


# ---------------------------------------------------------------------------
# AliasRoute
# ---------------------------------------------------------------------------


class TestAliasRoute:
    """Tests for AliasRoute validation."""

    def test_valid_alias(self) -> None:
        ar = AliasRoute(
            alias="fast",
            backends=[BackendEntry(provider="openai", model="gpt-4o-mini", priority=1)],
        )
        assert ar.alias == "fast"
        assert len(ar.backends) == 1

    def test_empty_alias_raises(self) -> None:
        with pytest.raises(Exception):
            AliasRoute(
                alias="",
                backends=[BackendEntry(provider="openai", model="gpt-4o", priority=1)],
            )

    def test_empty_backends_raises(self) -> None:
        with pytest.raises(Exception):
            AliasRoute(alias="fast", backends=[])


# ---------------------------------------------------------------------------
# RoutingConfig
# ---------------------------------------------------------------------------


class TestRoutingConfig:
    """Tests for RoutingConfig validation."""

    def test_minimal_routing(self) -> None:
        rc = RoutingConfig(default_provider="openai")
        assert rc.default_provider == "openai"
        assert rc.strategy == "priority"
        assert rc.aliases == []
        assert rc.model_routes == []

    def test_round_robin_strategy(self) -> None:
        rc = RoutingConfig(default_provider="openai", strategy="round_robin")
        assert rc.strategy == "round_robin"

    def test_invalid_strategy(self) -> None:
        with pytest.raises(Exception):
            RoutingConfig(default_provider="openai", strategy="random")  # type: ignore[arg-type]

    def test_duplicate_aliases_raise(self) -> None:
        backends = [BackendEntry(provider="openai", model="gpt-4o", priority=1)]
        with pytest.raises(Exception, match="Duplicate alias"):
            RoutingConfig(
                default_provider="openai",
                aliases=[
                    AliasRoute(alias="fast", backends=backends),
                    AliasRoute(alias="fast", backends=backends),
                ],
            )

    def test_duplicate_model_routes_raise(self) -> None:
        backends = [BackendEntry(provider="openai", model="gpt-4o", priority=1)]
        with pytest.raises(Exception, match="Duplicate model route"):
            RoutingConfig(
                default_provider="openai",
                model_routes=[
                    ModelRoute(model="gpt-4o", backends=backends),
                    ModelRoute(model="gpt-4o", backends=backends),
                ],
            )


# ---------------------------------------------------------------------------
# Config (top-level)
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the top-level Config model."""

    def test_minimal_valid_config(self) -> None:
        cfg = load_config_from_dict(_minimal_dict())
        assert isinstance(cfg, Config)
        assert len(cfg.providers) == 1
        assert cfg.routing.default_provider == "openai"

    def test_default_sub_configs_created(self) -> None:
        cfg = load_config_from_dict(_minimal_dict())
        assert isinstance(cfg.server, ServerConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.health_check, HealthCheckConfig)

    def test_duplicate_provider_names_raise(self) -> None:
        data = _minimal_dict()
        data["providers"] = [
            {"name": "openai", "type": "openai", "base_url": "https://api.openai.com/v1"},
            {"name": "openai", "type": "mistral", "base_url": "https://api.mistral.ai/v1"},
        ]
        with pytest.raises(ConfigError, match="Duplicate provider"):
            load_config_from_dict(data)

    def test_unknown_default_provider_raises(self) -> None:
        data = _minimal_dict()
        data["routing"]["default_provider"] = "nonexistent"
        with pytest.raises(ConfigError, match="default_provider"):
            load_config_from_dict(data)

    def test_alias_references_unknown_provider(self) -> None:
        data = _minimal_dict()
        data["routing"]["aliases"] = [
            {
                "alias": "fast",
                "backends": [
                    {"provider": "ghost_provider", "model": "gpt-4o", "priority": 1}
                ],
            }
        ]
        with pytest.raises(ConfigError, match="unknown provider"):
            load_config_from_dict(data)

    def test_model_route_references_unknown_provider(self) -> None:
        data = _minimal_dict()
        data["routing"]["model_routes"] = [
            {
                "model": "gpt-4o",
                "backends": [
                    {"provider": "no_such", "model": "gpt-4o", "priority": 1}
                ],
            }
        ]
        with pytest.raises(ConfigError, match="unknown provider"):
            load_config_from_dict(data)

    def test_empty_providers_list_raises(self) -> None:
        data = _minimal_dict()
        data["providers"] = []
        with pytest.raises(ConfigError):
            load_config_from_dict(data)

    def test_get_provider_found(self) -> None:
        cfg = load_config_from_dict(_minimal_dict())
        p = cfg.get_provider("openai")
        assert p is not None
        assert p.name == "openai"

    def test_get_provider_not_found(self) -> None:
        cfg = load_config_from_dict(_minimal_dict())
        p = cfg.get_provider("missing")
        assert p is None

    def test_full_config_round_trip(self) -> None:
        """A fully specified config dict should parse and round-trip without errors."""
        data: Dict[str, Any] = {
            "server": {"host": "0.0.0.0", "port": 9000, "log_level": "debug"},
            "logging": {"enabled": True, "log_format": "text"},
            "health_check": {"enabled": False},
            "providers": [
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                    "timeout": 60,
                    "max_retries": 3,
                },
                {
                    "name": "ollama",
                    "type": "ollama",
                    "base_url": "http://localhost:11434",
                    "timeout": 120,
                    "max_retries": 1,
                },
            ],
            "routing": {
                "default_provider": "openai",
                "strategy": "round_robin",
                "aliases": [
                    {
                        "alias": "local",
                        "backends": [
                            {"provider": "ollama", "model": "llama3", "priority": 1}
                        ],
                    }
                ],
                "model_routes": [
                    {
                        "model": "gpt-4o",
                        "backends": [
                            {"provider": "openai", "model": "gpt-4o", "priority": 1}
                        ],
                    }
                ],
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.server.port == 9000
        assert cfg.server.log_level == "debug"
        assert cfg.logging.log_format == "text"
        assert cfg.health_check.enabled is False
        assert len(cfg.providers) == 2
        assert cfg.routing.strategy == "round_robin"
        assert len(cfg.routing.aliases) == 1
        assert cfg.routing.aliases[0].alias == "local"
        assert len(cfg.routing.model_routes) == 1


# ---------------------------------------------------------------------------
# Environment-variable substitution
# ---------------------------------------------------------------------------


class TestSubstituteEnvVars:
    """Tests for _substitute_env_vars helper."""

    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret")
        result = _substitute_env_vars("value is ${MY_KEY}")
        assert result == "value is secret"

    def test_unset_variable_becomes_empty_string(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.delenv("UNSET_VAR", raising=False)
        result = _substitute_env_vars("${UNSET_VAR}")
        assert result == ""
        captured = capsys.readouterr()
        assert "UNSET_VAR" in captured.err

    def test_nested_dict_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_KEY", "abc123")
        data = {"provider": {"api_key": "${API_KEY}"}}
        result = _substitute_env_vars(data)
        assert result == {"provider": {"api_key": "abc123"}}

    def test_list_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAL", "x")
        result = _substitute_env_vars(["${VAL}", "plain"])
        assert result == ["x", "plain"]

    def test_non_string_values_pass_through(self) -> None:
        assert _substitute_env_vars(42) == 42
        assert _substitute_env_vars(3.14) == 3.14
        assert _substitute_env_vars(True) is True
        assert _substitute_env_vars(None) is None

    def test_multiple_substitutions_in_one_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        result = _substitute_env_vars("http://${HOST}:${PORT}/api")
        assert result == "http://localhost:8080/api"

    def test_string_without_placeholders(self) -> None:
        result = _substitute_env_vars("no placeholders here")
        assert result == "no placeholders here"


# ---------------------------------------------------------------------------
# load_config (file-based)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the file-based load_config function."""

    def test_load_valid_yaml_file(self, tmp_path: Path) -> None:
        yaml_content = """
            providers:
              - name: openai
                type: openai
                api_key: sk-test
                base_url: https://api.openai.com/v1
            routing:
              default_provider: openai
        """
        p = _write_yaml(tmp_path, yaml_content)
        cfg = load_config(p)
        assert cfg.providers[0].name == "openai"

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_yaml_file_raises_config_error(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, "")
        with pytest.raises(ConfigError, match="empty"):
            load_config(p)

    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, ": invalid: yaml: [")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(p)

    def test_yaml_not_a_mapping_raises_config_error(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(p)

    def test_env_var_substitution_in_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        yaml_content = """
            providers:
              - name: openai
                type: openai
                api_key: "${TEST_API_KEY}"
                base_url: https://api.openai.com/v1
            routing:
              default_provider: openai
        """
        p = _write_yaml(tmp_path, yaml_content)
        cfg = load_config(p)
        assert cfg.providers[0].api_key == "sk-from-env"

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        yaml_content = """
            providers:
              - name: openai
                type: openai
                api_key: sk-test
                base_url: https://api.openai.com/v1
            routing:
              default_provider: openai
        """
        p = _write_yaml(tmp_path, yaml_content)
        cfg = load_config(Path(p))  # explicitly pass a Path
        assert cfg is not None

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        yaml_content = """
            providers:
              - name: openai
                type: openai
                api_key: sk-test
                base_url: https://api.openai.com/v1
            routing:
              default_provider: openai
        """
        p = _write_yaml(tmp_path, yaml_content)
        cfg = load_config(str(p))  # explicitly pass a str
        assert cfg is not None

    def test_validation_error_raises_config_error(self, tmp_path: Path) -> None:
        yaml_content = """
            providers:
              - name: openai
                type: openai
                api_key: sk-test
                base_url: https://api.openai.com/v1
            routing:
              default_provider: nobody_home
        """
        p = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigError):
            load_config(p)

    def test_path_is_directory_raises_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a file"):
            load_config(tmp_path)  # tmp_path is a directory


# ---------------------------------------------------------------------------
# load_config_from_dict edge cases
# ---------------------------------------------------------------------------


class TestLoadConfigFromDict:
    """Tests for the in-memory load_config_from_dict helper."""

    def test_valid_minimal(self) -> None:
        cfg = load_config_from_dict(_minimal_dict())
        assert isinstance(cfg, Config)

    def test_invalid_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            load_config_from_dict({})  # missing required fields

    def test_server_overrides_propagate(self) -> None:
        data = _minimal_dict()
        data["server"] = {"port": 1234}
        cfg = load_config_from_dict(data)
        assert cfg.server.port == 1234
        # Other server defaults should still be present
        assert cfg.server.host == "127.0.0.1"

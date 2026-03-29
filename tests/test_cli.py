"""Unit and integration tests for llm_swap.cli — Click-based CLI entry point.

Tests cover:
- ``serve`` sub-command argument parsing, config loading, and uvicorn invocation.
- ``check-config`` sub-command success and failure paths.
- ``list-providers`` sub-command table and JSON output formats.
- Version flag.
- Default config path handling.

All tests use Click's ``CliRunner`` for isolation; no real servers are started
and no real network calls are made.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_swap import __version__
from llm_swap.cli import check_config, cli, list_providers, serve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, content: str) -> Path:
    """Write a YAML config string to a temp file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _minimal_config_yaml(default_provider: str = "openai") -> str:
    return f"""
        providers:
          - name: openai
            type: openai
            api_key: sk-test
            base_url: https://api.openai.com/v1
          - name: mistral
            type: mistral
            api_key: sk-mistral
            base_url: https://api.mistral.ai/v1
          - name: ollama
            type: ollama
            base_url: http://localhost:11434
        routing:
          default_provider: {default_provider}
          strategy: priority
          aliases:
            - alias: fast
              backends:
                - provider: openai
                  model: gpt-4o-mini
                  priority: 1
                - provider: mistral
                  model: mistral-small
                  priority: 2
            - alias: smart
              backends:
                - provider: openai
                  model: gpt-4o
                  priority: 1
          model_routes:
            - model: gpt-4o
              backends:
                - provider: openai
                  model: gpt-4o
                  priority: 1
        health_check:
          enabled: false
        logging:
          enabled: false
    """


# ---------------------------------------------------------------------------
# Version flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    """Tests for the --version flag on the root CLI group."""

    def test_version_flag_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_version_flag_shows_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert __version__ in result.output

    def test_version_flag_shows_prog_name(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert "llm_swap" in result.output


# ---------------------------------------------------------------------------
# Root help
# ---------------------------------------------------------------------------


class TestRootHelp:
    """Tests for the root CLI help message."""

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_serve_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "serve" in result.output

    def test_help_lists_check_config_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "check-config" in result.output

    def test_help_lists_list_providers_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "list-providers" in result.output


# ---------------------------------------------------------------------------
# check-config
# ---------------------------------------------------------------------------


class TestCheckConfig:
    """Tests for the check-config sub-command."""

    def test_valid_config_exits_zero(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        assert result.exit_code == 0

    def test_valid_config_prints_ok(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        assert "[OK]" in result.output

    def test_valid_config_prints_summary(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        assert "Providers" in result.output or "providers" in result.output.lower()

    def test_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            check_config, ["--config", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code != 0

    def test_missing_config_prints_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            check_config, ["--config", str(tmp_path / "nonexistent.yaml")]
        )
        # Error message goes to stderr; CliRunner mixes them by default
        assert "ERROR" in result.output or "ERROR" in (result.exception or "")

    def test_invalid_yaml_exits_nonzero(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, ": bad yaml [")
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        assert result.exit_code != 0

    def test_invalid_routing_reference_exits_nonzero(self, tmp_path: Path) -> None:
        content = """
            providers:
              - name: openai
                type: openai
                api_key: sk-test
                base_url: https://api.openai.com/v1
            routing:
              default_provider: nobody_home
        """
        cfg = _write_config(tmp_path, content)
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        assert result.exit_code != 0

    def test_verbose_flag_shows_provider_details(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg), "--verbose"])
        assert result.exit_code == 0
        assert "openai" in result.output
        assert "mistral" in result.output

    def test_verbose_flag_shows_aliases(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg), "--verbose"])
        assert "fast" in result.output
        assert "smart" in result.output

    def test_verbose_flag_shows_model_routes(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg), "--verbose"])
        assert "gpt-4o" in result.output

    def test_summary_shows_server_info(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        # Should mention the default host/port
        assert "127.0.0.1" in result.output or "8000" in result.output

    def test_summary_shows_default_provider(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(check_config, ["--config", str(cfg)])
        assert "openai" in result.output

    def test_check_config_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(check_config, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()


# ---------------------------------------------------------------------------
# list-providers
# ---------------------------------------------------------------------------


class TestListProviders:
    """Tests for the list-providers sub-command."""

    def test_exits_zero_on_valid_config(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        assert result.exit_code == 0

    def test_lists_all_provider_names(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        assert "openai" in result.output
        assert "mistral" in result.output
        assert "ollama" in result.output

    def test_shows_provider_types(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        # Provider types should be visible
        assert "openai" in result.output
        assert "mistral" in result.output

    def test_shows_base_urls(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        assert "api.openai.com" in result.output

    def test_shows_api_key_status(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        # Should show whether API key is present (yes/no)
        assert "yes" in result.output or "no" in result.output

    def test_shows_routing_strategy(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        assert "priority" in result.output

    def test_shows_aliases(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        assert "fast" in result.output
        assert "smart" in result.output

    def test_json_format_exits_zero(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(cfg), "--format", "json"]
        )
        assert result.exit_code == 0

    def test_json_format_produces_valid_json(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(cfg), "--format", "json"]
        )
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_json_format_includes_all_providers(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(cfg), "--format", "json"]
        )
        data = json.loads(result.output)
        names = {p["name"] for p in data}
        assert "openai" in names
        assert "mistral" in names
        assert "ollama" in names

    def test_json_format_has_required_keys(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(cfg), "--format", "json"]
        )
        data = json.loads(result.output)
        for entry in data:
            assert "name" in entry
            assert "type" in entry
            assert "base_url" in entry
            assert "has_api_key" in entry
            assert "timeout" in entry
            assert "max_retries" in entry

    def test_json_format_has_api_key_bool(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(cfg), "--format", "json"]
        )
        data = json.loads(result.output)
        openai_entry = next(p for p in data if p["name"] == "openai")
        assert openai_entry["has_api_key"] is True

    def test_json_no_api_key_provider(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(cfg), "--format", "json"]
        )
        data = json.loads(result.output)
        ollama_entry = next(p for p in data if p["name"] == "ollama")
        assert ollama_entry["has_api_key"] is False

    def test_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            list_providers, ["--config", str(tmp_path / "missing.yaml")]
        )
        assert result.exit_code != 0

    def test_list_providers_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--help"])
        assert result.exit_code == 0
        assert "format" in result.output.lower()

    def test_table_format_is_default(self, tmp_path: Path) -> None:
        """Default output should NOT be JSON."""
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        # Should not raise on non-JSON parse, confirm it's table format
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.output)

    def test_model_routes_shown_in_table(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(list_providers, ["--config", str(cfg)])
        assert "gpt-4o" in result.output


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for the serve sub-command."""

    def test_serve_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0

    def test_serve_help_shows_config_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "config" in result.output.lower()

    def test_serve_help_shows_host_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "host" in result.output.lower()

    def test_serve_help_shows_port_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "port" in result.output.lower()

    def test_serve_help_shows_log_level_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "log-level" in result.output.lower()

    def test_serve_help_shows_reload_option(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert "reload" in result.output.lower()

    def test_serve_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            serve, ["--config", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code != 0

    def test_serve_missing_config_prints_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            serve, ["--config", str(tmp_path / "nonexistent.yaml")]
        )
        # mix_stderr=True is default in CliRunner so error appears in output
        combined = result.output + (result.exception or "")
        assert "ERROR" in combined.upper() or result.exit_code != 0

    def test_serve_calls_uvicorn_run(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app") as mock_create_app:
                mock_app = MagicMock()
                mock_create_app.return_value = mock_app
                result = runner.invoke(serve, ["--config", str(cfg)])

        assert mock_run.called, f"uvicorn.run was not called. Output: {result.output}"

    def test_serve_passes_host_to_uvicorn(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                runner.invoke(serve, ["--config", str(cfg), "--host", "0.0.0.0"])

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("host") == "0.0.0.0"

    def test_serve_passes_port_to_uvicorn(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                runner.invoke(serve, ["--config", str(cfg), "--port", "9999"])

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("port") == 9999

    def test_serve_passes_log_level_to_uvicorn(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                runner.invoke(
                    serve, ["--config", str(cfg), "--log-level", "debug"]
                )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("log_level") == "debug"

    def test_serve_uses_config_host_when_not_overridden(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                runner.invoke(serve, ["--config", str(cfg)])

        call_kwargs = mock_run.call_args[1]
        # Default host from config is 127.0.0.1
        assert call_kwargs.get("host") == "127.0.0.1"

    def test_serve_uses_config_port_when_not_overridden(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                runner.invoke(serve, ["--config", str(cfg)])

        call_kwargs = mock_run.call_args[1]
        # Default port from config is 8000
        assert call_kwargs.get("port") == 8000

    def test_serve_passes_reload_flag(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run") as mock_run:
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                runner.invoke(serve, ["--config", str(cfg), "--reload"])

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("reload") is True

    def test_serve_prints_startup_banner(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run"):
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                result = runner.invoke(serve, ["--config", str(cfg)])

        assert "llm_swap" in result.output.lower() or "Starting" in result.output

    def test_serve_prints_provider_names(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()

        with patch("uvicorn.run"):
            with patch("llm_swap.proxy.create_app", return_value=MagicMock()):
                result = runner.invoke(serve, ["--config", str(cfg)])

        assert "openai" in result.output

    def test_serve_invalid_yaml_exits_nonzero(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, ": bad yaml [")
        runner = CliRunner()
        result = runner.invoke(serve, ["--config", str(cfg)])
        assert result.exit_code != 0

    def test_serve_calls_create_app_with_config(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        captured_configs: list = []

        def fake_create_app(config: Any) -> MagicMock:
            captured_configs.append(config)
            return MagicMock()

        with patch("uvicorn.run"):
            with patch("llm_swap.proxy.create_app", side_effect=fake_create_app):
                runner.invoke(serve, ["--config", str(cfg)])

        assert len(captured_configs) == 1
        assert captured_configs[0].providers[0].name == "openai"


# ---------------------------------------------------------------------------
# CLI invoked via the root group
# ---------------------------------------------------------------------------


class TestCLIGroup:
    """Tests for sub-commands invoked through the root cli group."""

    def test_cli_check_config_via_group(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(cli, ["check-config", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_cli_list_providers_via_group(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path, _minimal_config_yaml())
        runner = CliRunner()
        result = runner.invoke(cli, ["list-providers", "--config", str(cfg)])
        assert result.exit_code == 0

    def test_cli_serve_missing_config_via_group(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["serve", "--config", str(tmp_path / "missing.yaml")]
        )
        assert result.exit_code != 0

    def test_unknown_subcommand_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["unknown-command"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# conftest.ini marker (anyio)
# ---------------------------------------------------------------------------
# (No async tests in this file — all CLI tests are sync via CliRunner)

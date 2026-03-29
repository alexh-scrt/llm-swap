"""Click-based CLI entry point for llm_swap.

This module provides the ``llm_swap`` command-line interface with three
sub-commands:

- ``serve``        — Start the reverse-proxy HTTP server.
- ``check-config`` — Validate a configuration file and report any errors.
- ``list-providers`` — Display all configured providers and their settings.

Typical usage::

    # Start the server on the default port
    llm_swap serve --config config.yaml

    # Validate a config file without starting the server
    llm_swap check-config --config config.yaml

    # List all configured providers
    llm_swap list-providers --config config.yaml

All sub-commands accept a ``--config`` option that defaults to
``config.yaml`` in the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

from llm_swap import __version__
from llm_swap.config import ConfigError, load_config


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="llm_swap")
def cli() -> None:
    """llm_swap — a local OpenAI-compatible LLM reverse proxy.

    Route requests to OpenAI, Anthropic, Mistral, or a local Ollama instance
    based on configurable routing rules defined in a YAML file.

    Use ``llm_swap COMMAND --help`` for details on each sub-command.
    """


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    type=click.Path(dir_okay=False, readable=True),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--host",
    default=None,
    help="Override the host to bind on (default: from config).",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Override the TCP port to listen on (default: from config).",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"],
        case_sensitive=False,
    ),
    help="Override the uvicorn log level (default: from config).",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload on code changes (development only).",
)
def serve(
    config_path: str,
    host: Optional[str],
    port: Optional[int],
    log_level: Optional[str],
    reload: bool,
) -> None:
    """Start the llm_swap reverse-proxy server.

    Loads the configuration from CONFIG_PATH, validates it, then starts a
    Uvicorn ASGI server hosting the FastAPI proxy application.

    CLI options (--host, --port, --log-level) override the corresponding
    values from the configuration file when provided.

    Examples:

    \b
        llm_swap serve
        llm_swap serve --config /etc/llm_swap/config.yaml
        llm_swap serve --host 0.0.0.0 --port 9000 --log-level debug
    """
    # ── Load and validate configuration ────────────────────────────────
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"[llm_swap] ERROR: {exc}", err=True)
        sys.exit(1)

    # ── Apply CLI overrides ─────────────────────────────────────────────
    effective_host = host if host is not None else config.server.host
    effective_port = port if port is not None else config.server.port
    effective_log_level = (
        log_level.lower() if log_level is not None else config.server.log_level
    )

    # ── Print startup banner ────────────────────────────────────────────
    click.echo(
        f"[llm_swap] Starting proxy v{__version__} "
        f"on http://{effective_host}:{effective_port}"
    )
    click.echo(
        f"[llm_swap] Config: {Path(config_path).resolve()}"
    )
    click.echo(
        f"[llm_swap] Providers: "
        + ", ".join(p.name for p in config.providers)
    )
    click.echo(
        f"[llm_swap] Default provider: {config.routing.default_provider}"
    )
    click.echo(
        f"[llm_swap] Routing strategy: {config.routing.strategy}"
    )
    if config.routing.aliases:
        alias_names = ", ".join(a.alias for a in config.routing.aliases)
        click.echo(f"[llm_swap] Aliases: {alias_names}")
    if config.health_check.enabled:
        click.echo(
            f"[llm_swap] Health checks: enabled "
            f"(interval={config.health_check.interval_seconds}s)"
        )
    else:
        click.echo("[llm_swap] Health checks: disabled")

    # ── Build the FastAPI application ───────────────────────────────────
    # Import here to avoid circular imports at module load time.
    from llm_swap.proxy import create_app  # noqa: PLC0415

    app = create_app(config)

    # ── Launch Uvicorn ──────────────────────────────────────────────────
    uvicorn.run(
        app,
        host=effective_host,
        port=effective_port,
        log_level=effective_log_level,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# check-config
# ---------------------------------------------------------------------------


@cli.command("check-config")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print the parsed configuration details after validation.",
)
def check_config(config_path: str, verbose: bool) -> None:
    """Validate the configuration file and report any errors.

    Exits with code 0 on success and code 1 on validation failure.

    Examples:

    \b
        llm_swap check-config
        llm_swap check-config --config /etc/llm_swap/config.yaml
        llm_swap check-config --verbose
    """
    config_file = Path(config_path)
    click.echo(f"Checking configuration: {config_file.resolve()}")

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"\n[FAIL] Configuration is INVALID:\n", err=True)
        click.echo(f"  {exc}", err=True)
        sys.exit(1)

    click.echo("[OK]  Configuration is valid.\n")

    # Always print a brief summary
    click.echo(f"  Server         : {config.server.host}:{config.server.port}")
    click.echo(f"  Log level      : {config.server.log_level}")
    click.echo(f"  Request timeout: {config.server.request_timeout}s")
    click.echo(f"  Providers      : {len(config.providers)}")
    click.echo(f"  Aliases        : {len(config.routing.aliases)}")
    click.echo(f"  Model routes   : {len(config.routing.model_routes)}")
    click.echo(f"  Default provider: {config.routing.default_provider}")
    click.echo(f"  Routing strategy: {config.routing.strategy}")
    click.echo(
        f"  Health checks  : "
        f"{'enabled' if config.health_check.enabled else 'disabled'}"
    )
    click.echo(
        f"  Structured log : "
        f"{'enabled' if config.logging.enabled else 'disabled'} "
        f"({config.logging.log_format})"
    )

    if verbose:
        click.echo("\n--- Providers ---")
        for p in config.providers:
            key_hint = (
                f"{'*' * 8}" if p.api_key else "(none)"
            )
            click.echo(
                f"  [{p.name}] type={p.type}  "
                f"base_url={p.base_url}  "
                f"api_key={key_hint}  "
                f"timeout={p.timeout}s  "
                f"max_retries={p.max_retries}"
            )

        if config.routing.aliases:
            click.echo("\n--- Aliases ---")
            for alias in config.routing.aliases:
                backends_str = ", ".join(
                    f"{b.provider}/{b.model}(p={b.priority})"
                    for b in sorted(alias.backends, key=lambda b: b.priority)
                )
                click.echo(f"  {alias.alias!r:20s} → {backends_str}")

        if config.routing.model_routes:
            click.echo("\n--- Model Routes ---")
            for mr in config.routing.model_routes:
                backends_str = ", ".join(
                    f"{b.provider}/{b.model}(p={b.priority})"
                    for b in sorted(mr.backends, key=lambda b: b.priority)
                )
                click.echo(f"  {mr.model!r:40s} → {backends_str}")


# ---------------------------------------------------------------------------
# list-providers
# ---------------------------------------------------------------------------


@cli.command("list-providers")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    show_default=True,
    type=click.Choice(["table", "json"], case_sensitive=False),
    help="Output format: table (human-readable) or json.",
)
def list_providers(config_path: str, output_format: str) -> None:
    """List all configured backend providers.

    Displays provider names, types, base URLs, and whether an API key is
    configured.  Use ``--format json`` for machine-readable output.

    Examples:

    \b
        llm_swap list-providers
        llm_swap list-providers --config /etc/llm_swap/config.yaml
        llm_swap list-providers --format json
    """
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        click.echo(f"[llm_swap] ERROR: {exc}", err=True)
        sys.exit(1)

    providers = config.providers

    if not providers:
        click.echo("No providers configured.")
        return

    if output_format.lower() == "json":
        import json  # noqa: PLC0415

        data = [
            {
                "name": p.name,
                "type": p.type,
                "base_url": p.base_url,
                "has_api_key": bool(p.api_key),
                "timeout": p.timeout,
                "max_retries": p.max_retries,
                "extra_headers": list(p.headers.keys()),
            }
            for p in providers
        ]
        click.echo(json.dumps(data, indent=2))
        return

    # ── Table format ────────────────────────────────────────────────────
    # Calculate column widths dynamically
    name_w = max(len("NAME"), max(len(p.name) for p in providers))
    type_w = max(len("TYPE"), max(len(p.type) for p in providers))
    url_w = max(len("BASE URL"), max(len(p.base_url) for p in providers))

    header = (
        f"{'NAME':<{name_w}}  "
        f"{'TYPE':<{type_w}}  "
        f"{'BASE URL':<{url_w}}  "
        f"{'API KEY':8}  "
        f"{'TIMEOUT':>8}  "
        f"{'RETRIES':>7}"
    )
    separator = "-" * len(header)

    click.echo(f"\nConfigured providers ({len(providers)} total):\n")
    click.echo(header)
    click.echo(separator)

    for p in providers:
        key_status = "yes" if p.api_key else "no"
        row = (
            f"{p.name:<{name_w}}  "
            f"{p.type:<{type_w}}  "
            f"{p.base_url:<{url_w}}  "
            f"{key_status:<8}  "
            f"{p.timeout:>6}s  "
            f"{p.max_retries:>7}"
        )
        click.echo(row)

    click.echo()

    # Print routing summary
    click.echo(f"Default provider : {config.routing.default_provider}")
    click.echo(f"Routing strategy : {config.routing.strategy}")

    if config.routing.aliases:
        click.echo(f"\nAliases ({len(config.routing.aliases)}):\n")
        alias_name_w = max(
            len("ALIAS"),
            max(len(a.alias) for a in config.routing.aliases),
        )
        for alias in config.routing.aliases:
            backends = sorted(alias.backends, key=lambda b: b.priority)
            chain = " → ".join(
                f"{b.provider}/{b.model}" for b in backends
            )
            click.echo(f"  {alias.alias:<{alias_name_w}}  {chain}")

    if config.routing.model_routes:
        click.echo(f"\nModel routes ({len(config.routing.model_routes)}):\n")
        mr_name_w = max(
            len("MODEL"),
            max(len(mr.model) for mr in config.routing.model_routes),
        )
        for mr in config.routing.model_routes:
            backends = sorted(mr.backends, key=lambda b: b.priority)
            chain = " → ".join(
                f"{b.provider}/{b.model}" for b in backends
            )
            click.echo(f"  {mr.model:<{mr_name_w}}  {chain}")


# ---------------------------------------------------------------------------
# Module-level guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    cli()

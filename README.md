# llm_swap

> Drop-in LLM router — one endpoint, any backend, zero code changes.

llm_swap is a local reverse-proxy server that exposes a single OpenAI-compatible REST API and intelligently routes requests to any configured LLM backend — OpenAI, Anthropic, Mistral, or a local Ollama instance. Point your existing OpenAI SDK clients at the proxy and swap or mix providers without touching a single line of application code. Routing is controlled by a simple YAML config file with support for model aliases, priority chains, health checks, and structured logging.

---

## Quick Start

```bash
# Install
pip install llm_swap

# Copy and edit the example config
cp config.example.yaml config.yaml
# Add your API keys and routing rules to config.yaml

# Validate your config
llm_swap check-config --config config.yaml

# Start the proxy (default: http://127.0.0.1:8000)
llm_swap serve --config config.yaml
```

That's it. Your existing OpenAI SDK code works without modification:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-used",  # llm_swap uses keys from config.yaml
)

response = client.chat.completions.create(
    model="fast",  # resolves to whichever backend you mapped to "fast"
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Features

- **Zero-change client compatibility** — Fully OpenAI-compatible `/v1/chat/completions` endpoint with streaming (SSE) support; your existing SDK calls work as-is.
- **YAML-driven routing** — Map model aliases like `fast`, `smart`, or `cheap` to specific providers and models with priority ordering and fallback chains.
- **Multi-provider adapters** — Native support for OpenAI, Anthropic (Claude), Mistral, and local Ollama, with transparent request/response format normalization.
- **Automatic health checks** — Background tasks ping each provider, remove degraded backends from the routing pool, and restore them automatically on recovery.
- **Structured request logging** — Every request records the provider selected, latency, prompt/completion token counts, and errors for cost and performance visibility.

---

## Usage Examples

### CLI

```bash
# Start the server on a custom host and port
llm_swap serve --config config.yaml --host 0.0.0.0 --port 9000

# Validate config without starting the server
llm_swap check-config --config config.yaml

# List all configured providers and their status
llm_swap list-providers --config config.yaml

# List providers as JSON (useful for scripting)
llm_swap list-providers --config config.yaml --format json
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

with client.chat.completions.stream(
    model="smart",
    messages=[{"role": "user", "content": "Explain async/await in Python."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Check provider health

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "openai": {"healthy": true, "consecutive_failures": 0},
  "anthropic": {"healthy": true, "consecutive_failures": 0},
  "ollama": {"healthy": false, "consecutive_failures": 3}
}
```

### List available models/aliases

```bash
curl http://127.0.0.1:8000/v1/models
```

---

## Project Structure

```
llm_swap/
├── pyproject.toml          # Project metadata, dependencies, and entry-point
├── config.example.yaml     # Annotated example configuration file
├── conftest.py             # Pytest configuration (anyio backend)
│
├── llm_swap/
│   ├── __init__.py         # Package initializer, version string
│   ├── cli.py              # Click CLI: serve, check-config, list-providers
│   ├── proxy.py            # FastAPI app: /v1/chat/completions, /v1/models, /health
│   ├── router.py           # Routing engine: alias resolution, priority, fallback
│   ├── providers.py        # Provider adapters: OpenAI, Anthropic, Mistral, Ollama
│   ├── config.py           # Pydantic config models and YAML loader
│   ├── health.py           # Background provider health-check task
│   └── logger.py           # Structured request/response logging
│
└── tests/
    ├── test_config.py      # Config loading, validation, and error cases
    ├── test_router.py      # Routing engine: aliases, priority, round-robin, fallback
    ├── test_providers.py   # Provider adapters: request translation, response normalization
    ├── test_proxy.py       # FastAPI endpoint integration tests
    ├── test_health.py      # Health-check lifecycle and threshold transitions
    └── test_cli.py         # CLI sub-command tests
```

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit it. All options are annotated in the example file.

```yaml
# Server settings
server:
  host: "127.0.0.1"
  port: 8000
  log_level: "info"
  request_timeout: 120

# Structured logging
logging:
  enabled: true
  log_file: "llm_swap.log"  # null = stdout only
  log_format: "json"         # json | text

# Provider credentials
providers:
  - name: openai
    type: openai
    api_key: "${OPENAI_API_KEY}"  # env-var substitution supported

  - name: anthropic
    type: anthropic
    api_key: "${ANTHROPIC_API_KEY}"

  - name: ollama
    type: ollama
    base_url: "http://localhost:11434"

# Model aliases and routing rules
routing:
  default_provider: openai
  aliases:
    - alias: "fast"
      backends:
        - provider: ollama
          model: "llama3"
          priority: 1
        - provider: openai
          model: "gpt-4o-mini"
          priority: 2       # fallback if ollama is degraded

    - alias: "smart"
      strategy: "priority"  # priority | round_robin
      backends:
        - provider: anthropic
          model: "claude-3-5-sonnet-20241022"
          priority: 1
        - provider: openai
          model: "gpt-4o"
          priority: 2

    - alias: "cheap"
      backends:
        - provider: openai
          model: "gpt-4o-mini"
          priority: 1

# Health check settings
health_checks:
  enabled: true
  interval_seconds: 30
  unhealthy_threshold: 3    # failures before marking degraded
  healthy_threshold: 2      # successes before restoring
```

### Key configuration options

| Option | Description | Default |
|---|---|---|
| `server.host` | Interface to bind | `127.0.0.1` |
| `server.port` | TCP port | `8000` |
| `server.request_timeout` | Upstream timeout (seconds) | `120` |
| `logging.log_format` | `json` or `text` | `json` |
| `routing.default_provider` | Fallback when no alias matches | required |
| `health_checks.interval_seconds` | Ping frequency | `30` |
| `health_checks.unhealthy_threshold` | Failures before degraded | `3` |

Environment variables can be injected into any string value using `${VAR_NAME}` syntax.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*

"""llm_swap — a local reverse-proxy that exposes a single OpenAI-compatible REST API
and intelligently routes requests to any configured LLM backend.

Supported backends:
- OpenAI
- Anthropic (Claude)
- Mistral
- Ollama (local)

Clients can point their existing OpenAI SDK at this proxy and swap or mix providers
without changing any application code. Routing is controlled via a YAML config file.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]

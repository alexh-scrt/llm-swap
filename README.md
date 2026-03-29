# llm-swap
llm_swap is a local reverse-proxy server that exposes a single OpenAI-compatible REST API and intelligently routes requests to any configured LLM backend — OpenAI, Anthropic, Mistral, or a local Ollama instance — based on configurable rules such as model alias, cost tier, latency preference, or round-robin fallback. Teams can point their existing O

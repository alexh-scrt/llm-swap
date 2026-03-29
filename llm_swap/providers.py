"""Provider adapter layer for llm_swap.

This module implements adapters that translate the unified OpenAI-compatible
request format to each provider's native API format, send the HTTP request,
and normalize the response back to the OpenAI response schema.

Supported providers:
- ``openai``: Native OpenAI API (also compatible with OpenAI-like endpoints).
- ``anthropic``: Anthropic Claude API (Messages endpoint).
- ``mistral``: Mistral AI API (OpenAI-compatible, minor differences).
- ``ollama``: Local Ollama server (OpenAI-compatible /api/chat endpoint).

All adapters implement the :class:`BaseAdapter` interface which exposes two
async methods:

- :meth:`BaseAdapter.chat_completion` – non-streaming chat completion.
- :meth:`BaseAdapter.chat_completion_stream` – streaming chat completion
  yielding raw SSE ``data:`` lines.

Typical usage::

    from llm_swap.providers import get_adapter
    from llm_swap.config import ProviderConfig

    adapter = get_adapter(provider_config)
    response = await adapter.chat_completion(request_body, target_model="gpt-4o")
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from llm_swap.config import ProviderConfig


# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------

RequestBody = Dict[str, Any]
ResponseBody = Dict[str, Any]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Raised when a provider returns an error response or the request fails.

    Attributes:
        status_code: HTTP status code returned by the provider, or ``None``
            if the error occurred before a response was received.
        provider_name: The name of the provider that raised the error.
        message: Human-readable error description.
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ProviderError(provider={self.provider_name!r}, "
            f"status_code={self.status_code}, message={self.message!r})"
        )


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


class ProviderConnectionError(ProviderError):
    """Raised when the connection to a provider cannot be established."""


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """Abstract base class for all provider adapters.

    Concrete subclasses must implement :meth:`chat_completion` and
    :meth:`chat_completion_stream`.

    Args:
        config: The :class:`~llm_swap.config.ProviderConfig` for this adapter.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    @property
    def provider_name(self) -> str:
        """Return the provider's unique name."""
        return self._config.name

    def _build_client(self) -> httpx.AsyncClient:
        """Create a configured :class:`httpx.AsyncClient` for this provider.

        The client is constructed with the provider's timeout and any
        extra headers defined in the config.  Callers are responsible for
        closing the client (use as an async context manager).

        Returns:
            A ready-to-use :class:`httpx.AsyncClient`.
        """
        headers = dict(self._config.headers)
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=headers,
            timeout=float(self._config.timeout),
        )

    @abstractmethod
    async def chat_completion(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> ResponseBody:
        """Perform a non-streaming chat completion request.

        Args:
            request_body: The OpenAI-format request body from the client.
            target_model: The exact model name to send to the provider.

        Returns:
            A normalized OpenAI-format response body dict.

        Raises:
            ProviderError: On any provider-level error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """

    @abstractmethod
    async def chat_completion_stream(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> AsyncIterator[str]:
        """Perform a streaming chat completion request.

        Yields raw SSE ``data:`` lines (each is a complete JSON string or the
        sentinel ``"[DONE]"``).

        Args:
            request_body: The OpenAI-format request body from the client.
            target_model: The exact model name to send to the provider.

        Yields:
            Strings suitable for forwarding as ``data: <string>\\n\\n`` SSE frames.

        Raises:
            ProviderError: On any provider-level error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _openai_error_message(response: httpx.Response) -> str:
    """Extract a human-readable error message from an OpenAI-style error response."""
    try:
        body = response.json()
        error = body.get("error", {})
        return error.get("message") or response.text
    except Exception:
        return response.text


def _raise_for_provider(
    response: httpx.Response,
    provider_name: str,
    error_extractor: Any = None,
) -> None:
    """Raise a :class:`ProviderError` if the response status indicates failure.

    Args:
        response: The HTTP response to inspect.
        provider_name: Provider name for error attribution.
        error_extractor: Optional callable ``(response) -> str`` to extract
            the error message.  Defaults to :func:`_openai_error_message`.
    """
    if response.is_success:
        return
    extractor = error_extractor or _openai_error_message
    msg = extractor(response)
    raise ProviderError(
        message=f"{provider_name} returned HTTP {response.status_code}: {msg}",
        provider_name=provider_name,
        status_code=response.status_code,
    )


def _wrap_httpx_errors(func: Any) -> Any:  # pragma: no cover
    """Decorator placeholder – we handle httpx errors inline for clarity."""
    return func


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAIAdapter(BaseAdapter):
    """Adapter for the OpenAI Chat Completions API.

    The request format is already OpenAI-compatible so this adapter only
    overrides the ``model`` field and forwards everything else verbatim.
    """

    async def chat_completion(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> ResponseBody:
        """Send a non-streaming chat completion to OpenAI.

        Args:
            request_body: OpenAI-format request body.
            target_model: Model name to use.

        Returns:
            OpenAI-format response body.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = {**request_body, "model": target_model, "stream": False}
        try:
            async with self._build_client() as client:
                response = await client.post("/chat/completions", json=payload)
            _raise_for_provider(response, self.provider_name)
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request to {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc

    async def chat_completion_stream(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from OpenAI.

        Args:
            request_body: OpenAI-format request body.
            target_model: Model name to use.

        Yields:
            Raw SSE data strings.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = {**request_body, "model": target_model, "stream": True}
        try:
            async with self._build_client() as client:
                async with client.stream("POST", "/chat/completions", json=payload) as response:
                    _raise_for_provider(response, self.provider_name)
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                            if data:
                                yield data
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Stream from {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


class AnthropicAdapter(BaseAdapter):
    """Adapter for the Anthropic Messages API (Claude models).

    Translates OpenAI-format requests to Anthropic's ``/v1/messages`` format
    and normalizes responses back to OpenAI format.

    Key differences handled:
    - ``system`` role is extracted into a top-level ``system`` field.
    - ``max_tokens`` is required by Anthropic (defaults to 4096 if absent).
    - Response ``id``, ``model``, and ``usage`` fields are remapped.
    - Streaming uses Anthropic's own SSE event types.
    """

    _API_VERSION = "2023-06-01"
    _DEFAULT_MAX_TOKENS = 4096

    def _build_client(self) -> httpx.AsyncClient:
        """Build an Anthropic-specific client with required API version header."""
        headers = dict(self._config.headers)
        if self._config.api_key:
            headers["x-api-key"] = self._config.api_key
        headers["anthropic-version"] = self._API_VERSION
        # Remove Authorization header set by parent (Anthropic uses x-api-key)
        headers.pop("Authorization", None)
        return httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=headers,
            timeout=float(self._config.timeout),
        )

    def _translate_request(self, request_body: RequestBody, target_model: str) -> RequestBody:
        """Convert an OpenAI-format request body to Anthropic Messages format.

        Args:
            request_body: OpenAI-format chat completion request.
            target_model: Anthropic model identifier.

        Returns:
            Anthropic Messages API request body.
        """
        messages: List[Dict[str, Any]] = request_body.get("messages", [])
        system_content: Optional[str] = None
        filtered_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Concatenate multiple system messages
                if system_content is None:
                    system_content = content
                else:
                    system_content += "\n" + content
            else:
                # Map "assistant" and "user" roles directly
                filtered_messages.append({"role": role, "content": content})

        payload: RequestBody = {
            "model": target_model,
            "messages": filtered_messages,
            "max_tokens": request_body.get("max_tokens", self._DEFAULT_MAX_TOKENS),
        }

        if system_content is not None:
            payload["system"] = system_content

        # Forward optional parameters that Anthropic supports
        for key in ("temperature", "top_p", "top_k", "stop_sequences", "metadata"):
            if key in request_body:
                payload[key] = request_body[key]

        # Map OpenAI's ``stop`` to Anthropic's ``stop_sequences``
        if "stop" in request_body and "stop_sequences" not in payload:
            stop_val = request_body["stop"]
            if isinstance(stop_val, str):
                stop_val = [stop_val]
            payload["stop_sequences"] = stop_val

        return payload

    def _normalize_response(self, anthropic_response: Dict[str, Any]) -> ResponseBody:
        """Convert an Anthropic Messages response to OpenAI format.

        Args:
            anthropic_response: Raw Anthropic Messages API response.

        Returns:
            OpenAI-format chat completion response.
        """
        content_blocks: List[Dict[str, Any]] = anthropic_response.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        stop_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        anthropic_stop = anthropic_response.get("stop_reason", "end_turn")
        finish_reason = stop_reason_map.get(anthropic_stop, "stop")

        usage = anthropic_response.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return {
            "id": anthropic_response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": anthropic_response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _anthropic_error_message(self, response: httpx.Response) -> str:
        """Extract error message from an Anthropic error response."""
        try:
            body = response.json()
            error = body.get("error", {})
            return error.get("message") or response.text
        except Exception:
            return response.text

    async def chat_completion(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> ResponseBody:
        """Send a non-streaming chat completion to Anthropic.

        Args:
            request_body: OpenAI-format request body.
            target_model: Anthropic model identifier.

        Returns:
            OpenAI-format response body.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = self._translate_request(request_body, target_model)
        try:
            async with self._build_client() as client:
                response = await client.post("/v1/messages", json=payload)
            _raise_for_provider(response, self.provider_name, self._anthropic_error_message)
            return self._normalize_response(response.json())
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request to {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc

    async def chat_completion_stream(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from Anthropic, normalized to OpenAI SSE format.

        Anthropic uses its own SSE event types (``content_block_delta``,
        ``message_delta``, etc.). This method translates them to OpenAI-style
        ``chat.completion.chunk`` objects.

        Args:
            request_body: OpenAI-format request body.
            target_model: Anthropic model identifier.

        Yields:
            OpenAI-format SSE data strings.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = self._translate_request(request_body, target_model)
        payload["stream"] = True
        completion_id = f"chatcmpl-anthropic-{int(time.time())}"
        try:
            async with self._build_client() as client:
                async with client.stream("POST", "/v1/messages", json=payload) as response:
                    _raise_for_provider(response, self.provider_name, self._anthropic_error_message)
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[len("data: "):].strip()
                        if not data_str:
                            continue
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event_data.get("type", "")

                        if event_type == "content_block_delta":
                            delta = event_data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_chunk = delta.get("text", "")
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": target_model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": text_chunk,
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield json.dumps(chunk)

                        elif event_type == "message_delta":
                            delta = event_data.get("delta", {})
                            stop_reason = delta.get("stop_reason", "end_turn")
                            stop_reason_map = {
                                "end_turn": "stop",
                                "max_tokens": "length",
                                "stop_sequence": "stop",
                            }
                            finish_reason = stop_reason_map.get(stop_reason, "stop")
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": target_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason,
                                    }
                                ],
                            }
                            yield json.dumps(chunk)

                        elif event_type == "message_stop":
                            yield "[DONE]"

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Stream from {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc


# ---------------------------------------------------------------------------
# Mistral adapter
# ---------------------------------------------------------------------------


class MistralAdapter(BaseAdapter):
    """Adapter for the Mistral AI API.

    Mistral exposes an OpenAI-compatible API so this adapter is nearly
    identical to :class:`OpenAIAdapter`.  The main difference is that the
    base URL and authentication header are set per-config.
    """

    async def chat_completion(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> ResponseBody:
        """Send a non-streaming chat completion to Mistral AI.

        Args:
            request_body: OpenAI-format request body.
            target_model: Mistral model identifier.

        Returns:
            OpenAI-format response body.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = {**request_body, "model": target_model, "stream": False}
        try:
            async with self._build_client() as client:
                response = await client.post("/chat/completions", json=payload)
            _raise_for_provider(response, self.provider_name)
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request to {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc

    async def chat_completion_stream(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from Mistral AI.

        Args:
            request_body: OpenAI-format request body.
            target_model: Mistral model identifier.

        Yields:
            Raw SSE data strings.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = {**request_body, "model": target_model, "stream": True}
        try:
            async with self._build_client() as client:
                async with client.stream("POST", "/chat/completions", json=payload) as response:
                    _raise_for_provider(response, self.provider_name)
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                            if data:
                                yield data
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Stream from {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------


class OllamaAdapter(BaseAdapter):
    """Adapter for a local Ollama server.

    Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint
    since version 0.1.24.  This adapter targets that endpoint and does not
    require authentication.
    """

    def _build_client(self) -> httpx.AsyncClient:
        """Build an Ollama client without an Authorization header."""
        headers = dict(self._config.headers)
        # Ollama does not use API keys
        return httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=headers,
            timeout=float(self._config.timeout),
        )

    async def chat_completion(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> ResponseBody:
        """Send a non-streaming chat completion to a local Ollama server.

        Args:
            request_body: OpenAI-format request body.
            target_model: Ollama model name (e.g. ``"llama3"``)

        Returns:
            OpenAI-format response body.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = {**request_body, "model": target_model, "stream": False}
        try:
            async with self._build_client() as client:
                response = await client.post("/v1/chat/completions", json=payload)
            _raise_for_provider(response, self.provider_name)
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request to {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc

    async def chat_completion_stream(
        self,
        request_body: RequestBody,
        target_model: str,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from Ollama.

        Args:
            request_body: OpenAI-format request body.
            target_model: Ollama model name.

        Yields:
            Raw SSE data strings.

        Raises:
            ProviderError: On API error.
            ProviderTimeoutError: On timeout.
            ProviderConnectionError: On connection failure.
        """
        payload = {**request_body, "model": target_model, "stream": True}
        try:
            async with self._build_client() as client:
                async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    _raise_for_provider(response, self.provider_name)
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):].strip()
                            if data:
                                yield data
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Stream from {self.provider_name} timed out: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Cannot connect to {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming error from {self.provider_name}: {exc}",
                provider_name=self.provider_name,
            ) from exc


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: Dict[str, type] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "mistral": MistralAdapter,
    "ollama": OllamaAdapter,
}


def get_adapter(config: ProviderConfig) -> BaseAdapter:
    """Return the appropriate adapter instance for the given provider config.

    Args:
        config: A validated :class:`~llm_swap.config.ProviderConfig` instance.

    Returns:
        A concrete :class:`BaseAdapter` subclass instance.

    Raises:
        ValueError: If the provider type is not supported.
    """
    adapter_cls = _ADAPTER_REGISTRY.get(config.type)
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported provider type {config.type!r}. "
            f"Supported types: {sorted(_ADAPTER_REGISTRY)}"
        )
    return adapter_cls(config)

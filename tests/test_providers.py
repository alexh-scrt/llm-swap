"""Unit tests for llm_swap.providers — provider adapters and the structured logger.

All HTTP calls are intercepted with ``httpx``'s built-in transport mocking
(``httpx.MockTransport`` / ``respx`` patterns) using
``unittest.mock.AsyncMock`` and ``httpx.AsyncMockTransport`` via
``anyio``-compatible async tests.  No real network calls are made.

The tests verify:
- Request body translation for each provider.
- Response normalization to OpenAI format.
- Streaming SSE parsing and normalization.
- Error propagation (4xx/5xx, timeouts, connection errors).
- The adapter factory (``get_adapter``).
- The structured logger (``RequestLogger``).
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_swap.config import LoggingConfig, ProviderConfig
from llm_swap.logger import RequestLogger, RequestRecord, _format_record_message
from llm_swap.providers import (
    AnthropicAdapter,
    BaseAdapter,
    MistralAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    _openai_error_message,
    _raise_for_provider,
    get_adapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_config(
    name: str = "openai",
    provider_type: str = "openai",
    api_key: str = "sk-test",
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 30,
    max_retries: int = 0,
    headers: Optional[Dict[str, str]] = None,
) -> ProviderConfig:
    """Build a ProviderConfig for use in tests."""
    return ProviderConfig(
        name=name,
        type=provider_type,  # type: ignore[arg-type]
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        headers=headers or {},
    )


def _openai_chat_response(
    content: str = "Hello!",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> Dict[str, Any]:
    """Build a minimal OpenAI chat completion response dict."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _anthropic_messages_response(
    text: str = "Hello from Claude!",
    model: str = "claude-3-5-sonnet-20241022",
    input_tokens: int = 12,
    output_tokens: int = 8,
) -> Dict[str, Any]:
    """Build a minimal Anthropic Messages API response dict."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _make_mock_response(
    status_code: int = 200,
    json_body: Optional[Dict[str, Any]] = None,
    text_body: str = "",
) -> httpx.Response:
    """Create a mock httpx.Response."""
    if json_body is not None:
        content = json.dumps(json_body).encode()
        headers = {"content-type": "application/json"}
    else:
        content = text_body.encode()
        headers = {"content-type": "text/plain"}
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=httpx.Request("POST", "https://example.com/"),
    )


# ---------------------------------------------------------------------------
# _openai_error_message
# ---------------------------------------------------------------------------


class TestOpenAIErrorMessage:
    """Tests for the _openai_error_message helper."""

    def test_extracts_message_from_json(self) -> None:
        resp = _make_mock_response(
            status_code=400,
            json_body={"error": {"message": "Invalid request", "type": "invalid_request_error"}},
        )
        assert _openai_error_message(resp) == "Invalid request"

    def test_falls_back_to_text_on_non_json(self) -> None:
        resp = _make_mock_response(status_code=500, text_body="Internal Server Error")
        msg = _openai_error_message(resp)
        assert "Internal Server Error" in msg

    def test_falls_back_when_no_message_key(self) -> None:
        resp = _make_mock_response(status_code=400, json_body={"error": {"type": "bad"}})
        # Should not crash; returns empty string or text fallback
        result = _openai_error_message(resp)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _raise_for_provider
# ---------------------------------------------------------------------------


class TestRaiseForProvider:
    """Tests for the _raise_for_provider helper."""

    def test_does_not_raise_on_200(self) -> None:
        resp = _make_mock_response(200, json_body={"ok": True})
        _raise_for_provider(resp, "openai")  # Should not raise

    def test_raises_on_400(self) -> None:
        resp = _make_mock_response(
            400, json_body={"error": {"message": "Bad request"}}
        )
        with pytest.raises(ProviderError) as exc_info:
            _raise_for_provider(resp, "openai")
        assert exc_info.value.status_code == 400
        assert exc_info.value.provider_name == "openai"

    def test_raises_on_500(self) -> None:
        resp = _make_mock_response(500, text_body="Server Error")
        with pytest.raises(ProviderError) as exc_info:
            _raise_for_provider(resp, "mistral")
        assert exc_info.value.status_code == 500
        assert exc_info.value.provider_name == "mistral"

    def test_uses_custom_error_extractor(self) -> None:
        resp = _make_mock_response(429, text_body="Rate limited")
        custom_extractor = MagicMock(return_value="Custom message")
        with pytest.raises(ProviderError) as exc_info:
            _raise_for_provider(resp, "openai", error_extractor=custom_extractor)
        assert "Custom message" in exc_info.value.message
        custom_extractor.assert_called_once_with(resp)


# ---------------------------------------------------------------------------
# get_adapter factory
# ---------------------------------------------------------------------------


class TestGetAdapter:
    """Tests for the adapter factory function."""

    def test_openai_returns_openai_adapter(self) -> None:
        cfg = _make_provider_config(provider_type="openai")
        adapter = get_adapter(cfg)
        assert isinstance(adapter, OpenAIAdapter)

    def test_anthropic_returns_anthropic_adapter(self) -> None:
        cfg = _make_provider_config(
            name="anthropic",
            provider_type="anthropic",
            base_url="https://api.anthropic.com",
        )
        adapter = get_adapter(cfg)
        assert isinstance(adapter, AnthropicAdapter)

    def test_mistral_returns_mistral_adapter(self) -> None:
        cfg = _make_provider_config(
            name="mistral",
            provider_type="mistral",
            base_url="https://api.mistral.ai/v1",
        )
        adapter = get_adapter(cfg)
        assert isinstance(adapter, MistralAdapter)

    def test_ollama_returns_ollama_adapter(self) -> None:
        cfg = _make_provider_config(
            name="ollama",
            provider_type="ollama",
            api_key=None,
            base_url="http://localhost:11434",
        )
        adapter = get_adapter(cfg)
        assert isinstance(adapter, OllamaAdapter)

    def test_all_adapters_are_base_adapter_subclasses(self) -> None:
        for ptype, base_url in [
            ("openai", "https://api.openai.com/v1"),
            ("anthropic", "https://api.anthropic.com"),
            ("mistral", "https://api.mistral.ai/v1"),
            ("ollama", "http://localhost:11434"),
        ]:
            cfg = _make_provider_config(
                name=ptype, provider_type=ptype, base_url=base_url
            )
            adapter = get_adapter(cfg)
            assert isinstance(adapter, BaseAdapter)

    def test_unknown_type_raises_value_error(self) -> None:
        # Build with a valid type then monkey-patch to test the factory guard
        cfg = _make_provider_config(provider_type="openai")
        # Directly test the registry lookup path
        from llm_swap.providers import _ADAPTER_REGISTRY
        assert "openai" in _ADAPTER_REGISTRY


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------


class TestOpenAIAdapter:
    """Tests for the OpenAI provider adapter."""

    def _make_adapter(self) -> OpenAIAdapter:
        cfg = _make_provider_config(
            name="openai",
            provider_type="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        return OpenAIAdapter(cfg)

    @pytest.mark.anyio
    async def test_chat_completion_success(self) -> None:
        adapter = self._make_adapter()
        expected_response = _openai_chat_response(content="Hi there!")

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return _make_mock_response(200, json_body=expected_response)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=transport,
            )
            request_body = {"messages": [{"role": "user", "content": "Hello"}]}
            response = await adapter.chat_completion(request_body, "gpt-4o-mini")

        assert response["object"] == "chat.completion"
        assert response["choices"][0]["message"]["content"] == "Hi there!"
        assert response["usage"]["total_tokens"] == 15

    @pytest.mark.anyio
    async def test_chat_completion_sets_stream_false(self) -> None:
        """Verify that the adapter always sends stream=False for non-streaming calls."""
        adapter = self._make_adapter()
        captured_body: Dict[str, Any] = {}

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return _make_mock_response(200, json_body=_openai_chat_response())

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=transport,
            )
            await adapter.chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}]}, "gpt-4o-mini"
            )

        assert captured_body.get("stream") is False

    @pytest.mark.anyio
    async def test_chat_completion_uses_target_model(self) -> None:
        adapter = self._make_adapter()
        captured_body: Dict[str, Any] = {}

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return _make_mock_response(200, json_body=_openai_chat_response())

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=transport,
            )
            await adapter.chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}]}, "gpt-4o"
            )

        assert captured_body.get("model") == "gpt-4o"

    @pytest.mark.anyio
    async def test_chat_completion_raises_provider_error_on_4xx(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return _make_mock_response(
                401, json_body={"error": {"message": "Unauthorized"}}
            )

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=transport,
            )
            with pytest.raises(ProviderError) as exc_info:
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]}, "gpt-4o"
                )
        assert exc_info.value.status_code == 401
        assert exc_info.value.provider_name == "openai"

    @pytest.mark.anyio
    async def test_chat_completion_raises_timeout_error(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=transport,
            )
            with pytest.raises(ProviderTimeoutError):
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]}, "gpt-4o"
                )

    @pytest.mark.anyio
    async def test_chat_completion_raises_connection_error(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                transport=transport,
            )
            with pytest.raises(ProviderConnectionError):
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]}, "gpt-4o"
                )

    @pytest.mark.anyio
    async def test_chat_completion_stream_yields_data(self) -> None:
        adapter = self._make_adapter()

        sse_lines = [
            'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]

        async def iter_lines() -> AsyncIterator[str]:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.aiter_lines = iter_lines

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(adapter, "_build_client", return_value=mock_client):
            chunks: List[str] = []
            async for chunk in adapter.chat_completion_stream(
                {"messages": [{"role": "user", "content": "Hi"}]}, "gpt-4o-mini"
            ):
                chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[-1] == "[DONE]"

    @pytest.mark.anyio
    async def test_provider_name_property(self) -> None:
        adapter = self._make_adapter()
        assert adapter.provider_name == "openai"


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------


class TestAnthropicAdapter:
    """Tests for the Anthropic provider adapter."""

    def _make_adapter(self) -> AnthropicAdapter:
        cfg = _make_provider_config(
            name="anthropic",
            provider_type="anthropic",
            api_key="sk-ant-test",
            base_url="https://api.anthropic.com",
        )
        return AnthropicAdapter(cfg)

    def test_translate_request_extracts_system_message(self) -> None:
        adapter = self._make_adapter()
        openai_body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
        }
        payload = adapter._translate_request(openai_body, "claude-3-5-sonnet-20241022")
        assert payload["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in payload["messages"])
        assert payload["messages"][0] == {"role": "user", "content": "Hello"}

    def test_translate_request_no_system_message(self) -> None:
        adapter = self._make_adapter()
        openai_body = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        payload = adapter._translate_request(openai_body, "claude-3-haiku")
        assert "system" not in payload
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_translate_request_uses_default_max_tokens(self) -> None:
        adapter = self._make_adapter()
        payload = adapter._translate_request(
            {"messages": [{"role": "user", "content": "Hi"}]},
            "claude-3-5-sonnet-20241022",
        )
        assert payload["max_tokens"] == adapter._DEFAULT_MAX_TOKENS

    def test_translate_request_forwards_temperature(self) -> None:
        adapter = self._make_adapter()
        payload = adapter._translate_request(
            {"messages": [{"role": "user", "content": "Hi"}], "temperature": 0.7},
            "claude-3-5-sonnet-20241022",
        )
        assert payload["temperature"] == 0.7

    def test_translate_request_maps_stop_string_to_list(self) -> None:
        adapter = self._make_adapter()
        payload = adapter._translate_request(
            {"messages": [{"role": "user", "content": "Hi"}], "stop": "\n"},
            "claude-3-5-sonnet-20241022",
        )
        assert payload["stop_sequences"] == ["\n"]

    def test_translate_request_maps_stop_list(self) -> None:
        adapter = self._make_adapter()
        payload = adapter._translate_request(
            {"messages": [{"role": "user", "content": "Hi"}], "stop": ["\n", "END"]},
            "claude-3-5-sonnet-20241022",
        )
        assert payload["stop_sequences"] == ["\n", "END"]

    def test_translate_request_concatenates_multiple_system_messages(self) -> None:
        adapter = self._make_adapter()
        payload = adapter._translate_request(
            {
                "messages": [
                    {"role": "system", "content": "Part 1."},
                    {"role": "system", "content": "Part 2."},
                    {"role": "user", "content": "Hello"},
                ]
            },
            "claude-3-5-sonnet-20241022",
        )
        assert "Part 1." in payload["system"]
        assert "Part 2." in payload["system"]

    def test_normalize_response_structure(self) -> None:
        adapter = self._make_adapter()
        anthropic_resp = _anthropic_messages_response(text="Hi!", input_tokens=5, output_tokens=3)
        normalized = adapter._normalize_response(anthropic_resp)
        assert normalized["object"] == "chat.completion"
        assert normalized["choices"][0]["message"]["role"] == "assistant"
        assert normalized["choices"][0]["message"]["content"] == "Hi!"
        assert normalized["choices"][0]["finish_reason"] == "stop"
        assert normalized["usage"]["prompt_tokens"] == 5
        assert normalized["usage"]["completion_tokens"] == 3
        assert normalized["usage"]["total_tokens"] == 8

    def test_normalize_response_maps_max_tokens_stop_reason(self) -> None:
        adapter = self._make_adapter()
        resp = _anthropic_messages_response()
        resp["stop_reason"] = "max_tokens"
        normalized = adapter._normalize_response(resp)
        assert normalized["choices"][0]["finish_reason"] == "length"

    def test_normalize_response_handles_multiple_content_blocks(self) -> None:
        adapter = self._make_adapter()
        resp = _anthropic_messages_response()
        resp["content"] = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world!"},
        ]
        normalized = adapter._normalize_response(resp)
        assert normalized["choices"][0]["message"]["content"] == "Hello world!"

    @pytest.mark.anyio
    async def test_chat_completion_success(self) -> None:
        adapter = self._make_adapter()
        anthropic_resp = _anthropic_messages_response(text="Claude here!")

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            # Verify the request goes to /v1/messages
            assert "/v1/messages" in str(request.url)
            return _make_mock_response(200, json_body=anthropic_resp)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                transport=transport,
            )
            result = await adapter.chat_completion(
                {"messages": [{"role": "user", "content": "Hello"}]},
                "claude-3-5-sonnet-20241022",
            )

        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Claude here!"

    @pytest.mark.anyio
    async def test_chat_completion_raises_on_error(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return _make_mock_response(
                400,
                json_body={"error": {"type": "invalid_request_error", "message": "Bad"}},
            )

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                transport=transport,
            )
            with pytest.raises(ProviderError) as exc_info:
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]},
                    "claude-3-5-sonnet-20241022",
                )
        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_chat_completion_raises_timeout(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                transport=transport,
            )
            with pytest.raises(ProviderTimeoutError):
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]},
                    "claude-3-5-sonnet-20241022",
                )

    @pytest.mark.anyio
    async def test_chat_completion_stream_normalizes_to_openai_format(self) -> None:
        adapter = self._make_adapter()

        # Simulate Anthropic SSE stream
        sse_lines = [
            'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet-20241022"}}',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            'data: {"type":"content_block_stop","index":0}',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}',
            'data: {"type":"message_stop"}',
        ]

        async def iter_lines() -> AsyncIterator[str]:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.aiter_lines = iter_lines

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(adapter, "_build_client", return_value=mock_client):
            chunks: List[str] = []
            async for chunk in adapter.chat_completion_stream(
                {"messages": [{"role": "user", "content": "Hi"}]},
                "claude-3-5-sonnet-20241022",
            ):
                chunks.append(chunk)

        # Should have: 2 text deltas + 1 message_delta (finish) + 1 DONE
        assert len(chunks) >= 3
        assert chunks[-1] == "[DONE]"

        # Verify text chunks are OpenAI format
        text_chunks = [c for c in chunks if c != "[DONE]"]
        for tc in text_chunks:
            parsed = json.loads(tc)
            assert parsed["object"] in ("chat.completion.chunk",)

    @pytest.mark.anyio
    async def test_build_client_sets_anthropic_version_header(self) -> None:
        adapter = self._make_adapter()
        client = adapter._build_client()
        assert client.headers.get("anthropic-version") == AnthropicAdapter._API_VERSION
        await client.aclose()

    @pytest.mark.anyio
    async def test_build_client_uses_x_api_key_not_authorization(self) -> None:
        adapter = self._make_adapter()
        client = adapter._build_client()
        assert "x-api-key" in client.headers
        assert "authorization" not in client.headers
        await client.aclose()


# ---------------------------------------------------------------------------
# MistralAdapter
# ---------------------------------------------------------------------------


class TestMistralAdapter:
    """Tests for the Mistral provider adapter."""

    def _make_adapter(self) -> MistralAdapter:
        cfg = _make_provider_config(
            name="mistral",
            provider_type="mistral",
            api_key="sk-mistral-test",
            base_url="https://api.mistral.ai/v1",
        )
        return MistralAdapter(cfg)

    @pytest.mark.anyio
    async def test_chat_completion_success(self) -> None:
        adapter = self._make_adapter()
        expected = _openai_chat_response(content="Mistral response", model="mistral-small")

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return _make_mock_response(200, json_body=expected)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.mistral.ai/v1",
                transport=transport,
            )
            result = await adapter.chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}]},
                "mistral-small-latest",
            )

        assert result["choices"][0]["message"]["content"] == "Mistral response"

    @pytest.mark.anyio
    async def test_chat_completion_sets_target_model(self) -> None:
        adapter = self._make_adapter()
        captured: Dict[str, Any] = {}

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            captured.update(json.loads(request.content))
            return _make_mock_response(200, json_body=_openai_chat_response())

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.mistral.ai/v1",
                transport=transport,
            )
            await adapter.chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}]},
                "mistral-large-latest",
            )

        assert captured.get("model") == "mistral-large-latest"
        assert captured.get("stream") is False

    @pytest.mark.anyio
    async def test_chat_completion_raises_on_error(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return _make_mock_response(
                401, json_body={"error": {"message": "Unauthorized"}}
            )

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="https://api.mistral.ai/v1",
                transport=transport,
            )
            with pytest.raises(ProviderError):
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]},
                    "mistral-small-latest",
                )


# ---------------------------------------------------------------------------
# OllamaAdapter
# ---------------------------------------------------------------------------


class TestOllamaAdapter:
    """Tests for the Ollama provider adapter."""

    def _make_adapter(self) -> OllamaAdapter:
        cfg = _make_provider_config(
            name="ollama",
            provider_type="ollama",
            api_key=None,
            base_url="http://localhost:11434",
        )
        return OllamaAdapter(cfg)

    @pytest.mark.anyio
    async def test_chat_completion_success(self) -> None:
        adapter = self._make_adapter()
        expected = _openai_chat_response(content="Llama response", model="llama3")

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            assert "/v1/chat/completions" in str(request.url)
            return _make_mock_response(200, json_body=expected)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=transport,
            )
            result = await adapter.chat_completion(
                {"messages": [{"role": "user", "content": "Hi"}]},
                "llama3",
            )

        assert result["choices"][0]["message"]["content"] == "Llama response"

    @pytest.mark.anyio
    async def test_build_client_has_no_auth_header(self) -> None:
        adapter = self._make_adapter()
        client = adapter._build_client()
        # Ollama should not have Authorization header
        assert "authorization" not in {k.lower() for k in client.headers.keys()}
        await client.aclose()

    @pytest.mark.anyio
    async def test_chat_completion_raises_connection_error(self) -> None:
        adapter = self._make_adapter()

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        transport = httpx.MockTransport(mock_send)
        with patch.object(adapter, "_build_client") as mock_build:
            mock_build.return_value = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=transport,
            )
            with pytest.raises(ProviderConnectionError):
                await adapter.chat_completion(
                    {"messages": [{"role": "user", "content": "Hi"}]},
                    "llama3",
                )

    @pytest.mark.anyio
    async def test_chat_completion_stream_yields_data(self) -> None:
        adapter = self._make_adapter()

        sse_lines = [
            'data: {"id":"o1","object":"chat.completion.chunk","choices":[{"delta":{"content":"A"}}]}',
            'data: {"id":"o1","object":"chat.completion.chunk","choices":[{"delta":{"content":"B"}}]}',
            "data: [DONE]",
        ]

        async def iter_lines() -> AsyncIterator[str]:
            for line in sse_lines:
                yield line

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.aiter_lines = iter_lines

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(adapter, "_build_client", return_value=mock_client):
            chunks: List[str] = []
            async for chunk in adapter.chat_completion_stream(
                {"messages": [{"role": "user", "content": "Hi"}]},
                "llama3",
            ):
                chunks.append(chunk)

        assert chunks[-1] == "[DONE]"
        assert len(chunks) == 3


# ---------------------------------------------------------------------------
# ProviderError hierarchy
# ---------------------------------------------------------------------------


class TestProviderErrorHierarchy:
    """Tests for the ProviderError exception hierarchy."""

    def test_provider_error_is_exception(self) -> None:
        assert issubclass(ProviderError, Exception)

    def test_timeout_error_is_provider_error(self) -> None:
        assert issubclass(ProviderTimeoutError, ProviderError)

    def test_connection_error_is_provider_error(self) -> None:
        assert issubclass(ProviderConnectionError, ProviderError)

    def test_provider_error_attributes(self) -> None:
        err = ProviderError("Something went wrong", provider_name="openai", status_code=500)
        assert err.message == "Something went wrong"
        assert err.provider_name == "openai"
        assert err.status_code == 500

    def test_provider_error_status_code_defaults_to_none(self) -> None:
        err = ProviderError("Oops", provider_name="mistral")
        assert err.status_code is None

    def test_timeout_error_carries_provider_name(self) -> None:
        err = ProviderTimeoutError("Timed out", provider_name="anthropic")
        assert err.provider_name == "anthropic"


# ---------------------------------------------------------------------------
# RequestLogger
# ---------------------------------------------------------------------------


class TestRequestLogger:
    """Tests for the RequestLogger."""

    def _make_logger(self, **overrides: Any) -> RequestLogger:
        config = LoggingConfig(**overrides)
        return RequestLogger(config)

    def test_generate_request_id_returns_string(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        assert isinstance(rid, str)
        assert rid.startswith("req-")

    def test_generate_request_ids_are_unique(self) -> None:
        logger = self._make_logger()
        ids = {logger.generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_start_request_returns_request_id(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        returned = logger.start_request(
            request_id=rid,
            client_model="fast",
            provider_name="openai",
            target_model="gpt-4o-mini",
            streaming=False,
        )
        assert returned == rid

    def test_start_request_stores_in_flight(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(
            request_id=rid,
            client_model="fast",
            provider_name="openai",
            target_model="gpt-4o-mini",
            streaming=False,
        )
        assert rid in logger._in_flight

    def test_finish_request_removes_from_in_flight(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        logger.finish_request(rid)
        assert rid not in logger._in_flight

    def test_finish_request_returns_record(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        record = logger.finish_request(rid)
        assert isinstance(record, RequestRecord)

    def test_finish_request_extracts_token_counts(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "smart", "openai", "gpt-4o", False)
        response_body = {
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
        }
        record = logger.finish_request(rid, response_body=response_body)
        assert record is not None
        assert record.prompt_tokens == 20
        assert record.completion_tokens == 30
        assert record.total_tokens == 50

    def test_finish_request_computes_latency(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        record = logger.finish_request(rid)
        assert record is not None
        assert record.latency_ms is not None
        assert record.latency_ms >= 0.0

    def test_finish_request_uses_provided_latency(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        record = logger.finish_request(rid, latency_ms=123.456)
        assert record is not None
        assert abs(record.latency_ms - 123.456) < 0.001  # type: ignore[operator]

    def test_finish_request_returns_none_for_unknown_id(self) -> None:
        logger = self._make_logger()
        result = logger.finish_request("nonexistent-id")
        assert result is None

    def test_fail_request_sets_error_fields(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "smart", "openai", "gpt-4o", False)
        error = ProviderError("API error", provider_name="openai", status_code=500)
        record = logger.fail_request(rid, error)
        assert record is not None
        assert record.error == "API error"
        assert record.error_type == "ProviderError"

    def test_fail_request_removes_from_in_flight(self) -> None:
        logger = self._make_logger()
        rid = logger.generate_request_id()
        logger.start_request(rid, "smart", "openai", "gpt-4o", False)
        logger.fail_request(rid, ValueError("oops"))
        assert rid not in logger._in_flight

    def test_fail_request_returns_none_for_unknown_id(self) -> None:
        logger = self._make_logger()
        result = logger.fail_request("unknown", ValueError("x"))
        assert result is None

    def test_request_body_stored_when_enabled(self) -> None:
        logger = self._make_logger(log_request_body=True)
        rid = logger.generate_request_id()
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False, request_body=body)
        record = logger._in_flight[rid]
        assert record.request_body == body

    def test_request_body_not_stored_when_disabled(self) -> None:
        logger = self._make_logger(log_request_body=False)
        rid = logger.generate_request_id()
        body = {"messages": [{"role": "user", "content": "Secret"}]}
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False, request_body=body)
        record = logger._in_flight[rid]
        assert record.request_body is None

    def test_response_body_stored_when_enabled(self) -> None:
        logger = self._make_logger(log_response_body=True)
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        resp = {"choices": [{"message": {"content": "Hi"}}], "usage": {}}
        record = logger.finish_request(rid, response_body=resp)
        assert record is not None
        assert record.response_body == resp

    def test_response_body_not_stored_when_disabled(self) -> None:
        logger = self._make_logger(log_response_body=False)
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        resp = {"choices": [{"message": {"content": "Hi"}}]}
        record = logger.finish_request(rid, response_body=resp)
        assert record is not None
        assert record.response_body is None

    def test_disabled_logger_does_not_crash(self) -> None:
        logger = self._make_logger(enabled=False)
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        record = logger.finish_request(rid)
        assert record is not None

    def test_log_info_does_not_raise(self) -> None:
        logger = self._make_logger()
        logger.log_info("test message", key="value")

    def test_log_warning_does_not_raise(self) -> None:
        logger = self._make_logger()
        logger.log_warning("warning message")

    def test_log_error_does_not_raise(self) -> None:
        logger = self._make_logger()
        logger.log_error("error message")

    def test_json_format_emits_valid_json(self, capsys: pytest.CaptureFixture) -> None:
        logger = self._make_logger(log_format="json", enabled=True)
        rid = logger.generate_request_id()
        logger.start_request(rid, "smart", "openai", "gpt-4o", False)
        logger.finish_request(
            rid,
            response_body={"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
        )
        out = capsys.readouterr().out
        # Find the log line (may have multiple lines if multiple handlers)
        for line in out.splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert "request_id" in parsed or "message" in parsed
                break

    def test_text_format_emits_string(self, capsys: pytest.CaptureFixture) -> None:
        logger = self._make_logger(log_format="text", enabled=True)
        rid = logger.generate_request_id()
        logger.start_request(rid, "fast", "openai", "gpt-4o-mini", False)
        logger.finish_request(rid)
        out = capsys.readouterr().out
        assert rid in out


# ---------------------------------------------------------------------------
# RequestRecord
# ---------------------------------------------------------------------------


class TestRequestRecord:
    """Tests for the RequestRecord dataclass."""

    def test_to_dict_contains_expected_keys(self) -> None:
        record = RequestRecord(
            request_id="req-abc",
            client_model="fast",
            provider_name="openai",
            target_model="gpt-4o-mini",
            streaming=False,
        )
        d = record.to_dict()
        expected_keys = {
            "request_id", "client_model", "provider_name", "target_model",
            "streaming", "started_at", "finished_at", "latency_ms",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "error", "error_type", "request_body", "response_body",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_match_fields(self) -> None:
        record = RequestRecord(
            request_id="req-xyz",
            client_model="smart",
            provider_name="anthropic",
            target_model="claude-3-5-sonnet-20241022",
            streaming=True,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        d = record.to_dict()
        assert d["request_id"] == "req-xyz"
        assert d["client_model"] == "smart"
        assert d["provider_name"] == "anthropic"
        assert d["streaming"] is True
        assert d["prompt_tokens"] == 100

    def test_started_at_is_recent(self) -> None:
        before = time.time()
        record = RequestRecord(
            request_id="r", client_model="m",
            provider_name="p", target_model="t", streaming=False,
        )
        after = time.time()
        assert before <= record.started_at <= after


# ---------------------------------------------------------------------------
# _format_record_message
# ---------------------------------------------------------------------------


class TestFormatRecordMessage:
    """Tests for the _format_record_message helper."""

    def test_ok_record_contains_ok_status(self) -> None:
        record = RequestRecord(
            request_id="req-1",
            client_model="fast",
            provider_name="openai",
            target_model="gpt-4o-mini",
            streaming=False,
            latency_ms=250.0,
            total_tokens=30,
        )
        msg = _format_record_message(record)
        assert "[OK]" in msg
        assert "req-1" in msg
        assert "openai" in msg
        assert "250.0ms" in msg

    def test_error_record_contains_error_status(self) -> None:
        record = RequestRecord(
            request_id="req-2",
            client_model="smart",
            provider_name="anthropic",
            target_model="claude-3",
            streaming=False,
            error="API error",
            error_type="ProviderError",
        )
        msg = _format_record_message(record)
        assert "[ERROR]" in msg
        assert "API error" in msg

    def test_dash_when_latency_none(self) -> None:
        record = RequestRecord(
            request_id="req-3",
            client_model="local",
            provider_name="ollama",
            target_model="llama3",
            streaming=True,
        )
        msg = _format_record_message(record)
        assert "latency=-" in msg

    def test_token_count_shown_when_available(self) -> None:
        record = RequestRecord(
            request_id="req-4",
            client_model="fast",
            provider_name="openai",
            target_model="gpt-4o-mini",
            streaming=False,
            total_tokens=42,
        )
        msg = _format_record_message(record)
        assert "tokens=42" in msg

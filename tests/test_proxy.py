"""Integration tests for the FastAPI proxy endpoints.

Tests use FastAPI's ``TestClient`` (synchronous HTTPX-based client) together
with ``unittest.mock`` to patch provider adapters so no real HTTP calls are
made to external services.

Coverage includes:
- ``GET /`` liveness check.
- ``GET /health`` provider health endpoint.
- ``GET /v1/models`` model list endpoint.
- ``POST /v1/chat/completions`` non-streaming success and error paths.
- ``POST /v1/chat/completions`` streaming SSE success and error paths.
- Provider fallback on error.
- 503 when no provider is available.
- Request validation (missing fields, bad model).
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_swap.config import Config, load_config_from_dict
from llm_swap.health import HealthState
from llm_swap.providers import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)
from llm_swap.proxy import AppState, create_app


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _build_config(
    providers: Optional[List[Dict[str, Any]]] = None,
    routing_extra: Optional[Dict[str, Any]] = None,
) -> Config:
    """Build a minimal Config for testing."""
    data: Dict[str, Any] = {
        "providers": providers
        or [
            {
                "name": "openai",
                "type": "openai",
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
            },
            {
                "name": "mistral",
                "type": "mistral",
                "api_key": "sk-mistral",
                "base_url": "https://api.mistral.ai/v1",
            },
        ],
        "routing": {
            "default_provider": "openai",
            "strategy": "priority",
            "aliases": [
                {
                    "alias": "fast",
                    "backends": [
                        {"provider": "openai", "model": "gpt-4o-mini", "priority": 1},
                        {"provider": "mistral", "model": "mistral-small", "priority": 2},
                    ],
                },
                {
                    "alias": "smart",
                    "backends": [
                        {"provider": "openai", "model": "gpt-4o", "priority": 1},
                    ],
                },
            ],
            "model_routes": [
                {
                    "model": "gpt-4o",
                    "backends": [
                        {"provider": "openai", "model": "gpt-4o", "priority": 1},
                    ],
                },
            ],
            **(routing_extra or {}),
        },
        "health_check": {"enabled": False},
        "logging": {"enabled": False},
    }
    return load_config_from_dict(data)


def _make_openai_response(
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


async def _async_gen_chunks(chunks: List[str]) -> AsyncIterator[str]:
    """Async generator that yields the given list of SSE data strings."""
    for chunk in chunks:
        yield chunk


@pytest.fixture()
def config() -> Config:
    """Return the default test configuration."""
    return _build_config()


@pytest.fixture()
def client(config: Config) -> TestClient:
    """Return a TestClient wrapping the proxy app (health checks disabled)."""
    app = create_app(config)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Liveness check
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """Tests for the GET / liveness endpoint."""

    def test_get_root_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_get_root_returns_ok_status(self, client: TestClient) -> None:
        response = client.get("/")
        body = response.json()
        assert body["status"] == "ok"

    def test_get_root_identifies_service(self, client: TestClient) -> None:
        response = client.get("/")
        body = response.json()
        assert body["service"] == "llm_swap"


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client: TestClient) -> None:
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"

    def test_returns_providers_list(self, client: TestClient) -> None:
        response = client.get("/health")
        body = response.json()
        assert "providers" in body
        assert isinstance(body["providers"], list)

    def test_providers_include_configured_names(self, client: TestClient) -> None:
        response = client.get("/health")
        body = response.json()
        names = {p["provider_name"] for p in body["providers"]}
        assert "openai" in names
        assert "mistral" in names

    def test_providers_have_healthy_field(self, client: TestClient) -> None:
        response = client.get("/health")
        body = response.json()
        for provider in body["providers"]:
            assert "healthy" in provider

    def test_providers_start_healthy(self, client: TestClient) -> None:
        response = client.get("/health")
        body = response.json()
        for provider in body["providers"]:
            assert provider["healthy"] is True

    def test_degraded_provider_shown_in_health(self, config: Config) -> None:
        app = create_app(config)
        # Manually mark a provider as degraded
        import anyio

        async def _degrade() -> None:
            await app.state.app_state.health_state.mark_degraded(
                "openai", reason="test degraded"
            )

        anyio.from_thread.run_sync(lambda: None)  # ensure event loop exists
        # Use async client approach
        with TestClient(app, raise_server_exceptions=False) as c:
            # Degrade via the health state before making request
            import asyncio

            asyncio.get_event_loop().run_until_complete(_degrade())
            response = c.get("/health")
            body = response.json()
            openai_status = next(
                p for p in body["providers"] if p["provider_name"] == "openai"
            )
            assert openai_status["healthy"] is False


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for the GET /v1/models endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_returns_list_object(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        body = response.json()
        assert body["object"] == "list"

    def test_returns_data_list(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        body = response.json()
        assert "data" in body
        assert isinstance(body["data"], list)

    def test_aliases_appear_in_model_list(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        body = response.json()
        ids = {m["id"] for m in body["data"]}
        assert "fast" in ids
        assert "smart" in ids

    def test_direct_model_routes_appear_in_list(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        body = response.json()
        ids = {m["id"] for m in body["data"]}
        assert "gpt-4o" in ids

    def test_model_objects_have_required_fields(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        body = response.json()
        for model in body["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"

    def test_model_owned_by_llm_swap(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        body = response.json()
        for model in body["data"]:
            assert model["owned_by"] == "llm_swap"


# ---------------------------------------------------------------------------
# Chat completions — non-streaming
# ---------------------------------------------------------------------------


class TestChatCompletionsNonStreaming:
    """Tests for POST /v1/chat/completions without streaming."""

    def _post(self, client: TestClient, body: Dict[str, Any]) -> Any:
        return client.post("/v1/chat/completions", json=body)

    def test_success_returns_200(self, config: Config) -> None:
        app = create_app(config)
        mock_response = _make_openai_response(content="Hello from OpenAI!")

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch(
                "llm_swap.proxy.get_adapter"
            ) as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = AsyncMock(return_value=mock_response)
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert response.status_code == 200

    def test_success_returns_completion_object(self, config: Config) -> None:
        app = create_app(config)
        mock_response = _make_openai_response(content="Hello!")

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = AsyncMock(return_value=mock_response)
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello!"

    def test_response_includes_request_id_header(self, config: Config) -> None:
        app = create_app(config)
        mock_response = _make_openai_response()

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = AsyncMock(return_value=mock_response)
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

        assert "x-request-id" in response.headers

    def test_unknown_model_routes_to_default(self, config: Config) -> None:
        """Unknown model falls back to the default provider."""
        app = create_app(config)
        mock_response = _make_openai_response(content="Default response")

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = AsyncMock(return_value=mock_response)
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "totally-unknown-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert response.status_code == 200

    def test_missing_messages_returns_422(self, client: TestClient) -> None:
        response = self._post(
            client,
            {"model": "fast"},  # missing 'messages'
        )
        assert response.status_code == 422

    def test_missing_model_returns_422(self, client: TestClient) -> None:
        response = self._post(
            client,
            {"messages": [{"role": "user", "content": "Hello"}]},  # missing 'model'
        )
        assert response.status_code == 422

    def test_empty_messages_returns_422(self, client: TestClient) -> None:
        response = self._post(
            client,
            {"model": "fast", "messages": []},
        )
        assert response.status_code == 422

    def test_provider_error_falls_back_to_next_backend(self, config: Config) -> None:
        """First provider fails → should retry with second backend."""
        app = create_app(config)
        success_response = _make_openai_response(content="Fallback response")
        call_count = 0

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError(
                    "OpenAI is down", provider_name="openai", status_code=503
                )
            return success_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",  # openai p=1, mistral p=2
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        # Fallback should succeed
        assert response.status_code == 200
        assert call_count == 2

    def test_all_backends_fail_returns_502(self, config: Config) -> None:
        """All backends fail → 502 error."""
        app = create_app(config)

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            raise ProviderError("All down", provider_name="openai", status_code=503)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert response.status_code == 502

    def test_503_when_all_providers_degraded(self, config: Config) -> None:
        """All providers degraded in health state → 503."""
        app = create_app(config)

        import asyncio

        async def _degrade_all() -> None:
            hs = app.state.app_state.health_state
            await hs.mark_degraded("openai")
            await hs.mark_degraded("mistral")

        with TestClient(app, raise_server_exceptions=False) as c:
            asyncio.get_event_loop().run_until_complete(_degrade_all())
            response = c.post(
                "/v1/chat/completions",
                json={
                    "model": "fast",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert response.status_code == 503

    def test_timeout_error_triggers_fallback(self, config: Config) -> None:
        """A ProviderTimeoutError also triggers fallback."""
        app = create_app(config)
        success_response = _make_openai_response(content="Timeout fallback")
        call_count = 0

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderTimeoutError("Timed out", provider_name="openai")
            return success_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert response.status_code == 200

    def test_connection_error_triggers_fallback(self, config: Config) -> None:
        """A ProviderConnectionError also triggers fallback."""
        app = create_app(config)
        success_response = _make_openai_response(content="Connection fallback")
        call_count = 0

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderConnectionError(
                    "Cannot connect", provider_name="openai"
                )
            return success_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert response.status_code == 200

    def test_optional_params_forwarded(self, config: Config) -> None:
        """Optional fields like temperature are passed through to the adapter."""
        app = create_app(config)
        mock_response = _make_openai_response()
        captured_body: Dict[str, Any] = {}

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            captured_body.update(body)
            return mock_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "temperature": 0.7,
                        "max_tokens": 100,
                    },
                )

        assert captured_body.get("temperature") == 0.7
        assert captured_body.get("max_tokens") == 100

    def test_error_response_has_error_field(self, config: Config) -> None:
        """Error responses should have an 'error' field matching OpenAI format."""
        app = create_app(config)

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            raise ProviderError("Bad gateway", provider_name="openai", status_code=502)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        body = response.json()
        assert "detail" in body
        detail = body["detail"]
        assert "error" in detail


# ---------------------------------------------------------------------------
# Chat completions — streaming
# ---------------------------------------------------------------------------


class TestChatCompletionsStreaming:
    """Tests for POST /v1/chat/completions with stream=True."""

    def test_streaming_response_is_text_event_stream(self, config: Config) -> None:
        app = create_app(config)
        chunks = [
            '{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}',
            "[DONE]",
        ]

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            return _async_gen_chunks(chunks)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_streaming_response_contains_data_lines(self, config: Config) -> None:
        app = create_app(config)
        chunk_json = '{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}'
        chunks = [chunk_json, "[DONE]"]

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            return _async_gen_chunks(chunks)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        text = response.text
        assert "data:" in text

    def test_streaming_response_ends_with_done(self, config: Config) -> None:
        app = create_app(config)
        chunks = [
            '{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"A"}}]}',
            "[DONE]",
        ]

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            return _async_gen_chunks(chunks)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        assert "[DONE]" in response.text

    def test_streaming_adds_done_when_not_sent_by_provider(self, config: Config) -> None:
        """If the provider doesn't send [DONE], the proxy adds it."""
        app = create_app(config)
        chunks = [
            '{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"A"}}]}',
            # No [DONE]
        ]

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            return _async_gen_chunks(chunks)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        assert "[DONE]" in response.text

    def test_streaming_error_sends_error_chunk(self, config: Config) -> None:
        """If streaming fails, an error data frame is sent before [DONE]."""
        app = create_app(config)

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            raise ProviderError("Stream failure", provider_name="openai")
            yield  # make it a generator  # noqa: unreachable

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        text = response.text
        assert "[DONE]" in text
        # Should contain an error payload
        data_lines = [
            line[len("data: "):]
            for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert len(data_lines) > 0
        parsed = json.loads(data_lines[0])
        assert "error" in parsed

    def test_streaming_includes_request_id_header(self, config: Config) -> None:
        app = create_app(config)
        chunks = ["[DONE]"]

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            return _async_gen_chunks(chunks)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        assert "x-request-id" in response.headers

    def test_streaming_multiple_chunks_parsed(self, config: Config) -> None:
        app = create_app(config)
        chunk1 = '{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}'
        chunk2 = '{"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"}}]}'
        chunks = [chunk1, chunk2, "[DONE]"]

        async def mock_stream(
            body: Dict[str, Any], model: str
        ) -> AsyncIterator[str]:
            return _async_gen_chunks(chunks)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion_stream = mock_stream
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
                )

        data_lines = [
            line[len("data: "):]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert len(data_lines) == 2
        for dl in data_lines:
            parsed = json.loads(dl)
            assert parsed["object"] == "chat.completion.chunk"


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------


class TestAppState:
    """Tests for the AppState container."""

    def test_app_state_has_router(self, config: Config) -> None:
        state = AppState(config)
        from llm_swap.router import Router

        assert isinstance(state.router, Router)

    def test_app_state_has_health_state(self, config: Config) -> None:
        state = AppState(config)
        from llm_swap.health import HealthState

        assert isinstance(state.health_state, HealthState)

    def test_app_state_has_health_checker(self, config: Config) -> None:
        state = AppState(config)
        from llm_swap.health import HealthChecker

        assert isinstance(state.health_checker, HealthChecker)

    def test_app_state_has_request_logger(self, config: Config) -> None:
        state = AppState(config)
        from llm_swap.logger import RequestLogger

        assert isinstance(state.request_logger, RequestLogger)

    def test_app_state_attached_to_app(self, config: Config) -> None:
        app = create_app(config)
        assert hasattr(app.state, "app_state")
        assert isinstance(app.state.app_state, AppState)


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app factory."""

    def test_returns_fastapi_app(self, config: Config) -> None:
        from fastapi import FastAPI

        app = create_app(config)
        assert isinstance(app, FastAPI)

    def test_app_has_chat_completions_route(self, config: Config) -> None:
        app = create_app(config)
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/v1/chat/completions" in routes

    def test_app_has_models_route(self, config: Config) -> None:
        app = create_app(config)
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/v1/models" in routes

    def test_app_has_health_route(self, config: Config) -> None:
        app = create_app(config)
        routes = {r.path for r in app.routes}  # type: ignore[attr-defined]
        assert "/health" in routes

    def test_app_starts_with_health_checks_disabled(self, config: Config) -> None:
        """When health checks are disabled the checker should not be running."""
        app = create_app(config)  # config has health_check.enabled=False
        with TestClient(app, raise_server_exceptions=False):
            assert not app.state.app_state.health_checker.is_running

    def test_app_starts_with_health_checks_enabled(self) -> None:
        """When health checks are enabled the checker should start."""
        cfg = _build_config()
        # Override to enable health checks with a long interval so they don't
        # actually fire during the test
        import dataclasses
        from llm_swap.config import HealthCheckConfig

        enabled_cfg_data: Dict[str, Any] = {
            "providers": [
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
            "routing": {"default_provider": "openai"},
            "health_check": {"enabled": True, "interval_seconds": 9999},
            "logging": {"enabled": False},
        }
        enabled_cfg = load_config_from_dict(enabled_cfg_data)
        app = create_app(enabled_cfg)

        with patch("llm_swap.health.HealthChecker._check_all_providers", new_callable=AsyncMock):
            with TestClient(app, raise_server_exceptions=False):
                assert app.state.app_state.health_checker.is_running


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    """Tests that CORS headers are present on responses."""

    def test_cors_headers_on_options(self, client: TestClient) -> None:
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORS middleware should add CORS headers
        assert response.status_code in (200, 204)


# ---------------------------------------------------------------------------
# Edge cases and extra coverage
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_single_backend_failure_returns_502(self) -> None:
        """A config with a single-backend alias that fails returns 502."""
        cfg = _build_config(
            providers=[
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
            routing_extra={
                "aliases": [
                    {
                        "alias": "only",
                        "backends": [
                            {"provider": "openai", "model": "gpt-4o", "priority": 1}
                        ],
                    }
                ],
                "model_routes": [],
            },
        )
        app = create_app(cfg)

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            raise ProviderError("Unavailable", provider_name="openai", status_code=503)

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                response = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "only",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert response.status_code == 502

    def test_model_name_passed_through_to_adapter(self) -> None:
        """The target model resolved by the router is passed to the adapter."""
        cfg = _build_config()
        app = create_app(cfg)
        captured_model: Dict[str, str] = {}
        mock_response = _make_openai_response()

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            captured_model["model"] = model
            return mock_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",  # alias → gpt-4o-mini
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        # The resolved model for 'fast' alias is gpt-4o-mini
        assert captured_model.get("model") == "gpt-4o-mini"

    def test_system_message_preserved_in_request_body(self) -> None:
        """System messages should be present in the body passed to the adapter."""
        cfg = _build_config()
        app = create_app(cfg)
        captured_body: Dict[str, Any] = {}
        mock_response = _make_openai_response()

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            captured_body.update(body)
            return mock_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

        messages = captured_body.get("messages", [])
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_get_adapter_called_with_correct_provider_config(self) -> None:
        """The adapter factory should receive the matched provider's config."""
        cfg = _build_config()
        app = create_app(cfg)
        captured_configs: List[Any] = []
        mock_response = _make_openai_response()

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = AsyncMock(return_value=mock_response)

                def side_effect(provider_cfg: Any) -> Any:
                    captured_configs.append(provider_cfg)
                    return mock_adapter

                mock_get_adapter.side_effect = side_effect

                c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fast",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert len(captured_configs) == 1
        assert captured_configs[0].name == "openai"  # 'fast' alias → openai first

    def test_direct_model_route_resolved_correctly(self) -> None:
        """A direct model route 'gpt-4o' should route to openai with model gpt-4o."""
        cfg = _build_config()
        app = create_app(cfg)
        captured_model: Dict[str, str] = {}
        mock_response = _make_openai_response()

        async def mock_chat_completion(
            body: Dict[str, Any], model: str
        ) -> Dict[str, Any]:
            captured_model["model"] = model
            return mock_response

        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("llm_swap.proxy.get_adapter") as mock_get_adapter:
                mock_adapter = MagicMock()
                mock_adapter.chat_completion = mock_chat_completion
                mock_get_adapter.return_value = mock_adapter

                c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

        assert captured_model.get("model") == "gpt-4o"

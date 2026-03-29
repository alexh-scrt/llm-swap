"""FastAPI proxy application for llm_swap.

This module exposes the core HTTP API for the llm_swap reverse proxy:

- ``POST /v1/chat/completions`` — OpenAI-compatible chat completions endpoint
  supporting both regular (JSON) and streaming (SSE) responses.
- ``GET /v1/models`` — Returns a list of available model aliases and direct
  model routes configured in the YAML file.
- ``GET /health`` — Returns the current health status of all providers.
- ``GET /`` — Simple liveness check.

The application wires together:

- :class:`~llm_swap.router.Router` for backend selection and fallback.
- :func:`~llm_swap.providers.get_adapter` for per-provider HTTP translation.
- :class:`~llm_swap.health.HealthState` for health-aware routing.
- :class:`~llm_swap.logger.RequestLogger` for structured request logging.

Typical usage (programmatic)::

    from llm_swap.config import load_config
    from llm_swap.proxy import create_app

    config = load_config("config.yaml")
    app = create_app(config)

Or via the CLI::

    llm_swap serve --config config.yaml
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_swap.config import Config
from llm_swap.health import HealthChecker, HealthState
from llm_swap.logger import RequestLogger
from llm_swap.providers import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    get_adapter,
)
from llm_swap.router import NoAvailableProviderError, Router


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in an OpenAI-format chat conversation."""

    role: str = Field(description="Message role: system, user, or assistant.")
    content: str = Field(description="Text content of the message.")
    name: Optional[str] = Field(default=None, description="Optional participant name.")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str = Field(description="Model name or alias to route the request to.")
    messages: List[ChatMessage] = Field(
        min_length=1, description="Conversation messages."
    )
    stream: bool = Field(default=False, description="Enable SSE streaming.")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Any] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    user: Optional[str] = Field(default=None)
    # Allow arbitrary extra fields so clients can pass provider-specific params
    model_config = {"extra": "allow"}


class ModelObject(BaseModel):
    """An entry in the OpenAI-compatible model list response."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm_swap"


class ModelListResponse(BaseModel):
    """Response body for ``GET /v1/models``."""

    object: str = "list"
    data: List[ModelObject]


# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------


class AppState:
    """Holds shared application-level objects across requests.

    Attributes:
        config: The loaded configuration.
        router: The routing engine.
        health_state: The shared provider health map.
        health_checker: The background health-check task.
        request_logger: The structured request logger.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.router = Router(config)
        self.health_state = HealthState(config.providers)
        self.health_checker = HealthChecker(config, self.health_state)
        self.request_logger = RequestLogger(config.logging)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application.

    Sets up the lifespan context (starts/stops the health checker), registers
    all route handlers, adds CORS middleware, and attaches the
    :class:`AppState` to ``app.state``.

    Args:
        config: A fully validated :class:`~llm_swap.config.Config` instance.

    Returns:
        A configured :class:`fastapi.FastAPI` application ready to serve.
    """
    app_state = AppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # type: ignore[misc]
        """FastAPI lifespan: start health checker on startup, stop on shutdown."""
        app_state.request_logger.log_info(
            "llm_swap proxy starting up",
            host=config.server.host,
            port=config.server.port,
        )
        await app_state.health_checker.start()
        try:
            yield
        finally:
            await app_state.health_checker.stop()
            app_state.request_logger.log_info("llm_swap proxy shut down.")

    app = FastAPI(
        title="llm_swap",
        description=(
            "A local reverse-proxy that routes OpenAI-compatible requests "
            "to any configured LLM backend."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Attach shared state so route handlers can access it
    app.state.app_state = app_state

    # CORS — permissive by default so SDK clients work from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Register routes
    # ------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, str]:
        """Liveness check."""
        return {"status": "ok", "service": "llm_swap"}

    @app.get("/health", response_model=Dict[str, Any])
    async def health_endpoint() -> Dict[str, Any]:
        """Return the current health status of all configured providers."""
        statuses = app_state.health_state.all_statuses()
        return {
            "status": "ok",
            "providers": [s.to_dict() for s in statuses],
        }

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        """Return all configured model aliases and direct model routes.

        This endpoint enumerates the aliases and model routes defined in the
        YAML configuration so that OpenAI-SDK clients can discover available
        ``model`` values.
        """
        model_ids: List[str] = []

        # Add aliases
        for alias_route in config.routing.aliases:
            model_ids.append(alias_route.alias)

        # Add direct model routes (deduplicated)
        for model_route in config.routing.model_routes:
            if model_route.model not in model_ids:
                model_ids.append(model_route.model)

        # Also expose provider names as virtual models (for direct pass-through)
        for provider in config.providers:
            if provider.name not in model_ids:
                model_ids.append(provider.name)

        data = [
            ModelObject(id=mid, owned_by="llm_swap", created=int(time.time()))
            for mid in model_ids
        ]
        return ModelListResponse(object="list", data=data)

    @app.post("/v1/chat/completions")
    async def chat_completions(  # noqa: C901  (complexity OK here)
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Response:
        """Proxy a chat completion request to the best available backend.

        Supports both streaming (SSE) and non-streaming responses.  The
        router selects a provider based on the ``model`` field, health state,
        and configured routing rules.  On provider error the proxy
        automatically retries with the next available backend.

        Args:
            request: Parsed :class:`ChatCompletionRequest` body.
            raw_request: The raw :class:`fastapi.Request` for header access.

        Returns:
            A :class:`fastapi.responses.JSONResponse` for non-streaming
            requests or a :class:`fastapi.responses.StreamingResponse` for
            streaming (``stream=True``) requests.

        Raises:
            HTTPException 400: If the request body is malformed.
            HTTPException 503: If no healthy backend is available.
            HTTPException 502: If all backends return errors.
        """
        router: Router = app_state.router
        health_state = app_state.health_state
        req_logger: RequestLogger = app_state.request_logger

        # Convert Pydantic model to plain dict for the adapters
        request_body = request.model_dump(exclude_none=True)

        request_id = req_logger.generate_request_id()
        client_model = request.model
        streaming = request.stream

        # Collect health snapshot once per request
        health_snapshot = health_state.snapshot()

        # Determine all viable backends (ordered by priority)
        try:
            candidates = router.all_candidates(
                client_model,
                health_state=health_snapshot,
            )
        except Exception:
            # Unexpected error in routing — treat as no provider
            candidates = []

        if not candidates:
            req_logger.log_warning(
                "No available providers for model",
                request_id=request_id,
                client_model=client_model,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": f"No available provider for model {client_model!r}.",
                        "type": "service_unavailable",
                        "code": "no_available_provider",
                    }
                },
            )

        # Try each candidate in order, retrying on provider errors
        excluded: Set[str] = set()
        last_error: Optional[ProviderError] = None

        for candidate in candidates:
            if candidate.provider_name in excluded:
                continue

            provider_cfg = candidate.provider_config
            target_model = candidate.model
            adapter = get_adapter(provider_cfg)

            req_logger.start_request(
                request_id=request_id,
                client_model=client_model,
                provider_name=candidate.provider_name,
                target_model=target_model,
                streaming=streaming,
                request_body=request_body,
            )

            if streaming:
                # Return a streaming SSE response
                return StreamingResponse(
                    _stream_completion(
                        adapter=adapter,
                        request_body=request_body,
                        target_model=target_model,
                        request_id=request_id,
                        req_logger=req_logger,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "X-Request-Id": request_id,
                    },
                )

            # Non-streaming path
            start_time = time.monotonic()
            try:
                response_body = await adapter.chat_completion(
                    request_body, target_model
                )
                latency_ms = (time.monotonic() - start_time) * 1000.0
                req_logger.finish_request(
                    request_id=request_id,
                    response_body=response_body,
                    latency_ms=latency_ms,
                )
                return JSONResponse(
                    content=response_body,
                    headers={"X-Request-Id": request_id},
                )

            except (ProviderError, ProviderTimeoutError, ProviderConnectionError) as exc:
                latency_ms = (time.monotonic() - start_time) * 1000.0
                last_error = exc
                excluded.add(candidate.provider_name)
                req_logger.fail_request(
                    request_id=request_id,
                    error=exc,
                    latency_ms=latency_ms,
                )
                req_logger.log_warning(
                    f"Provider {candidate.provider_name!r} failed; trying next backend.",
                    request_id=request_id,
                    provider=candidate.provider_name,
                    error=str(exc),
                )
                # Re-register for the next candidate (different provider/model)
                continue

        # All candidates exhausted
        error_detail: Dict[str, Any] = {
            "error": {
                "message": (
                    f"All backends failed for model {client_model!r}. "
                    f"Last error: {last_error}"
                ),
                "type": "upstream_error",
                "code": "all_backends_failed",
            }
        }
        if last_error is not None and isinstance(last_error.status_code, int):
            # Surface the upstream status code hint
            http_status = 502
        else:
            http_status = 502

        raise HTTPException(status_code=http_status, detail=error_detail)

    return app


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


async def _stream_completion(
    adapter: Any,
    request_body: Dict[str, Any],
    target_model: str,
    request_id: str,
    req_logger: RequestLogger,
) -> AsyncIterator[bytes]:
    """Async generator that proxies SSE chunks from a provider adapter.

    Wraps each data chunk in the SSE wire format (``data: ...\\n\\n``) and
    emits a final ``data: [DONE]\\n\\n`` if the provider doesn't send one.

    Args:
        adapter: A provider adapter instance.
        request_body: The OpenAI-format request body.
        target_model: The model name to forward to the provider.
        request_id: Unique request identifier for logging.
        req_logger: The :class:`~llm_swap.logger.RequestLogger` instance.

    Yields:
        UTF-8 encoded SSE frames.
    """
    start_time = time.monotonic()
    got_done = False
    try:
        async for chunk in adapter.chat_completion_stream(request_body, target_model):
            if chunk == "[DONE]":
                got_done = True
                yield b"data: [DONE]\n\n"
            else:
                yield f"data: {chunk}\n\n".encode("utf-8")

        if not got_done:
            yield b"data: [DONE]\n\n"

        latency_ms = (time.monotonic() - start_time) * 1000.0
        req_logger.finish_request(
            request_id=request_id,
            latency_ms=latency_ms,
        )

    except (ProviderError, ProviderTimeoutError, ProviderConnectionError) as exc:
        latency_ms = (time.monotonic() - start_time) * 1000.0
        req_logger.fail_request(
            request_id=request_id,
            error=exc,
            latency_ms=latency_ms,
        )
        # Send an error chunk in OpenAI error format before closing the stream
        error_payload = json.dumps(
            {
                "error": {
                    "message": str(exc),
                    "type": "upstream_error",
                    "code": "provider_error",
                }
            }
        )
        yield f"data: {error_payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - start_time) * 1000.0
        req_logger.fail_request(
            request_id=request_id,
            error=exc,
            latency_ms=latency_ms,
        )
        error_payload = json.dumps(
            {
                "error": {
                    "message": f"Internal proxy error: {exc}",
                    "type": "internal_error",
                    "code": "proxy_error",
                }
            }
        )
        yield f"data: {error_payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

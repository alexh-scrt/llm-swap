"""Structured request/response logging for llm_swap.

This module provides a :class:`RequestLogger` that records details about each
proxied request including:

- Timestamp and unique request ID.
- The model name requested by the client.
- The provider chosen and the target model sent to that provider.
- Upstream latency in milliseconds.
- Prompt and completion token counts (when available in the response).
- Whether the request was streaming.
- Error information on failure.
- Optionally the full request body and/or response body.

Log records can be written in JSON or human-readable text format to either
stdout or a file (or both).

Typical usage::

    from llm_swap.logger import RequestLogger
    from llm_swap.config import LoggingConfig

    logger = RequestLogger(LoggingConfig())

    record_id = logger.start_request(
        request_id="req-001",
        client_model="fast",
        provider_name="openai",
        target_model="gpt-4o-mini",
        streaming=False,
        request_body={"messages": [...]},
    )

    logger.finish_request(
        request_id="req-001",
        response_body={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        latency_ms=342.1,
    )
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from llm_swap.config import LoggingConfig


# ---------------------------------------------------------------------------
# Log record dataclass
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """Captures all relevant metadata for a single proxied request.

    Attributes:
        request_id: Unique identifier for this request (UUID4).
        client_model: The ``model`` value sent by the client.
        provider_name: The provider that was selected to handle the request.
        target_model: The model name forwarded to the provider.
        streaming: Whether SSE streaming was requested.
        started_at: Unix timestamp (float) when the request was initiated.
        finished_at: Unix timestamp (float) when the response was received,
            or ``None`` if not yet finished.
        latency_ms: Round-trip latency in milliseconds, or ``None``.
        prompt_tokens: Number of prompt tokens reported by the provider.
        completion_tokens: Number of completion tokens reported by the provider.
        total_tokens: Total token count reported by the provider.
        error: Error message if the request failed, otherwise ``None``.
        error_type: Exception class name if the request failed.
        request_body: Full request body (only populated when
            ``log_request_body=True``).
        response_body: Full response body (only populated when
            ``log_response_body=True``).
    """

    request_id: str
    client_model: str
    provider_name: str
    target_model: str
    streaming: bool
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    request_body: Optional[Dict[str, Any]] = None
    response_body: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this record to a plain dictionary.

        Returns:
            A JSON-serializable dictionary representing the record.
        """
        return {
            "request_id": self.request_id,
            "client_model": self.client_model,
            "provider_name": self.provider_name,
            "target_model": self.target_model,
            "streaming": self.streaming,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "error_type": self.error_type,
            "request_body": self.request_body,
            "response_body": self.response_body,
        }


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class RequestLogger:
    """Structured logger for proxied LLM requests.

    Manages in-flight request records and writes finalized records to the
    configured output (stdout and/or a file) in JSON or text format.

    Args:
        config: A :class:`~llm_swap.config.LoggingConfig` instance.
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._in_flight: Dict[str, RequestRecord] = {}
        self._logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        """Configure and return the underlying :class:`logging.Logger`.

        Creates a dedicated logger named ``llm_swap.requests`` with handlers
        for stdout and optionally a log file.

        Returns:
            Configured :class:`logging.Logger` instance.
        """
        logger = logging.getLogger("llm_swap.requests")
        logger.setLevel(logging.DEBUG)
        # Prevent propagation to root logger to avoid duplicate output
        logger.propagate = False
        # Remove any existing handlers (e.g. from previous test runs)
        logger.handlers.clear()

        if not self._config.enabled:
            logger.addHandler(logging.NullHandler())
            return logger

        formatter = self._build_formatter()

        # Always log to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        # Optionally also log to a file
        if self._config.log_file:
            file_path = Path(self._config.log_file)
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(file_path, encoding="utf-8")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as exc:
                # If we cannot open the log file, warn on stderr and continue
                print(
                    f"[llm_swap] WARNING: Cannot open log file {file_path}: {exc}",
                    file=sys.stderr,
                )

        return logger

    def _build_formatter(self) -> logging.Formatter:
        """Return a :class:`logging.Formatter` appropriate for the configured format.

        Returns:
            A :class:`logging.Formatter` for JSON or text output.
        """
        if self._config.log_format == "json":
            return _JsonFormatter()
        return logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_request_id(self) -> str:
        """Generate a new unique request identifier.

        Returns:
            A UUID4 string prefixed with ``req-``.
        """
        return f"req-{uuid.uuid4().hex[:12]}"

    def start_request(
        self,
        request_id: str,
        client_model: str,
        provider_name: str,
        target_model: str,
        streaming: bool,
        request_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new in-flight request and optionally log the request body.

        Args:
            request_id: Unique identifier for this request.  Use
                :meth:`generate_request_id` to create one.
            client_model: Model name sent by the client.
            provider_name: Provider selected for this request.
            target_model: Model name to be sent to the provider.
            streaming: Whether SSE streaming was requested.
            request_body: Full request body dict; stored only if
                ``log_request_body`` is enabled in the config.

        Returns:
            The ``request_id`` (passed through for convenience).
        """
        stored_body: Optional[Dict[str, Any]] = None
        if self._config.log_request_body and request_body is not None:
            stored_body = request_body

        record = RequestRecord(
            request_id=request_id,
            client_model=client_model,
            provider_name=provider_name,
            target_model=target_model,
            streaming=streaming,
            started_at=time.time(),
            request_body=stored_body,
        )
        self._in_flight[request_id] = record
        return request_id

    def finish_request(
        self,
        request_id: str,
        response_body: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
    ) -> Optional[RequestRecord]:
        """Mark a request as successfully completed and emit a log record.

        Token counts are extracted automatically from the ``usage`` field in
        ``response_body`` if present.

        Args:
            request_id: The identifier returned by :meth:`start_request`.
            response_body: The provider's response body dict (OpenAI format).
            latency_ms: Latency in milliseconds; computed from ``started_at``
                if ``None``.

        Returns:
            The finalized :class:`RequestRecord`, or ``None`` if ``request_id``
            was not found (e.g. already finalized).
        """
        record = self._in_flight.pop(request_id, None)
        if record is None:
            return None

        record.finished_at = time.time()
        record.latency_ms = (
            latency_ms
            if latency_ms is not None
            else (record.finished_at - record.started_at) * 1000.0
        )

        if response_body is not None:
            usage = response_body.get("usage", {})
            if usage:
                record.prompt_tokens = usage.get("prompt_tokens")
                record.completion_tokens = usage.get("completion_tokens")
                record.total_tokens = usage.get("total_tokens")
            if self._config.log_response_body:
                record.response_body = response_body

        self._emit(record)
        return record

    def fail_request(
        self,
        request_id: str,
        error: Exception,
        latency_ms: Optional[float] = None,
    ) -> Optional[RequestRecord]:
        """Mark a request as failed and emit a log record.

        Args:
            request_id: The identifier returned by :meth:`start_request`.
            error: The exception that caused the failure.
            latency_ms: Latency in milliseconds; computed from ``started_at``
                if ``None``.

        Returns:
            The finalized :class:`RequestRecord`, or ``None`` if ``request_id``
            was not found.
        """
        record = self._in_flight.pop(request_id, None)
        if record is None:
            return None

        record.finished_at = time.time()
        record.latency_ms = (
            latency_ms
            if latency_ms is not None
            else (record.finished_at - record.started_at) * 1000.0
        )
        record.error = str(error)
        record.error_type = type(error).__name__

        self._emit(record)
        return record

    def log_info(self, message: str, **context: Any) -> None:
        """Emit a free-form informational log message.

        Args:
            message: Human-readable log message.
            **context: Additional key-value pairs to include in the record.
        """
        extra_data = {"message": message, **context}
        self._logger.info(message, extra={"_extra": extra_data})

    def log_warning(self, message: str, **context: Any) -> None:
        """Emit a free-form warning log message.

        Args:
            message: Human-readable log message.
            **context: Additional key-value pairs to include in the record.
        """
        extra_data = {"message": message, **context}
        self._logger.warning(message, extra={"_extra": extra_data})

    def log_error(self, message: str, **context: Any) -> None:
        """Emit a free-form error log message.

        Args:
            message: Human-readable log message.
            **context: Additional key-value pairs to include in the record.
        """
        extra_data = {"message": message, **context}
        self._logger.error(message, extra={"_extra": extra_data})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, record: RequestRecord) -> None:
        """Write a finalized :class:`RequestRecord` to the logger.

        Uses ``WARNING`` level for failed requests and ``INFO`` for successful
        ones so that error records stand out in log files.

        Args:
            record: The finalized record to emit.
        """
        record_dict = record.to_dict()
        level = logging.WARNING if record.error else logging.INFO
        self._logger.log(
            level,
            _format_record_message(record),
            extra={"_extra": record_dict},
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """A :class:`logging.Formatter` that serializes log records as JSON.

    If the ``LogRecord`` has an ``_extra`` attribute (set via the ``extra``
    parameter to :meth:`logging.Logger.log`), its contents are merged into
    the top-level JSON object.  Otherwise only ``time``, ``level``, and
    ``message`` are emitted.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Serialize the log record to a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string.
        """
        base: Dict[str, Any] = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "_extra", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, default=str)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _format_record_message(record: RequestRecord) -> str:
    """Build a human-readable one-liner for a finalized request record.

    Used as the ``message`` field in both JSON and text log output.

    Args:
        record: A finalized :class:`RequestRecord`.

    Returns:
        A compact summary string.
    """
    status = "ERROR" if record.error else "OK"
    latency = f"{record.latency_ms:.1f}ms" if record.latency_ms is not None else "-"
    tokens = "-"
    if record.total_tokens is not None:
        tokens = str(record.total_tokens)
    parts = [
        f"[{status}]",
        f"req={record.request_id}",
        f"model={record.client_model!r}",
        f"provider={record.provider_name!r}",
        f"target_model={record.target_model!r}",
        f"stream={record.streaming}",
        f"latency={latency}",
        f"tokens={tokens}",
    ]
    if record.error:
        parts.append(f"error={record.error!r}")
    return " ".join(parts)

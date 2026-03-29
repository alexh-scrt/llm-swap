"""Background health-check task for llm_swap.

This module implements a background task that periodically pings each
configured provider and maintains a shared health-state mapping that the
router uses to skip degraded backends.

Health state lifecycle:

1. All providers start as **healthy** (assumed available until proven otherwise).
2. A dedicated async loop pings each provider on a configurable interval.
3. After ``unhealthy_threshold`` consecutive failures a provider is marked
   **degraded** (``health_state[name] = False``).
4. After ``healthy_threshold`` consecutive successes a degraded provider is
   restored to **healthy** (``health_state[name] = True``).
5. The shared :class:`HealthState` object is safe to read from multiple
   async/threaded contexts simultaneously.

Each provider type is pinged with a lightweight request:

- **openai / mistral**: ``GET /models`` (standard OpenAI-compatible endpoint).
- **anthropic**: ``GET /v1/models`` with the required ``anthropic-version`` header.
- **ollama**: ``GET /api/tags`` (Ollama-native tag listing endpoint).

Typical usage::

    from llm_swap.config import load_config
    from llm_swap.health import HealthChecker, HealthState

    cfg = load_config("config.yaml")
    state = HealthState(cfg.providers)
    checker = HealthChecker(cfg, state)

    # In an async context (e.g. FastAPI lifespan):
    await checker.start()
    # ... serve requests ...
    await checker.stop()

    # Read current state (e.g. inside the router):
    is_healthy = state.is_healthy("openai")
    all_statuses = state.snapshot()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from llm_swap.config import Config, HealthCheckConfig, ProviderConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProviderStatus
# ---------------------------------------------------------------------------


@dataclass
class ProviderStatus:
    """Runtime health status for a single provider.

    Attributes:
        provider_name: Unique name of the provider.
        healthy: Whether the provider is currently considered available.
        consecutive_failures: Number of consecutive failed health checks.
        consecutive_successes: Number of consecutive successful health checks
            since the last failure (used to restore degraded providers).
        last_check_at: Unix timestamp of the most recent health check attempt,
            or ``None`` if no check has been performed yet.
        last_error: The error message from the most recent failed check, or
            ``None`` if the last check succeeded.
    """

    provider_name: str
    healthy: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_at: Optional[float] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize this status to a plain dictionary.

        Returns:
            A JSON-serializable dictionary.
        """
        return {
            "provider_name": self.provider_name,
            "healthy": self.healthy,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_check_at": self.last_check_at,
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# HealthState
# ---------------------------------------------------------------------------


class HealthState:
    """Shared, concurrency-safe health-state store for all providers.

    :class:`HealthState` is the single source of truth for which providers
    are currently available.  It is written by the background
    :class:`HealthChecker` task and read (via :meth:`snapshot`) by the
    router on every request.

    The implementation uses an :class:`asyncio.Lock` to protect mutations;
    reads via :meth:`is_healthy` and :meth:`snapshot` acquire the same lock
    so they always observe a consistent view.

    Args:
        providers: The list of :class:`~llm_swap.config.ProviderConfig`
            objects from the application configuration.  All providers
            start as healthy.
    """

    def __init__(self, providers: List[ProviderConfig]) -> None:
        self._statuses: Dict[str, ProviderStatus] = {
            p.name: ProviderStatus(provider_name=p.name)
            for p in providers
        }
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    async def is_healthy_async(self, provider_name: str) -> bool:
        """Return whether *provider_name* is currently healthy (async).

        Providers not found in the state are assumed healthy.

        Args:
            provider_name: The unique provider name to check.

        Returns:
            ``True`` if the provider is healthy or unknown, ``False`` if degraded.
        """
        async with self._lock:
            status = self._statuses.get(provider_name)
            return status.healthy if status is not None else True

    def is_healthy(self, provider_name: str) -> bool:
        """Return whether *provider_name* is currently healthy (synchronous).

        This method does **not** acquire the async lock and is safe to call
        from synchronous code (e.g. the router).  Because Python's GIL
        protects dictionary reads, the worst that can happen is reading a
        slightly stale boolean value, which is acceptable for routing.

        Providers not found in the state are assumed healthy.

        Args:
            provider_name: The unique provider name to check.

        Returns:
            ``True`` if the provider is healthy or unknown, ``False`` if degraded.
        """
        status = self._statuses.get(provider_name)
        return status.healthy if status is not None else True

    def snapshot(self) -> Dict[str, bool]:
        """Return a point-in-time copy of the health map.

        The returned dictionary maps provider names to boolean health values
        and is safe to pass to :meth:`~llm_swap.router.Router.route` as the
        ``health_state`` argument.

        Returns:
            A ``{provider_name: is_healthy}`` dictionary.
        """
        return {name: status.healthy for name, status in self._statuses.items()}

    def get_status(self, provider_name: str) -> Optional[ProviderStatus]:
        """Return the full :class:`ProviderStatus` for *provider_name*.

        Args:
            provider_name: The unique provider name.

        Returns:
            The :class:`ProviderStatus` or ``None`` if not found.
        """
        return self._statuses.get(provider_name)

    def all_statuses(self) -> List[ProviderStatus]:
        """Return a list of all :class:`ProviderStatus` objects.

        Returns:
            A list of :class:`ProviderStatus` instances, one per provider.
        """
        return list(self._statuses.values())

    # ------------------------------------------------------------------
    # Write API (used by HealthChecker only)
    # ------------------------------------------------------------------

    async def record_success(
        self,
        provider_name: str,
        healthy_threshold: int,
    ) -> bool:
        """Record a successful health-check ping for *provider_name*.

        If the provider was previously degraded and has now accumulated
        ``healthy_threshold`` consecutive successes, it is restored to healthy.

        Args:
            provider_name: The unique provider name.
            healthy_threshold: Number of consecutive successes required to
                restore a degraded provider.

        Returns:
            ``True`` if the provider transitioned from degraded → healthy,
            ``False`` otherwise.
        """
        async with self._lock:
            status = self._statuses.get(provider_name)
            if status is None:
                return False

            status.last_check_at = time.time()
            status.last_error = None
            status.consecutive_failures = 0
            status.consecutive_successes += 1

            transitioned = False
            if not status.healthy and status.consecutive_successes >= healthy_threshold:
                status.healthy = True
                transitioned = True

            return transitioned

    async def record_failure(
        self,
        provider_name: str,
        error: str,
        unhealthy_threshold: int,
    ) -> bool:
        """Record a failed health-check ping for *provider_name*.

        If the provider has now accumulated ``unhealthy_threshold`` consecutive
        failures it is marked as degraded.

        Args:
            provider_name: The unique provider name.
            error: Human-readable error description.
            unhealthy_threshold: Number of consecutive failures required to
                mark a provider as degraded.

        Returns:
            ``True`` if the provider transitioned from healthy → degraded,
            ``False`` otherwise.
        """
        async with self._lock:
            status = self._statuses.get(provider_name)
            if status is None:
                return False

            status.last_check_at = time.time()
            status.last_error = error
            status.consecutive_successes = 0
            status.consecutive_failures += 1

            transitioned = False
            if status.healthy and status.consecutive_failures >= unhealthy_threshold:
                status.healthy = False
                transitioned = True

            return transitioned

    async def mark_healthy(self, provider_name: str) -> None:
        """Unconditionally mark *provider_name* as healthy.

        Resets all counters.  Useful for initial setup or manual overrides.

        Args:
            provider_name: The unique provider name.
        """
        async with self._lock:
            status = self._statuses.get(provider_name)
            if status is not None:
                status.healthy = True
                status.consecutive_failures = 0
                status.consecutive_successes = 0
                status.last_error = None

    async def mark_degraded(self, provider_name: str, reason: str = "") -> None:
        """Unconditionally mark *provider_name* as degraded.

        Resets all counters.  Useful for manual overrides.

        Args:
            provider_name: The unique provider name.
            reason: Optional reason string stored in ``last_error``.
        """
        async with self._lock:
            status = self._statuses.get(provider_name)
            if status is not None:
                status.healthy = False
                status.consecutive_failures = 0
                status.consecutive_successes = 0
                status.last_error = reason or None


# ---------------------------------------------------------------------------
# Ping helpers
# ---------------------------------------------------------------------------


async def _ping_provider(
    provider: ProviderConfig,
    timeout: float,
) -> None:
    """Send a lightweight HTTP request to verify a provider is reachable.

    The exact endpoint and headers depend on the provider type:

    - **openai** / **mistral**: ``GET {base_url}/models``
    - **anthropic**: ``GET {base_url}/v1/models`` with
      ``anthropic-version: 2023-06-01`` and ``x-api-key`` headers.
    - **ollama**: ``GET {base_url}/api/tags`` (no authentication).

    A response with any 2xx or 4xx status code (except 401 for providers that
    require authentication) is considered a successful ping — it proves the
    server is reachable and responding.  Only network-level errors or 5xx
    responses are treated as failures.

    Args:
        provider: The provider configuration to ping.
        timeout: Maximum seconds to wait for a response.

    Raises:
        Exception: Any exception indicates the ping failed.
    """
    headers: Dict[str, str] = {}

    if provider.type == "anthropic":
        path = "/v1/models"
        if provider.api_key:
            headers["x-api-key"] = provider.api_key
        headers["anthropic-version"] = "2023-06-01"
    elif provider.type == "ollama":
        path = "/api/tags"
    else:
        # openai and mistral both support GET /models
        path = "/models"
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

    # Add any custom provider headers
    headers.update(provider.headers)

    url = f"{provider.base_url}{path}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)

    # Accept any 2xx or 4xx (server is up but may reject the request for
    # auth/param reasons).  Only raise on 5xx or network errors.
    if response.status_code >= 500:
        raise RuntimeError(
            f"Provider {provider.name!r} returned HTTP {response.status_code}"
        )


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------


class HealthChecker:
    """Background task that periodically pings all configured providers.

    :class:`HealthChecker` manages its own :class:`asyncio.Task` and writes
    results into the shared :class:`HealthState`.  It is designed to be
    started once at application startup and stopped on shutdown.

    Example usage inside a FastAPI lifespan::

        async def lifespan(app: FastAPI):
            checker = HealthChecker(config, health_state)
            await checker.start()
            yield
            await checker.stop()

    Args:
        config: The application :class:`~llm_swap.config.Config`.
        state: The shared :class:`HealthState` to update.
    """

    def __init__(self, config: Config, state: HealthState) -> None:
        self._config = config
        self._hc_config: HealthCheckConfig = config.health_check
        self._state = state
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background health-check loop.

        If health checks are disabled in the configuration (``enabled=False``)
        this method returns immediately without starting any background task.

        Calling :meth:`start` on an already-running checker has no effect.
        """
        if not self._hc_config.enabled:
            logger.info(
                "Health checks are disabled; skipping background health-check task."
            )
            return

        if self._task is not None and not self._task.done():
            logger.warning("HealthChecker.start() called but checker is already running.")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(
            self._run_loop(), name="llm_swap.health_checker"
        )
        logger.info(
            "Health checker started (interval=%ds, timeout=%ds, "
            "unhealthy_threshold=%d, healthy_threshold=%d).",
            self._hc_config.interval_seconds,
            self._hc_config.timeout_seconds,
            self._hc_config.unhealthy_threshold,
            self._hc_config.healthy_threshold,
        )

    async def stop(self) -> None:
        """Stop the background health-check loop gracefully.

        Sets the internal stop event and waits for the background task to
        finish.  Safe to call even if the checker was never started or has
        already stopped.
        """
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Health checker task did not finish within 5 s; cancelling.")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Health checker stopped.")

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the background loop is currently active."""
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main health-check loop — runs until :meth:`stop` is called.

        On each iteration all providers are checked concurrently using
        :func:`asyncio.gather`.  The loop then sleeps for
        ``interval_seconds`` (or until the stop event fires).
        """
        while not self._stop_event.is_set():
            await self._check_all_providers()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(self._hc_config.interval_seconds),
                )
            except asyncio.TimeoutError:
                # Normal case: interval elapsed, loop again
                pass

    async def _check_all_providers(self) -> None:
        """Ping all providers concurrently and update the health state.

        Uses :func:`asyncio.gather` with ``return_exceptions=True`` so that
        a failure in one provider check does not prevent others from running.
        """
        providers = self._config.providers
        if not providers:
            return

        tasks = [
            self._check_provider(provider)
            for provider in providers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_provider(self, provider: ProviderConfig) -> None:
        """Ping a single provider and update :attr:`_state` accordingly.

        Args:
            provider: The provider to check.
        """
        timeout = float(self._hc_config.timeout_seconds)
        try:
            await _ping_provider(provider, timeout=timeout)
            transitioned = await self._state.record_success(
                provider.name,
                healthy_threshold=self._hc_config.healthy_threshold,
            )
            if transitioned:
                logger.info(
                    "Provider %r recovered and is now healthy.",
                    provider.name,
                )
            else:
                logger.debug("Health check OK for provider %r.", provider.name)

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            transitioned = await self._state.record_failure(
                provider.name,
                error=error_msg,
                unhealthy_threshold=self._hc_config.unhealthy_threshold,
            )
            if transitioned:
                logger.warning(
                    "Provider %r is now DEGRADED after %d consecutive failures. "
                    "Last error: %s",
                    provider.name,
                    self._hc_config.unhealthy_threshold,
                    error_msg,
                )
            else:
                logger.debug(
                    "Health check FAILED for provider %r: %s",
                    provider.name,
                    error_msg,
                )

    # ------------------------------------------------------------------
    # Manual check
    # ------------------------------------------------------------------

    async def check_now(self) -> Dict[str, bool]:
        """Immediately run a health check for all providers and return the result.

        This method can be called at any time (e.g. on startup before the
        periodic loop is running) to get the current health picture without
        waiting for the next scheduled check.

        Returns:
            A ``{provider_name: is_healthy}`` dictionary reflecting the
            state after this synchronous check completes.
        """
        await self._check_all_providers()
        return self._state.snapshot()

"""Routing engine for llm_swap.

This module implements the routing logic that selects a backend provider and
model for each incoming request. It supports:

- **Alias resolution**: maps logical names (e.g. ``"fast"``, ``"smart"``) to
  an ordered list of backend entries defined in the configuration.
- **Direct model routing**: maps known model names to specific backends without
  requiring an alias.
- **Default provider fallback**: when no alias or model route matches, routes
  to the configured ``default_provider``.
- **Priority ordering**: backends are sorted by their ``priority`` field
  (ascending); lower numbers are tried first.
- **Round-robin selection**: when ``strategy="round_robin"`` and multiple
  backends share the same priority level, requests are distributed evenly
  across them.
- **Health-aware routing**: backends whose provider is marked as degraded in
  the supplied health-state mapping are skipped (with fallback to the next
  available entry).
- **Fallback chains**: if a selected backend fails (the caller raises an
  exception and calls :py:meth:`Router.next_backend`), the engine returns the
  next candidate from the ordered list.

Typical usage::

    from llm_swap.config import load_config
    from llm_swap.router import Router, RoutingResult

    cfg = load_config("config.yaml")
    router = Router(cfg)

    result = router.route(model="fast")
    print(result.provider_name, result.model)

    # On failure, ask for the next candidate:
    result2 = router.route(model="fast", exclude_providers={result.provider_name})
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from llm_swap.config import BackendEntry, Config, ProviderConfig


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingResult:
    """The outcome of a routing decision.

    Attributes:
        provider_name: The unique name of the selected provider (matches a
            :class:`~llm_swap.config.ProviderConfig` ``name`` field).
        model: The model name to pass to the provider's API.
        provider_config: The full :class:`~llm_swap.config.ProviderConfig`
            for the selected provider.
        priority: The priority level of the selected backend entry.
    """

    provider_name: str
    model: str
    provider_config: ProviderConfig
    priority: int


class NoAvailableProviderError(Exception):
    """Raised when no backend provider can be selected for a request.

    This occurs when all candidates in the fallback chain are either degraded
    or explicitly excluded by the caller.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
_RoundRobinState:
    """Per-alias/route round-robin counter (protected by a lock)."""
    counter: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class _RoundRobinState:  # noqa: F811  (redefine with real implementation)
    """Per-routing-key round-robin counter, thread-safe."""

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock: threading.Lock = threading.Lock()

    def next_index(self, length: int) -> int:
        """Return the next index in ``[0, length)`` and advance the counter.

        Args:
            length: Number of candidates at the current priority level.

        Returns:
            Index into the candidates list.
        """
        with self._lock:
            idx = self._counter % length
            self._counter += 1
            return idx


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Stateful routing engine that resolves model names to backend providers.

    The router is initialised once with a :class:`~llm_swap.config.Config`
    object and is designed to be shared across request handlers (it is
    thread-safe for round-robin state).

    Args:
        config: A fully validated :class:`~llm_swap.config.Config` instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._strategy = config.routing.strategy

        # Build lookup maps for O(1) access
        self._alias_map: Dict[str, List[BackendEntry]] = {
            ar.alias: ar.backends for ar in config.routing.aliases
        }
        self._model_route_map: Dict[str, List[BackendEntry]] = {
            mr.model: mr.backends for mr in config.routing.model_routes
        }
        self._provider_map: Dict[str, ProviderConfig] = {
            p.name: p for p in config.providers
        }

        # Round-robin counters keyed by routing key (alias or model name)
        self._rr_state: Dict[str, _RoundRobinState] = {}
        self._rr_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        model: str,
        health_state: Optional[Dict[str, bool]] = None,
        exclude_providers: Optional[Set[str]] = None,
    ) -> RoutingResult:
        """Select the best available backend for the given model name.

        Resolution order:

        1. Check the alias map (``routing.aliases`` in config).
        2. Check the model-route map (``routing.model_routes`` in config).
        3. Fall back to the ``routing.default_provider`` with the model name
           passed through verbatim.

        Within each candidate list the backends are ordered by priority
        (ascending). If ``strategy="round_robin"`` backends at the *same*
        priority level are rotated on successive calls.

        Degraded providers (``health_state[name] is False``) and explicitly
        excluded providers are skipped.  If no candidate survives filtering a
        :class:`NoAvailableProviderError` is raised.

        Args:
            model: The model name or alias supplied by the client.
            health_state: Optional mapping of ``provider_name -> is_healthy``.
                Providers absent from the map are assumed healthy.
            exclude_providers: Optional set of provider names to skip
                (e.g. providers that already failed for this request).

        Returns:
            A :class:`RoutingResult` describing the selected backend.

        Raises:
            NoAvailableProviderError: If no healthy, non-excluded backend
                can be found.
        """
        _health = health_state or {}
        _exclude: FrozenSet[str] = frozenset(exclude_providers or set())

        candidates = self._resolve_candidates(model)

        # Filter by health and exclusion
        available = [
            be for be in candidates
            if self._is_available(be.provider, _health, _exclude)
        ]

        if not available:
            raise NoAvailableProviderError(
                f"No available backend for model={model!r}. "
                f"Excluded: {sorted(_exclude) or 'none'}. "
                f"All candidates: {[be.provider for be in candidates]}."
            )

        selected = self._select(model, available)
        provider_cfg = self._provider_map[selected.provider]
        return RoutingResult(
            provider_name=selected.provider,
            model=selected.model,
            provider_config=provider_cfg,
            priority=selected.priority,
        )

    def all_candidates(
        self,
        model: str,
        health_state: Optional[Dict[str, bool]] = None,
        exclude_providers: Optional[Set[str]] = None,
    ) -> List[RoutingResult]:
        """Return all available backends for *model* in priority order.

        Unlike :meth:`route`, this method returns the full list of viable
        backends so that callers can implement their own retry loops.

        Args:
            model: The model name or alias supplied by the client.
            health_state: Optional mapping of ``provider_name -> is_healthy``.
            exclude_providers: Optional set of provider names to skip.

        Returns:
            A list of :class:`RoutingResult` objects, ordered by priority
            (ascending).  May be empty if all candidates are degraded or
            excluded.
        """
        _health = health_state or {}
        _exclude: FrozenSet[str] = frozenset(exclude_providers or set())

        candidates = self._resolve_candidates(model)
        results: List[RoutingResult] = []
        for be in candidates:
            if self._is_available(be.provider, _health, _exclude):
                provider_cfg = self._provider_map[be.provider]
                results.append(
                    RoutingResult(
                        provider_name=be.provider,
                        model=be.model,
                        provider_config=provider_cfg,
                        priority=be.priority,
                    )
                )
        return results

    def list_providers(self) -> List[ProviderConfig]:
        """Return all configured provider definitions.

        Returns:
            A list of :class:`~llm_swap.config.ProviderConfig` objects in the
            order they were declared in the configuration file.
        """
        return list(self._config.providers)

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Look up a provider by name.

        Args:
            name: The unique provider name.

        Returns:
            The :class:`~llm_swap.config.ProviderConfig` or ``None``.
        """
        return self._provider_map.get(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_candidates(self, model: str) -> List[BackendEntry]:
        """Return the ordered list of :class:`BackendEntry` objects for *model*.

        Resolution order: alias map → model-route map → default provider.

        The returned list is sorted by priority (ascending).  Round-robin
        shuffling within priority levels happens later in :meth:`_select`.

        Args:
            model: The model name or alias from the client request.

        Returns:
            A non-empty list of :class:`BackendEntry` objects.  The default
            provider fallback guarantees the list always has at least one
            entry (assuming the config was validated successfully).
        """
        backends: Optional[List[BackendEntry]] = None

        if model in self._alias_map:
            backends = self._alias_map[model]
        elif model in self._model_route_map:
            backends = self._model_route_map[model]

        if backends is not None:
            # Sort by priority ascending (stable sort preserves declaration order
            # for entries at the same priority level)
            return sorted(backends, key=lambda be: be.priority)

        # Default fallback: route to the default provider, pass model verbatim
        default_name = self._config.routing.default_provider
        return [
            BackendEntry(
                provider=default_name,
                model=model,
                priority=1,
            )
        ]

    @staticmethod
    def _is_available(
        provider_name: str,
        health_state: Dict[str, bool],
        exclude: FrozenSet[str],
    ) -> bool:
        """Return ``True`` if *provider_name* is healthy and not excluded.

        Args:
            provider_name: Name of the provider to check.
            health_state: Mapping of provider name to health boolean.
            exclude: Set of provider names to treat as unavailable.

        Returns:
            ``True`` if the provider can receive requests, ``False`` otherwise.
        """
        if provider_name in exclude:
            return False
        # Absent from health_state → assume healthy
        return health_state.get(provider_name, True)

    def _select(self, routing_key: str, candidates: List[BackendEntry]) -> BackendEntry:
        """Pick one backend from *candidates* according to the configured strategy.

        *candidates* is already filtered (healthy, not excluded) and sorted by
        priority ascending.

        - ``priority`` strategy: always return the first (lowest-priority-number)
          entry.  If multiple entries share the lowest priority, the first one
          in declaration order is chosen.
        - ``round_robin`` strategy: group the entries at the lowest priority
          level and rotate among them.

        Args:
            routing_key: The alias or model name (used as the round-robin key).
            candidates: Filtered, priority-sorted list of backends.

        Returns:
            The selected :class:`BackendEntry`.
        """
        assert candidates, "_select called with empty candidates list"

        if self._strategy == "priority":
            return candidates[0]

        # round_robin: collect all entries sharing the minimum priority
        min_priority = candidates[0].priority
        top_tier = [be for be in candidates if be.priority == min_priority]

        if len(top_tier) == 1:
            return top_tier[0]

        rr = self._get_rr_state(routing_key)
        idx = rr.next_index(len(top_tier))
        return top_tier[idx]

    def _get_rr_state(self, key: str) -> _RoundRobinState:
        """Retrieve or create the :class:`_RoundRobinState` for *key*.

        Thread-safe via a module-level lock.

        Args:
            key: Routing key (alias or model name).

        Returns:
            The :class:`_RoundRobinState` for that key.
        """
        with self._rr_lock:
            if key not in self._rr_state:
                self._rr_state[key] = _RoundRobinState()
            return self._rr_state[key]

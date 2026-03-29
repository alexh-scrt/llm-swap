"""Unit tests for llm_swap.router — routing engine covering alias resolution,
priority ordering, round-robin distribution, health-aware filtering,
exclusion sets, and fallback to the default provider.

All tests are fully self-contained and do not touch the filesystem or any
live provider; they construct :class:`~llm_swap.config.Config` objects
in-memory via :func:`~llm_swap.config.load_config_from_dict`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from llm_swap.config import Config, load_config_from_dict
from llm_swap.router import NoAvailableProviderError, Router, RoutingResult


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _build_config(routing_extra: Dict[str, Any] | None = None) -> Config:
    """Return a Config with several providers and routing rules.

    Providers defined:
    - ``openai``   (type: openai)
    - ``anthropic`` (type: anthropic)
    - ``mistral``  (type: mistral)
    - ``ollama``   (type: ollama)

    Aliases defined:
    - ``smart``  → openai(p=1), anthropic(p=2), mistral(p=3)
    - ``fast``   → openai(p=1), mistral(p=2)
    - ``cheap``  → ollama(p=1), mistral(p=2)
    - ``local``  → ollama(p=1)
    - ``tied``   → openai(p=1), anthropic(p=1)   ← two at same priority

    Model routes defined:
    - ``gpt-4o``   → openai
    - ``llama3``   → ollama
    """
    data: Dict[str, Any] = {
        "providers": [
            {
                "name": "openai",
                "type": "openai",
                "api_key": "sk-openai",
                "base_url": "https://api.openai.com/v1",
            },
            {
                "name": "anthropic",
                "type": "anthropic",
                "api_key": "sk-anthropic",
                "base_url": "https://api.anthropic.com",
            },
            {
                "name": "mistral",
                "type": "mistral",
                "api_key": "sk-mistral",
                "base_url": "https://api.mistral.ai/v1",
            },
            {
                "name": "ollama",
                "type": "ollama",
                "api_key": None,
                "base_url": "http://localhost:11434",
            },
        ],
        "routing": {
            "default_provider": "openai",
            "strategy": "priority",
            "aliases": [
                {
                    "alias": "smart",
                    "backends": [
                        {"provider": "openai", "model": "gpt-4o", "priority": 1},
                        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "priority": 2},
                        {"provider": "mistral", "model": "mistral-large-latest", "priority": 3},
                    ],
                },
                {
                    "alias": "fast",
                    "backends": [
                        {"provider": "openai", "model": "gpt-4o-mini", "priority": 1},
                        {"provider": "mistral", "model": "mistral-small-latest", "priority": 2},
                    ],
                },
                {
                    "alias": "cheap",
                    "backends": [
                        {"provider": "ollama", "model": "llama3", "priority": 1},
                        {"provider": "mistral", "model": "open-mistral-7b", "priority": 2},
                    ],
                },
                {
                    "alias": "local",
                    "backends": [
                        {"provider": "ollama", "model": "llama3", "priority": 1},
                    ],
                },
                {
                    "alias": "tied",
                    "backends": [
                        {"provider": "openai", "model": "gpt-4o", "priority": 1},
                        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "priority": 1},
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
                {
                    "model": "llama3",
                    "backends": [
                        {"provider": "ollama", "model": "llama3", "priority": 1},
                    ],
                },
            ],
            **(routing_extra or {}),
        },
    }
    return load_config_from_dict(data)


@pytest.fixture()
def router() -> Router:
    """Return a Router built from the default test config (priority strategy)."""
    return Router(_build_config())


@pytest.fixture()
def rr_router() -> Router:
    """Return a Router with round-robin strategy."""
    return Router(_build_config(routing_extra={"strategy": "round_robin"}))


# ---------------------------------------------------------------------------
# RoutingResult
# ---------------------------------------------------------------------------


class TestRoutingResult:
    """Verify the RoutingResult dataclass behaves as expected."""

    def test_is_frozen(self, router: Router) -> None:
        result = router.route("smart")
        with pytest.raises((AttributeError, TypeError)):
            result.provider_name = "other"  # type: ignore[misc]

    def test_fields_populated(self, router: Router) -> None:
        result = router.route("smart")
        assert isinstance(result, RoutingResult)
        assert result.provider_name
        assert result.model
        assert result.provider_config is not None
        assert result.priority >= 1


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------


class TestAliasResolution:
    """Test that alias → backend mapping is resolved correctly."""

    def test_smart_alias_selects_openai(self, router: Router) -> None:
        result = router.route("smart")
        assert result.provider_name == "openai"
        assert result.model == "gpt-4o"
        assert result.priority == 1

    def test_fast_alias_selects_openai(self, router: Router) -> None:
        result = router.route("fast")
        assert result.provider_name == "openai"
        assert result.model == "gpt-4o-mini"

    def test_cheap_alias_selects_ollama(self, router: Router) -> None:
        result = router.route("cheap")
        assert result.provider_name == "ollama"
        assert result.model == "llama3"

    def test_local_alias_selects_ollama(self, router: Router) -> None:
        result = router.route("local")
        assert result.provider_name == "ollama"

    def test_provider_config_is_correct_type(self, router: Router) -> None:
        result = router.route("smart")
        assert result.provider_config.type == "openai"

    def test_provider_config_name_matches(self, router: Router) -> None:
        result = router.route("cheap")
        assert result.provider_config.name == "ollama"


# ---------------------------------------------------------------------------
# Model-route resolution
# ---------------------------------------------------------------------------


class TestModelRouteResolution:
    """Test that exact model names are routed via model_routes."""

    def test_gpt4o_routes_to_openai(self, router: Router) -> None:
        result = router.route("gpt-4o")
        assert result.provider_name == "openai"
        assert result.model == "gpt-4o"

    def test_llama3_routes_to_ollama(self, router: Router) -> None:
        result = router.route("llama3")
        assert result.provider_name == "ollama"
        assert result.model == "llama3"


# ---------------------------------------------------------------------------
# Default provider fallback
# ---------------------------------------------------------------------------


class TestDefaultProviderFallback:
    """Test that unknown model names fall back to the default provider."""

    def test_unknown_model_routes_to_default(self, router: Router) -> None:
        result = router.route("totally-unknown-model")
        assert result.provider_name == "openai"  # default_provider
        # Model name should be passed through verbatim
        assert result.model == "totally-unknown-model"

    def test_unknown_model_priority_is_1(self, router: Router) -> None:
        result = router.route("mystery-model")
        assert result.priority == 1

    def test_default_provider_config_populated(self, router: Router) -> None:
        result = router.route("mystery-model")
        assert result.provider_config is not None
        assert result.provider_config.name == "openai"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Test that backends are tried in ascending priority order."""

    def test_priority_1_selected_over_priority_2(self, router: Router) -> None:
        result = router.route("smart")
        assert result.priority == 1

    def test_fallback_to_priority_2_when_p1_excluded(self, router: Router) -> None:
        result = router.route("smart", exclude_providers={"openai"})
        assert result.provider_name == "anthropic"
        assert result.priority == 2

    def test_fallback_to_priority_3_when_p1_and_p2_excluded(self, router: Router) -> None:
        result = router.route("smart", exclude_providers={"openai", "anthropic"})
        assert result.provider_name == "mistral"
        assert result.priority == 3

    def test_all_excluded_raises(self, router: Router) -> None:
        with pytest.raises(NoAvailableProviderError):
            router.route("smart", exclude_providers={"openai", "anthropic", "mistral"})

    def test_candidates_returned_in_priority_order(self, router: Router) -> None:
        results = router.all_candidates("smart")
        priorities = [r.priority for r in results]
        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# Health-aware routing
# ---------------------------------------------------------------------------


class TestHealthAwareRouting:
    """Test that degraded providers are skipped in routing."""

    def test_degraded_provider_is_skipped(self, router: Router) -> None:
        result = router.route("fast", health_state={"openai": False})
        assert result.provider_name == "mistral"

    def test_healthy_provider_is_selected_normally(self, router: Router) -> None:
        result = router.route("fast", health_state={"openai": True})
        assert result.provider_name == "openai"

    def test_absent_provider_assumed_healthy(self, router: Router) -> None:
        # openai is not in health_state → assumed healthy
        result = router.route("fast", health_state={})
        assert result.provider_name == "openai"

    def test_all_degraded_raises(self, router: Router) -> None:
        with pytest.raises(NoAvailableProviderError):
            router.route(
                "fast",
                health_state={"openai": False, "mistral": False},
            )

    def test_single_backend_alias_degraded_raises(self, router: Router) -> None:
        with pytest.raises(NoAvailableProviderError):
            router.route("local", health_state={"ollama": False})

    def test_health_state_combined_with_exclusion(self, router: Router) -> None:
        # openai degraded, anthropic excluded → mistral selected
        result = router.route(
            "smart",
            health_state={"openai": False},
            exclude_providers={"anthropic"},
        )
        assert result.provider_name == "mistral"


# ---------------------------------------------------------------------------
# Exclusion sets
# ---------------------------------------------------------------------------


class TestExclusionSets:
    """Test that explicitly excluded providers are skipped."""

    def test_exclude_single_provider(self, router: Router) -> None:
        result = router.route("cheap", exclude_providers={"ollama"})
        assert result.provider_name == "mistral"

    def test_exclude_empty_set_has_no_effect(self, router: Router) -> None:
        result = router.route("cheap", exclude_providers=set())
        assert result.provider_name == "ollama"

    def test_exclude_none_has_no_effect(self, router: Router) -> None:
        result = router.route("cheap", exclude_providers=None)
        assert result.provider_name == "ollama"

    def test_exclude_all_raises_no_available(self, router: Router) -> None:
        with pytest.raises(NoAvailableProviderError):
            router.route("local", exclude_providers={"ollama"})


# ---------------------------------------------------------------------------
# all_candidates
# ---------------------------------------------------------------------------


class TestAllCandidates:
    """Test the all_candidates helper method."""

    def test_returns_all_healthy_backends(self, router: Router) -> None:
        results = router.all_candidates("smart")
        provider_names = [r.provider_name for r in results]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "mistral" in provider_names

    def test_excludes_degraded_backends(self, router: Router) -> None:
        results = router.all_candidates("smart", health_state={"openai": False})
        provider_names = [r.provider_name for r in results]
        assert "openai" not in provider_names
        assert "anthropic" in provider_names

    def test_excludes_explicitly_excluded(self, router: Router) -> None:
        results = router.all_candidates("smart", exclude_providers={"mistral"})
        provider_names = [r.provider_name for r in results]
        assert "mistral" not in provider_names

    def test_returns_empty_when_all_excluded(self, router: Router) -> None:
        results = router.all_candidates(
            "fast",
            exclude_providers={"openai", "mistral"},
        )
        assert results == []

    def test_ordered_by_priority(self, router: Router) -> None:
        results = router.all_candidates("smart")
        priorities = [r.priority for r in results]
        assert priorities == sorted(priorities)

    def test_default_fallback_returns_one_candidate(self, router: Router) -> None:
        results = router.all_candidates("mystery-model")
        assert len(results) == 1
        assert results[0].provider_name == "openai"


# ---------------------------------------------------------------------------
# Round-robin strategy
# ---------------------------------------------------------------------------


class TestRoundRobinStrategy:
    """Test that round-robin distributes load among equal-priority backends."""

    def test_rr_alternates_between_tied_backends(self, rr_router: Router) -> None:
        """The 'tied' alias has openai and anthropic both at priority=1."""
        seen: List[str] = []
        for _ in range(4):
            result = rr_router.route("tied")
            seen.append(result.provider_name)
        # Both providers should appear
        assert "openai" in seen
        assert "anthropic" in seen

    def test_rr_cycles_through_all_tied(self, rr_router: Router) -> None:
        first = rr_router.route("tied").provider_name
        second = rr_router.route("tied").provider_name
        third = rr_router.route("tied").provider_name
        # Should cycle: first == third for two backends
        assert first != second
        assert first == third

    def test_rr_with_one_backend_always_same(self, rr_router: Router) -> None:
        results = [rr_router.route("local").provider_name for _ in range(3)]
        assert all(r == "ollama" for r in results)

    def test_rr_skips_degraded_in_top_tier(self, rr_router: Router) -> None:
        """If one of the tied backends is degraded, only the healthy one is used."""
        for _ in range(4):
            result = rr_router.route("tied", health_state={"openai": False})
            assert result.provider_name == "anthropic"

    def test_rr_falls_back_to_lower_priority_when_all_top_degraded(
        self, rr_router: Router
    ) -> None:
        """fast: openai(p=1), mistral(p=2). With rr and openai degraded → mistral."""
        result = rr_router.route("fast", health_state={"openai": False})
        assert result.provider_name == "mistral"
        assert result.priority == 2

    def test_priority_strategy_ignores_round_robin(self, router: Router) -> None:
        """Priority router should always pick the first tied backend, not rotate."""
        first_providers = [router.route("tied").provider_name for _ in range(4)]
        # All should be the same (whichever declaration comes first)
        assert len(set(first_providers)) == 1


# ---------------------------------------------------------------------------
# list_providers / get_provider
# ---------------------------------------------------------------------------


class TestProviderAccessors:
    """Tests for list_providers and get_provider helper methods."""

    def test_list_providers_returns_all(self, router: Router) -> None:
        providers = router.list_providers()
        names = {p.name for p in providers}
        assert names == {"openai", "anthropic", "mistral", "ollama"}

    def test_list_providers_count(self, router: Router) -> None:
        assert len(router.list_providers()) == 4

    def test_get_provider_found(self, router: Router) -> None:
        p = router.get_provider("mistral")
        assert p is not None
        assert p.name == "mistral"
        assert p.type == "mistral"

    def test_get_provider_not_found(self, router: Router) -> None:
        p = router.get_provider("nonexistent")
        assert p is None

    def test_list_providers_returns_copies_of_config(self, router: Router) -> None:
        providers = router.list_providers()
        # Modifying the returned list should not affect the router's internal state
        providers.clear()
        assert len(router.list_providers()) == 4


# ---------------------------------------------------------------------------
# NoAvailableProviderError
# ---------------------------------------------------------------------------


class TestNoAvailableProviderError:
    """Test error cases that produce NoAvailableProviderError."""

    def test_error_is_exception_subclass(self) -> None:
        assert issubclass(NoAvailableProviderError, Exception)

    def test_error_message_includes_model_name(self, router: Router) -> None:
        with pytest.raises(NoAvailableProviderError, match="fast"):
            router.route("fast", exclude_providers={"openai", "mistral"})

    def test_default_fallback_also_degraded_raises(self, router: Router) -> None:
        """Unknown model → default provider; if that's degraded → error."""
        with pytest.raises(NoAvailableProviderError):
            router.route("unknown-model", health_state={"openai": False})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_alias_takes_precedence_over_model_route(self) -> None:
        """If an alias and a model route share the same name, alias wins."""
        data: Dict[str, Any] = {
            "providers": [
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk",
                    "base_url": "https://api.openai.com/v1",
                },
                {
                    "name": "mistral",
                    "type": "mistral",
                    "api_key": "sk",
                    "base_url": "https://api.mistral.ai/v1",
                },
            ],
            "routing": {
                "default_provider": "openai",
                "aliases": [
                    {
                        "alias": "overlap",
                        "backends": [
                            {"provider": "mistral", "model": "mistral-small", "priority": 1}
                        ],
                    }
                ],
                "model_routes": [
                    {
                        "model": "overlap",
                        "backends": [
                            {"provider": "openai", "model": "gpt-4o", "priority": 1}
                        ],
                    }
                ],
            },
        }
        cfg = load_config_from_dict(data)
        r = Router(cfg)
        result = r.route("overlap")
        # Alias map is checked first → mistral
        assert result.provider_name == "mistral"

    def test_single_provider_config(self) -> None:
        """A config with just one provider should route without errors."""
        data: Dict[str, Any] = {
            "providers": [
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk",
                    "base_url": "https://api.openai.com/v1",
                },
            ],
            "routing": {
                "default_provider": "openai",
            },
        }
        cfg = load_config_from_dict(data)
        r = Router(cfg)
        result = r.route("gpt-4o")
        assert result.provider_name == "openai"

    def test_health_state_none_treated_as_all_healthy(self, router: Router) -> None:
        result = router.route("smart", health_state=None)
        assert result.provider_name == "openai"

    def test_route_returns_routing_result_instance(self, router: Router) -> None:
        result = router.route("local")
        assert isinstance(result, RoutingResult)

    def test_out_of_order_priorities_sorted_correctly(self) -> None:
        """Backends declared in descending priority order are still sorted ascending."""
        data: Dict[str, Any] = {
            "providers": [
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk",
                    "base_url": "https://api.openai.com/v1",
                },
                {
                    "name": "mistral",
                    "type": "mistral",
                    "api_key": "sk",
                    "base_url": "https://api.mistral.ai/v1",
                },
            ],
            "routing": {
                "default_provider": "openai",
                "aliases": [
                    {
                        "alias": "rev",
                        "backends": [
                            # Declared with higher priority number first
                            {"provider": "mistral", "model": "mistral-small", "priority": 2},
                            {"provider": "openai", "model": "gpt-4o", "priority": 1},
                        ],
                    }
                ],
            },
        }
        cfg = load_config_from_dict(data)
        r = Router(cfg)
        result = r.route("rev")
        # openai has priority=1 so it should be selected despite being declared second
        assert result.provider_name == "openai"
        assert result.priority == 1

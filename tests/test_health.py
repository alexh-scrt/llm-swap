"""Unit tests for llm_swap.health — background health-check task.

Tests cover:
- HealthState read/write operations and thread safety.
- ProviderStatus dataclass serialization.
- _ping_provider routing logic (correct URL/headers per provider type).
- HealthChecker lifecycle (start, stop, check_now).
- Threshold-based state transitions (healthy → degraded, degraded → healthy).
- Concurrent check behaviour.

All HTTP calls are intercepted; no real network connections are made.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_swap.config import HealthCheckConfig, ProviderConfig, load_config_from_dict
from llm_swap.health import HealthChecker, HealthState, ProviderStatus, _ping_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    name: str = "openai",
    provider_type: str = "openai",
    api_key: Optional[str] = "sk-test",
    base_url: str = "https://api.openai.com/v1",
) -> ProviderConfig:
    """Create a minimal ProviderConfig for testing."""
    return ProviderConfig(
        name=name,
        type=provider_type,  # type: ignore[arg-type]
        api_key=api_key,
        base_url=base_url,
    )


def _make_config(
    providers: Optional[List[Dict[str, Any]]] = None,
    health_check_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return a minimal Config object with optional overrides."""
    data: Dict[str, Any] = {
        "providers": providers
        or [
            {
                "name": "openai",
                "type": "openai",
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
            }
        ],
        "routing": {"default_provider": "openai"},
    }
    if health_check_overrides:
        data["health_check"] = health_check_overrides
    return load_config_from_dict(data)


def _make_mock_response(status_code: int = 200) -> httpx.Response:
    """Build a minimal httpx.Response with the given status code."""
    return httpx.Response(
        status_code=status_code,
        content=b"{}",
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", "https://example.com/"),
    )


# ---------------------------------------------------------------------------
# ProviderStatus
# ---------------------------------------------------------------------------


class TestProviderStatus:
    """Tests for the ProviderStatus dataclass."""

    def test_defaults(self) -> None:
        status = ProviderStatus(provider_name="openai")
        assert status.healthy is True
        assert status.consecutive_failures == 0
        assert status.consecutive_successes == 0
        assert status.last_check_at is None
        assert status.last_error is None

    def test_to_dict_keys(self) -> None:
        status = ProviderStatus(provider_name="openai")
        d = status.to_dict()
        expected = {
            "provider_name",
            "healthy",
            "consecutive_failures",
            "consecutive_successes",
            "last_check_at",
            "last_error",
        }
        assert set(d.keys()) == expected

    def test_to_dict_values(self) -> None:
        status = ProviderStatus(
            provider_name="mistral",
            healthy=False,
            consecutive_failures=3,
            last_error="timeout",
        )
        d = status.to_dict()
        assert d["provider_name"] == "mistral"
        assert d["healthy"] is False
        assert d["consecutive_failures"] == 3
        assert d["last_error"] == "timeout"


# ---------------------------------------------------------------------------
# HealthState — basic operations
# ---------------------------------------------------------------------------


class TestHealthStateBasic:
    """Tests for HealthState read API."""

    def _make_state(self, names: Optional[List[str]] = None) -> HealthState:
        providers = [
            _make_provider(name=n) for n in (names or ["openai", "anthropic", "ollama"])
        ]
        return HealthState(providers)

    def test_all_providers_start_healthy(self) -> None:
        state = self._make_state()
        for name in ["openai", "anthropic", "ollama"]:
            assert state.is_healthy(name) is True

    def test_unknown_provider_assumed_healthy(self) -> None:
        state = self._make_state()
        assert state.is_healthy("nonexistent") is True

    def test_snapshot_returns_all_providers(self) -> None:
        state = self._make_state(["openai", "mistral"])
        snap = state.snapshot()
        assert set(snap.keys()) == {"openai", "mistral"}

    def test_snapshot_all_true_initially(self) -> None:
        state = self._make_state()
        snap = state.snapshot()
        assert all(v is True for v in snap.values())

    def test_get_status_returns_provider_status(self) -> None:
        state = self._make_state(["openai"])
        status = state.get_status("openai")
        assert isinstance(status, ProviderStatus)
        assert status.provider_name == "openai"

    def test_get_status_returns_none_for_unknown(self) -> None:
        state = self._make_state()
        assert state.get_status("unknown") is None

    def test_all_statuses_returns_list(self) -> None:
        state = self._make_state(["openai", "mistral"])
        statuses = state.all_statuses()
        assert len(statuses) == 2
        names = {s.provider_name for s in statuses}
        assert names == {"openai", "mistral"}


# ---------------------------------------------------------------------------
# HealthState — record_success / record_failure
# ---------------------------------------------------------------------------


class TestHealthStateTransitions:
    """Tests for HealthState state transitions."""

    @pytest.mark.anyio
    async def test_record_success_increments_counter(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.record_success("openai", healthy_threshold=2)
        status = state.get_status("openai")
        assert status is not None
        assert status.consecutive_successes == 1

    @pytest.mark.anyio
    async def test_record_success_clears_failure_counter(self) -> None:
        state = HealthState([_make_provider("openai")])
        # Simulate prior failures
        status = state.get_status("openai")
        assert status is not None
        status.consecutive_failures = 5
        await state.record_success("openai", healthy_threshold=2)
        assert status.consecutive_failures == 0

    @pytest.mark.anyio
    async def test_record_success_clears_error(self) -> None:
        state = HealthState([_make_provider("openai")])
        status = state.get_status("openai")
        assert status is not None
        status.last_error = "previous error"
        await state.record_success("openai", healthy_threshold=1)
        assert status.last_error is None

    @pytest.mark.anyio
    async def test_record_success_updates_last_check_at(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.record_success("openai", healthy_threshold=1)
        status = state.get_status("openai")
        assert status is not None
        assert status.last_check_at is not None

    @pytest.mark.anyio
    async def test_record_failure_increments_counter(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.record_failure("openai", error="timeout", unhealthy_threshold=3)
        status = state.get_status("openai")
        assert status is not None
        assert status.consecutive_failures == 1

    @pytest.mark.anyio
    async def test_record_failure_stores_error(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.record_failure("openai", error="connection refused", unhealthy_threshold=3)
        status = state.get_status("openai")
        assert status is not None
        assert status.last_error == "connection refused"

    @pytest.mark.anyio
    async def test_record_failure_clears_success_counter(self) -> None:
        state = HealthState([_make_provider("openai")])
        status = state.get_status("openai")
        assert status is not None
        status.consecutive_successes = 5
        await state.record_failure("openai", error="err", unhealthy_threshold=3)
        assert status.consecutive_successes == 0

    @pytest.mark.anyio
    async def test_transitions_healthy_to_degraded_at_threshold(self) -> None:
        state = HealthState([_make_provider("openai")])
        # First failure: below threshold (threshold=2)
        transitioned = await state.record_failure(
            "openai", error="err", unhealthy_threshold=2
        )
        assert transitioned is False
        assert state.is_healthy("openai") is True

        # Second failure: reaches threshold
        transitioned = await state.record_failure(
            "openai", error="err", unhealthy_threshold=2
        )
        assert transitioned is True
        assert state.is_healthy("openai") is False

    @pytest.mark.anyio
    async def test_does_not_re_transition_already_degraded(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("openai", reason="manual")

        # Further failures should NOT signal another transition
        for _ in range(5):
            transitioned = await state.record_failure(
                "openai", error="err", unhealthy_threshold=2
            )
            assert transitioned is False

    @pytest.mark.anyio
    async def test_transitions_degraded_to_healthy_at_threshold(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("openai", reason="test")
        assert state.is_healthy("openai") is False

        # First success: below threshold (threshold=2)
        transitioned = await state.record_success("openai", healthy_threshold=2)
        assert transitioned is False
        assert state.is_healthy("openai") is False

        # Second success: reaches threshold
        transitioned = await state.record_success("openai", healthy_threshold=2)
        assert transitioned is True
        assert state.is_healthy("openai") is True

    @pytest.mark.anyio
    async def test_does_not_re_transition_already_healthy(self) -> None:
        state = HealthState([_make_provider("openai")])
        # Starts healthy
        for _ in range(5):
            transitioned = await state.record_success("openai", healthy_threshold=1)
            assert transitioned is False

    @pytest.mark.anyio
    async def test_record_success_returns_false_for_unknown_provider(self) -> None:
        state = HealthState([_make_provider("openai")])
        result = await state.record_success("ghost", healthy_threshold=1)
        assert result is False

    @pytest.mark.anyio
    async def test_record_failure_returns_false_for_unknown_provider(self) -> None:
        state = HealthState([_make_provider("openai")])
        result = await state.record_failure("ghost", error="err", unhealthy_threshold=1)
        assert result is False


# ---------------------------------------------------------------------------
# HealthState — mark_healthy / mark_degraded
# ---------------------------------------------------------------------------


class TestHealthStateManualMarking:
    """Tests for the manual mark_healthy and mark_degraded helpers."""

    @pytest.mark.anyio
    async def test_mark_degraded_sets_healthy_false(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("openai", reason="manual test")
        assert state.is_healthy("openai") is False

    @pytest.mark.anyio
    async def test_mark_degraded_stores_reason(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("openai", reason="network partition")
        status = state.get_status("openai")
        assert status is not None
        assert status.last_error == "network partition"

    @pytest.mark.anyio
    async def test_mark_degraded_resets_counters(self) -> None:
        state = HealthState([_make_provider("openai")])
        status = state.get_status("openai")
        assert status is not None
        status.consecutive_failures = 10
        status.consecutive_successes = 5
        await state.mark_degraded("openai")
        assert status.consecutive_failures == 0
        assert status.consecutive_successes == 0

    @pytest.mark.anyio
    async def test_mark_healthy_sets_healthy_true(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("openai")
        await state.mark_healthy("openai")
        assert state.is_healthy("openai") is True

    @pytest.mark.anyio
    async def test_mark_healthy_resets_counters(self) -> None:
        state = HealthState([_make_provider("openai")])
        status = state.get_status("openai")
        assert status is not None
        status.consecutive_failures = 7
        status.last_error = "some error"
        await state.mark_healthy("openai")
        assert status.consecutive_failures == 0
        assert status.last_error is None

    @pytest.mark.anyio
    async def test_mark_healthy_on_unknown_provider_does_not_raise(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_healthy("nonexistent")  # Should not raise

    @pytest.mark.anyio
    async def test_mark_degraded_on_unknown_provider_does_not_raise(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("nonexistent")  # Should not raise


# ---------------------------------------------------------------------------
# HealthState — is_healthy_async
# ---------------------------------------------------------------------------


class TestHealthStateAsyncRead:
    """Tests for the async variant of is_healthy."""

    @pytest.mark.anyio
    async def test_is_healthy_async_returns_true_initially(self) -> None:
        state = HealthState([_make_provider("openai")])
        result = await state.is_healthy_async("openai")
        assert result is True

    @pytest.mark.anyio
    async def test_is_healthy_async_returns_false_when_degraded(self) -> None:
        state = HealthState([_make_provider("openai")])
        await state.mark_degraded("openai")
        result = await state.is_healthy_async("openai")
        assert result is False

    @pytest.mark.anyio
    async def test_is_healthy_async_unknown_assumed_healthy(self) -> None:
        state = HealthState([_make_provider("openai")])
        result = await state.is_healthy_async("nobody")
        assert result is True


# ---------------------------------------------------------------------------
# _ping_provider
# ---------------------------------------------------------------------------


class TestPingProvider:
    """Tests for the _ping_provider helper."""

    @pytest.mark.anyio
    async def test_openai_pings_models_endpoint(self) -> None:
        provider = _make_provider(
            name="openai",
            provider_type="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        captured_requests: List[httpx.Request] = []

        async def mock_send(request: httpx.Request, **kwargs: Any) -> httpx.Response:
            captured_requests.append(request)
            return _make_mock_response(200)

        transport = httpx.MockTransport(mock_send)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(200))
            mock_client_cls.return_value = mock_client

            await _ping_provider(provider, timeout=5.0)

            call_args = mock_client.get.call_args
            url = call_args[0][0]
            assert "/models" in url

    @pytest.mark.anyio
    async def test_anthropic_pings_v1_models_with_headers(self) -> None:
        provider = _make_provider(
            name="anthropic",
            provider_type="anthropic",
            api_key="sk-ant",
            base_url="https://api.anthropic.com",
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(200))
            mock_client_cls.return_value = mock_client

            await _ping_provider(provider, timeout=5.0)

            call_args = mock_client.get.call_args
            url = call_args[0][0]
            headers = call_args[1].get("headers", {})

            assert "/v1/models" in url
            assert "anthropic-version" in headers
            assert "x-api-key" in headers
            assert "Authorization" not in headers

    @pytest.mark.anyio
    async def test_ollama_pings_api_tags(self) -> None:
        provider = _make_provider(
            name="ollama",
            provider_type="ollama",
            api_key=None,
            base_url="http://localhost:11434",
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(200))
            mock_client_cls.return_value = mock_client

            await _ping_provider(provider, timeout=5.0)

            call_args = mock_client.get.call_args
            url = call_args[0][0]
            assert "/api/tags" in url

    @pytest.mark.anyio
    async def test_mistral_pings_models_endpoint(self) -> None:
        provider = _make_provider(
            name="mistral",
            provider_type="mistral",
            api_key="sk-mistral",
            base_url="https://api.mistral.ai/v1",
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(200))
            mock_client_cls.return_value = mock_client

            await _ping_provider(provider, timeout=5.0)

            call_args = mock_client.get.call_args
            url = call_args[0][0]
            assert "/models" in url

    @pytest.mark.anyio
    async def test_5xx_response_raises_exception(self) -> None:
        provider = _make_provider("openai", "openai", "sk", "https://api.openai.com/v1")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(503))
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError):
                await _ping_provider(provider, timeout=5.0)

    @pytest.mark.anyio
    async def test_4xx_response_does_not_raise(self) -> None:
        """4xx means the server is up but rejected the request (e.g. wrong key)."""
        provider = _make_provider("openai", "openai", "sk", "https://api.openai.com/v1")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(401))
            mock_client_cls.return_value = mock_client

            # Should NOT raise — 4xx means the server is reachable
            await _ping_provider(provider, timeout=5.0)

    @pytest.mark.anyio
    async def test_network_error_propagates(self) -> None:
        provider = _make_provider("openai", "openai", "sk", "https://api.openai.com/v1")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("connection refused")
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await _ping_provider(provider, timeout=5.0)

    @pytest.mark.anyio
    async def test_custom_headers_forwarded(self) -> None:
        provider = ProviderConfig(
            name="openai",
            type="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            headers={"X-Custom": "custom-value"},
        )
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=_make_mock_response(200))
            mock_client_cls.return_value = mock_client

            await _ping_provider(provider, timeout=5.0)

            call_args = mock_client.get.call_args
            headers = call_args[1].get("headers", {})
            assert headers.get("X-Custom") == "custom-value"


# ---------------------------------------------------------------------------
# HealthChecker — lifecycle
# ---------------------------------------------------------------------------


class TestHealthCheckerLifecycle:
    """Tests for HealthChecker start/stop/is_running."""

    @pytest.mark.anyio
    async def test_starts_and_stops_cleanly(self) -> None:
        cfg = _make_config(health_check_overrides={"enabled": True, "interval_seconds": 60})
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        with patch.object(checker, "_check_all_providers", new_callable=AsyncMock):
            await checker.start()
            assert checker.is_running is True
            await checker.stop()
            assert checker.is_running is False

    @pytest.mark.anyio
    async def test_disabled_checker_does_not_start_task(self) -> None:
        cfg = _make_config(health_check_overrides={"enabled": False})
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        await checker.start()
        assert checker.is_running is False
        await checker.stop()  # Should be a no-op

    @pytest.mark.anyio
    async def test_double_start_does_not_create_second_task(self) -> None:
        cfg = _make_config(health_check_overrides={"enabled": True, "interval_seconds": 60})
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        with patch.object(checker, "_check_all_providers", new_callable=AsyncMock):
            await checker.start()
            task_before = checker._task
            await checker.start()  # Second call should be ignored
            task_after = checker._task
            assert task_before is task_after
            await checker.stop()

    @pytest.mark.anyio
    async def test_stop_before_start_does_not_raise(self) -> None:
        cfg = _make_config()
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)
        await checker.stop()  # Should not raise

    @pytest.mark.anyio
    async def test_is_running_false_initially(self) -> None:
        cfg = _make_config()
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)
        assert checker.is_running is False


# ---------------------------------------------------------------------------
# HealthChecker — check_now
# ---------------------------------------------------------------------------


class TestHealthCheckerCheckNow:
    """Tests for the manual check_now method."""

    @pytest.mark.anyio
    async def test_check_now_returns_health_snapshot(self) -> None:
        cfg = _make_config()
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        with patch("llm_swap.health._ping_provider", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = None  # Success
            result = await checker.check_now()

        assert isinstance(result, dict)
        assert "openai" in result

    @pytest.mark.anyio
    async def test_check_now_marks_provider_healthy_on_success(self) -> None:
        cfg = _make_config()
        state = HealthState(cfg.providers)
        await state.mark_degraded("openai", reason="test")
        checker = HealthChecker(cfg, state)

        # Need healthy_threshold=1 so one success restores
        checker._hc_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=30,
            timeout_seconds=5,
            unhealthy_threshold=2,
            healthy_threshold=1,
        )

        with patch("llm_swap.health._ping_provider", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = None
            result = await checker.check_now()

        assert result["openai"] is True

    @pytest.mark.anyio
    async def test_check_now_marks_provider_degraded_on_failure(self) -> None:
        cfg = _make_config()
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        # unhealthy_threshold=1 so one failure marks degraded
        checker._hc_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=30,
            timeout_seconds=5,
            unhealthy_threshold=1,
            healthy_threshold=1,
        )

        with patch(
            "llm_swap.health._ping_provider",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection refused"),
        ):
            result = await checker.check_now()

        assert result["openai"] is False

    @pytest.mark.anyio
    async def test_check_now_multiple_providers(self) -> None:
        cfg = _make_config(
            providers=[
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk1",
                    "base_url": "https://api.openai.com/v1",
                },
                {
                    "name": "mistral",
                    "type": "mistral",
                    "api_key": "sk2",
                    "base_url": "https://api.mistral.ai/v1",
                },
            ]
        )
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        with patch("llm_swap.health._ping_provider", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = None
            result = await checker.check_now()

        assert "openai" in result
        assert "mistral" in result

    @pytest.mark.anyio
    async def test_check_now_partial_failure(self) -> None:
        """If one provider fails, others should still be updated."""
        cfg = _make_config(
            providers=[
                {
                    "name": "openai",
                    "type": "openai",
                    "api_key": "sk1",
                    "base_url": "https://api.openai.com/v1",
                },
                {
                    "name": "ollama",
                    "type": "ollama",
                    "api_key": None,
                    "base_url": "http://localhost:11434",
                },
            ],
            health_check_overrides={"unhealthy_threshold": 1, "healthy_threshold": 1},
        )
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        async def selective_ping(
            provider: Any, timeout: float
        ) -> None:
            if provider.name == "ollama":
                raise RuntimeError("ollama is down")

        with patch("llm_swap.health._ping_provider", side_effect=selective_ping):
            result = await checker.check_now()

        assert result["openai"] is True
        assert result["ollama"] is False


# ---------------------------------------------------------------------------
# HealthChecker — threshold behaviour during loop
# ---------------------------------------------------------------------------


class TestHealthCheckerThresholds:
    """Tests verifying threshold logic is respected during the check loop."""

    @pytest.mark.anyio
    async def test_provider_degraded_only_after_threshold_failures(self) -> None:
        cfg = _make_config(
            health_check_overrides={"unhealthy_threshold": 3, "healthy_threshold": 1}
        )
        state = HealthState(cfg.providers)
        checker = HealthChecker(cfg, state)

        failure_exc = RuntimeError("error")

        with patch(
            "llm_swap.health._ping_provider",
            new_callable=AsyncMock,
            side_effect=failure_exc,
        ):
            # First check: 1 failure, still healthy
            await checker._check_all_providers()
            assert state.is_healthy("openai") is True

            # Second check: 2 failures, still healthy
            await checker._check_all_providers()
            assert state.is_healthy("openai") is True

            # Third check: 3 failures, now degraded
            await checker._check_all_providers()
            assert state.is_healthy("openai") is False

    @pytest.mark.anyio
    async def test_provider_restored_only_after_threshold_successes(self) -> None:
        cfg = _make_config(
            health_check_overrides={"unhealthy_threshold": 1, "healthy_threshold": 2}
        )
        state = HealthState(cfg.providers)
        await state.mark_degraded("openai", reason="pre-degraded")
        checker = HealthChecker(cfg, state)

        with patch("llm_swap.health._ping_provider", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = None

            # First success: 1 success, still degraded
            await checker._check_all_providers()
            assert state.is_healthy("openai") is False

            # Second success: 2 successes, now healthy
            await checker._check_all_providers()
            assert state.is_healthy("openai") is True

    @pytest.mark.anyio
    async def test_recovery_resets_on_new_failure(self) -> None:
        """If a recovering provider fails again, the success counter resets."""
        cfg = _make_config(
            health_check_overrides={"unhealthy_threshold": 1, "healthy_threshold": 3}
        )
        state = HealthState(cfg.providers)
        await state.mark_degraded("openai")
        checker = HealthChecker(cfg, state)

        success_mock = AsyncMock(return_value=None)
        failure_mock = AsyncMock(side_effect=RuntimeError("err"))

        with patch("llm_swap.health._ping_provider", success_mock):
            await checker._check_all_providers()  # 1 success
            await checker._check_all_providers()  # 2 successes

        status = state.get_status("openai")
        assert status is not None
        assert status.consecutive_successes == 2

        with patch("llm_swap.health._ping_provider", failure_mock):
            await checker._check_all_providers()  # fails → resets success counter

        assert status.consecutive_successes == 0
        # Still degraded (was degraded before, failure counter incremented)
        assert state.is_healthy("openai") is False


# ---------------------------------------------------------------------------
# HealthChecker — snapshot integration with router health_state
# ---------------------------------------------------------------------------


class TestHealthStateSnapshotIntegration:
    """Verify snapshot() produces the dict expected by the router."""

    @pytest.mark.anyio
    async def test_snapshot_reflects_degraded_provider(self) -> None:
        providers = [
            _make_provider("openai"),
            _make_provider("mistral", "mistral", "sk", "https://api.mistral.ai/v1"),
        ]
        state = HealthState(providers)
        await state.mark_degraded("mistral")

        snap = state.snapshot()
        assert snap["openai"] is True
        assert snap["mistral"] is False

    @pytest.mark.anyio
    async def test_snapshot_after_recovery(self) -> None:
        providers = [_make_provider("openai")]
        state = HealthState(providers)
        await state.mark_degraded("openai")
        await state.mark_healthy("openai")

        snap = state.snapshot()
        assert snap["openai"] is True

    def test_snapshot_does_not_mutate_internal_state(self) -> None:
        providers = [_make_provider("openai")]
        state = HealthState(providers)
        snap = state.snapshot()
        snap["openai"] = False  # Mutate the copy
        # Internal state should not be affected
        assert state.is_healthy("openai") is True

"""Unit tests for CircuitBreaker runtime guardrails."""

import pytest
from agentlens.guardrails import (
    CircuitBreaker,
    CircuitBreakerAction,
    CircuitBreakerTrippedError,
)
from agentlens.models import AgentStep, ToolCall, ToolCallStatus


def test_circuit_breaker_max_steps():
    breaker = CircuitBreaker(max_steps=3, action_on_trip=CircuitBreakerAction.RAISE_EXCEPTION)

    for i in range(3):
        res = breaker.intercept_step(AgentStep(step_index=i, step_id=f"s{i}"))
        assert res["allowed"] is True

    # 4th step should trip
    with pytest.raises(CircuitBreakerTrippedError) as exc_info:
        breaker.intercept_step(AgentStep(step_index=3, step_id="s3"))

    assert "Step limit exceeded" in str(exc_info.value)
    assert breaker.is_tripped is True


def test_circuit_breaker_token_budget():
    breaker = CircuitBreaker(max_tokens=1000, action_on_trip=CircuitBreakerAction.RAISE_EXCEPTION)

    step1 = AgentStep(step_index=0, step_id="s0", prompt_tokens=600, completion_tokens=300)
    assert breaker.intercept_step(step1)["allowed"] is True

    # Exceeds 1000 tokens
    step2 = AgentStep(step_index=1, step_id="s1", prompt_tokens=200, completion_tokens=100)
    with pytest.raises(CircuitBreakerTrippedError) as exc_info:
        breaker.intercept_step(step2)

    assert "Token budget exceeded" in str(exc_info.value)


def test_circuit_breaker_consecutive_failures():
    breaker = CircuitBreaker(
        max_consecutive_tool_failures=2,
        action_on_trip=CircuitBreakerAction.HALT_GRACEFULLY,
    )

    failed_tc = ToolCall(id="c0", tool_name="api", arguments={}, status=ToolCallStatus.FAILED)
    step1 = AgentStep(step_index=0, step_id="s0", tool_calls=[failed_tc])
    res1 = breaker.intercept_step(step1)
    assert res1["allowed"] is True

    step2 = AgentStep(step_index=1, step_id="s1", tool_calls=[failed_tc])
    res2 = breaker.intercept_step(step2)
    assert res2["allowed"] is False
    assert res2["status"] == "TRIPPED"
    assert "Consecutive tool failure limit" in res2["reason"]


def test_circuit_breaker_inject_intervention():
    breaker = CircuitBreaker(
        max_repeated_tool_calls=2,
        action_on_trip=CircuitBreakerAction.INJECT_INTERVENTION,
    )

    tc = ToolCall(id="c0", tool_name="search", arguments={"q": "same"})
    step1 = AgentStep(step_index=0, step_id="s0", tool_calls=[tc])
    step2 = AgentStep(step_index=1, step_id="s1", tool_calls=[tc])

    breaker.intercept_step(step1)
    res = breaker.intercept_step(step2)

    assert res["allowed"] is False
    assert res["status"] == "TRIPPED"
    assert res["intervention"] is not None
    assert "[SYSTEM GUARD INTERVENTION]" in res["intervention"]


def test_circuit_breaker_reset():
    breaker = CircuitBreaker(max_steps=2, action_on_trip=CircuitBreakerAction.HALT_GRACEFULLY)
    breaker.intercept_step(AgentStep(step_index=0, step_id="s0"))
    breaker.intercept_step(AgentStep(step_index=1, step_id="s1"))
    res = breaker.intercept_step(AgentStep(step_index=2, step_id="s2"))
    assert res["status"] == "TRIPPED"

    breaker.reset()
    assert breaker.is_tripped is False
    assert breaker.trip_reason is None
    res_after = breaker.intercept_step(AgentStep(step_index=0, step_id="s0"))
    assert res_after["allowed"] is True

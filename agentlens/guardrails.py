"""Runtime Guardrails & Circuit Breaker Middleware.

Provides real-time runtime safety interceptors to halt runaway loops,
enforce token budgets, and prevent cascade failures in production agent systems.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from agentlens.detectors.loop_detector import LoopDetector
from agentlens.models import AgentStep, ToolCallStatus


class CircuitBreakerAction(str, Enum):
    """Action to take when a guardrail threshold is violated."""
    RAISE_EXCEPTION = "raise_exception"
    HALT_GRACEFULLY = "halt_gracefully"
    INJECT_INTERVENTION = "inject_intervention"


class CircuitBreakerTrippedError(Exception):
    """Raised when an agent violates safety guardrails or enters an infinite loop."""
    def __init__(self, reason: str, step_index: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"AgentLens CircuitBreaker Tripped at Step {step_index}: {reason}")
        self.reason = reason
        self.step_index = step_index
        self.details = details or {}


class CircuitBreaker:
    """Active runtime interceptor that prevents runaway agent loops and budget overflows."""

    def __init__(
        self,
        max_steps: int = 15,
        max_repeated_tool_calls: int = 3,
        max_consecutive_tool_failures: int = 3,
        max_tokens: int = 50_000,
        max_cost_usd: float = 0.50,
        action_on_trip: CircuitBreakerAction = CircuitBreakerAction.RAISE_EXCEPTION,
    ):
        self.max_steps = max_steps
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.max_consecutive_tool_failures = max_consecutive_tool_failures
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.action_on_trip = action_on_trip

        self._steps_history: List[AgentStep] = []
        self._loop_detector = LoopDetector(max_repeated_tool_calls=self.max_repeated_tool_calls)
        self._consecutive_tool_failures = 0
        self._total_tokens_consumed = 0
        self._is_tripped = False
        self._trip_reason: Optional[str] = None

    @property
    def is_tripped(self) -> bool:
        return self._is_tripped

    @property
    def trip_reason(self) -> Optional[str]:
        return self._trip_reason

    def reset(self) -> None:
        """Resets circuit breaker state for a new session."""
        self._steps_history.clear()
        self._consecutive_tool_failures = 0
        self._total_tokens_consumed = 0
        self._is_tripped = False
        self._trip_reason = None

    def intercept_step(self, step: AgentStep) -> Dict[str, Any]:
        """Inspects an in-flight step. Returns status dictionary and optional intervention directive."""
        self._steps_history.append(step)
        step_idx = step.step_index
        self._total_tokens_consumed += (step.prompt_tokens + step.completion_tokens)

        # 1. Check Max Steps
        if len(self._steps_history) > self.max_steps:
            return self._trip(
                reason=f"Step limit exceeded ({len(self._steps_history)} > {self.max_steps})",
                step_index=step_idx,
            )

        # 2. Check Token Budget
        if self._total_tokens_consumed > self.max_tokens:
            return self._trip(
                reason=f"Token budget exceeded ({self._total_tokens_consumed} > {self.max_tokens} tokens)",
                step_index=step_idx,
            )

        # 3. Check Consecutive Tool Failures
        has_failed_tool = any(tc.status == ToolCallStatus.FAILED for tc in step.tool_calls)
        if has_failed_tool:
            self._consecutive_tool_failures += 1
            if self._consecutive_tool_failures >= self.max_consecutive_tool_failures:
                return self._trip(
                    reason=(
                        f"Consecutive tool failure limit reached "
                        f"({self._consecutive_tool_failures} failed steps)"
                    ),
                    step_index=step_idx,
                )
        else:
            self._consecutive_tool_failures = 0

        # 4. Check Cyclic / Repeated Tool Calls
        loop_anomalies = self._loop_detector.detect_tool_loops(self._steps_history)
        if loop_anomalies:
            first_loop = loop_anomalies[0]
            return self._trip(
                reason=first_loop.message,
                step_index=step_idx,
            )

        return {
            "allowed": True,
            "status": "PASS",
            "message": "Step executed safely within guardrail limits.",
            "intervention": None,
        }

    def _trip(self, reason: str, step_index: int) -> Dict[str, Any]:
        """Handles tripping of the circuit breaker based on configured action."""
        self._is_tripped = True
        self._trip_reason = reason

        intervention_text = (
            f"[SYSTEM GUARD INTERVENTION] Circuit breaker tripped at step {step_index}. "
            f"Reason: {reason}. Halt further tool calls and synthesize final answer immediately."
        )

        if self.action_on_trip == CircuitBreakerAction.RAISE_EXCEPTION:
            raise CircuitBreakerTrippedError(reason=reason, step_index=step_index)

        return {
            "allowed": False,
            "status": "TRIPPED",
            "reason": reason,
            "step_index": step_index,
            "intervention": intervention_text if self.action_on_trip == CircuitBreakerAction.INJECT_INTERVENTION else None,
        }

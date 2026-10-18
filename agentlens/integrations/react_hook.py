"""Generic ReAct Loop Execution Wrapper.

Wraps custom or LangChain-style Thought-Action-Observation loops with
AgentLens tracing and real-time CircuitBreaker protection.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

from agentlens.guardrails import CircuitBreaker
from agentlens.models import ToolCallStatus
from agentlens.tracer import AgentTracer


class ReActWrapper:
    """Wraps and observes iterative ReAct reasoning cycles."""

    def __init__(self, tracer: AgentTracer, circuit_breaker: Optional[CircuitBreaker] = None):
        self.tracer = tracer
        self.circuit_breaker = circuit_breaker

    def parse_react_chunk(self, llm_output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extracts Thought, Action, and Action Input from standard ReAct formatted strings."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", llm_output, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*(.*?)(?=Action Input:|$)", llm_output, re.DOTALL | re.IGNORECASE)
        input_match = re.search(r"Action Input:\s*(.*?)(?=Observation:|$)", llm_output, re.DOTALL | re.IGNORECASE)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        action_input = input_match.group(1).strip() if input_match else None

        return thought, action, action_input

    def step(
        self,
        agent_name: str,
        thought: Optional[str],
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_fn: Optional[Callable] = None,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes a single ReAct step, recording trace and evaluating guardrails."""
        with self.tracer.step(agent_name=agent_name, role=role, thought=thought) as step_ctx:
            result = None
            status = ToolCallStatus.SUCCESS
            err_msg = None

            if tool_name and tool_fn:
                try:
                    result = tool_fn(**(tool_args or {}))
                except Exception as exc:
                    status = ToolCallStatus.FAILED
                    err_msg = str(exc)
                    result = f"Error: {exc}"

                step_ctx.record_tool_call(
                    tool_name=tool_name,
                    arguments=tool_args or {},
                    result=result,
                    status=status,
                    error_message=err_msg,
                )
                step_ctx.set_observation(str(result))

            # Inspect step with circuit breaker if configured
            if self.circuit_breaker:
                intercept_res = self.circuit_breaker.intercept_step(step_ctx.step)
                return {
                    "result": result,
                    "circuit_breaker": intercept_res,
                }

            return {"result": result, "status": status}

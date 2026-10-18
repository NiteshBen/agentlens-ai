"""CrewAI Step Callback Integration Hook for AgentLens.

Allows turnkey tracing and guardrail interception of CrewAI multi-agent crews.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agentlens.models import AgentStep, ToolCall, ToolCallStatus
from agentlens.tracer import AgentTracer


class CrewAITracerHook:
    """Callback listener designed to intercept CrewAI step and task executions."""

    def __init__(self, tracer: AgentTracer):
        self.tracer = tracer

    def step_callback(self, agent_output: Any) -> None:
        """Invoked after an individual agent step in CrewAI."""
        try:
            # Handle CrewAI output objects or dicts
            agent_name = getattr(agent_output, "agent", "CrewAgent")
            thought = getattr(agent_output, "thought", "")
            tool_name = getattr(agent_output, "tool", None)
            tool_input = getattr(agent_output, "tool_input", {})
            text_result = getattr(agent_output, "text", "")

            step = AgentStep(
                step_index=self.tracer.current_step_count,
                step_id=f"crew_step_{self.tracer.current_step_count}",
                agent_name=str(agent_name),
                thought=str(thought) if thought else None,
                observation=str(text_result) if text_result else None,
            )

            if tool_name:
                tc = ToolCall(
                    id=f"crew_call_{self.tracer.current_step_count}",
                    tool_name=str(tool_name),
                    arguments=tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
                    result=str(text_result),
                    status=ToolCallStatus.SUCCESS,
                )
                step.tool_calls.append(tc)

            self.tracer.record_step(step)
        except Exception:
            pass  # Non-intrusive safety guarantee

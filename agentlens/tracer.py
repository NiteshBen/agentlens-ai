"""AgentTracer - Live Execution Trajectory Recorder and Context Manager.

Provides non-intrusive runtime tracking, function decorators, and context managers
to record multi-agent reasoning steps, tool calls, latencies, and token costs.
"""

from __future__ import annotations

import functools
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

from agentlens.models import AgentStep, ToolCall, ToolCallStatus, Trajectory


class StepContext:
    """Active context manager for an in-flight agent step."""

    def __init__(
        self,
        step_index: int,
        agent_name: str,
        role: Optional[str] = None,
        thought: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.step = AgentStep(
            step_index=step_index,
            step_id=f"step_{uuid.uuid4().hex[:8]}",
            agent_name=agent_name,
            role=role,
            thought=thought,
            metadata=metadata or {},
        )
        self.start_time = time.perf_counter()

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Any] = None,
        latency_ms: float = 0.0,
        status: ToolCallStatus = ToolCallStatus.SUCCESS,
        error_message: Optional[str] = None,
    ) -> ToolCall:
        """Records a tool invocation executed during this step."""
        tc = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )
        self.step.tool_calls.append(tc)
        return tc

    def set_thought(self, thought: str) -> None:
        """Updates or sets the step's internal thought string."""
        self.step.thought = thought

    def set_observation(self, observation: str) -> None:
        """Sets the observation observed by the agent."""
        self.step.observation = observation

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Records token usage for this step."""
        self.step.prompt_tokens += prompt_tokens
        self.step.completion_tokens += completion_tokens

    def close(self) -> AgentStep:
        """Finalizes step duration and returns the completed AgentStep model."""
        self.step.latency_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return self.step


class AgentTracer:
    """Execution tracer for multi-agent trajectories."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        workflow_name: str = "Multi-Agent System",
        goal: str = "",
        cost_per_1k_prompt_tokens: float = 0.00015,
        cost_per_1k_completion_tokens: float = 0.00060,
    ):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self.workflow_name = workflow_name
        self.goal = goal
        self.cost_per_1k_prompt_tokens = cost_per_1k_prompt_tokens
        self.cost_per_1k_completion_tokens = cost_per_1k_completion_tokens

        self._steps: List[AgentStep] = []
        self._active_step_ctx: Optional[StepContext] = None
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._end_time: Optional[str] = None
        self._final_answer: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    @property
    def current_step_count(self) -> int:
        return len(self._steps)

    def set_goal(self, goal: str) -> None:
        """Updates the target goal for the session."""
        self.goal = goal

    def set_final_answer(self, final_answer: str) -> None:
        """Records the synthesized final response and marks completion timestamp."""
        self._final_answer = final_answer
        self._end_time = datetime.now(timezone.utc).isoformat()

    def add_metadata(self, key: str, value: Any) -> None:
        """Adds custom metadata to trajectory."""
        self._metadata[key] = value

    @contextmanager
    def step(
        self,
        agent_name: str = "Agent",
        role: Optional[str] = None,
        thought: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StepContext]:
        """Context manager for tracing an individual agent step."""
        ctx = StepContext(
            step_index=len(self._steps),
            agent_name=agent_name,
            role=role,
            thought=thought,
            metadata=metadata,
        )
        self._active_step_ctx = ctx
        try:
            yield ctx
        finally:
            finished_step = ctx.close()
            self._steps.append(finished_step)
            self._active_step_ctx = None

    def record_step(self, step: AgentStep) -> None:
        """Directly appends a pre-built AgentStep."""
        step.step_index = len(self._steps)
        self._steps.append(step)

    def trace_tool(self, name: Optional[str] = None) -> Callable:
        """Decorator to automatically trace a standalone tool function call."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_t = time.perf_counter()
                status = ToolCallStatus.SUCCESS
                err_msg = None
                result = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as exc:
                    status = ToolCallStatus.FAILED
                    err_msg = str(exc)
                    raise
                finally:
                    duration_ms = round((time.perf_counter() - start_t) * 1000, 2)
                    tc = ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        tool_name=tool_name,
                        arguments={"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}},
                        result=str(result) if result is not None else None,
                        latency_ms=duration_ms,
                        status=status,
                        error_message=err_msg,
                    )
                    # If currently inside an active step, record to active context
                    if self._active_step_ctx is not None:
                        self._active_step_ctx.step.tool_calls.append(tc)
                    elif self._steps:
                        self._steps[-1].tool_calls.append(tc)

            return wrapper
        return decorator

    def build_trajectory(self) -> Trajectory:
        """Synthesizes the complete immutable Trajectory object."""
        total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in self._steps)
        prompt_tokens = sum(s.prompt_tokens for s in self._steps)
        comp_tokens = sum(s.completion_tokens for s in self._steps)

        cost = (
            (prompt_tokens / 1000.0) * self.cost_per_1k_prompt_tokens
            + (comp_tokens / 1000.0) * self.cost_per_1k_completion_tokens
        )

        return Trajectory(
            session_id=self.session_id,
            workflow_name=self.workflow_name,
            goal=self.goal,
            steps=list(self._steps),
            start_time=self._start_time,
            end_time=self._end_time or datetime.now(timezone.utc).isoformat(),
            final_answer=self._final_answer,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost, 6),
            metadata=dict(self._metadata),
        )

    def save_json(self, file_path: str) -> str:
        """Saves current trajectory to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        traj = self.build_trajectory()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(traj.to_json(indent=2))
        return os.path.abspath(file_path)

    @classmethod
    def load_json(cls, file_path: str) -> Trajectory:
        """Loads a Trajectory from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return Trajectory.from_json(f.read())

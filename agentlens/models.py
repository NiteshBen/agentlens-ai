"""Core Data Models for AgentLens-AI.

Defines schemas for agent execution trajectories, steps, tool calls,
anomalies, and comprehensive evaluation reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event types in an agent execution lifecycle."""
    THOUGHT = "thought"
    ACTION_START = "action_start"
    ACTION_END = "action_end"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    INTER_AGENT_MSG = "inter_agent_msg"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    SYSTEM = "system"


class ToolCallStatus(str, Enum):
    """Status of an executed or attempted tool call."""
    SUCCESS = "success"
    FAILED = "failed"
    HALLUCINATED = "hallucinated"
    BLOCKED = "blocked"


class AnomalyType(str, Enum):
    """Taxonomy of agent execution anomalies and failure modes."""
    CIRCULAR_TOOL_CALL = "circular_tool_call"
    REPETITIVE_THOUGHT = "repetitive_thought"
    STATE_OSCILLATION = "state_oscillation"
    HALLUCINATED_TOOL = "hallucinated_tool"
    SCHEMA_VIOLATION = "schema_violation"
    CONTEXT_DRIFT = "context_drift"
    BUDGET_EXCEEDED = "budget_exceeded"
    INFINITE_LOOP = "infinite_loop"


class Severity(str, Enum):
    """Severity levels for detected anomalies."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Composite health state of an evaluated trajectory."""
    OPTIMAL = "optimal"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL_FAILURE = "critical_failure"


class ToolCall(BaseModel):
    """Represents a discrete tool invocation by an agent."""
    id: str = Field(..., description="Unique tool call identifier")
    tool_name: str = Field(..., description="Name of the invoked tool")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments passed to the tool")
    raw_arguments: Optional[str] = Field(default=None, description="Original raw arguments string from LLM")
    result: Optional[Any] = Field(default=None, description="Output returned by the tool")
    latency_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    status: ToolCallStatus = Field(default=ToolCallStatus.SUCCESS, description="Execution status")
    error_message: Optional[str] = Field(default=None, description="Error details if invocation failed")


class AgentStep(BaseModel):
    """Represents a single ReAct step or turn in an agent's execution."""
    step_index: int = Field(..., description="Sequential zero-based index of this step")
    step_id: str = Field(..., description="Unique ID for the step")
    agent_name: str = Field(default="Agent", description="Name of the agent taking this step")
    role: Optional[str] = Field(default=None, description="Agent role (e.g. Researcher, Coder)")
    thought: Optional[str] = Field(default=None, description="Internal reasoning or CoT reflection")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tools invoked in this step")
    observation: Optional[str] = Field(default=None, description="Observation from tool results or environment")
    prompt_tokens: int = Field(default=0, description="Prompt tokens consumed")
    completion_tokens: int = Field(default=0, description="Completion tokens generated")
    latency_ms: float = Field(default=0.0, description="Step duration in milliseconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom contextual metadata")


class Trajectory(BaseModel):
    """Complete execution trajectory of an agentic workflow."""
    session_id: str = Field(..., description="Unique session identifier")
    workflow_name: str = Field(default="Agentic Workflow", description="Name of the multi-agent task")
    goal: str = Field(..., description="The user objective, query, or task specification")
    steps: List[AgentStep] = Field(default_factory=list, description="Chronological sequence of agent steps")
    start_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Start time ISO timestamp",
    )
    end_time: Optional[str] = Field(default=None, description="End time ISO timestamp")
    final_answer: Optional[str] = Field(default=None, description="Synthesized final response or output")
    total_tokens: int = Field(default=0, description="Sum of tokens across all steps")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated total API spend in USD")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary workflow metadata")

    def to_json(self, indent: int = 2) -> str:
        """Serializes trajectory to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Trajectory":
        """Deserializes trajectory from JSON string."""
        return cls.model_validate_json(json_str)


class Anomaly(BaseModel):
    """An identified flaw, loop, or safety risk within an agent's execution."""
    anomaly_type: AnomalyType = Field(..., description="Category of anomaly")
    severity: Severity = Field(..., description="Severity level")
    title: str = Field(..., description="Concise anomaly title")
    message: str = Field(..., description="Detailed diagnostic description")
    step_indices: List[int] = Field(default_factory=list, description="Indices of steps involved")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of detection")
    recommendation: Optional[str] = Field(default=None, description="Remediation guidance")


class TrajectoryMetrics(BaseModel):
    """Quantified performance and reliability metrics of an agent trajectory."""
    total_steps: int = Field(default=0)
    total_tool_calls: int = Field(default=0)
    successful_tool_calls: int = Field(default=0)
    failed_or_hallucinated_tool_calls: int = Field(default=0)
    tool_reliability_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    step_efficiency_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    goal_alignment_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning_coherence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    loop_risk_index: float = Field(default=0.0, ge=0.0, le=1.0)
    drift_divergence_index: float = Field(default=0.0, ge=0.0, le=1.0)
    total_tokens: int = Field(default=0)
    total_latency_ms: float = Field(default=0.0)
    estimated_cost_usd: float = Field(default=0.0)
    overall_health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    health_status: HealthStatus = Field(default=HealthStatus.OPTIMAL)


class EvaluationReport(BaseModel):
    """Full diagnostic evaluation report for an agent trajectory."""
    session_id: str
    workflow_name: str
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: TrajectoryMetrics
    anomalies: List[Anomaly] = Field(default_factory=list)
    summary: str
    actionable_recommendations: List[str] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to JSON."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationReport":
        """Deserializes report from JSON."""
        return cls.model_validate_json(json_str)

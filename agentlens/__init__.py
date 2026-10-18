"""AgentLens-AI: Production Multi-Agent Observability, Trajectory Evaluation & Reasoning Drift Engine.

Empowers developers to monitor, evaluate, benchmark, and guard multi-agent systems
against infinite loops, tool-call hallucinations, context drift, and unbudgeted token costs.
"""

from agentlens.detectors import (
    DriftDetector,
    HallucinationDetector,
    LoopDetector,
    ToolParameterSchema,
    ToolSchema,
)
from agentlens.evaluator import TrajectoryEvaluator
from agentlens.guardrails import (
    CircuitBreaker,
    CircuitBreakerAction,
    CircuitBreakerTrippedError,
)
from agentlens.integrations import CrewAITracerHook, ReActWrapper
from agentlens.models import (
    AgentStep,
    Anomaly,
    AnomalyType,
    EvaluationReport,
    EventType,
    HealthStatus,
    Severity,
    ToolCall,
    ToolCallStatus,
    Trajectory,
    TrajectoryMetrics,
)
from agentlens.tracer import AgentTracer, StepContext
from agentlens.visualizer import render_html_dashboard

__version__ = "0.1.0"

__all__ = [
    # Core Tracer & Context
    "AgentTracer",
    "StepContext",
    # Evaluation Engine
    "TrajectoryEvaluator",
    "render_html_dashboard",
    # Guardrails
    "CircuitBreaker",
    "CircuitBreakerAction",
    "CircuitBreakerTrippedError",
    # Detectors
    "LoopDetector",
    "HallucinationDetector",
    "ToolSchema",
    "ToolParameterSchema",
    "DriftDetector",
    # Integrations
    "CrewAITracerHook",
    "ReActWrapper",
    # Data Models
    "Trajectory",
    "AgentStep",
    "ToolCall",
    "EvaluationReport",
    "TrajectoryMetrics",
    "Anomaly",
    "AnomalyType",
    "Severity",
    "HealthStatus",
    "ToolCallStatus",
    "EventType",
]

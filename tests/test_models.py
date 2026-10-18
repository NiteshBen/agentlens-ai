"""Unit tests for AgentLens data models."""

import pytest
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


def test_tool_call_model():
    tc = ToolCall(
        id="call_1",
        tool_name="search",
        arguments={"q": "pytest"},
        result="Success",
        latency_ms=45.2,
        status=ToolCallStatus.SUCCESS,
    )
    assert tc.tool_name == "search"
    assert tc.arguments == {"q": "pytest"}
    assert tc.latency_ms == 45.2
    assert tc.status == ToolCallStatus.SUCCESS


def test_agent_step_model():
    step = AgentStep(
        step_index=0,
        step_id="step_0",
        agent_name="Researcher",
        role="Search Analyst",
        thought="Need to check docs",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert step.step_index == 0
    assert step.agent_name == "Researcher"
    assert step.role == "Search Analyst"
    assert step.tool_calls == []
    assert step.prompt_tokens == 100


def test_trajectory_serialization():
    step = AgentStep(
        step_index=0,
        step_id="step_0",
        agent_name="AgentA",
        tool_calls=[
            ToolCall(id="tc1", tool_name="tool_a", arguments={"x": 1}, result="ok")
        ],
    )
    traj = Trajectory(
        session_id="sess_123",
        workflow_name="Test Flow",
        goal="Validate models",
        steps=[step],
        final_answer="All good",
        total_tokens=150,
        estimated_cost_usd=0.0002,
    )

    json_str = traj.to_json()
    assert "sess_123" in json_str
    assert "Validate models" in json_str

    deserialized = Trajectory.from_json(json_str)
    assert deserialized.session_id == traj.session_id
    assert len(deserialized.steps) == 1
    assert deserialized.steps[0].tool_calls[0].tool_name == "tool_a"


def test_anomaly_and_metrics_models():
    anom = Anomaly(
        anomaly_type=AnomalyType.CIRCULAR_TOOL_CALL,
        severity=Severity.HIGH,
        title="Loop Detected",
        message="Tool called 3x",
        step_indices=[1, 2, 3],
        confidence=0.95,
        recommendation="Break loop",
    )
    assert anom.severity == Severity.HIGH
    assert anom.step_indices == [1, 2, 3]

    metrics = TrajectoryMetrics(
        total_steps=5,
        total_tool_calls=4,
        successful_tool_calls=4,
        overall_health_score=92.5,
        health_status=HealthStatus.OPTIMAL,
    )
    assert metrics.overall_health_score == 92.5
    assert metrics.health_status == HealthStatus.OPTIMAL


def test_evaluation_report_serialization():
    metrics = TrajectoryMetrics(overall_health_score=88.0)
    rep = EvaluationReport(
        session_id="sess_abc",
        workflow_name="Report Flow",
        metrics=metrics,
        anomalies=[],
        summary="Clean run",
        actionable_recommendations=["Keep going"],
    )
    json_out = rep.to_json()
    loaded = EvaluationReport.from_json(json_out)
    assert loaded.session_id == "sess_abc"
    assert loaded.summary == "Clean run"
    assert loaded.metrics.overall_health_score == 88.0

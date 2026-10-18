"""Unit tests for TrajectoryEvaluator."""

import pytest
from agentlens.detectors.hallucination_detector import ToolSchema
from agentlens.evaluator import TrajectoryEvaluator
from agentlens.models import (
    AgentStep,
    HealthStatus,
    ToolCall,
    ToolCallStatus,
    Trajectory,
)


def test_evaluator_clean_trajectory():
    evaluator = TrajectoryEvaluator()

    traj = Trajectory(
        session_id="clean_sess",
        workflow_name="Clean Pipeline",
        goal="Compile weekly project status report",
        steps=[
            AgentStep(
                step_index=0, step_id="s0",
                thought="Gathering commit activity from repository",
                tool_calls=[ToolCall(id="c0", tool_name="git_log", arguments={"days": 7}, result="12 commits")],
                prompt_tokens=100, completion_tokens=50, latency_ms=45.0,
            ),
            AgentStep(
                step_index=1, step_id="s1",
                thought="Summarizing pull request metrics for project status",
                tool_calls=[ToolCall(id="c1", tool_name="gh_prs", arguments={"state": "merged"}, result="4 PRs")],
                prompt_tokens=150, completion_tokens=80, latency_ms=60.0,
            ),
        ],
        final_answer="Weekly project status report generated with 12 commits and 4 PRs merged.",
    )

    report = evaluator.evaluate(traj)
    m = report.metrics

    assert m.overall_health_score >= 85.0
    assert m.health_status in (HealthStatus.OPTIMAL, HealthStatus.HEALTHY)
    assert m.tool_reliability_rate == 1.0
    assert m.loop_risk_index == 0.0
    assert len(report.anomalies) == 0
    assert "Clean Pipeline" in report.summary


def test_evaluator_looping_trajectory_penalized():
    evaluator = TrajectoryEvaluator()

    # Trajectory with 3 identical repeated tool calls
    steps = []
    for i in range(4):
        steps.append(
            AgentStep(
                step_index=i, step_id=f"s{i}",
                thought="Trying to query the same broken endpoint",
                tool_calls=[
                    ToolCall(
                        id=f"c{i}",
                        tool_name="flaky_api",
                        arguments={"endpoint": "/status"},
                        status=ToolCallStatus.FAILED,
                        error_message="503 Service Unavailable",
                    )
                ],
            )
        )

    traj = Trajectory(
        session_id="loop_sess",
        workflow_name="Broken Loop Flow",
        goal="Query endpoint status",
        steps=steps,
    )

    report = evaluator.evaluate(traj)
    m = report.metrics

    assert m.overall_health_score < 60.0
    assert m.health_status in (HealthStatus.DEGRADED, HealthStatus.CRITICAL_FAILURE)
    assert m.tool_reliability_rate == 0.0
    assert m.loop_risk_index > 0.0
    assert len(report.anomalies) >= 1


def test_evaluator_register_tool_schemas():
    evaluator = TrajectoryEvaluator()
    evaluator.register_tool_schemas([
        ToolSchema(name="authorized_tool", required=["param1"])
    ])

    traj = Trajectory(
        session_id="unauth_sess",
        workflow_name="Hallucination Test",
        goal="Run tool",
        steps=[
            AgentStep(
                step_index=0, step_id="s0",
                tool_calls=[ToolCall(id="c0", tool_name="hallucinated_tool", arguments={})],
            )
        ],
    )

    report = evaluator.evaluate(traj)
    assert any(a.title.startswith("Hallucinated Tool") for a in report.anomalies)

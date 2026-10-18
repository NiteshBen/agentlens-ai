"""Unit tests for AgentTracer."""

import os
import pytest
from agentlens.models import ToolCallStatus
from agentlens.tracer import AgentTracer


def test_tracer_basic_step_recording():
    tracer = AgentTracer(session_id="test_sess_01", workflow_name="Tracer Test", goal="Test tracing")
    assert tracer.current_step_count == 0

    with tracer.step(agent_name="Analyst", role="Researcher", thought="Initial thought") as step_ctx:
        step_ctx.record_tool_call(
            tool_name="ping",
            arguments={"host": "127.0.0.1"},
            result="pong",
            latency_ms=15.0,
        )
        step_ctx.record_tokens(50, 25)
        step_ctx.set_observation("Ping received")

    assert tracer.current_step_count == 1
    traj = tracer.build_trajectory()
    assert len(traj.steps) == 1
    assert traj.steps[0].agent_name == "Analyst"
    assert traj.steps[0].thought == "Initial thought"
    assert len(traj.steps[0].tool_calls) == 1
    assert traj.steps[0].tool_calls[0].tool_name == "ping"
    assert traj.total_tokens == 75
    assert traj.estimated_cost_usd > 0


def test_tracer_tool_decorator():
    tracer = AgentTracer()

    @tracer.trace_tool(name="multiply_tool")
    def multiply(a: int, b: int) -> int:
        return a * b

    with tracer.step(agent_name="MathAgent") as s:
        res = multiply(4, 5)
        assert res == 20

    traj = tracer.build_trajectory()
    assert len(traj.steps) == 1
    assert len(traj.steps[0].tool_calls) == 1
    tc = traj.steps[0].tool_calls[0]
    assert tc.tool_name == "multiply_tool"
    assert tc.result == "20"
    assert tc.status == ToolCallStatus.SUCCESS


def test_tracer_tool_decorator_handles_error():
    tracer = AgentTracer()

    @tracer.trace_tool(name="failing_tool")
    def fail_op():
        raise ValueError("Division by zero")

    with tracer.step(agent_name="ErrorAgent") as s:
        with pytest.raises(ValueError):
            fail_op()

    traj = tracer.build_trajectory()
    assert len(traj.steps) == 1
    tc = traj.steps[0].tool_calls[0]
    assert tc.tool_name == "failing_tool"
    assert tc.status == ToolCallStatus.FAILED
    assert "Division by zero" in tc.error_message


def test_tracer_save_and_load_json(tmp_path):
    tracer = AgentTracer(workflow_name="Persist Test", goal="Save to disk")
    with tracer.step(agent_name="SaveAgent", thought="Thinking...") as s:
        s.record_tool_call("write", {"file": "a.txt"}, "ok")
    tracer.set_final_answer("Done saving")

    temp_file = str(tmp_path / "trajectory_test.json")
    saved_path = tracer.save_json(temp_file)
    assert os.path.exists(saved_path)

    loaded_traj = AgentTracer.load_json(saved_path)
    assert loaded_traj.workflow_name == "Persist Test"
    assert loaded_traj.final_answer == "Done saving"
    assert len(loaded_traj.steps) == 1

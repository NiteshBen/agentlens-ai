"""Unit tests for AgentLens integration hooks (CrewAI & ReAct)."""

import pytest
from types import SimpleNamespace
from agentlens.guardrails import CircuitBreaker, CircuitBreakerAction
from agentlens.integrations.crewai_hook import CrewAITracerHook
from agentlens.integrations.react_hook import ReActWrapper
from agentlens.models import ToolCallStatus
from agentlens.tracer import AgentTracer


def test_crewai_tracer_hook():
    tracer = AgentTracer(workflow_name="CrewAI Test")
    hook = CrewAITracerHook(tracer)

    # Mock CrewAI step output object
    mock_output = SimpleNamespace(
        agent="Researcher",
        thought="Need to search arXiv for papers",
        tool="search_arxiv",
        tool_input={"query": "multi-agent benchmarks"},
        text="Found 5 papers on multi-agent evaluation",
    )

    hook.step_callback(mock_output)

    assert tracer.current_step_count == 1
    traj = tracer.build_trajectory()
    assert traj.steps[0].agent_name == "Researcher"
    assert traj.steps[0].thought == "Need to search arXiv for papers"
    assert len(traj.steps[0].tool_calls) == 1
    assert traj.steps[0].tool_calls[0].tool_name == "search_arxiv"
    assert traj.steps[0].observation == "Found 5 papers on multi-agent evaluation"


def test_react_wrapper_parser():
    tracer = AgentTracer()
    wrapper = ReActWrapper(tracer)

    raw_llm = """
    Thought: I should search the database for current pricing.
    Action: sql_query
    Action Input: {"table": "prices"}
    Observation: Will be filled next
    """

    thought, action, action_input = wrapper.parse_react_chunk(raw_llm)
    assert thought == "I should search the database for current pricing."
    assert action == "sql_query"
    assert action_input == '{"table": "prices"}'


def test_react_wrapper_step_execution():
    tracer = AgentTracer()
    breaker = CircuitBreaker(max_steps=5, action_on_trip=CircuitBreakerAction.HALT_GRACEFULLY)
    wrapper = ReActWrapper(tracer, circuit_breaker=breaker)

    def dummy_tool(x: int) -> int:
        return x * 2

    res = wrapper.step(
        agent_name="MathAgent",
        thought="Multiplying value",
        tool_name="dummy_tool",
        tool_args={"x": 10},
        tool_fn=dummy_tool,
    )

    assert res["result"] == 20
    assert res["circuit_breaker"]["allowed"] is True
    assert tracer.current_step_count == 1

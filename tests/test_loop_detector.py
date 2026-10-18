"""Unit tests for LoopDetector."""

import pytest
from agentlens.detectors.loop_detector import LoopDetector, _jaccard_similarity
from agentlens.models import AgentStep, AnomalyType, ToolCall, ToolCallStatus


def test_jaccard_similarity_helper():
    assert _jaccard_similarity("hello world", "hello world") == 1.0
    assert _jaccard_similarity("completely different", "xyz 12345") == 0.0
    sim = _jaccard_similarity("I should search the web for documents", "I should search the web for files")
    assert sim >= 0.65


def test_detect_consecutive_tool_loop():
    detector = LoopDetector(max_repeated_tool_calls=2)

    # Agent executes 'web_search' with same argument 3 times
    steps = []
    for i in range(3):
        tc = ToolCall(id=f"c{i}", tool_name="web_search", arguments={"q": "AI agents 2026"})
        step = AgentStep(step_index=i, step_id=f"s{i}", agent_name="LoopBot", tool_calls=[tc])
        steps.append(step)

    anomalies = detector.analyze(steps)
    assert len(anomalies) >= 1
    loop_anom = [a for a in anomalies if a.anomaly_type == AnomalyType.CIRCULAR_TOOL_CALL][0]
    assert "web_search" in loop_anom.title
    assert set(loop_anom.step_indices) == {0, 1, 2}


def test_detect_cyclical_multi_step_pattern():
    detector = LoopDetector(max_cycle_length=3)

    # Pattern: ToolA -> ToolB -> ToolA -> ToolB
    steps = [
        AgentStep(
            step_index=0, step_id="s0",
            tool_calls=[ToolCall(id="c0", tool_name="tool_a", arguments={"mode": 1})],
        ),
        AgentStep(
            step_index=1, step_id="s1",
            tool_calls=[ToolCall(id="c1", tool_name="tool_b", arguments={"mode": 2})],
        ),
        AgentStep(
            step_index=2, step_id="s2",
            tool_calls=[ToolCall(id="c2", tool_name="tool_a", arguments={"mode": 1})],
        ),
        AgentStep(
            step_index=3, step_id="s3",
            tool_calls=[ToolCall(id="c3", tool_name="tool_b", arguments={"mode": 2})],
        ),
    ]

    anomalies = detector.analyze(steps)
    cycle_anoms = [a for a in anomalies if a.anomaly_type == AnomalyType.STATE_OSCILLATION]
    assert len(cycle_anoms) >= 1
    assert "tool_a -> tool_b" in cycle_anoms[0].title.lower()


def test_detect_repetitive_thoughts():
    detector = LoopDetector(thought_similarity_threshold=0.80)

    steps = [
        AgentStep(
            step_index=0, step_id="s0",
            thought="I need to inspect the database schema to find the right column name.",
        ),
        AgentStep(
            step_index=1, step_id="s1",
            thought="I need to inspect the database schema to find the right column name.",
        ),
    ]

    anomalies = detector.analyze(steps)
    thought_anoms = [a for a in anomalies if a.anomaly_type == AnomalyType.REPETITIVE_THOUGHT]
    assert len(thought_anoms) == 1
    assert thought_anoms[0].step_indices == [0, 1]


def test_clean_execution_has_no_loops():
    detector = LoopDetector()

    steps = [
        AgentStep(
            step_index=0, step_id="s0",
            thought="First I should search the web for current documentation.",
            tool_calls=[ToolCall(id="c0", tool_name="search", arguments={"q": "fastapi docs"})],
        ),
        AgentStep(
            step_index=1, step_id="s1",
            thought="Now I will write the code to disk based on what I discovered.",
            tool_calls=[ToolCall(id="c1", tool_name="write_file", arguments={"filename": "main.py"})],
        ),
    ]

    anomalies = detector.analyze(steps)
    assert len(anomalies) == 0

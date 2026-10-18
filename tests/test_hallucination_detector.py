"""Unit tests for HallucinationDetector."""

import pytest
from agentlens.detectors.hallucination_detector import HallucinationDetector
from agentlens.models import AgentStep, AnomalyType, ToolCall, ToolCallStatus


def test_hallucinated_tool_name():
    detector = HallucinationDetector()
    detector.register_tool(name="web_search", parameters={"query": "string"}, required=["query"])
    detector.register_tool(name="file_writer", parameters={"path": "string", "content": "string"})

    # Agent calls unknown tool 'web_search_database'
    step = AgentStep(
        step_index=0,
        step_id="s0",
        tool_calls=[ToolCall(id="c0", tool_name="web_search_database", arguments={"query": "test"})],
    )

    anomalies = detector.analyze([step])
    assert len(anomalies) == 1
    anom = anomalies[0]
    assert anom.anomaly_type == AnomalyType.HALLUCINATED_TOOL
    assert "web_search_database" in anom.title
    assert "web_search" in anom.message  # Fuzzy match suggestion


def test_missing_required_parameter():
    detector = HallucinationDetector()
    detector.register_tool(
        name="database_query",
        parameters={"sql": "string", "timeout": "integer"},
        required=["sql"],
    )

    # Tool called without 'sql'
    step = AgentStep(
        step_index=1,
        step_id="s1",
        tool_calls=[ToolCall(id="c1", tool_name="database_query", arguments={"timeout": 10})],
    )

    anomalies = detector.analyze([step])
    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == AnomalyType.SCHEMA_VIOLATION
    assert "Missing Required Parameter: 'sql'" in anomalies[0].title


def test_parameter_type_mismatch():
    detector = HallucinationDetector()
    detector.register_tool(
        name="resize_image",
        parameters={"width": "integer", "height": "integer"},
        required=["width", "height"],
    )

    # Agent passes string instead of integer
    step = AgentStep(
        step_index=0,
        step_id="s0",
        tool_calls=[ToolCall(id="c0", tool_name="resize_image", arguments={"width": "1080px", "height": 1080})],
    )

    anomalies = detector.analyze([step])
    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == AnomalyType.SCHEMA_VIOLATION
    assert "Type Mismatch" in anomalies[0].title


def test_tool_failure_status_detected():
    detector = HallucinationDetector()
    step = AgentStep(
        step_index=0,
        step_id="s0",
        tool_calls=[
            ToolCall(
                id="c0",
                tool_name="api_fetch",
                arguments={"url": "https://example.com"},
                status=ToolCallStatus.FAILED,
                error_message="Connection timed out",
            )
        ],
    )

    anomalies = detector.analyze([step])
    assert len(anomalies) == 1
    assert "Tool Execution Failed" in anomalies[0].title

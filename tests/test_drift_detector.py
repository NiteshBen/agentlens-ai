"""Unit tests for DriftDetector."""

import pytest
from collections import Counter
from agentlens.detectors.drift_detector import (
    DriftDetector,
    _cosine_similarity,
    _tokenize,
)
from agentlens.models import AgentStep, AnomalyType, ToolCall


def test_tokenize_filters_stopwords():
    tokens = _tokenize("What is the latest revenue for the quarterly report?")
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "revenue" in tokens
    assert "quarterly" in tokens
    assert "report" in tokens


def test_cosine_similarity_calculation():
    v1 = Counter(["python", "pytest", "testing"])
    v2 = Counter(["python", "pytest", "testing"])
    assert _cosine_similarity(v1, v2) == pytest.approx(1.0)

    v3 = Counter(["cooking", "recipe", "pasta"])
    assert _cosine_similarity(v1, v3) == 0.0


def test_drift_detection_on_divergent_steps():
    detector = DriftDetector(min_similarity_threshold=0.15, max_consecutive_drift=2)
    goal = "Analyze financial quarterly reports and extract net margins"

    steps = [
        # Step 0: Relevant
        AgentStep(
            step_index=0, step_id="s0",
            thought="Looking for financial quarterly reports online",
            tool_calls=[ToolCall(id="c0", tool_name="search", arguments={"q": "quarterly earnings report"})],
        ),
        # Step 1: Drifting
        AgentStep(
            step_index=1, step_id="s1",
            thought="Thinking about booking vacation flights to the tropical islands",
            tool_calls=[ToolCall(id="c1", tool_name="flights", arguments={"dest": "hawaii"})],
        ),
        # Step 2: Drifting
        AgentStep(
            step_index=2, step_id="s2",
            thought="Checking hotel reviews and surfing spots",
            tool_calls=[ToolCall(id="c2", tool_name="hotel_search", arguments={"city": "honolulu"})],
        ),
    ]

    alignments = detector.compute_step_alignments(goal, steps)
    assert len(alignments) == 3
    assert alignments[0] > alignments[1]
    assert alignments[0] > alignments[2]

    anomalies = detector.analyze(goal, steps)
    assert len(anomalies) >= 1
    assert any(a.anomaly_type == AnomalyType.CONTEXT_DRIFT for a in anomalies)


def test_aligned_steps_produce_no_drift_anomaly():
    detector = DriftDetector()
    goal = "Optimize PyTorch convolutional neural network training latency"

    steps = [
        AgentStep(
            step_index=0, step_id="s0",
            thought="Profiling PyTorch neural network training loop for bottleneck operations",
        ),
        AgentStep(
            step_index=1, step_id="s1",
            thought="Benchmarking mixed precision FP16 convolutional layers to improve latency",
        ),
        AgentStep(
            step_index=2, step_id="s2",
            thought="Reviewing PyTorch CUDA kernel profiler metrics",
        ),
    ]

    anomalies = detector.analyze(goal, steps)
    assert len(anomalies) == 0

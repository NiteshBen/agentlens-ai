"""Loop & Circular Reasoning Detector for Multi-Agent Systems.

Detects infinite loops, cyclic tool invocations, repetitive thoughts,
and oscillatory traps using sliding-window pattern matching and semantic hashing.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agentlens.models import AgentStep, Anomaly, AnomalyType, Severity


def _normalize_string(text: str) -> str:
    """Normalizes string for robust similarity comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _jaccard_similarity(str1: str, str2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity between two strings."""
    s1, s2 = _normalize_string(str1), _normalize_string(str2)
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    ngrams1 = {s1[i : i + n] for i in range(max(1, len(s1) - n + 1))}
    ngrams2 = {s2[i : i + n] for i in range(max(1, len(s2) - n + 1))}

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def _hash_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Generates canonical deterministic hash for a tool call."""
    canonical_args = json.dumps(arguments, sort_keys=True, default=str)
    raw = f"{tool_name.lower().strip()}::{canonical_args.lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class LoopDetector:
    """Analyzes agent execution steps for circular traps, loops, and repetitive reasoning."""

    def __init__(
        self,
        max_repeated_tool_calls: int = 2,
        thought_similarity_threshold: float = 0.82,
        max_cycle_length: int = 3,
    ):
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.thought_similarity_threshold = thought_similarity_threshold
        self.max_cycle_length = max_cycle_length

    def detect_tool_loops(self, steps: List[AgentStep]) -> List[Anomaly]:
        """Detects consecutive or cyclical repetitions of identical or near-identical tool calls."""
        anomalies: List[Anomaly] = []
        if len(steps) < 2:
            return anomalies

        # Flatten all tool calls with step mapping
        calls: List[Tuple[int, str, Dict[str, Any], str]] = []
        for step in steps:
            for tc in step.tool_calls:
                call_hash = _hash_tool_call(tc.tool_name, tc.arguments)
                calls.append((step.step_index, tc.tool_name, tc.arguments, call_hash))

        if len(calls) < 2:
            return anomalies

        # 1. Exact consecutive repetition: Tool(X, Y) repeated consecutively
        current_streak = 1
        streak_tool = calls[0][1]
        streak_steps = [calls[0][0]]

        for i in range(1, len(calls)):
            prev_call = calls[i - 1]
            curr_call = calls[i]

            # Check if exact hash match or high argument similarity for same tool
            is_same_hash = prev_call[3] == curr_call[3]
            is_same_tool = prev_call[1] == curr_call[1]
            arg_sim = (
                _jaccard_similarity(json.dumps(prev_call[2]), json.dumps(curr_call[2]))
                if is_same_tool
                else 0.0
            )

            if is_same_hash or (is_same_tool and arg_sim > 0.88):
                current_streak += 1
                streak_steps.append(curr_call[0])
            else:
                if current_streak >= self.max_repeated_tool_calls:
                    unique_steps = sorted(list(set(streak_steps)))
                    severity = Severity.CRITICAL if current_streak >= 4 else Severity.HIGH
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.CIRCULAR_TOOL_CALL,
                            severity=severity,
                            title=f"Repeated Tool Invocation Loop: '{streak_tool}'",
                            message=(
                                f"Tool '{streak_tool}' was executed {current_streak} times in succession "
                                f"with identical or near-identical parameters across steps {unique_steps}."
                            ),
                            step_indices=unique_steps,
                            confidence=min(1.0, 0.75 + (current_streak * 0.08)),
                            recommendation=(
                                f"Agent appears stuck in a loop calling '{streak_tool}'. Consider providing "
                                f"clearer stopping criteria, enforcing diverse search queries, or enabling "
                                f"AgentLens CircuitBreaker."
                            ),
                        )
                    )
                current_streak = 1
                streak_tool = curr_call[1]
                streak_steps = [curr_call[0]]

        # Check trailing streak
        if current_streak >= self.max_repeated_tool_calls:
            unique_steps = sorted(list(set(streak_steps)))
            severity = Severity.CRITICAL if current_streak >= 4 else Severity.HIGH
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.CIRCULAR_TOOL_CALL,
                    severity=severity,
                    title=f"Repeated Tool Invocation Loop: '{streak_tool}'",
                    message=(
                        f"Tool '{streak_tool}' was executed {current_streak} times consecutively "
                        f"across steps {unique_steps}."
                    ),
                    step_indices=unique_steps,
                    confidence=min(1.0, 0.75 + (current_streak * 0.08)),
                    recommendation=f"Halt repetitive calls to '{streak_tool}' and prompt agent to summarize findings.",
                )
            )

        # 2. Multi-step cyclic patterns (e.g., A -> B -> A -> B)
        hash_seq = [c[3] for c in calls]
        for cycle_len in range(2, min(self.max_cycle_length + 1, len(hash_seq) // 2 + 1)):
            for start in range(len(hash_seq) - (cycle_len * 2) + 1):
                p1 = hash_seq[start : start + cycle_len]
                p2 = hash_seq[start + cycle_len : start + (cycle_len * 2)]
                if p1 == p2:
                    involved_steps = sorted(
                        list({calls[idx][0] for idx in range(start, start + (cycle_len * 2))})
                    )
                    pattern_tools = [calls[start + k][1] for k in range(cycle_len)]
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.STATE_OSCILLATION,
                            severity=Severity.HIGH,
                            title=f"Cyclic Tool Pattern ({' -> '.join(pattern_tools)})",
                            message=(
                                f"Agent exhibits a periodic cycle of length {cycle_len}: "
                                f"[{' -> '.join(pattern_tools)}] repeating across steps {involved_steps}."
                            ),
                            step_indices=involved_steps,
                            confidence=0.92,
                            recommendation="Intervene with a reflection step to break the oscillatory loop.",
                        )
                    )
                    break

        return anomalies

    def detect_thought_loops(self, steps: List[AgentStep]) -> List[Anomaly]:
        """Detects repetitive or near-verbatim reasoning thoughts across agent steps."""
        anomalies: List[Anomaly] = []
        if len(steps) < 2:
            return anomalies

        for i in range(len(steps)):
            t1 = steps[i].thought
            if not t1 or len(t1.strip()) < 15:
                continue

            for j in range(i + 1, min(i + 4, len(steps))):
                t2 = steps[j].thought
                if not t2 or len(t2.strip()) < 15:
                    continue

                sim = _jaccard_similarity(t1, t2)
                if sim >= self.thought_similarity_threshold:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.REPETITIVE_THOUGHT,
                            severity=Severity.MEDIUM if sim < 0.95 else Severity.HIGH,
                            title="Repetitive Agent Reasoning",
                            message=(
                                f"Step {steps[i].step_index} and Step {steps[j].step_index} share "
                                f"{sim:.1%} reasoning similarity, indicating cognitive stagnation."
                            ),
                            step_indices=[steps[i].step_index, steps[j].step_index],
                            confidence=round(sim, 2),
                            recommendation="Inject intermediate feedback or re-prompt agent with different directives.",
                        )
                    )

        return anomalies

    def analyze(self, steps: List[AgentStep]) -> List[Anomaly]:
        """Executes full suite of loop detection rules."""
        tool_anomalies = self.detect_tool_loops(steps)
        thought_anomalies = self.detect_thought_loops(steps)

        # De-duplicate anomalies with overlapping titles and steps
        seen = set()
        unique: List[Anomaly] = []
        for anom in tool_anomalies + thought_anomalies:
            key = (anom.anomaly_type, tuple(anom.step_indices))
            if key not in seen:
                seen.add(key)
                unique.append(anom)

        return unique

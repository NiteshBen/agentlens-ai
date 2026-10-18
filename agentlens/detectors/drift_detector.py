"""Reasoning & Context Drift Detector.

Measures semantic alignment between the user's initial objective and agent reasoning
steps over time, detecting goal divergence, topic hijacking, and cognitive derailment.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from agentlens.models import AgentStep, Anomaly, AnomalyType, Severity

# Stopwords for lightweight, zero-dependency tokenization
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


def _tokenize(text: str) -> List[str]:
    """Tokenizes and filters stopwords from text."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """Computes cosine similarity between two term frequency count vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator / denominator)


class DriftDetector:
    """Monitors semantic alignment between task goal and agent step progression."""

    def __init__(self, min_similarity_threshold: float = 0.12, max_consecutive_drift: int = 2):
        self.min_similarity_threshold = min_similarity_threshold
        self.max_consecutive_drift = max_consecutive_drift

    def compute_step_alignments(self, goal: str, steps: List[AgentStep]) -> List[float]:
        """Calculates normalized cosine similarity of each step with the goal."""
        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return [1.0] * len(steps)

        goal_vec = Counter(goal_tokens)
        alignments: List[float] = []

        for step in steps:
            # Combine thought, tool names, arguments, and observations for context
            text_parts = [step.thought or ""]
            for tc in step.tool_calls:
                text_parts.append(tc.tool_name)
                text_parts.append(str(tc.arguments))
            if step.observation:
                text_parts.append(step.observation[:200])

            combined_text = " ".join(text_parts)
            step_tokens = _tokenize(combined_text)

            if not step_tokens:
                alignments.append(0.5)
                continue

            step_vec = Counter(step_tokens)
            sim = _cosine_similarity(goal_vec, step_vec)
            # Add slight boost if any key goal keyword appears directly
            overlap_bonus = 0.15 if set(goal_tokens) & set(step_tokens) else 0.0
            score = min(1.0, sim + overlap_bonus)
            alignments.append(round(score, 3))

        return alignments

    def compute_drift_index(self, alignments: List[float]) -> float:
        """Computes aggregate drift index (0.0 = aligned, 1.0 = total divergence)."""
        if not alignments:
            return 0.0
        avg_alignment = sum(alignments) / len(alignments)
        return round(max(0.0, min(1.0, 1.0 - avg_alignment)), 3)

    def analyze(self, goal: str, steps: List[AgentStep]) -> List[Anomaly]:
        """Detects sustained drift away from the system goal."""
        anomalies: List[Anomaly] = []
        if len(steps) < 3 or not goal:
            return anomalies

        alignments = self.compute_step_alignments(goal, steps)
        drifting_steps: List[int] = []

        for i, score in enumerate(alignments):
            # Later steps should remain reasonably aligned unless concluding
            if score < self.min_similarity_threshold:
                drifting_steps.append(steps[i].step_index)
            else:
                if len(drifting_steps) >= self.max_consecutive_drift:
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.CONTEXT_DRIFT,
                            severity=Severity.HIGH if len(drifting_steps) >= 3 else Severity.MEDIUM,
                            title="Context & Goal Drift Detected",
                            message=(
                                f"Agent reasoning diverged from user goal across consecutive steps "
                                f"{drifting_steps} (average semantic alignment < {self.min_similarity_threshold:.2f})."
                            ),
                            step_indices=list(drifting_steps),
                            confidence=0.88,
                            recommendation="Reinforce original user objective in agent memory or scratchpad.",
                        )
                    )
                drifting_steps = []

        # Check trailing drift
        if len(drifting_steps) >= self.max_consecutive_drift:
            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.CONTEXT_DRIFT,
                    severity=Severity.HIGH,
                    title="Terminal Context Drift",
                    message=(
                        f"Agent wandered off-topic during final steps {drifting_steps} "
                        f"without resolving original task."
                    ),
                    step_indices=list(drifting_steps),
                    confidence=0.90,
                    recommendation="Remind agent of initial user requirements before final answer generation.",
                )
            )

        return anomalies

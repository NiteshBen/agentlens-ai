"""TrajectoryEvaluator - Comprehensive Health and Quality Assessment Engine.

Computes quantitative trajectory metrics, runs multi-dimensional anomaly detectors,
and generates structured diagnostic evaluation reports.
"""

from __future__ import annotations

import os
from typing import List, Optional

from agentlens.detectors.drift_detector import DriftDetector
from agentlens.detectors.hallucination_detector import HallucinationDetector, ToolSchema
from agentlens.detectors.loop_detector import LoopDetector
from agentlens.models import (
    Anomaly,
    AnomalyType,
    EvaluationReport,
    HealthStatus,
    Severity,
    ToolCallStatus,
    Trajectory,
    TrajectoryMetrics,
)


class TrajectoryEvaluator:
    """Evaluates agent execution trajectories for reliability, safety, and efficiency."""

    def __init__(
        self,
        loop_detector: Optional[LoopDetector] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        drift_detector: Optional[DriftDetector] = None,
    ):
        self.loop_detector = loop_detector or LoopDetector()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.drift_detector = drift_detector or DriftDetector()

    def register_tool_schemas(self, schemas: List[ToolSchema]) -> None:
        """Registers expected schemas for tool hallucination checking."""
        for s in schemas:
            self.hallucination_detector._schemas[s.name.lower()] = s

    def evaluate(self, trajectory: Trajectory) -> EvaluationReport:
        """Runs full diagnostic evaluation suite across the given trajectory."""
        steps = trajectory.steps
        total_steps = len(steps)

        # 1. Run all anomaly detectors
        anomalies: List[Anomaly] = []
        anomalies.extend(self.loop_detector.analyze(steps))
        anomalies.extend(self.hallucination_detector.analyze(steps))
        anomalies.extend(self.drift_detector.analyze(trajectory.goal, steps))

        # 2. Tool Metrics
        total_tool_calls = sum(len(s.tool_calls) for s in steps)
        successful_calls = sum(
            1 for s in steps for tc in s.tool_calls if tc.status == ToolCallStatus.SUCCESS
        )
        failed_or_hallucinated = total_tool_calls - successful_calls

        tool_reliability_rate = (
            successful_calls / total_tool_calls if total_tool_calls > 0 else 1.0
        )

        # 3. Drift & Goal Alignment
        alignments = self.drift_detector.compute_step_alignments(trajectory.goal, steps)
        drift_divergence_index = self.drift_detector.compute_drift_index(alignments)
        base_alignment = max(0.0, 1.0 - drift_divergence_index)

        # Incorporate final answer alignment if provided
        if trajectory.final_answer and trajectory.goal:
            from collections import Counter
            from agentlens.detectors.drift_detector import _cosine_similarity, _tokenize
            g_toks = _tokenize(trajectory.goal)
            fa_toks = _tokenize(trajectory.final_answer)
            if g_toks and fa_toks:
                fa_sim = _cosine_similarity(Counter(g_toks), Counter(fa_toks))
                overlap_ratio = len(set(g_toks) & set(fa_toks)) / len(set(g_toks))
                fa_score = min(1.0, fa_sim + (overlap_ratio * 0.35))
                goal_alignment_score = round(min(1.0, (base_alignment * 0.5) + (fa_score * 0.5)), 3)
            else:
                goal_alignment_score = round(base_alignment, 3)
        else:
            goal_alignment_score = round(base_alignment, 3)

        # 4. Loop Risk Index
        loop_anoms = [
            a for a in anomalies if a.anomaly_type in (
                AnomalyType.CIRCULAR_TOOL_CALL,
                AnomalyType.REPETITIVE_THOUGHT,
                AnomalyType.STATE_OSCILLATION,
            )
        ]
        loop_risk_index = min(1.0, len(loop_anoms) * 0.35)

        # 5. Step Efficiency Ratio
        # Ideal steps heuristic: for tasks with tools, ~1-2 calls per step.
        # Penalizes excessive step counts or trajectories dominated by failed steps.
        if total_steps <= 1:
            step_efficiency_ratio = 1.0
        else:
            effective_steps = max(1, total_steps - len(loop_anoms) * 2 - failed_or_hallucinated)
            step_efficiency_ratio = round(max(0.1, min(1.0, effective_steps / total_steps)), 3)

        # 6. Reasoning Coherence Score
        repetitive_thoughts = [a for a in anomalies if a.anomaly_type == AnomalyType.REPETITIVE_THOUGHT]
        coherence_penalties = len(repetitive_thoughts) * 0.2
        reasoning_coherence_score = round(max(0.1, 1.0 - coherence_penalties), 3)

        # 7. Overall Health Score (0 - 100)
        # Weighted formula:
        #   30% Tool Reliability
        #   25% Loop Immunity (1.0 - loop_risk_index)
        #   20% Goal Alignment
        #   15% Step Efficiency
        #   10% Reasoning Coherence
        health_raw = (
            (tool_reliability_rate * 30.0)
            + ((1.0 - loop_risk_index) * 25.0)
            + (goal_alignment_score * 20.0)
            + (step_efficiency_ratio * 15.0)
            + (reasoning_coherence_score * 10.0)
        )

        # Deduct critical severity penalties
        critical_count = sum(1 for a in anomalies if a.severity == Severity.CRITICAL)
        health_raw -= critical_count * 25.0
        overall_health_score = round(max(0.0, min(100.0, health_raw)), 1)

        # Determine health status
        if overall_health_score >= 85.0:
            health_status = HealthStatus.OPTIMAL
        elif overall_health_score >= 70.0:
            health_status = HealthStatus.HEALTHY
        elif overall_health_score >= 45.0:
            health_status = HealthStatus.DEGRADED
        else:
            health_status = HealthStatus.CRITICAL_FAILURE

        total_latency = sum(s.latency_ms for s in steps)

        metrics = TrajectoryMetrics(
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            successful_tool_calls=successful_calls,
            failed_or_hallucinated_tool_calls=failed_or_hallucinated,
            tool_reliability_rate=round(tool_reliability_rate, 3),
            step_efficiency_ratio=step_efficiency_ratio,
            goal_alignment_score=goal_alignment_score,
            reasoning_coherence_score=reasoning_coherence_score,
            loop_risk_index=round(loop_risk_index, 3),
            drift_divergence_index=drift_divergence_index,
            total_tokens=trajectory.total_tokens,
            total_latency_ms=round(total_latency, 2),
            estimated_cost_usd=trajectory.estimated_cost_usd,
            overall_health_score=overall_health_score,
            health_status=health_status,
        )

        # 8. Executive Diagnostic Summary
        summary = self._generate_summary(trajectory, metrics, anomalies)

        # 9. Actionable Recommendations
        recs: List[str] = []
        for anom in anomalies:
            if anom.recommendation and anom.recommendation not in recs:
                recs.append(anom.recommendation)

        if not recs:
            recs.append("Trajectory demonstrated robust execution with zero detected loops or schema faults.")

        return EvaluationReport(
            session_id=trajectory.session_id,
            workflow_name=trajectory.workflow_name,
            metrics=metrics,
            anomalies=anomalies,
            summary=summary,
            actionable_recommendations=recs,
        )

    def _generate_summary(
        self,
        trajectory: Trajectory,
        metrics: TrajectoryMetrics,
        anomalies: List[Anomaly],
    ) -> str:
        """Synthesizes human-readable executive diagnostic commentary."""
        lines = [
            f"Trajectory '{trajectory.workflow_name}' completed {metrics.total_steps} steps "
            f"with an overall health score of {metrics.overall_health_score}/100 ({metrics.health_status.value.upper()})."
        ]

        if metrics.health_status in (HealthStatus.OPTIMAL, HealthStatus.HEALTHY):
            lines.append(
                f"Execution was efficient with a {metrics.tool_reliability_rate:.1%} tool reliability rate "
                f"and {metrics.goal_alignment_score:.1%} goal alignment."
            )
        else:
            lines.append(
                f"Execution suffered degradation: detected {len(anomalies)} anomalies "
                f"impacting reliability and cost."
            )

        if anomalies:
            criticals = [a for a in anomalies if a.severity in (Severity.CRITICAL, Severity.HIGH)]
            if criticals:
                lines.append(
                    f"Primary issue: {criticals[0].title} - {criticals[0].message}"
                )

        return " ".join(lines)

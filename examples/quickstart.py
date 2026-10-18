"""Quickstart Example for AgentLens-AI.

Demonstrates tracing a multi-agent research workflow, evaluating health metrics,
and generating a self-contained HTML dashboard.
"""

import os
from agentlens import AgentTracer, TrajectoryEvaluator, render_html_dashboard

def main():
    print("=" * 60)
    print("  AgentLens-AI Quickstart Demo")
    print("=" * 60)

    # 1. Initialize Tracer
    tracer = AgentTracer(
        workflow_name="Autonomous Tech Briefing Crew",
        goal="Investigate 2026 AI Agent framework advances and summarize key takeaways",
    )

    # 2. Step 1: Research Agent searches for trends
    print("\n[*] Step 1: Researcher agent running...")
    with tracer.step(agent_name="Researcher", role="Literature Analyst", thought="Searching arXiv and GitHub for new multi-agent evaluation frameworks") as step:
        step.record_tool_call(
            tool_name="arxiv_search",
            arguments={"query": "multi-agent reasoning loops and trajectory benchmarks", "max_results": 5},
            result="Found 5 papers detailing cyclic reasoning in ReAct loops and hallucinated tool calls.",
            latency_ms=132.5,
        )
        step.record_tokens(prompt_tokens=180, completion_tokens=95)
        step.set_observation("Identified key papers on agent trajectory drift and circuit-breaker patterns.")

    # 3. Step 2: Writer Agent synthesizes briefing
    print("[*] Step 2: Writer agent drafting executive summary...")
    with tracer.step(agent_name="Writer", role="Technical Writer", thought="Distilling findings into actionable executive takeaways") as step:
        step.record_tool_call(
            tool_name="markdown_exporter",
            arguments={"filename": "agent_report_2026.md", "title": "2026 Multi-Agent Evaluation Trends"},
            result="File written: agent_report_2026.md (1,240 words)",
            latency_ms=18.2,
        )
        step.record_tokens(prompt_tokens=240, completion_tokens=310)
        step.set_observation("Document drafted and saved successfully.")

    # 4. Finalize Trajectory
    tracer.set_final_answer(
        "Executive Summary: In 2026, enterprise multi-agent adoption requires runtime trajectory "
        "observability and loop circuit-breakers to mitigate unbudgeted token consumption and tool failures."
    )
    trajectory = tracer.build_trajectory()

    # 5. Evaluate Trajectory
    print("\n[*] Evaluating execution trajectory...")
    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(trajectory)

    print(f"[+] Health Score:       {report.metrics.overall_health_score}/100.0 ({report.metrics.health_status.value.upper()})")
    print(f"[+] Tool Reliability:   {report.metrics.tool_reliability_rate * 100:.1f}%")
    print(f"[+] Goal Alignment:     {report.metrics.goal_alignment_score * 100:.1f}%")
    print(f"[+] Total Tokens:       {report.metrics.total_tokens:,}")
    print(f"[+] Anomalies Detected: {len(report.anomalies)}")

    # 6. Render Dashboard
    out_html = os.path.join(os.path.dirname(__file__), "quickstart_dashboard.html")
    render_html_dashboard(trajectory, report, out_html)
    print(f"\n[+] Standalone HTML Dashboard rendered to:\n    {os.path.abspath(out_html)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

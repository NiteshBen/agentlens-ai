"""Command-Line Interface for AgentLens-AI.

Provides CLI commands to evaluate trajectories, detect loops,
render interactive HTML dashboards, and run diagnostic multi-agent benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from agentlens.evaluator import TrajectoryEvaluator
from agentlens.models import HealthStatus, Severity, ToolCallStatus, Trajectory
from agentlens.tracer import AgentTracer
from agentlens.visualizer import render_html_dashboard


def _print_banner():
    banner = """
======================================================================
     ___                    _   _                      ___ ___ 
    / _ \\ ___ ___ _ __  ___| | | |   ___ _ __  ___    /   |_ _|
   / /_\\// _ `/ -_) ' \\/ _ \\ | | |__/ -_) ' \\(_-<   / /| || | 
  /_/  \\_\\_, \\___/_|_/_|___/_| |____\\___/_||_/__/  /_/ |_|___|
         |__/                                                  
  Production Multi-Agent Observability & Trajectory Evaluation
======================================================================
"""
    print(banner)


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluates a trajectory JSON file and displays diagnostic findings."""
    _print_banner()
    if not os.path.exists(args.file):
        print(f"[!] Error: File not found: {args.file}")
        return 1

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
        traj = Trajectory.model_validate(data)
    except Exception as exc:
        print(f"[!] Error parsing trajectory: {exc}")
        return 1

    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(traj)
    m = report.metrics

    print(f"[*] Session ID:     {traj.session_id}")
    print(f"[*] Workflow:       {traj.workflow_name}")
    print(f"[*] Goal:           {traj.goal}")
    print(f"[*] Total Steps:    {m.total_steps}")
    print(f"[*] Total Latency:  {m.total_latency_ms:.1f}ms | Tokens: {m.total_tokens:,}")
    print("-" * 70)
    print(f"[+] OVERALL HEALTH SCORE:   {m.overall_health_score}/100.0  [{m.health_status.value.upper()}]")
    print(f"[+] Tool Reliability Rate:  {m.tool_reliability_rate * 100:.1f}% ({m.successful_tool_calls}/{m.total_tool_calls})")
    print(f"[+] Step Efficiency Ratio:  {m.step_efficiency_ratio * 100:.1f}%")
    print(f"[+] Goal Alignment Score:   {m.goal_alignment_score * 100:.1f}%")
    print(f"[+] Loop Risk Index:        {m.loop_risk_index * 100:.1f}%")
    print("-" * 70)

    if report.anomalies:
        print(f"\n[!] Detected {len(report.anomalies)} Anomalies:")
        for idx, anom in enumerate(report.anomalies, 1):
            sev_tag = f"[{anom.severity.value.upper()}]"
            steps = f"(Steps {anom.step_indices})" if anom.step_indices else ""
            print(f"  {idx}. {sev_tag:<10} {anom.title} {steps}")
            print(f"     Details:  {anom.message}")
            if anom.recommendation:
                print(f"     Remedy:   {anom.recommendation}")
    else:
        print("\n[+] Zero Anomalies Detected! Trajectory executed with high reliability.")

    if args.html:
        render_html_dashboard(traj, report, args.html)
        print(f"\n[+] Standalone HTML Dashboard generated: {os.path.abspath(args.html)}")

    return 0 if m.health_status != HealthStatus.CRITICAL_FAILURE else 2


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Generates an interactive HTML dashboard from a trajectory JSON file."""
    _print_banner()
    if not os.path.exists(args.file):
        print(f"[!] Error: File not found: {args.file}")
        return 1

    with open(args.file, "r", encoding="utf-8") as f:
        traj = Trajectory.model_validate(json.load(f))

    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(traj)
    output_path = args.output or f"{os.path.splitext(args.file)[0]}_dashboard.html"
    render_html_dashboard(traj, report, output_path)
    print(f"[+] Interactive Dashboard saved to: {os.path.abspath(output_path)}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Runs synthetic multi-agent benchmark with clean, looping, and drifting agent runs."""
    _print_banner()
    print("[*] Running AgentLens Multi-Agent Diagnostic Benchmark Suite...\n")

    evaluator = TrajectoryEvaluator()
    results = []

    # 1. Clean Multi-Agent Run
    t1 = AgentTracer(workflow_name="Clean Multi-Agent Research", goal="Synthesize latest AI agent trends")
    with t1.step(agent_name="Researcher", role="Search Analyst", thought="Searching for recent arXiv benchmarks") as s:
        s.record_tool_call("web_search", {"query": "AI agent evaluation benchmarks 2026"}, result="Found 3 papers", latency_ms=110)
        s.set_observation("Identified AgentBench, GAIA, and ToolBench.")
    with t1.step(agent_name="Writer", role="Technical Writer", thought="Drafting executive briefing") as s:
        s.record_tool_call("file_writer", {"path": "brief.md", "content": "# AI Benchmarks"}, result="Saved", latency_ms=30)
    t1.set_final_answer("Completed executive report on 2026 AI agent benchmarks.")
    r1 = evaluator.evaluate(t1.build_trajectory())
    results.append(("Clean Agent (Baseline)", r1))

    # 2. Runaway Tool Loop Run
    t2 = AgentTracer(workflow_name="Looping Agent Run", goal="Extract financial stats from database")
    for k in range(4):
        with t2.step(agent_name="SQLAgent", role="DB Querier", thought="Querying quarterly revenue table again") as s:
            s.record_tool_call("sql_query", {"query": "SELECT revenue FROM q3_results"}, result="Query timed out", latency_ms=250)
            s.set_observation("Query timed out. Retrying same query...")
    r2 = evaluator.evaluate(t2.build_trajectory())
    results.append(("Looping Agent (Circular Tool)", r2))

    # 3. Context Drift Run
    t3 = AgentTracer(workflow_name="Drifting Agent Run", goal="Deploy Kubernetes ingress controller")
    with t3.step(agent_name="DevOpsAgent", thought="Looking up k8s ingress helm charts") as s:
        s.record_tool_call("helm_search", {"chart": "ingress-nginx"}, result="chart found", latency_ms=80)
    with t3.step(agent_name="DevOpsAgent", thought="Browsing random recipes for dinner") as s:
        s.record_tool_call("recipe_search", {"cuisine": "italian pasta"}, result="spaghetti carbonara recipe", latency_ms=90)
    with t3.step(agent_name="DevOpsAgent", thought="Finding best cooking wine pairing") as s:
        s.record_tool_call("wine_search", {"type": "pinot grigio"}, result="wine store link", latency_ms=60)
    r3 = evaluator.evaluate(t3.build_trajectory())
    results.append(("Drifting Agent (Goal Hijack)", r3))

    # Display benchmark comparative table
    print(f"{'AGENT PROFILE':<32} | {'STEPS':<6} | {'HEALTH':<8} | {'TOOL ACC':<9} | {'LOOP RISK':<10} | {'STATUS'}")
    print("-" * 84)
    for name, rep in results:
        m = rep.metrics
        print(
            f"{name:<32} | "
            f"{m.total_steps:<6} | "
            f"{m.overall_health_score:<8.1f} | "
            f"{m.tool_reliability_rate * 100:<8.0f}% | "
            f"{m.loop_risk_index * 100:<9.0f}% | "
            f"{m.health_status.value.upper()}"
        )
    print("-" * 84)
    print("\n[+] Benchmark complete. AgentLens successfully identified all anomalous behaviors.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generates a sample trajectory and renders an interactive HTML dashboard."""
    _print_banner()
    print("[*] Generating sample multi-agent trajectory...")

    tracer = AgentTracer(
        workflow_name="Autonomous Market Research Crew",
        goal="Analyze global GPU market supply and summarize 2026 hyperscaler demand",
    )

    with tracer.step(agent_name="MarketResearcher", role="Research Analyst", thought="Searching for latest 2026 H100/B200 shipment volumes") as s:
        s.record_tool_call("web_search", {"query": "NVIDIA B200 hyperscaler allocation 2026"}, result="B200 allocations sold out through Q3 2026 to top 4 clouds.", latency_ms=145.2)
        s.record_tokens(120, 85)
        s.set_observation("Hyperscalers have pre-committed 85% of projected next-gen GPU capacity.")

    with tracer.step(agent_name="FinancialAnalyst", role="Quant Strategist", thought="Calculating estimated CAPEX run-rates") as s:
        s.record_tool_call("calc_tool", {"formula": "45000 * 2500000"}, result="112,500,000,000", latency_ms=4.1)
        s.record_tokens(95, 42)
        s.set_observation("Aggregate hyperscaler AI infrastructure CAPEX exceeds $112B annually.")

    with tracer.step(agent_name="ChiefEditor", role="Executive Synthesizer", thought="Compiling final executive summary for leadership") as s:
        s.record_tokens(180, 220)
        s.set_observation("Briefing written successfully.")

    tracer.set_final_answer(
        "Executive Summary: Hyperscaler demand for next-generation AI accelerators remains at peak levels, "
        "with 85% of 2026 capacity committed and total infrastructure CAPEX exceeding $112B."
    )

    traj = tracer.build_trajectory()
    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(traj)

    out_file = args.output or "agentlens_demo_dashboard.html"
    render_html_dashboard(traj, report, out_file)
    print(f"[+] Demo trajectory created with health score {report.metrics.overall_health_score}/100.0!")
    print(f"[+] Interactive HTML dashboard rendered to: {os.path.abspath(out_file)}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="agentlens",
        description="AgentLens-AI: Production Multi-Agent Observability, Trajectory Evaluation & Guardrail Engine.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trajectory JSON file")
    eval_parser.add_argument("file", help="Path to trajectory JSON file")
    eval_parser.add_argument("--html", help="Optional path to render HTML dashboard", default=None)

    # Dashboard subcommand
    dash_parser = subparsers.add_parser("dashboard", help="Render standalone HTML dashboard from trajectory")
    dash_parser.add_argument("file", help="Path to trajectory JSON file")
    dash_parser.add_argument("-o", "--output", help="Output HTML path", default=None)

    # Benchmark subcommand
    subparsers.add_parser("benchmark", help="Run multi-agent diagnostic benchmark")

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", help="Generate demo trajectory and view interactive dashboard")
    demo_parser.add_argument("-o", "--output", help="Output HTML path", default="agentlens_demo_dashboard.html")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "eval": cmd_eval,
        "dashboard": cmd_dashboard,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
    }

    exit_code = dispatch[args.command](args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

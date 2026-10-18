"""Zero-Dependency Interactive HTML Dashboard Visualizer.

Generates self-contained, responsive, dark-mode diagnostic dashboards
for multi-agent execution trajectories and evaluation reports without any CDN dependencies.
"""

from __future__ import annotations

import html
import json
import os
from typing import Optional

from agentlens.models import EvaluationReport, HealthStatus, Severity, Trajectory


def render_html_dashboard(
    trajectory: Trajectory,
    report: EvaluationReport,
    output_path: Optional[str] = None,
) -> str:
    """Renders a pixel-perfect, standalone HTML dashboard for trajectory diagnostics."""
    m = report.metrics

    # Health status color scheme
    if m.health_status == HealthStatus.OPTIMAL:
        status_color = "#10b981"  # Emerald
        status_bg = "rgba(16, 185, 129, 0.15)"
        status_text = "OPTIMAL"
    elif m.health_status == HealthStatus.HEALTHY:
        status_color = "#38bdf8"  # Cyan
        status_bg = "rgba(56, 189, 248, 0.15)"
        status_text = "HEALTHY"
    elif m.health_status == HealthStatus.DEGRADED:
        status_color = "#f59e0b"  # Amber
        status_bg = "rgba(245, 158, 11, 0.15)"
        status_text = "DEGRADED"
    else:
        status_color = "#f43f5e"  # Rose
        status_bg = "rgba(244, 63, 94, 0.15)"
        status_text = "CRITICAL FAILURE"

    # Build Anomalies HTML
    anomalies_html = ""
    if not report.anomalies:
        anomalies_html = """
        <div class="empty-state">
            <div class="icon-circle success">✓</div>
            <h4>Clean Execution</h4>
            <p>Zero loops, schema violations, or context drift anomalies detected.</p>
        </div>
        """
    else:
        for anom in report.anomalies:
            sev = anom.severity.value
            sev_badge_color = "#f43f5e" if sev in ("critical", "high") else ("#f59e0b" if sev == "medium" else "#38bdf8")
            steps_tag = ", ".join(f"Step {s}" for s in anom.step_indices) if anom.step_indices else "Global"
            anomalies_html += f"""
            <div class="anomaly-card {sev}">
                <div class="anom-header">
                    <span class="anom-badge" style="background: {sev_badge_color}22; color: {sev_badge_color}; border: 1px solid {sev_badge_color}66;">
                        {anom.severity.value.upper()}
                    </span>
                    <span class="anom-title">{html.escape(anom.title)}</span>
                    <span class="anom-steps">📍 {html.escape(steps_tag)}</span>
                </div>
                <div class="anom-body">{html.escape(anom.message)}</div>
                {f'<div class="anom-rec">💡 <strong>Remedy:</strong> {html.escape(anom.recommendation)}</div>' if anom.recommendation else ''}
            </div>
            """

    # Build Steps Timeline HTML
    steps_html = ""
    for step in trajectory.steps:
        tool_cards = ""
        for tc in step.tool_calls:
            tc_status = tc.status.value
            status_cls = "badge-success" if tc_status == "success" else "badge-error"
            args_str = json.dumps(tc.arguments, indent=2) if tc.arguments else "{}"
            result_str = str(tc.result)[:400] if tc.result is not None else "(No output recorded)"
            tool_cards += f"""
            <div class="tool-call-box">
                <div class="tool-call-hdr">
                    <span class="tool-name">⚡ {html.escape(tc.tool_name)}</span>
                    <span class="tool-latency">{tc.latency_ms:.1f}ms</span>
                    <span class="badge {status_cls}">{html.escape(tc_status.upper())}</span>
                </div>
                <div class="tool-args"><code>{html.escape(args_str)}</code></div>
                <div class="tool-res"><strong>Result:</strong> {html.escape(result_str)}</div>
            </div>
            """

        steps_html += f"""
        <div class="step-card" id="step-node-{step.step_index}">
            <div class="step-header">
                <div class="step-title-wrap">
                    <span class="step-num">Step {step.step_index}</span>
                    <span class="step-agent-role">{html.escape(step.agent_name)}{f' • {html.escape(step.role)}' if step.role else ''}</span>
                </div>
                <div class="step-meta">
                    <span>⏱ {step.latency_ms:.1f}ms</span>
                    <span>🪙 {step.prompt_tokens + step.completion_tokens} tokens</span>
                </div>
            </div>
            {f'<div class="step-thought">💭 <em>"{html.escape(step.thought)}"</em></div>' if step.thought else ''}
            {tool_cards}
            {f'<div class="step-obs">👁️ <strong>Observation:</strong> {html.escape(step.observation)}</div>' if step.observation else ''}
        </div>
        """

    # Build Recommendations HTML
    recs_html = "".join(f"<li>{html.escape(r)}</li>" for r in report.actionable_recommendations)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentLens-AI — {html.escape(trajectory.workflow_name)}</title>
    <style>
        :root {{
            --bg-base: #090d16;
            --bg-card: #111827;
            --bg-card-hover: #1e293b;
            --border: #1f293d;
            --border-highlight: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-base);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 32px 24px;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}
        /* Top Navigation Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 32px;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .brand-group {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .brand-logo {{
            background: linear-gradient(135deg, #0ea5e9, #a855f7);
            color: white;
            font-weight: 800;
            font-size: 22px;
            padding: 10px 18px;
            border-radius: 12px;
            letter-spacing: -0.5px;
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.3);
        }}
        .brand-titles h1 {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .brand-titles p {{
            color: var(--text-muted);
            font-size: 14px;
        }}
        .status-pill {{
            font-weight: 700;
            font-size: 13px;
            padding: 6px 16px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
            border: 1px solid currentColor;
        }}
        /* Goal Bar */
        .goal-bar {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 32px;
        }}
        .goal-bar strong {{
            color: var(--accent-cyan);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: block;
            margin-bottom: 4px;
        }}
        .goal-bar span {{
            font-size: 16px;
            color: var(--text-main);
        }}
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 36px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }}
        .metric-label {{
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -1px;
            line-height: 1;
            margin-bottom: 6px;
        }}
        .metric-sub {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        /* Main Layout */
        .main-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 28px;
        }}
        @media (max-width: 960px) {{
            .main-layout {{ grid-template-columns: 1fr; }}
        }}
        .section-header {{
            font-size: 18px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        /* Anomaly Cards */
        .anomaly-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 14px;
        }}
        .anomaly-card.critical, .anomaly-card.high {{
            border-left: 4px solid var(--accent-rose);
        }}
        .anomaly-card.medium {{
            border-left: 4px solid var(--accent-amber);
        }}
        .anom-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .anom-badge {{
            font-size: 10px;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .anom-title {{
            font-weight: 700;
            font-size: 15px;
            flex: 1;
        }}
        .anom-steps {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .anom-body {{
            font-size: 14px;
            color: #cbd5e1;
            margin-bottom: 8px;
        }}
        .anom-rec {{
            font-size: 13px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 6px;
            padding: 8px 12px;
            color: #38bdf8;
        }}
        /* Steps Timeline */
        .step-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            transition: border-color 0.2s;
        }}
        .step-card:hover {{
            border-color: rgba(56, 189, 248, 0.4);
        }}
        .step-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .step-num {{
            font-size: 12px;
            font-weight: 700;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            padding: 3px 8px;
            border-radius: 6px;
            margin-right: 8px;
        }}
        .step-agent-role {{
            font-weight: 700;
            font-size: 15px;
        }}
        .step-meta {{
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            gap: 12px;
        }}
        .step-thought {{
            font-size: 14px;
            color: #cbd5e1;
            margin-bottom: 12px;
            background: rgba(30, 41, 59, 0.5);
            padding: 10px 14px;
            border-radius: 8px;
            border-left: 2px solid var(--accent-purple);
        }}
        .tool-call-box {{
            background: #0d121f;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px;
            margin-top: 8px;
        }}
        .tool-call-hdr {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .tool-name {{
            font-weight: 700;
            color: var(--accent-cyan);
            font-size: 14px;
        }}
        .tool-latency {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .badge {{
            font-size: 10px;
            font-weight: 800;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #10b981; }}
        .badge-error {{ background: rgba(244, 63, 94, 0.2); color: #f43f5e; }}
        .tool-args code {{
            font-size: 12px;
            color: #94a3b8;
            font-family: Consolas, monospace;
            display: block;
            background: #090d16;
            padding: 6px 8px;
            border-radius: 4px;
            margin: 6px 0;
            white-space: pre-wrap;
        }}
        .tool-res {{
            font-size: 12px;
            color: #cbd5e1;
        }}
        .step-obs {{
            margin-top: 10px;
            font-size: 13px;
            color: #94a3b8;
        }}
        /* Summary & Recommendations */
        .summary-box {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 24px;
        }}
        .summary-box h3 {{
            font-size: 16px;
            margin-bottom: 8px;
            color: var(--accent-cyan);
        }}
        .recs-box ul {{
            padding-left: 20px;
            margin-top: 8px;
        }}
        .recs-box li {{
            font-size: 14px;
            color: #cbd5e1;
            margin-bottom: 6px;
        }}
        .empty-state {{
            text-align: center;
            padding: 32px 16px;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px dashed var(--border);
        }}
        .icon-circle {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            margin: 0 auto 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 700;
        }}
        .icon-circle.success {{ background: rgba(16, 185, 129, 0.2); color: #10b981; }}
        footer {{
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
            font-size: 13px;
            flex-wrap: wrap;
            gap: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand-group">
                <div class="brand-logo">AgentLens</div>
                <div class="brand-titles">
                    <h1>{html.escape(trajectory.workflow_name)}</h1>
                    <p>Session: <code>{html.escape(trajectory.session_id)}</code> • Evaluated: {report.evaluated_at[:19]}Z</p>
                </div>
            </div>
            <div>
                <span class="status-pill" style="color: {status_color}; background: {status_bg};">
                    ● {status_text}
                </span>
            </div>
        </header>

        <div class="goal-bar">
            <strong>Target Objective / System Goal</strong>
            <span>{html.escape(trajectory.goal)}</span>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Health Score</div>
                <div class="metric-value" style="color: {status_color};">{m.overall_health_score}</div>
                <div class="metric-sub">Out of 100.0 (Weighted)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Tool Reliability</div>
                <div class="metric-value" style="color: #10b981;">{m.tool_reliability_rate * 100:.0f}%</div>
                <div class="metric-sub">{m.successful_tool_calls}/{m.total_tool_calls} Successful Calls</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Step Efficiency</div>
                <div class="metric-value" style="color: #38bdf8;">{m.step_efficiency_ratio * 100:.0f}%</div>
                <div class="metric-sub">{m.total_steps} Total Execution Steps</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Loop Risk</div>
                <div class="metric-value" style="color: {'#10b981' if m.loop_risk_index < 0.2 else '#f43f5e'};">{m.loop_risk_index * 100:.0f}%</div>
                <div class="metric-sub">Oscillation & Repeat Risk</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Tokens & Spend</div>
                <div class="metric-value" style="color: #a855f7;">{m.total_tokens:,}</div>
                <div class="metric-sub">${m.estimated_cost_usd:.4f} Est. Spend ({m.total_latency_ms:.0f}ms)</div>
            </div>
        </div>

        <div class="summary-box">
            <h3>Executive Diagnostic Summary</h3>
            <p>{html.escape(report.summary)}</p>
        </div>

        <div class="main-layout">
            <div>
                <div class="section-header">
                    <span>⚠️ Detected Anomalies & Diagnostics ({len(report.anomalies)})</span>
                </div>
                {anomalies_html}

                <div class="summary-box recs-box" style="margin-top: 24px;">
                    <h3>Actionable Recommendations</h3>
                    <ul>{recs_html}</ul>
                </div>
            </div>

            <div>
                <div class="section-header">
                    <span>📜 Execution Timeline & Steps ({len(trajectory.steps)})</span>
                </div>
                {steps_html}
            </div>
        </div>

        <footer>
            <span>AgentLens-AI — Production Multi-Agent Observability & Trajectory Evaluation</span>
            <span>Developed by Nitesh Beniwal</span>
        </footer>
    </div>
</body>
</html>
"""

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)

    return html_template

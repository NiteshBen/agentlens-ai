"""Unit tests for HTML Dashboard Visualizer."""

import os
import pytest
from agentlens.evaluator import TrajectoryEvaluator
from agentlens.models import AgentStep, ToolCall, Trajectory
from agentlens.visualizer import render_html_dashboard


def test_render_html_dashboard_string():
    traj = Trajectory(
        session_id="sess_vis_01",
        workflow_name="Visualizer Test",
        goal="Ensure clean rendering",
        steps=[
            AgentStep(
                step_index=0, step_id="s0",
                agent_name="VisAgent",
                role="UI Engineer",
                thought="Designing clean dashboard",
                tool_calls=[ToolCall(id="c0", tool_name="css_render", arguments={"theme": "dark"}, result="ok")],
            )
        ],
    )

    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(traj)

    html_str = render_html_dashboard(traj, report)
    assert "<!DOCTYPE html>" in html_str
    assert "AgentLens" in html_str
    assert "Visualizer Test" in html_str
    assert "sess_vis_01" in html_str
    assert "Ensure clean rendering" in html_str
    assert "VisAgent" in html_str
    assert "css_render" in html_str
    # Standalone check: zero external CDN scripts or stylesheet links
    assert "http://" not in html_str
    assert "https://" not in html_str


def test_render_html_dashboard_file(tmp_path):
    traj = Trajectory(
        session_id="sess_vis_02",
        workflow_name="File Render Test",
        goal="Test saving HTML",
        steps=[],
    )
    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(traj)

    out_file = str(tmp_path / "dashboard.html")
    render_html_dashboard(traj, report, out_file)

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "File Render Test" in content

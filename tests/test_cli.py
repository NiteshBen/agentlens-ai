"""Unit tests for AgentLens CLI commands."""

import argparse
import json
import os
import pytest
from agentlens.cli import cmd_benchmark, cmd_dashboard, cmd_demo, cmd_eval


def test_cli_eval_command(tmp_path):
    data = {
        "session_id": "cli_sess_01",
        "workflow_name": "CLI Test Flow",
        "goal": "Verify CLI eval",
        "steps": [],
    }
    json_path = str(tmp_path / "test_traj.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    args = argparse.Namespace(file=json_path, html=None)
    code = cmd_eval(args)
    assert code == 0


def test_cli_eval_missing_file():
    args = argparse.Namespace(file="non_existent_file.json", html=None)
    code = cmd_eval(args)
    assert code == 1


def test_cli_dashboard_command(tmp_path):
    data = {
        "session_id": "cli_sess_02",
        "workflow_name": "CLI Dashboard Flow",
        "goal": "Verify CLI dashboard",
        "steps": [],
    }
    json_path = str(tmp_path / "test_traj.json")
    html_path = str(tmp_path / "out.html")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    args = argparse.Namespace(file=json_path, output=html_path)
    code = cmd_dashboard(args)
    assert code == 0
    assert os.path.exists(html_path)


def test_cli_benchmark_command():
    args = argparse.Namespace()
    code = cmd_benchmark(args)
    assert code == 0


def test_cli_demo_command(tmp_path):
    out_html = str(tmp_path / "demo.html")
    args = argparse.Namespace(output=out_html)
    code = cmd_demo(args)
    assert code == 0
    assert os.path.exists(out_html)

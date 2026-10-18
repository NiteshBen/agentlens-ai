# AgentLens-AI 🔍⚡

[![Pytest Tests](https://img.shields.io/badge/pytest-40%20passed-10b981?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/NiteshBen/agentlens-ai)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-38bdf8?style=for-the-badge&logo=python&logoColor=white)](https://github.com/NiteshBen/agentlens-ai)
[![License: MIT](https://img.shields.io/badge/license-MIT-a855f7?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Local-First](https://img.shields.io/badge/cloud%20lock--in-zero%20%240%20cost-10b981?style=for-the-badge)](https://github.com/NiteshBen/agentlens-ai)
[![Platform](https://img.shields.io/badge/platform-cross--platform-slate?style=for-the-badge)](https://github.com/NiteshBen/agentlens-ai)

> **Production-Grade Multi-Agent Observability, Trajectory Evaluation & Reasoning Drift Engine in Python.**  
> Monitor, diagnose, benchmark, and guard autonomous multi-agent systems against infinite loops, tool hallucinations, context drift, and runaway token budgets.

---

## 🎯 The Production Multi-Agent Challenge

Multi-agent architectures (**CrewAI, LangGraph, AutoGen, and custom ReAct loops**) represent the frontier of generative AI engineering. However, deploying autonomous agents into enterprise production exposes severe reliability blind spots:

1. **Circular Reasoning & Tool Loops:** Agents repeatedly query identical or oscillating tools ($Tool_A \to Tool_B \to Tool_A \to Tool_B$) when unexpected results occur, draining token budgets.
2. **Tool-Call Hallucinations:** Large language models frequently invent non-existent tool names or pass malformed arguments that crash downstream services.
3. **Context & Semantic Drift:** Over multi-turn reasoning cycles, agents wander away from the original user objective and hijack context with tangential outputs.
4. **Lack of Runtime Safety & Observability:** Most workflows execute as black boxes with no automated circuit-breakers to halt runaway agents or audit execution trajectories.

**AgentLens-AI** solves this with an offline-first, mathematically grounded Python engine that provides real-time event tracing, sliding-window cycle detection, schema guardrails, trajectory scoring, and responsive dark-mode HTML dashboards.

---

## 🏗️ Architectural Overview

```
                      +---------------------------------------+
                      |         Target User Objective         |
                      +---------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            AgentLens-AI Runtime Engine                            |
|                                                                                   |
|  +--------------------+    +----------------------+    +-----------------------+  |
|  |    AgentTracer     |--->|    CircuitBreaker    |--->|  Framework Hooks      |  |
|  | Context & Decors   |    | Real-Time Interceptor|    | CrewAI / ReAct        |  |
|  +--------------------+    +----------------------+    +-----------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                     Multi-Dimensional Diagnostic Detectors                  |  |
|  |                                                                             |  |
|  |  +---------------------+  +-------------------------+  +-----------------+  |  |
|  |  |    LoopDetector     |  |  HallucinationDetector  |  |  DriftDetector  |  |  |
|  |  | Cycles & Oscillate  |  | Schemas & Type Checks   |  | Cosine Alignment|  |  |
|  |  +---------------------+  +-------------------------+  +-----------------+  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                          TrajectoryEvaluator (0 - 100)                      |  |
|  |   Health Score • Step Efficiency • Tool Reliability • Loop Risk • Token Cost|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Interactive Deliverables & Outputs                         |
|                                                                                   |
|  +--------------------------------+            +-------------------------------+  |
|  | Standalone Dark HTML Dashboard |            | Production CLI & JSON Audits  |  |
|  | Zero-CDN Interactive Analytics |            | CI/CD Gates & Terminal Reports|  |
|  +--------------------------------+            +-------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## ✨ Core Features

* **⚡ Zero Cloud Lock-In & 100% Local Execution:** Runs completely offline with deterministic algorithms and pure Python heuristics. $0 operational spend.
* **🔁 Sliding-Window Loop Detection:** Detects circular tool invocations, multi-step periodic cycles, and repetitive cognitive thoughts using canonical hashing and character n-gram Jaccard similarity.
* **🛡️ Tool Schema & Hallucination Guard:** Registers tool signatures (Pydantic/JSON schemas) and validates parameter names, mandatory arguments, types, and fuzzy matches hallucinated tools to valid candidates.
* **🧭 Semantic Context Drift Tracking:** Calculates TF-IDF cosine alignment between user objectives and step progressions, flagging topic hijacking before completion.
* **🚨 Real-Time CircuitBreaker:** Intercepts in-flight steps with configurable trip actions (`RAISE_EXCEPTION`, `HALT_GRACEFULLY`, `INJECT_INTERVENTION`) to stop runaway agents.
* **📊 Standalone Interactive HTML Dashboard:** Generates single-file, zero-CDN dark-mode diagnostic dashboards featuring KPI cards, anomaly alerts, step timelines, and JSON exports.
* **🔌 Native Turnkey Hooks:** Drop-in integration for **CrewAI** step callbacks and custom **ReAct** thought-action loops.
* **🧪 Production Test Suite:** 40 unit tests passing with `pytest` in <0.3s.

---

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/NiteshBen/agentlens-ai.git
cd agentlens-ai

# Install in editable mode
pip install -e .
```

### 1. Tracing an Autonomous Workflow

```python
from agentlens import AgentTracer, TrajectoryEvaluator, render_html_dashboard

# Initialize tracer with session objective
tracer = AgentTracer(
    workflow_name="Autonomous Market Research Crew",
    goal="Investigate 2026 AI accelerator demand and synthesize market brief",
)

# Trace Step 1: Research Agent
with tracer.step(agent_name="Researcher", role="Market Analyst", thought="Searching for next-gen GPU shipments") as step:
    step.record_tool_call(
        tool_name="web_search",
        arguments={"query": "hyperscaler AI CAPEX 2026"},
        result="Found hyperscaler commitments exceeding $112B annually.",
        latency_ms=120.4,
    )
    step.record_tokens(prompt_tokens=140, completion_tokens=85)
    step.set_observation("Hyperscaler CAPEX remains at record levels.")

# Trace Step 2: Writer Agent
with tracer.step(agent_name="Writer", role="Technical Writer", thought="Drafting executive summary") as step:
    step.record_tokens(prompt_tokens=220, completion_tokens=310)
    step.set_observation("Executive briefing compiled.")

tracer.set_final_answer("Completed market brief on 2026 AI infrastructure investments.")
trajectory = tracer.build_trajectory()

# Run Diagnostic Evaluation
evaluator = TrajectoryEvaluator()
report = evaluator.evaluate(trajectory)

print(f"Health Score: {report.metrics.overall_health_score}/100.0 [{report.metrics.health_status.value.upper()}]")
print(f"Tool Reliability: {report.metrics.tool_reliability_rate * 100:.1f}%")

# Generate Standalone Interactive HTML Dashboard
render_html_dashboard(trajectory, report, output_path="dashboard.html")
```

### 2. Runtime Loop Prevention with CircuitBreaker

```python
from agentlens import CircuitBreaker, CircuitBreakerAction, AgentStep, ToolCall

# Halt agent if the same tool is repeated 3 times
breaker = CircuitBreaker(
    max_repeated_tool_calls=3,
    action_on_trip=CircuitBreakerAction.INJECT_INTERVENTION,
)

step = AgentStep(
    step_index=0,
    step_id="step_01",
    tool_calls=[ToolCall(id="c1", tool_name="fetch_api", arguments={"id": 42})],
)

status = breaker.intercept_step(step)
if status["status"] == "TRIPPED":
    print(f"Emergency Alert: {status['intervention']}")
```

### 3. CrewAI Integration Hook

```python
from crewai import Agent, Crew, Task
from agentlens import AgentTracer, CrewAITracerHook

tracer = AgentTracer(workflow_name="CrewAI Content Pipeline", goal="Write AI report")
hook = CrewAITracerHook(tracer)

# Hook directly into CrewAI agent step callback
researcher = Agent(
    role="Researcher",
    goal="Search industry trends",
    step_callback=hook.step_callback,
)
```

---

## 💻 Command-Line Interface (CLI)

AgentLens includes a full-featured CLI for CI/CD pipelines and manual evaluation:

```bash
# 1. Evaluate a trajectory JSON file
agentlens eval examples/sample_trajectories.json

# 2. Generate an interactive HTML dashboard
agentlens dashboard examples/sample_trajectories.json -o audit_dashboard.html

# 3. Run multi-agent diagnostic benchmarks (Clean vs Looping vs Drifting)
agentlens benchmark

# 4. Generate demo trajectory and view live dashboard
agentlens demo -o demo_dashboard.html
```

---

## 📊 Benchmark Suite Results

AgentLens was benchmarked across three synthetic agent profiles representing real-world production scenarios:

| Agent Execution Profile | Steps | Health Score (/100) | Tool Accuracy | Loop Risk | Diagnostic Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Clean Agent (Baseline)** | 2 | **88.5** | 100% | 0% | `OPTIMAL` |
| **Looping Agent (Circular Tool)** | 4 | **9.8** | 100% | 100% | `CRITICAL_FAILURE` |
| **Drifting Agent (Goal Hijack)** | 3 | **83.2** | 100% | 0% | `HEALTHY` |

---

## 🧪 Testing

AgentLens-AI comes with a comprehensive unit test suite:

```bash
python -m pytest tests -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1
collected 40 items

tests/test_cli.py::test_cli_eval_command PASSED                          [  2%]
tests/test_cli.py::test_cli_eval_missing_file PASSED                     [  5%]
tests/test_cli.py::test_cli_dashboard_command PASSED                     [  7%]
tests/test_cli.py::test_cli_benchmark_command PASSED                     [ 10%]
tests/test_cli.py::test_cli_demo_command PASSED                          [ 12%]
tests/test_drift_detector.py::test_tokenize_filters_stopwords PASSED     [ 15%]
tests/test_drift_detector.py::test_cosine_similarity_calculation PASSED  [ 17%]
tests/test_drift_detector.py::test_drift_detection_on_divergent_steps PASSED [ 20%]
tests/test_drift_detector.py::test_aligned_steps_produce_no_drift_anomaly PASSED [ 22%]
tests/test_evaluator.py::test_evaluator_clean_trajectory PASSED          [ 25%]
tests/test_evaluator.py::test_evaluator_looping_trajectory_penalized PASSED [ 27%]
tests/test_evaluator.py::test_evaluator_register_tool_schemas PASSED     [ 30%]
tests/test_guardrails.py::test_circuit_breaker_max_steps PASSED          [ 32%]
tests/test_guardrails.py::test_circuit_breaker_token_budget PASSED       [ 35%]
tests/test_guardrails.py::test_circuit_breaker_consecutive_failures PASSED [ 37%]
tests/test_guardrails.py::test_circuit_breaker_inject_intervention PASSED [ 40%]
tests/test_guardrails.py::test_circuit_breaker_reset PASSED              [ 42%]
tests/test_hallucination_detector.py::test_hallucinated_tool_name PASSED [ 45%]
tests/test_hallucination_detector.py::test_missing_required_parameter PASSED [ 47%]
tests/test_hallucination_detector.py::test_parameter_type_mismatch PASSED [ 50%]
tests/test_hallucination_detector.py::test_tool_failure_status_detected PASSED [ 52%]
tests/test_integrations.py::test_crewai_tracer_hook PASSED               [ 55%]
tests/test_integrations.py::test_react_wrapper_parser PASSED             [ 57%]
tests/test_integrations.py::test_react_wrapper_step_execution PASSED     [ 60%]
tests/test_loop_detector.py::test_jaccard_similarity_helper PASSED       [ 62%]
tests/test_loop_detector.py::test_detect_consecutive_tool_loop PASSED    [ 65%]
tests/test_loop_detector.py::test_detect_cyclical_multi_step_pattern PASSED [ 67%]
tests/test_loop_detector.py::test_detect_repetitive_thoughts PASSED      [ 70%]
tests/test_loop_detector.py::test_clean_execution_has_no_loops PASSED    [ 72%]
tests/test_models.py::test_tool_call_model PASSED                        [ 75%]
tests/test_models.py::test_agent_step_model PASSED                       [ 77%]
tests/test_models.py::test_trajectory_serialization PASSED               [ 80%]
tests/test_models.py::test_anomaly_and_metrics_models PASSED             [ 82%]
tests/test_models.py::test_evaluation_report_serialization PASSED        [ 85%]
tests/test_tracer.py::test_tracer_basic_step_recording PASSED            [ 87%]
tests/test_tracer.py::test_tracer_tool_decorator PASSED                  [ 90%]
tests/test_tracer.py::test_tracer_tool_decorator_handles_error PASSED    [ 92%]
tests/test_tracer.py::test_tracer_save_and_load_json PASSED              [ 95%]
tests/test_visualizer.py::test_render_html_dashboard_string PASSED       [ 97%]
tests/test_visualizer.py::test_render_html_dashboard_file PASSED         [100%]

============================= 40 passed in 0.23s ==============================
```

---

## 📁 Repository Structure

```
agentlens-ai/
├── agentlens/
│   ├── __init__.py                  # Top-level API exports
│   ├── models.py                    # Pydantic schemas (Trajectory, Step, ToolCall, Report)
│   ├── tracer.py                    # AgentTracer runtime context manager & decorators
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── loop_detector.py         # Circular tool call & thought cycle detection
│   │   ├── hallucination_detector.py# Tool schema validation & hallucinated tool name detection
│   │   └── drift_detector.py        # Semantic goal alignment & topic drift detection
│   ├── evaluator.py                 # Multi-factor TrajectoryEvaluator & health scoring
│   ├── guardrails.py                # Runtime CircuitBreaker & budget interceptor
│   ├── visualizer.py                # Zero-dependency standalone HTML dashboard renderer
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── crewai_hook.py           # CrewAI step callback listener
│   │   └── react_hook.py            # Generic ReAct loop wrapper
│   └── cli.py                       # CLI commands (eval, dashboard, benchmark, demo)
├── tests/                           # 40 comprehensive unit tests
├── examples/
│   ├── quickstart.py                # Quickstart demonstration script
│   ├── loop_circuit_breaker_demo.py # Live loop interception demo
│   └── sample_trajectories.json     # Pre-built audit trajectories
├── pyproject.toml                   # Modern PEP 621 packaging metadata
├── setup.py                         # Editable installation configuration
├── requirements.txt                 # Runtime dependencies
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## 👨‍💻 Author

**Nitesh Beniwal**  
* Associate Software Developer | MCA (Master of Computer Applications)  
* Email: [niteshben01@gmail.com](mailto:niteshben01@gmail.com)  
* LinkedIn: [linkedin.com/in/nitesh-ben](https://linkedin.com/in/nitesh-ben)  
* GitHub: [github.com/NiteshBen](https://github.com/NiteshBen)  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

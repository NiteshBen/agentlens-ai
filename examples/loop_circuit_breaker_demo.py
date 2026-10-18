"""Circuit Breaker & Loop Prevention Demo.

Demonstrates how AgentLens CircuitBreaker intercepts and halts an agent
caught in an infinite circular tool invocation loop before wasting token budget.
"""

from agentlens import (
    AgentTracer,
    CircuitBreaker,
    CircuitBreakerAction,
    CircuitBreakerTrippedError,
    TrajectoryEvaluator,
)

def main():
    print("=" * 65)
    print("  AgentLens-AI: Runtime Circuit Breaker Loop Interception Demo")
    print("=" * 65)

    tracer = AgentTracer(
        workflow_name="Database Scraping Agent",
        goal="Extract real-time crypto prices from API",
    )

    # Initialize CircuitBreaker configured to halt on 3 repetitive calls
    breaker = CircuitBreaker(
        max_repeated_tool_calls=3,
        action_on_trip=CircuitBreakerAction.INJECT_INTERVENTION,
    )

    print("\n[*] Simulating agent attempting repetitive failed API calls...")
    for turn in range(1, 6):
        with tracer.step(agent_name="CryptoScraper", role="Data Ingestion", thought=f"Retrying price query for BTC (Attempt {turn})") as step:
            step.record_tool_call(
                tool_name="crypto_price_api",
                arguments={"symbol": "BTC-USD", "timeout": 5},
                result="HTTP 429 Too Many Requests: Rate limit exceeded",
                latency_ms=210.0,
            )
            step.record_tokens(110, 45)
            step.set_observation("Rate limited. Retrying...")

            # Check with circuit breaker in real time!
            intercept = breaker.intercept_step(step.step)
            print(f"  -> Turn {turn}: Status = {intercept['status']}")

            if intercept["status"] == "TRIPPED":
                print(f"\n[!] ALERT: CircuitBreaker Tripped at Turn {turn}!")
                print(f"    Reason: {intercept['reason']}")
                if intercept["intervention"]:
                    print(f"    Intervention: {intercept['intervention']}")
                print("\n[*] Gracefully halting runaway loop before token budget explodes.")
                break

    # Evaluate the trajectory
    traj = tracer.build_trajectory()
    evaluator = TrajectoryEvaluator()
    report = evaluator.evaluate(traj)

    print("\n[*] Diagnostic Report:")
    print(f"    Health Score: {report.metrics.overall_health_score}/100.0")
    print(f"    Loop Risk:    {report.metrics.loop_risk_index * 100:.0f}%")
    print(f"    Detected {len(report.anomalies)} anomalies:")
    for a in report.anomalies:
        print(f"    - [{a.severity.value.upper()}] {a.title}: {a.message}")
    print("=" * 65)

if __name__ == "__main__":
    main()

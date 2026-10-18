"""Tool Hallucination & Schema Violation Detector.

Validates tool invocations against registered tool schemas, identifying
hallucinated tool names, missing parameters, type mismatches, and execution errors.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

from agentlens.models import AgentStep, Anomaly, AnomalyType, Severity, ToolCallStatus


class ToolParameterSchema(BaseModel):
    """Specification for a tool parameter."""
    param_type: str = "string"  # string, integer, float, boolean, array, object
    required: bool = False
    description: Optional[str] = None


class ToolSchema(BaseModel):
    """Registered tool specification."""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, ToolParameterSchema] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


def _check_type_match(val: Any, expected_type: str) -> bool:
    """Validates if a value satisfies the expected primitive schema type."""
    expected = expected_type.lower()
    if expected in ("string", "str"):
        return isinstance(val, str)
    elif expected in ("integer", "int"):
        return isinstance(val, int) and not isinstance(val, bool)
    elif expected in ("float", "number"):
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    elif expected in ("boolean", "bool"):
        return isinstance(val, bool)
    elif expected in ("array", "list"):
        return isinstance(val, list)
    elif expected in ("object", "dict"):
        return isinstance(val, dict)
    return True


class HallucinationDetector:
    """Validates agent tool calls against expected schemas and detects hallucinations."""

    def __init__(self, schemas: Optional[List[ToolSchema]] = None):
        self._schemas: Dict[str, ToolSchema] = {}
        if schemas:
            for s in schemas:
                self._schemas[s.name.lower()] = s

    def register_tool(
        self,
        name: str,
        parameters: Optional[Dict[str, Union[str, ToolParameterSchema]]] = None,
        required: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Registers a recognized tool schema."""
        parsed_params: Dict[str, ToolParameterSchema] = {}
        if parameters:
            for p_name, p_def in parameters.items():
                if isinstance(p_def, str):
                    is_req = bool(required and p_name in required)
                    parsed_params[p_name] = ToolParameterSchema(param_type=p_def, required=is_req)
                elif isinstance(p_def, ToolParameterSchema):
                    parsed_params[p_name] = p_def
                elif isinstance(p_def, dict):
                    parsed_params[p_name] = ToolParameterSchema(
                        param_type=p_def.get("type", "string"),
                        required=p_def.get("required", False),
                        description=p_def.get("description"),
                    )

        self._schemas[name.lower()] = ToolSchema(
            name=name,
            description=description,
            parameters=parsed_params,
            required=required or [k for k, v in parsed_params.items() if v.required],
        )

    @property
    def registered_tools(self) -> List[str]:
        return list(self._schemas.keys())

    def analyze(self, steps: List[AgentStep]) -> List[Anomaly]:
        """Scans trajectory steps for tool hallucinations, schema violations, and runtime errors."""
        anomalies: List[Anomaly] = []
        registered_names = set(self._schemas.keys())

        for step in steps:
            for tc in step.tool_calls:
                tool_lower = tc.tool_name.lower().strip()

                # 1. Hallucinated Tool Name (if schemas are registered)
                if registered_names and tool_lower not in registered_names:
                    closest = difflib.get_close_matches(tool_lower, registered_names, n=1, cutoff=0.5)
                    hint = f" Did the agent mean '{closest[0]}'?" if closest else ""
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.HALLUCINATED_TOOL,
                            severity=Severity.HIGH,
                            title=f"Hallucinated Tool: '{tc.tool_name}'",
                            message=(
                                f"In Step {step.step_index}, agent attempted to invoke non-existent "
                                f"tool '{tc.tool_name}'.{hint}"
                            ),
                            step_indices=[step.step_index],
                            confidence=0.98,
                            recommendation=(
                                f"Update agent prompt with explicit list of available tools: "
                                f"{', '.join(sorted(registered_names))}."
                            ),
                        )
                    )
                    continue

                # 2. Schema Validation (if schema exists)
                if tool_lower in self._schemas:
                    schema = self._schemas[tool_lower]
                    args = tc.arguments or {}

                    # Missing required parameters
                    for req_param in schema.required:
                        if req_param not in args:
                            anomalies.append(
                                Anomaly(
                                    anomaly_type=AnomalyType.SCHEMA_VIOLATION,
                                    severity=Severity.HIGH,
                                    title=f"Missing Required Parameter: '{req_param}'",
                                    message=(
                                        f"In Step {step.step_index}, tool '{tc.tool_name}' was called "
                                        f"without mandatory parameter '{req_param}'."
                                    ),
                                    step_indices=[step.step_index],
                                    confidence=1.0,
                                    recommendation=f"Ensure agent prompt reinforces schema for '{tc.tool_name}'.",
                                )
                            )

                    # Type mismatches
                    for p_name, p_val in args.items():
                        if p_name in schema.parameters:
                            expected_t = schema.parameters[p_name].param_type
                            if not _check_type_match(p_val, expected_t):
                                anomalies.append(
                                    Anomaly(
                                        anomaly_type=AnomalyType.SCHEMA_VIOLATION,
                                        severity=Severity.MEDIUM,
                                        title=f"Parameter Type Mismatch in '{tc.tool_name}'",
                                        message=(
                                            f"Parameter '{p_name}' expected type '{expected_t}', but received "
                                            f"'{type(p_val).__name__}' ({repr(p_val)[:50]})."
                                        ),
                                        step_indices=[step.step_index],
                                        confidence=0.95,
                                        recommendation=f"Coerce or format parameter '{p_name}' as {expected_t}.",
                                    )
                                )

                # 3. Execution Failures / Status
                if tc.status in (ToolCallStatus.FAILED, ToolCallStatus.HALLUCINATED):
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.SCHEMA_VIOLATION,
                            severity=Severity.MEDIUM,
                            title=f"Tool Execution Failed: '{tc.tool_name}'",
                            message=(
                                f"Tool '{tc.tool_name}' in Step {step.step_index} returned an error status: "
                                f"{tc.error_message or 'Unknown runtime error'}"
                            ),
                            step_indices=[step.step_index],
                            confidence=1.0,
                            recommendation="Inspect tool implementation or add fallback error handling.",
                        )
                    )

        return anomalies

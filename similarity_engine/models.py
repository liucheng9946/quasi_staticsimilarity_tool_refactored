from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParameterDefinition:
    name: str
    label: str
    unit: str = ""
    description: str = ""
    group: str = "截面参数"
    default: Optional[float] = None
    prototype_default: Optional[float] = None
    model_default: Optional[float] = None
    positive: bool = True
    nonnegative: bool = False
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    step: float = 1.0
    format: str = "%.6f"
    order: int = 0


@dataclass
class SectionDefinition:
    id: str
    name: str
    component_version: str = "1.0"
    parameters: List[ParameterDefinition] = field(default_factory=list)
    area_expression: str = ""
    inertia_expression: str = ""
    characteristic_expression: str = ""
    constraints: List[Dict[str, str]] = field(default_factory=list)
    image: Optional[str] = None
    symbol_description: Dict[str, str] = field(default_factory=dict)
    legacy_id: Optional[int] = None


@dataclass
class MethodVariable:
    name: str
    label: str = ""
    role: str = "known"
    default: Optional[float] = None
    initial_value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    step: float = 0.05
    format: str = "%.8f"
    description: str = ""


@dataclass
class MethodDefinition:
    id: str
    name: str
    short_name: Optional[str] = None
    description: str = ""
    component_version: str = "1.0"
    variables: List[MethodVariable] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    legacy_id: Optional[int] = None


@dataclass
class ComponentRecord:
    definition: Any
    source: str
    path: Optional[str] = None


@dataclass
class CalculationOptions:
    override_policy: str = "validate"
    residual_tolerance: float = 1e-8
    solver_tolerance: float = 1e-10
    max_iterations: int = 1000
    allow_solver_fallback: bool = True


@dataclass
class CalculationResult:
    values: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    extra_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_dict(self) -> Dict[str, Any]:
        output = dict(self.values)
        if self.extra_outputs:
            output["extraOutputs"] = dict(self.extra_outputs)
        if self.metadata:
            output["componentMetadata"] = dict(self.metadata)
        return output

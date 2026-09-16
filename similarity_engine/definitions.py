from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .models import MethodDefinition, MethodVariable, ParameterDefinition, SectionDefinition
from .safe_expr import (
    SafeExpressionError,
    evaluate_expression,
    split_assignment,
    validate_expression_names,
)

_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,63}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_mapping(data: Any, label: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{label}必须是对象结构")
    return data


def _validate_id(component_id: str) -> None:
    if not _ID_PATTERN.fullmatch(component_id or ""):
        raise ValueError("组件 ID 必须以英文字母开头，仅包含字母、数字、下划线或连字符，长度为 2~64")


def _parse_parameter(item: Dict[str, Any], index: int) -> ParameterDefinition:
    name = str(item.get("name", "")).strip()
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"第 {index + 1} 个参数名称不合法：{name}")
    label = str(item.get("label") or name).strip()
    default = item.get("default")
    proto_default = item.get("prototype_default", item.get("proto_default", default))
    model_default = item.get("model_default", default)
    return ParameterDefinition(
        name=name,
        label=label,
        unit=str(item.get("unit", "")),
        description=str(item.get("description", "")),
        group=str(item.get("group", "截面参数")),
        default=float(default) if default is not None else None,
        prototype_default=float(proto_default) if proto_default is not None else None,
        model_default=float(model_default) if model_default is not None else None,
        positive=bool(item.get("positive", True)),
        nonnegative=bool(item.get("nonnegative", False)),
        lower_bound=float(item["lower_bound"]) if item.get("lower_bound") is not None else None,
        upper_bound=float(item["upper_bound"]) if item.get("upper_bound") is not None else None,
        step=float(item.get("step", 1.0)),
        format=str(item.get("format", "%.6f")),
        order=int(item.get("order", index)),
    )


def parse_section_definition(data: Any) -> SectionDefinition:
    data = _require_mapping(data, "截面配置")
    component_id = str(data.get("id", "")).strip()
    _validate_id(component_id)
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("截面配置缺少 name")
    parameters_data = data.get("parameters")
    if not isinstance(parameters_data, list) or not parameters_data:
        raise ValueError("截面配置必须包含非空 parameters 列表")
    parameters = [_parse_parameter(_require_mapping(item, "参数定义"), index) for index, item in enumerate(parameters_data)]
    names = [item.name for item in parameters]
    if len(names) != len(set(names)):
        raise ValueError("截面参数名称不能重复")

    formulas = data.get("formulas", {})
    area_expression = str(data.get("area_expression", formulas.get("area", ""))).strip()
    inertia_expression = str(data.get("inertia_expression", formulas.get("inertia", ""))).strip()
    characteristic_expression = str(
        data.get("characteristic_expression", formulas.get("characteristic", ""))
    ).strip()
    if not area_expression or not inertia_expression or not characteristic_expression:
        raise ValueError("截面配置必须定义面积、惯性矩和特征尺寸表达式")

    allowed = set(names)
    for label, expression in [
        ("面积", area_expression),
        ("惯性矩", inertia_expression),
        ("特征尺寸", characteristic_expression),
    ]:
        try:
            validate_expression_names(expression, allowed)
        except SafeExpressionError as exc:
            raise ValueError(f"{label}表达式不合法：{exc}") from exc

    constraints_raw = data.get("constraints", []) or []
    constraints: List[Dict[str, str]] = []
    for item in constraints_raw:
        if isinstance(item, str):
            expression, message = item, f"几何约束不满足：{item}"
        elif isinstance(item, dict):
            expression = str(item.get("expression", "")).strip()
            message = str(item.get("message", f"几何约束不满足：{expression}"))
        else:
            raise ValueError("constraints 中每项必须是字符串或对象")
        if not expression:
            raise ValueError("几何约束表达式不能为空")
        validate_expression_names(expression, allowed)
        constraints.append({"expression": expression, "message": message})

    definition = SectionDefinition(
        id=component_id,
        name=name,
        component_version=str(data.get("version", data.get("component_version", "1.0"))),
        parameters=parameters,
        area_expression=area_expression,
        inertia_expression=inertia_expression,
        characteristic_expression=characteristic_expression,
        constraints=constraints,
        image=str(data.get("image")) if data.get("image") else None,
        symbol_description={
            str(key): str(value)
            for key, value in (data.get("symbol_description") or {}).items()
        },
    )
    _validate_section_defaults(definition)
    return definition


def _validate_section_defaults(definition: SectionDefinition) -> None:
    for side in ("prototype", "model"):
        variables: Dict[str, float] = {}
        for parameter in definition.parameters:
            value = parameter.prototype_default if side == "prototype" else parameter.model_default
            if value is None:
                value = parameter.default
            if value is None:
                raise ValueError(f"参数 {parameter.name} 缺少默认值")
            value = float(value)
            if parameter.positive and value <= 0:
                raise ValueError(f"参数 {parameter.name} 默认值必须大于 0")
            if parameter.nonnegative and value < 0:
                raise ValueError(f"参数 {parameter.name} 默认值不能小于 0")
            if parameter.lower_bound is not None and value < parameter.lower_bound:
                raise ValueError(f"参数 {parameter.name} 默认值小于下界")
            if parameter.upper_bound is not None and value > parameter.upper_bound:
                raise ValueError(f"参数 {parameter.name} 默认值大于上界")
            variables[parameter.name] = value
        for constraint in definition.constraints:
            if not bool(evaluate_expression(constraint["expression"], variables)):
                raise ValueError(f"默认参数不满足约束：{constraint['message']}")
        for label, expression in [
            ("面积", definition.area_expression),
            ("惯性矩", definition.inertia_expression),
            ("特征尺寸", definition.characteristic_expression),
        ]:
            value = float(evaluate_expression(expression, variables))
            if value <= 0:
                raise ValueError(f"{label}表达式在默认参数下必须得到正数")


def _parse_variable(item: Dict[str, Any], index: int) -> MethodVariable:
    name = str(item.get("name", "")).strip()
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"第 {index + 1} 个方法变量名称不合法：{name}")
    role = str(item.get("role", "known")).strip().lower()
    if role not in {"known", "free", "output", "unknown"}:
        raise ValueError(f"方法变量 {name} 的 role 不合法")
    return MethodVariable(
        name=name,
        label=str(item.get("label") or name),
        role=role,
        default=float(item["default"]) if item.get("default") is not None else None,
        initial_value=float(item["initial_value"]) if item.get("initial_value") is not None else None,
        lower_bound=float(item["lower_bound"]) if item.get("lower_bound") is not None else None,
        upper_bound=float(item["upper_bound"]) if item.get("upper_bound") is not None else None,
        step=float(item.get("step", 0.05)),
        format=str(item.get("format", "%.8f")),
        description=str(item.get("description", "")),
    )


def parse_method_definition(data: Any) -> MethodDefinition:
    data = _require_mapping(data, "计算方法配置")
    component_id = str(data.get("id", "")).strip()
    _validate_id(component_id)
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("计算方法配置缺少 name")

    variables_raw = data.get("variables", []) or []
    if not isinstance(variables_raw, list):
        raise ValueError("variables 必须是列表")
    variables = [_parse_variable(_require_mapping(item, "变量定义"), index) for index, item in enumerate(variables_raw)]
    variable_names = [item.name for item in variables]
    if len(variable_names) != len(set(variable_names)):
        raise ValueError("方法变量名称不能重复")

    equations = [str(item).strip() for item in (data.get("equations") or [])]
    if not equations:
        raise ValueError("计算方法配置必须包含 equations")
    outputs = [str(item).strip() for item in (data.get("outputs") or [])]
    if not outputs:
        outputs = [item.name for item in variables if item.role in {"output", "unknown"}]
    if not outputs:
        raise ValueError("计算方法配置必须声明 outputs")

    common_names = {
        "SCB", "SH", "SE", "Ssigma", "SsigmaReal", "Sbs", "Sy", "SA", "SI", "SK",
        "Ap", "Am", "Ip", "Im",
    }
    allowed = common_names | set(variable_names) | set(outputs)
    assigned = set()
    for equation in equations:
        left, right = split_assignment(equation)
        assigned.add(left)
        validate_expression_names(right, allowed)
    missing = set(outputs) - assigned - set(variable_names)
    if missing:
        raise ValueError("以下输出变量没有对应方程或变量定义：" + "、".join(sorted(missing)))

    return MethodDefinition(
        id=component_id,
        name=name,
        short_name=str(data.get("short_name")) if data.get("short_name") else None,
        description=str(data.get("description", "")),
        component_version=str(data.get("version", data.get("component_version", "1.0"))),
        variables=variables,
        equations=equations,
        outputs=outputs,
    )

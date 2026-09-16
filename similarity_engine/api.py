from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .definitions import parse_method_definition, parse_section_definition
from .models import (
    CalculationOptions,
    CalculationResult,
    ComponentRecord,
    MethodDefinition,
    SectionDefinition,
)
from .safe_expr import SafeExpressionError, evaluate_expression, split_assignment


_BUILTIN_SECTIONS = [
    ("builtin_circle", "圆形", 1),
    ("builtin_circular_tube", "圆环", 2),
    ("builtin_square", "方形", 3),
    ("builtin_hollow_square", "空心方钢", 4),
    ("builtin_i_section", "工字钢", 5),
]

_BUILTIN_METHODS = [
    ("builtin_da", "传统量纲分析方法 DA", 2),
    ("builtin_csl_y", "已完成拟静力试验 CSL-Y", 3),
    ("builtin_csl_n", "拟进行拟静力试验 CSL-N", 4),
]


class SimilarityAPI:
    def __init__(self, user_config_root: Optional[Path] = None):
        project_root = Path(__file__).resolve().parents[1]
        self.user_config_root = Path(user_config_root or project_root / "user_components")
        self.section_root = self.user_config_root / "sections"
        self.method_root = self.user_config_root / "methods"
        self.section_root.mkdir(parents=True, exist_ok=True)
        self.method_root.mkdir(parents=True, exist_ok=True)
        self._sections: Dict[str, ComponentRecord] = {}
        self._methods: Dict[str, ComponentRecord] = {}
        self.reload_components()

    def reload_components(self) -> None:
        self._sections = {
            component_id: ComponentRecord(
                definition=SectionDefinition(
                    id=component_id,
                    name=name,
                    component_version="1.0",
                    legacy_id=legacy_id,
                ),
                source="builtin",
            )
            for component_id, name, legacy_id in _BUILTIN_SECTIONS
        }
        self._methods = {
            component_id: ComponentRecord(
                definition=MethodDefinition(
                    id=component_id,
                    name=name,
                    short_name=name,
                    component_version="1.0",
                    legacy_id=legacy_id,
                ),
                source="builtin",
            )
            for component_id, name, legacy_id in _BUILTIN_METHODS
        }
        self._load_user_directory(self.section_root, "section")
        self._load_user_directory(self.method_root, "method")

    def _load_user_directory(self, root: Path, kind: str) -> None:
        target = self._sections if kind == "section" else self._methods
        parser = parse_section_definition if kind == "section" else parse_method_definition
        for path in sorted(root.iterdir()):
            if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue
            try:
                data = json.loads(path.read_text("utf-8")) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text("utf-8"))
                definition = parser(data)
                if definition.id in target:
                    continue
                target[definition.id] = ComponentRecord(definition=definition, source="user", path=str(path))
            except Exception:
                # 已安装文件若被手工破坏，不阻断主程序启动；重新安装时会显示明确错误。
                continue

    def list_sections(self) -> List[ComponentRecord]:
        return list(self._sections.values())

    def list_methods(self) -> List[ComponentRecord]:
        return list(self._methods.values())

    def get_section(self, component_id: str) -> ComponentRecord:
        if component_id not in self._sections:
            raise ValueError(f"未找到截面组件：{component_id}")
        return self._sections[component_id]

    def get_method(self, component_id: str) -> ComponentRecord:
        if component_id not in self._methods:
            raise ValueError(f"未找到计算方法组件：{component_id}")
        return self._methods[component_id]

    def builtin_section_ids(self) -> set[str]:
        return {item[0] for item in _BUILTIN_SECTIONS}

    def builtin_method_ids(self) -> set[str]:
        return {item[0] for item in _BUILTIN_METHODS}

    def calculate(
        self,
        section: str,
        method: str,
        SCB: float,
        proto: Dict[str, Any],
        model: Dict[str, Any],
        user_inputs: Optional[Dict[str, Any]] = None,
        options: Optional[CalculationOptions] = None,
        show_detail: int = 1,
    ) -> CalculationResult:
        del options
        user_inputs = dict(user_inputs or {})
        section_record = self.get_section(section)
        method_record = self.get_method(method)
        if section_record.source != "user" and method_record.source != "user":
            raise ValueError("统一组件 API 仅用于至少包含一个用户扩展组件的计算")

        section_values = self._calculate_section(section_record, proto, model)
        values = self._calculate_common_ratios(
            section_record, method_record, SCB, proto, model, section_values, user_inputs, show_detail
        )
        extra_outputs: Dict[str, Any] = {}

        if method_record.source == "builtin":
            self._apply_builtin_method(method_record.definition, values, user_inputs)
        else:
            extra_outputs = self._apply_user_method(method_record.definition, values, user_inputs)

        metadata = {
            "sectionId": section_record.definition.id,
            "sectionVersion": section_record.definition.component_version,
            "sectionSource": section_record.source,
            "methodId": method_record.definition.id,
            "methodVersion": method_record.definition.component_version,
            "methodSource": method_record.source,
        }
        return CalculationResult(values=values, extra_outputs=extra_outputs, metadata=metadata)

    def _calculate_section(
        self,
        record: ComponentRecord,
        proto: Dict[str, Any],
        model: Dict[str, Any],
    ) -> Dict[str, float]:
        if record.source == "builtin":
            legacy_id = record.definition.legacy_id
            return self._calculate_builtin_section(legacy_id, proto["section"], model["section"])
        definition: SectionDefinition = record.definition
        proto_section = {key: float(value) for key, value in proto.get("section", {}).items()}
        model_section = {key: float(value) for key, value in model.get("section", {}).items()}
        required = {parameter.name for parameter in definition.parameters}
        for side_name, params in (("原型构件", proto_section), ("缩尺模型", model_section)):
            missing = required - set(params)
            if missing:
                raise ValueError(f"{side_name}缺少截面参数：{'、'.join(sorted(missing))}")
            for parameter in definition.parameters:
                value = float(params[parameter.name])
                if parameter.positive and value <= 0:
                    raise ValueError(f"{side_name}参数 {parameter.label} 必须大于 0")
                if parameter.nonnegative and value < 0:
                    raise ValueError(f"{side_name}参数 {parameter.label} 不能小于 0")
                if parameter.lower_bound is not None and value < parameter.lower_bound:
                    raise ValueError(f"{side_name}参数 {parameter.label} 小于下界")
                if parameter.upper_bound is not None and value > parameter.upper_bound:
                    raise ValueError(f"{side_name}参数 {parameter.label} 大于上界")
            for constraint in definition.constraints:
                if not bool(evaluate_expression(constraint["expression"], params)):
                    raise ValueError(f"{side_name}：{constraint['message']}")
        try:
            Ap = float(evaluate_expression(definition.area_expression, proto_section))
            Am = float(evaluate_expression(definition.area_expression, model_section))
            Ip = float(evaluate_expression(definition.inertia_expression, proto_section))
            Im = float(evaluate_expression(definition.inertia_expression, model_section))
            yp = float(evaluate_expression(definition.characteristic_expression, proto_section))
            ym = float(evaluate_expression(definition.characteristic_expression, model_section))
        except SafeExpressionError as exc:
            raise ValueError(f"截面表达式计算失败：{exc}") from exc
        if min(Ap, Am, Ip, Im, yp, ym) <= 0:
            raise ValueError("截面面积、惯性矩及特征尺寸必须为正数")
        return {"Ap": Ap, "Am": Am, "Ip": Ip, "Im": Im, "yp": yp, "ym": ym}

    @staticmethod
    def _calculate_builtin_section(section_type: int, p: Dict[str, float], m: Dict[str, float]) -> Dict[str, float]:
        def area(params: Dict[str, float]) -> float:
            if section_type == 1:
                return math.pi * params["D"] ** 2 / 4
            if section_type == 2:
                inner = params["D"] - 2 * params["t"]
                return math.pi * (params["D"] ** 2 - inner ** 2) / 4
            if section_type == 3:
                return params["B"] ** 2
            if section_type == 4:
                inner = params["B"] - 2 * params["tw"]
                return params["B"] ** 2 - inner ** 2
            return params["d"] * params["h"] - (params["d"] - params["tw"]) * (params["h"] - 2 * params["tf"])

        def inertia(params: Dict[str, float]) -> float:
            if section_type == 1:
                return math.pi * params["D"] ** 4 / 64
            if section_type == 2:
                inner = params["D"] - 2 * params["t"]
                return math.pi * (params["D"] ** 4 - inner ** 4) / 64
            if section_type == 3:
                return params["B"] ** 4 / 12
            if section_type == 4:
                inner = params["B"] - 2 * params["tw"]
                return (params["B"] ** 4 - inner ** 4) / 12
            return (params["d"] * params["h"] ** 3 - (params["d"] - params["tw"]) * (params["h"] - 2 * params["tf"]) ** 3) / 12

        def y(params: Dict[str, float]) -> float:
            if section_type in (1, 2):
                return params["D"]
            if section_type in (3, 4):
                return params["B"]
            return params["h"]

        return {"Ap": area(p), "Am": area(m), "Ip": inertia(p), "Im": inertia(m), "yp": y(p), "ym": y(m)}

    @staticmethod
    def _calculate_common_ratios(
        section_record: ComponentRecord,
        method_record: ComponentRecord,
        SCB: float,
        proto: Dict[str, Any],
        model: Dict[str, Any],
        section_values: Dict[str, float],
        user_inputs: Dict[str, Any],
        show_detail: int,
    ) -> Dict[str, Any]:
        if SCB <= 0:
            raise ValueError("几何基准相似比 SCB 必须大于 0")
        for side_name, params in (("原型构件", proto), ("缩尺模型", model)):
            for key in ("E", "fy", "bs", "l"):
                if params.get(key) is None or float(params[key]) <= 0:
                    raise ValueError(f"{side_name}{key} 必须大于 0")
        Ap, Am = section_values["Ap"], section_values["Am"]
        Ip, Im = section_values["Ip"], section_values["Im"]
        SH = float(model["l"]) / float(proto["l"])
        SE = float(model["E"]) / float(proto["E"])
        Ssigma_real = float(model["fy"]) / float(proto["fy"])
        Sbs = float(model["bs"]) / float(proto["bs"])
        Sy = section_values["ym"] / section_values["yp"]
        SA = Am / Ap
        SI = Im / Ip
        base = {
            "SH": SH, "SE": SE, "SsigmaReal": Ssigma_real, "Ssigma": Ssigma_real,
            "Sbs": Sbs, "Sy": Sy, "SA": SA, "SI": SI,
        }
        for key in ("SH", "SE", "Ssigma", "Sbs", "Sy", "SA", "SI"):
            if user_inputs.get(key) is not None:
                base[key] = float(user_inputs[key])
        if user_inputs.get("Ssigma") is not None:
            base["SsigmaReal"] = float(user_inputs["Ssigma"])
        base["SK"] = base["SE"] * base["SI"] / base["SH"] ** 3
        if user_inputs.get("SK") is not None:
            base["SK"] = float(user_inputs["SK"])
        return {
            "sectionType": section_record.definition.legacy_id,
            "sectionName": section_record.definition.name,
            "methodType": method_record.definition.legacy_id,
            "methodName": method_record.definition.short_name or method_record.definition.name,
            "SCB": float(SCB),
            **base,
            "Sd": None,
            "SF": None,
            "Ap": Ap,
            "Am": Am,
            "Ip": Ip,
            "Im": Im,
            "showDetail": show_detail,
        }

    @staticmethod
    def _apply_builtin_method(definition: MethodDefinition, values: Dict[str, Any], user_inputs: Dict[str, Any]) -> None:
        method_type = definition.legacy_id
        if method_type == 2:
            values["Sd"] = values["SCB"]
            values["SF"] = values["SCB"] ** 2
            values["Ssigma"] = 1.0
        elif method_type == 3:
            values["Sd"] = values["SCB"]
            values["SF"] = values["SK"] * values["Sd"]
            values["Ssigma"] = 1.0
        elif method_type == 4:
            values["Sd"] = values["SH"] ** 2 * values["SsigmaReal"] / values["SE"] / values["Sy"]
            values["SF"] = values["SI"] * values["SsigmaReal"] / values["Sy"] / values["SH"]
            values["Ssigma"] = values["SsigmaReal"]
        else:
            raise ValueError("未知内置计算方法")
        if user_inputs.get("Sd") is not None:
            values["Sd"] = float(user_inputs["Sd"])
        if user_inputs.get("SF") is not None:
            values["SF"] = float(user_inputs["SF"])

    @staticmethod
    def _apply_user_method(
        definition: MethodDefinition,
        values: Dict[str, Any],
        user_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = dict(values)
        for variable in definition.variables:
            if variable.role in {"known", "free"}:
                value = user_inputs.get(variable.name)
                if value is None:
                    value = variable.default if variable.default is not None else variable.initial_value
                if value is None:
                    raise ValueError(f"扩展方法缺少附加参数：{variable.label or variable.name}")
                value = float(value)
                if variable.lower_bound is not None and value < variable.lower_bound:
                    raise ValueError(f"附加参数 {variable.label or variable.name} 小于下界")
                if variable.upper_bound is not None and value > variable.upper_bound:
                    raise ValueError(f"附加参数 {variable.label or variable.name} 大于上界")
                context[variable.name] = value
        pending = [split_assignment(item) for item in definition.equations]
        for _ in range(len(pending) + 2):
            next_pending = []
            progressed = False
            for left, right in pending:
                try:
                    context[left] = float(evaluate_expression(right, context))
                    progressed = True
                except SafeExpressionError as exc:
                    if "未定义变量" in str(exc):
                        next_pending.append((left, right))
                    else:
                        raise ValueError(f"扩展方法方程计算失败：{left} = {right}；{exc}") from exc
            pending = next_pending
            if not pending:
                break
            if not progressed:
                unresolved = "；".join(f"{left} = {right}" for left, right in pending)
                raise ValueError(f"扩展方法方程存在循环依赖或缺少变量：{unresolved}")
        for output in definition.outputs:
            if output not in context:
                raise ValueError(f"扩展方法未得到输出变量：{output}")
        for key in ("SH", "SE", "Ssigma", "Sbs", "Sy", "SA", "SI", "SK", "Sd", "SF"):
            if key in context:
                values[key] = context[key]
        if user_inputs.get("Sd") is not None:
            values["Sd"] = float(user_inputs["Sd"])
        if user_inputs.get("SF") is not None:
            values["SF"] = float(user_inputs["SF"])
        standard = set(values)
        return {key: context[key] for key in definition.outputs if key not in standard}

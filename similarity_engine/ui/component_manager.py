from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import yaml

from ..definitions import parse_method_definition, parse_section_definition


class UserComponentManager:
    def __init__(self, user_config_root: Path):
        self.user_config_root = Path(user_config_root)
        self.section_root = self.user_config_root / "sections"
        self.method_root = self.user_config_root / "methods"
        self.section_root.mkdir(parents=True, exist_ok=True)
        self.method_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse(content: bytes, filename: str) -> Any:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".yaml", ".yml", ".json"}:
            raise ValueError("文件类型错误：仅支持 YAML、YML 或 JSON")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("配置文件必须使用 UTF-8 编码") from exc
        try:
            return json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        except Exception as exc:
            raise ValueError(f"配置解析失败：{exc}") from exc

    def install(self, content: bytes, filename: str, kind: str):
        if kind not in {"section", "method"}:
            raise ValueError("组件类型错误")
        data = self._parse(content, filename)
        definition = parse_section_definition(data) if kind == "section" else parse_method_definition(data)
        builtin_ids = {
            "builtin_circle", "builtin_circular_tube", "builtin_square", "builtin_hollow_square", "builtin_i_section",
            "builtin_da", "builtin_csl_y", "builtin_csl_n",
        }
        if definition.id in builtin_ids:
            raise ValueError("组件 ID 与软件内置组件冲突")
        root = self.section_root if kind == "section" else self.method_root
        existing = list(root.glob(f"{definition.id}.*"))
        if existing:
            raise ValueError(f"组件 ID 已存在：{definition.id}")
        target = root / f"{definition.id}.yaml"
        target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return definition, target

    def delete(self, kind: str, component_id: str) -> None:
        if kind not in {"section", "method"}:
            raise ValueError("组件类型错误")
        root = self.section_root if kind == "section" else self.method_root
        matches = list(root.glob(f"{component_id}.*"))
        if not matches:
            raise ValueError("未找到指定的用户扩展组件")
        for path in matches:
            path.unlink()

    @staticmethod
    def section_template() -> bytes:
        text = """# 用户自定义截面模板（仅允许安全数学表达式，不执行 Python 代码）
id: rectangular_section
name: 矩形截面
version: "1.0"
# image: images/rectangular_section.png  # 可选，路径相对于 app.py 所在目录
symbol_description:
  B: 截面宽度
  H: 截面高度
parameters:
  - name: B
    label: 截面宽度 B
    unit: mm
    description: 矩形截面的宽度
    group: 截面参数
    default: 300
    prototype_default: 300
    model_default: 150
    positive: true
    lower_bound: 0.001
    upper_bound: 100000
    step: 1
    format: "%.6f"
    order: 1
  - name: H
    label: 截面高度 H
    unit: mm
    description: 矩形截面的高度
    group: 截面参数
    default: 450
    prototype_default: 450
    model_default: 225
    positive: true
    lower_bound: 0.001
    upper_bound: 100000
    step: 1
    format: "%.6f"
    order: 2
formulas:
  area: B * H
  inertia: B * H**3 / 12
  characteristic: H
constraints:
  - expression: B > 0 and H > 0
    message: B 和 H 必须大于 0
"""
        return text.encode("utf-8")

    @staticmethod
    def method_template() -> bytes:
        text = """# 用户自定义计算方法模板
# 方程必须采用“变量 = 表达式”形式；表达式仅允许已声明变量和白名单数学函数。
id: custom_quasi_static_method
name: 用户自定义拟静力计算方法
short_name: 自定义拟静力方法
version: "1.0"
description: 根据已有基本参数与衍生参数计算位移、恢复力及应力相似比。
variables:
  - name: alpha
    label: 修正系数 alpha
    role: known
    default: 1.0
    initial_value: 1.0
    lower_bound: 0.000001
    upper_bound: 1000
    step: 0.05
    format: "%.8f"
    description: 用户方法的附加修正系数
  - name: Sd
    label: 位移相似比 Sd
    role: output
  - name: SF
    label: 恢复力相似比 SF
    role: output
  - name: Ssigma
    label: 应力相似比 Ssigma
    role: output
equations:
  - Sd = SCB
  - SF = alpha * SK * Sd
  - Ssigma = SsigmaReal
outputs:
  - Sd
  - SF
  - Ssigma
"""
        return text.encode("utf-8")

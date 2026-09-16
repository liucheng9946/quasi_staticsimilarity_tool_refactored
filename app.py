# -*- coding: utf-8 -*-
"""
基于分类相似律的缩尺拟静力试验非完全相似比确定工具
Streamlit 主界面程序

运行方式：
    streamlit run app.py

依赖：
    streamlit pandas numpy openpyxl PyYAML pydantic
"""

from __future__ import annotations

import base64
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

from similarity_engine.api import SimilarityAPI
from similarity_engine.models import CalculationOptions
from similarity_engine.ui.component_manager import UserComponentManager
from similarity_engine.ui.dynamic_form import render_dynamic_parameter_form
from corecode import (
    calculate_similarity,
    apply_simply_supported_to_base_result,
    MEMBER_NAMES as CORE_MEMBER_NAMES,
)

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# 1. 界面常量：核心计算统一调用 corecode.py
# =========================================================

SECTION_NAMES = {
    1: "圆形",
    2: "圆环",
    3: "方形",
    4: "空心方钢",
    5: "工字钢",
}

MEMBER_NAMES = dict(CORE_MEMBER_NAMES)

METHOD_NAMES = {
    2: "传统量纲分析方法 DA",
    3: "已完成拟静力试验 CSL-Y",
    4: "拟进行拟静力试验 CSL-N",
}

METHOD_DESCRIPTIONS = {
    "传统量纲分析方法 DA": "当前结果基于传统量纲分析方法得到，可作为分类相似律计算的对照。",
    "已完成拟静力试验 CSL-Y": "当前结果面向已完成缩尺拟静力试验，在既有位移比例基础上修正恢复力相似比。",
    "拟进行拟静力试验 CSL-N": "当前结果面向拟进行缩尺拟静力试验，联合考虑位移、恢复力和应力约束。",
}

OVERRIDE_KEYS = ["SH", "SE", "Ssigma", "Sbs", "Sy", "SA", "SI", "SW", "SK", "Srho", "Sq", "Sd", "SF"]

OVERRIDE_LABELS = {
    "SH": "高度/长度相似比 SH",
    "SE": "弹性模量相似比 SE",
    "Ssigma": "屈服强度/应力相似比 Ssigma",
    "Sbs": "硬化系数相似比 Sbs",
    "Sy": "截面特征尺寸相似比 Sy",
    "SA": "截面面积相似比 SA",
    "SI": "截面惯性矩相似比 SI",
    "SW": "截面抵抗矩相似比 SW",
    "SK": "等效刚度相似比 SK",
    "Srho": "材料密度相似比 Srho",
    "Sq": "均布荷载相似比 Sq",
    "Sd": "位移相似比 Sd",
    "SF": "恢复力相似比 SF",
}

SECTION_INPUTS = {
    1: [("D", "直径 D", "mm")],
    2: [("D", "外径 D", "mm"), ("t", "壁厚 t", "mm")],
    3: [("B", "边长 B", "mm")],
    4: [("B", "外边长 B", "mm"), ("tw", "壁厚 tw", "mm")],
    5: [("h", "截面高度 h", "mm"), ("d", "翼缘宽度 d", "mm"), ("tw", "腹板厚度 tw", "mm"), ("tf", "翼缘厚度 tf", "mm")],
}

DEFAULT_SECTION_VALUES = {
    1: {"proto": {"D": 450.0}, "model": {"D": 220.0}},
    2: {"proto": {"D": 450.0, "t": 14.0}, "model": {"D": 220.0, "t": 8.0}},
    3: {"proto": {"B": 450.0}, "model": {"B": 220.0}},
    4: {"proto": {"B": 450.0, "tw": 14.0}, "model": {"B": 220.0, "tw": 8.0}},
    5: {"proto": {"h": 350.0, "d": 350.0, "tw": 12.0, "tf": 19.0}, "model": {"h": 175.0, "d": 175.0, "tw": 6.0, "tf": 10.0}},
}

DEFAULT_MATERIAL_VALUES = {
    "proto": {"E": 210000.0, "fy": 270.0, "bs": 0.02, "l": 3600.0},
    "model": {"E": 210000.0, "fy": 270.0, "bs": 0.02, "l": 1800.0},
}

DEFAULT_DENSITY_VALUES = {"proto": 7850.0, "model": 7850.0}
DEFAULT_GRAVITY = 9.80665


class ValidationError(ValueError):
    """用于参数校验的明确错误类型。"""



# =========================================================
# 1.1 扩展组件 API 与兼容辅助
# =========================================================

@st.cache_resource
def get_api() -> SimilarityAPI:
    return SimilarityAPI()


def get_component_maps(api: SimilarityAPI):
    sections = api.list_sections()
    methods = api.list_methods()
    return (
        {item.definition.id: item for item in sections},
        {item.definition.id: item for item in methods},
    )


def format_component_source(source: str) -> str:
    return {"builtin": "软件内置", "user": "用户扩展"}.get(source, "—")


def component_display_name(record: Any) -> str:
    definition = record.definition
    return getattr(definition, "short_name", None) or definition.name


def safe_component_image_path(image: Optional[str]) -> Optional[Path]:
    if not image:
        return None
    raw = Path(str(image))
    if raw.is_absolute():
        return None
    candidate = (PROJECT_ROOT / raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
        return None
    return candidate if candidate.is_file() else None


def render_user_section_schematic(section: Any) -> None:
    image_path = safe_component_image_path(section.image)
    if image_path is not None:
        st.markdown(
            f'<div class="schematic-card"><div class="schematic-title">📐 截面示意图 — {section.name}</div>'
            '<div class="schematic-note">示意图由用户扩展截面配置提供。</div></div>',
            unsafe_allow_html=True,
        )
        st.image(str(image_path), use_container_width=False)
        return

    symbols = "、".join(section.symbol_description.keys())
    if not symbols:
        symbols = "、".join(parameter.name for parameter in section.parameters)
    st.markdown(
        f"""
        <div class="schematic-card" style="min-height:220px;display:flex;align-items:center;justify-content:center;text-align:center;">
            <div>
                <div style="font-size:3rem;margin-bottom:0.45rem;">📐</div>
                <div class="schematic-title">{section.name}</div>
                <div class="schematic-note">参数符号：{symbols or '—'}</div>
                <div style="color:#667085;font-size:0.88rem;">当前用户扩展截面未提供示意图</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_method_inputs(method: Any) -> Dict[str, float]:
    method_inputs: Dict[str, float] = {}
    variables = [item for item in method.variables if item.role in {"known", "free"}]
    if not variables:
        return method_inputs

    st.markdown("---")
    st.markdown('<div class="section-caption">用户扩展方法附加参数</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">以下输入项由当前计算方法配置文件自动生成。</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2, gap="large")
    for index, variable in enumerate(variables):
        kwargs: Dict[str, Any] = {
            "value": float(
                variable.default
                if variable.default is not None
                else variable.initial_value
                if variable.initial_value is not None
                else 0.0
            ),
            "step": float(variable.step or 0.05),
            "format": variable.format or "%.8f",
            "key": f"method_input_{method.id}_{variable.name}",
            "help": variable.description or None,
        }
        if variable.lower_bound is not None:
            kwargs["min_value"] = float(variable.lower_bound)
        if variable.upper_bound is not None:
            kwargs["max_value"] = float(variable.upper_bound)
        with cols[index % 2]:
            method_inputs[variable.name] = st.number_input(
                variable.label or variable.name,
                **kwargs,
            )
    return method_inputs


def normalize_component_result(result: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "sectionType", "sectionName", "methodType", "methodName", "SCB",
        "SH", "SE", "Ssigma", "Sbs", "Sy", "SA", "SI", "SK",
        "Sd", "SF", "Ap", "Am", "Ip", "Im", "showDetail",
    ]
    normalized = dict(result)
    for key in required:
        normalized.setdefault(key, None)
    return normalized


def calculate_with_component_api(
    api: SimilarityAPI,
    section_id: str,
    method_id: str,
    scb: float,
    proto: Dict[str, Any],
    model: Dict[str, Any],
    overrides: Dict[str, Optional[float]],
    method_inputs: Dict[str, float],
) -> Dict[str, Any]:
    merged_inputs = {key: value for key, value in overrides.items() if value is not None}
    merged_inputs.update(method_inputs)
    calculation_result = api.calculate(
        section=section_id,
        method=method_id,
        SCB=float(scb),
        proto=proto,
        model=model,
        user_inputs=merged_inputs,
        options=CalculationOptions(
            override_policy="validate",
            residual_tolerance=1e-8,
            solver_tolerance=1e-10,
            max_iterations=1000,
            allow_solver_fallback=True,
        ),
        show_detail=1,
    )
    result = normalize_component_result(calculation_result.to_legacy_dict())
    if calculation_result.warnings:
        result["calculationWarnings"] = list(calculation_result.warnings)
    return result


# =========================================================
# 2. 页面样式与通用工具函数
# =========================================================

def set_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --main-bg: #f5f7fb;
            --card-bg: #ffffff;
            --text-main: #1f2937;
            --text-muted: #667085;
            --border: #e5e7eb;
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --soft-blue: #eff6ff;
            --soft-green: #ecfdf3;
            --soft-orange: #fff7ed;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
            color: var(--text-main);
        }

        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.6rem;
            max-width: 1380px;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f4c81 0%, #2563eb 58%, #60a5fa 100%);
            color: white;
            padding: 1.8rem 2rem;
            border-radius: 24px;
            box-shadow: 0 18px 42px rgba(37, 99, 235, 0.22);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            font-weight: 500;
            opacity: 0.94;
            margin-bottom: 0.8rem;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.6rem;
        }

        .hero-tag {
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.26);
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.85rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .small-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
            height: 100%;
        }

        .section-caption {
            font-size: 1.02rem;
            font-weight: 750;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .section-note {
            font-size: 0.88rem;
            color: var(--text-muted);
            line-height: 1.55;
            margin-bottom: 0.8rem;
        }

        .metric-card {
            background: white;
            border: 1px solid #e8edf5;
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
            height: 128px;
            min-height: 128px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
        }

        .metric-title {
            font-size: 0.82rem;
            color: #667085;
            font-weight: 650;
            margin-bottom: 0.25rem;
            line-height: 1.25;
        }

        .metric-value {
            font-size: 1.22rem;
            color: #0f172a;
            font-weight: 800;
            word-break: break-word;
            line-height: 1.25;
        }

        .metric-foot {
            min-height: 1.1rem;
            margin-top: 0.25rem;
            color: #94a3b8;
            font-size: 0.78rem;
            line-height: 1.2;
        }

        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            line-height: 1.55;
            margin: 0.8rem 0 1.1rem 0;
        }

        .success-box {
            background: #ecfdf3;
            border: 1px solid #bbf7d0;
            color: #166534;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            line-height: 1.55;
            margin: 0.8rem 0 1.1rem 0;
        }

        .warning-box {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            line-height: 1.55;
            margin: 0.8rem 0 1.1rem 0;
        }

        .schematic-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe7f6;
            border-radius: 18px;
            padding: 0.95rem 1.05rem 0.75rem 1.05rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
            margin-top: 0.85rem;
        }

        .schematic-title {
            font-size: 0.98rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.15rem;
        }

        .schematic-note {
            color: #667085;
            font-size: 0.82rem;
            margin-bottom: 0.45rem;
        }

        .schematic-wrap {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            overflow: hidden;
        }

        .schematic-wrap svg text {
            font-family: Arial, "Microsoft YaHei", sans-serif;
            fill: #1f2937;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 800;
            color: #0f172a;
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid #d6e2f3;
            font-weight: 700;
            min-height: 2.65rem;
        }

        div[data-testid="stDownloadButton"] > button {
            border-radius: 14px;
            font-weight: 750;
            min-height: 2.7rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(255, 255, 255, 0.65);
            padding: 0.35rem;
            border-radius: 16px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        }

        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        if not np.isfinite(v):
            return "—"
        if abs(v) >= 1e6 or (0 < abs(v) < 1e-4):
            return f"{v:.{digits}e}"
        return f"{v:.{digits}g}"
    except Exception:
        return str(value)


def metric_html(title: str, value: Any, foot: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{format_num(value)}</div>
        <div class="metric-foot">{foot}</div>
    </div>
    """


# =========================================================
# 2.1 截面示意图：SVG + base64 data URI
# =========================================================
# ============================================================
# 三、截面示意图 (SVG)
# ============================================================
# 所有 SVG 尺寸均统一在 viewBox=0 0 280 260 的画布内绘制
# 颜色统一:
#   - 截面填充:    #DBEAFE (浅蓝)
#   - 截面描边:    #1F4E79 (深蓝)
#   - 标注线/箭头: #DC2626 (红色)
#   - 标注文字:    #1F2937 (深灰)
#
# 注意: Streamlit 的 st.markdown(unsafe_allow_html=True) 会过滤掉 <svg> 标签,
# 因此 SVG 必须通过 base64 data URI 编码后用 <img> 标签嵌入才能正常显示。

_SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'viewBox="0 0 280 260" width="280" height="260" '
    'preserveAspectRatio="xMidYMid meet">'
)

_SVG_DEFS = '''
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
          markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#DC2626"/>
  </marker>
</defs>
'''


def _svg_circle() -> str:
    """圆形截面示意图"""
    return _SVG_HEADER + _SVG_DEFS + '''
<!-- 截面圆 -->
<circle cx="140" cy="130" r="80" fill="#DBEAFE" stroke="#1F4E79" stroke-width="2"/>
<!-- 直径标注 (水平双向箭头) -->
<line x1="60" y1="130" x2="220" y2="130" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="140" y="123" text-anchor="middle" font-size="16" font-weight="bold"
      fill="#1F2937" font-family="Arial">D</text>
<!-- 圆心 -->
<circle cx="140" cy="130" r="2.2" fill="#1F2937"/>
<!-- 标题 -->
<text x="140" y="240" text-anchor="middle" font-size="13" fill="#1F4E79"
      font-weight="bold" font-family="Arial">圆形截面</text>
</svg>'''


def _svg_circular_tube() -> str:
    """圆环截面示意图"""
    return _SVG_HEADER + _SVG_DEFS + '''
<!-- 外圆 -->
<circle cx="140" cy="130" r="80" fill="#DBEAFE" stroke="#1F4E79" stroke-width="2"/>
<!-- 内圆 (空心) -->
<circle cx="140" cy="130" r="58" fill="#FFFFFF" stroke="#1F4E79" stroke-width="2"/>
<!-- 外径标注 -->
<line x1="60" y1="130" x2="220" y2="130" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="140" y="123" text-anchor="middle" font-size="15" font-weight="bold"
      fill="#1F2937" font-family="Arial">D</text>
<!-- 壁厚标注 (右上斜引出) -->
<line x1="218" y1="78" x2="245" y2="55" stroke="#DC2626" stroke-width="1.2"/>
<line x1="240" y1="80" x2="255" y2="60" stroke="#DC2626" stroke-width="1.2"
      marker-end="url(#arr)"/>
<text x="258" y="55" font-size="13" font-weight="bold" fill="#1F2937"
      font-family="Arial">t</text>
<!-- 标题 -->
<text x="140" y="240" text-anchor="middle" font-size="13" fill="#1F4E79"
      font-weight="bold" font-family="Arial">圆环截面</text>
</svg>'''


def _svg_square() -> str:
    """方形(实心方截面)示意图"""
    return _SVG_HEADER + _SVG_DEFS + '''
<!-- 方形截面 -->
<rect x="60" y="50" width="160" height="160" fill="#DBEAFE"
      stroke="#1F4E79" stroke-width="2"/>
<!-- 边长 B 标注 (顶部水平双向箭头) -->
<line x1="60" y1="35" x2="220" y2="35" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="140" y="28" text-anchor="middle" font-size="16" font-weight="bold"
      fill="#1F2937" font-family="Arial">B</text>
<!-- 左侧标注 B (说明等边) -->
<line x1="45" y1="50" x2="45" y2="210" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="34" y="135" text-anchor="middle" font-size="16" font-weight="bold"
      fill="#1F2937" font-family="Arial">B</text>
<!-- 标题 -->
<text x="140" y="240" text-anchor="middle" font-size="13" fill="#1F4E79"
      font-weight="bold" font-family="Arial">方形截面</text>
</svg>'''


def _svg_hollow_square() -> str:
    """空心方钢截面示意图"""
    return _SVG_HEADER + _SVG_DEFS + '''
<!-- 外方 -->
<rect x="60" y="50" width="160" height="160" fill="#DBEAFE"
      stroke="#1F4E79" stroke-width="2"/>
<!-- 内方 (空心) -->
<rect x="84" y="74" width="112" height="112" fill="#FFFFFF"
      stroke="#1F4E79" stroke-width="2"/>
<!-- 外边长 B 标注 (顶部) -->
<line x1="60" y1="35" x2="220" y2="35" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="140" y="28" text-anchor="middle" font-size="16" font-weight="bold"
      fill="#1F2937" font-family="Arial">B</text>
<!-- 壁厚 tw 标注 (右上斜引出) -->
<line x1="208" y1="62" x2="248" y2="40" stroke="#DC2626" stroke-width="1.2"
      marker-end="url(#arr)"/>
<text x="252" y="38" font-size="13" font-weight="bold" fill="#1F2937"
      font-family="Arial">tw</text>
<!-- 标题 -->
<text x="140" y="240" text-anchor="middle" font-size="13" fill="#1F4E79"
      font-weight="bold" font-family="Arial">空心方钢截面</text>
</svg>'''


def _svg_i_beam() -> str:
    """工字钢截面示意图 (含 h, d, tw, tf 四个标注)"""
    # 几何参数 (画布坐标)
    # 截面整体居中: 中心 (140, 130)
    # 外形: 高度 160 (y=50~210), 翼缘宽 d=140 (x=70~210)
    # 翼缘厚 tf=18 (上 50~68, 下 192~210)
    # 腹板宽 tw=22 (x=129~151)
    return _SVG_HEADER + _SVG_DEFS + '''
<!-- 工字钢轮廓 (单一闭合多边形) -->
<path d="M 70,50  L 210,50  L 210,68  L 151,68
         L 151,192 L 210,192 L 210,210 L 70,210
         L 70,192  L 129,192 L 129,68  L 70,68 Z"
      fill="#DBEAFE" stroke="#1F4E79" stroke-width="2"/>

<!-- 截面高度 h 标注 (左侧外部) -->
<line x1="50" y1="50" x2="50" y2="210" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="40" y="135" text-anchor="middle" font-size="14" font-weight="bold"
      fill="#1F2937" font-family="Arial">h</text>

<!-- 截面宽度 d 标注 (顶部外部) -->
<line x1="70" y1="35" x2="210" y2="35" stroke="#DC2626" stroke-width="1.5"
      marker-start="url(#arr)" marker-end="url(#arr)"/>
<text x="140" y="28" text-anchor="middle" font-size="14" font-weight="bold"
      fill="#1F2937" font-family="Arial">d</text>

<!-- 翼缘厚度 tf 标注 (右上斜引出) -->
<line x1="215" y1="59" x2="248" y2="59" stroke="#DC2626" stroke-width="1.2"
      marker-end="url(#arr)"/>
<text x="252" y="63" font-size="12" font-weight="bold" fill="#1F2937"
      font-family="Arial">tf</text>

<!-- 腹板厚度 tw 标注 (右下斜引出, 指向腹板) -->
<line x1="248" y1="155" x2="155" y2="155" stroke="#DC2626" stroke-width="1.2"
      marker-end="url(#arr)"/>
<text x="252" y="159" font-size="12" font-weight="bold" fill="#1F2937"
      font-family="Arial">tw</text>

<!-- 标题 -->
<text x="140" y="240" text-anchor="middle" font-size="13" fill="#1F4E79"
      font-weight="bold" font-family="Arial">工字钢截面</text>
</svg>'''


SECTION_SVG_MAP = {
    1: _svg_circle,
    2: _svg_circular_tube,
    3: _svg_square,
    4: _svg_hollow_square,
    5: _svg_i_beam,
}


def _svg_to_data_uri(svg_str: str) -> str:
    """将 SVG 字符串转换为 base64 编码的 data URI, 便于通过 <img> 标签显示"""
    b64 = base64.b64encode(svg_str.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def render_section_schematic(section_type: int) -> None:
    """在 Streamlit 中渲染当前截面的示意图与参数说明
    使用 base64 data URI + <img> 标签, 避免 Streamlit 过滤 <svg> 标签导致空白
    """
    svg_fn = SECTION_SVG_MAP.get(section_type)
    if svg_fn is None:
        st.warning("当前截面无可用示意图")
        return

    svg_str = svg_fn()
    img_uri = _svg_to_data_uri(svg_str)

    section_name = SECTION_NAMES.get(section_type, "截面")

    # 参数符号-含义映射 (用于在示意图下方显示符号说明)
    legend_map: Dict[int, List[Tuple[str, str]]] = {
        1: [("D", "直径")],
        2: [("D", "外径"), ("t", "壁厚")],
        3: [("B", "边长")],
        4: [("B", "外边长"), ("tw", "壁厚")],
        5: [("h", "截面高度"), ("d", "截面宽度"),
            ("tw", "腹板厚度"), ("tf", "翼缘厚度")],
    }
    legend_items = legend_map.get(section_type, [])
    legend_html = "".join(
        f'<span style="display:inline-block;background:#F0F9FF;'
        f'border:1px solid #BAE6FD;border-radius:6px;'
        f'padding:2px 8px;margin:2px 4px 2px 0;'
        f'font-size:12px;color:#0C4A6E;">'
        f'<b>{sym}</b> &nbsp; {desc}</span>'
        for sym, desc in legend_items
    )

    st.markdown(
        f'''
<div style="background:white;border:1px solid #E5E7EB;border-radius:10px;
            padding:0.85rem 0.95rem;margin-bottom:0.9rem;
            box-shadow:0 1px 2px rgba(15,23,42,0.04);">
  <div style="color:#1F4E79;font-weight:600;font-size:0.95rem;
              margin-bottom:0.5rem;">
    📐 截面示意图 — {section_name}
  </div>
  <div style="text-align:center;">
    <img src="{img_uri}" alt="截面示意图"
         style="width:100%;max-width:340px;height:auto;display:inline-block;
                background:#FFFFFF;border:1px solid #E5E7EB;
                border-radius:8px;padding:6px;" />
  </div>
  <div style="margin-top:0.55rem;text-align:center;">
    {legend_html}
  </div>
  <div style="color:#6B7280;font-size:0.78rem;margin-top:0.5rem;
              text-align:center;line-height:1.5;">
    示意图仅用于标识参数符号位置, 不代表实际比例
  </div>
</div>
''',
        unsafe_allow_html=True,
    )

def to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "数值" in out.columns:
        out["数值"] = out["数值"].apply(format_num)
    return out


def reset_input_state() -> None:
    prefixes = (
        "proto_", "model_", "override_", "use_override",
        "dynamic_", "method_input_",
    )
    keys_to_delete = [key for key in st.session_state.keys() if key.startswith(prefixes)]
    keys_to_delete += [
        "member_select", "section_select", "method_select", "scb_input", "result",
        "input_snapshot", "override_snapshot", "tables_snapshot",
        "calculation_result", "method_input_snapshot", "load_snapshot",
    ]
    for key in set(keys_to_delete):
        st.session_state.pop(key, None)
    st.session_state["form_version"] = int(st.session_state.get("form_version", 0)) + 1


# =========================================================
# 3. 界面输入构建函数
# =========================================================

def get_section_inputs(section_type: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """根据截面形式生成原型构件与缩尺模型的截面参数输入。"""
    proto_section: Dict[str, float] = {}
    model_section: Dict[str, float] = {}

    proto_defaults = DEFAULT_SECTION_VALUES[section_type]["proto"]
    model_defaults = DEFAULT_SECTION_VALUES[section_type]["model"]

    st.markdown('<div class="section-caption">截面参数</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">根据截面形式输入几何参数，单位均为 mm。</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### 原型构件")
        for name, label, unit in SECTION_INPUTS[section_type]:
            proto_section[name] = st.number_input(
                f"{label}（原型构件，{unit}）",
                min_value=1.0e-9,
                value=float(proto_defaults[name]),
                step=1.0,
                format="%.6f",
                key=f"proto_sec_{section_type}_{name}",
            )

    with right:
        st.markdown("##### 缩尺模型")
        for name, label, unit in SECTION_INPUTS[section_type]:
            model_section[name] = st.number_input(
                f"{label}（缩尺模型，{unit}）",
                min_value=1.0e-9,
                value=float(model_defaults[name]),
                step=1.0,
                format="%.6f",
                key=f"model_sec_{section_type}_{name}",
            )

    return proto_section, model_section


def get_material_inputs() -> Tuple[Dict[str, float], Dict[str, float]]:
    """生成材料与构件长度参数输入。"""
    st.markdown('<div class="section-caption">材料与构件参数</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">E、fy 的单位为 MPa；l 为构件高度或计算长度，单位为 mm；bs 为无量纲硬化系数。</div>',
        unsafe_allow_html=True,
    )

    proto_mat: Dict[str, float] = {}
    model_mat: Dict[str, float] = {}

    material_items = [
        ("E", "弹性模量 E", "MPa", 1000.0, "%.6f"),
        ("fy", "屈服强度 fy", "MPa", 10.0, "%.6f"),
        ("bs", "硬化系数 bs", "—", 0.005, "%.6f"),
        ("l", "构件高度/计算长度 l", "mm", 100.0, "%.6f"),
    ]

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### 原型构件")
        for key, label, unit, step, fmt in material_items:
            proto_mat[key] = st.number_input(
                f"{label}（原型构件，{unit}）",
                min_value=1.0e-9,
                value=float(DEFAULT_MATERIAL_VALUES["proto"][key]),
                step=float(step),
                format=fmt,
                key=f"proto_mat_{key}",
            )

    with right:
        st.markdown("##### 缩尺模型")
        for key, label, unit, step, fmt in material_items:
            model_mat[key] = st.number_input(
                f"{label}（缩尺模型，{unit}）",
                min_value=1.0e-9,
                value=float(DEFAULT_MATERIAL_VALUES["model"][key]),
                step=float(step),
                format=fmt,
                key=f"model_mat_{key}",
            )

    return proto_mat, model_mat


def get_simply_supported_load_inputs() -> Dict[str, Any]:
    """生成简支梁密度、集中荷载及均布荷载输入。"""
    st.markdown("---")
    st.markdown('<div class="section-caption">简支梁荷载参数</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">原型集中荷载 Fp 的单位为 N。均布荷载可由材料密度和截面面积自动计算，也可直接输入。</div>',
        unsafe_allow_html=True,
    )

    rho_col1, rho_col2 = st.columns(2, gap="large")
    with rho_col1:
        rho_p = st.number_input(
            "材料密度 rho（原型构件，kg/m³）",
            min_value=1.0e-9,
            value=float(DEFAULT_DENSITY_VALUES["proto"]),
            step=50.0,
            format="%.6f",
            key="proto_rho",
        )
    with rho_col2:
        rho_m = st.number_input(
            "材料密度 rho（缩尺模型，kg/m³）",
            min_value=1.0e-9,
            value=float(DEFAULT_DENSITY_VALUES["model"]),
            step=50.0,
            format="%.6f",
            key="model_rho",
        )

    load_col1, load_col2 = st.columns([1.0, 1.0], gap="large")
    with load_col1:
        Fp = st.number_input(
            "原型跨中集中荷载 Fp（N）",
            value=200000.0,
            step=10000.0,
            format="%.6f",
            key="load_Fp",
            help="可输入正值或负值；相似比计算保留荷载符号。",
        )
    with load_col2:
        q_mode = st.selectbox(
            "均布荷载输入方式",
            options=["由构件自重自动计算", "直接输入实际均布荷载"],
            key="load_q_mode",
        )

    load_params: Dict[str, Any] = {
        "Fp": float(Fp),
        "g": DEFAULT_GRAVITY,
        "q_unit": "N/m",
    }
    if q_mode == "直接输入实际均布荷载":
        q_col1, q_col2 = st.columns(2, gap="large")
        with q_col1:
            qp = st.number_input(
                "原型均布荷载 qp（N/m）",
                value=1879.597456,
                step=100.0,
                format="%.6f",
                key="load_qp",
            )
        with q_col2:
            qm = st.number_input(
                "模型均布荷载 qm（N/m）",
                value=469.899364,
                step=100.0,
                format="%.6f",
                key="load_qm",
            )
        load_params.update({"qp": float(qp), "qm": float(qm)})

    return {
        "rho_p": float(rho_p),
        "rho_m": float(rho_m),
        "load_params": load_params,
        "q_mode": q_mode,
    }


def build_param_dict(
    proto_mat: Dict[str, float],
    model_mat: Dict[str, float],
    proto_section: Dict[str, float],
    model_section: Dict[str, float],
    simply_supported_inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """把界面输入整理为核心函数所需的 proto 与 model 字典。"""
    proto = {
        "E": float(proto_mat["E"]),
        "fy": float(proto_mat["fy"]),
        "bs": float(proto_mat["bs"]),
        "l": float(proto_mat["l"]),
        "section": {k: float(v) for k, v in proto_section.items()},
    }
    model = {
        "E": float(model_mat["E"]),
        "fy": float(model_mat["fy"]),
        "bs": float(model_mat["bs"]),
        "l": float(model_mat["l"]),
        "section": {k: float(v) for k, v in model_section.items()},
    }
    if simply_supported_inputs:
        proto["rho"] = float(simply_supported_inputs["rho_p"])
        model["rho"] = float(simply_supported_inputs["rho_m"])
    return proto, model

def build_user_inputs() -> Dict[str, Optional[float]]:
    """整理高级设置中的覆盖参数。"""
    user_inputs: Dict[str, Optional[float]] = {key: None for key in OVERRIDE_KEYS}

    with st.expander("高级设置：手动指定部分相似比", expanded=False):
        st.markdown(
            """
            <div class="info-box">
            当已有部分相似比，或需要人工指定部分约束时，可启用本功能。未勾选的参数将由系统自动计算。
            </div>
            """,
            unsafe_allow_html=True,
        )
        use_override = st.checkbox("启用手动指定部分相似比", value=False, key="use_override_checkbox")

        cols = st.columns(2, gap="large")
        for idx, key in enumerate(OVERRIDE_KEYS):
            with cols[idx % 2]:
                checked = st.checkbox(
                    f"覆盖 {OVERRIDE_LABELS[key]}",
                    value=False,
                    key=f"override_check_{key}",
                    disabled=not use_override,
                )
                value = st.number_input(
                    f"{OVERRIDE_LABELS[key]} 的指定值",
                    min_value=1.0e-12,
                    value=1.0,
                    step=0.05,
                    format="%.8f",
                    key=f"override_value_{key}",
                    disabled=(not use_override or not checked),
                )
                if use_override and checked:
                    user_inputs[key] = float(value)

    return user_inputs


def method_label_to_type(method_label: str) -> int:
    mapping = {
        "传统量纲分析方法 DA": 2,
        "已完成拟静力试验 CSL-Y": 3,
        "拟进行拟静力试验 CSL-N": 4,
    }
    if method_label not in mapping:
        raise ValidationError("应用场景选择错误")
    return mapping[method_label]


# =========================================================
# 4. 结果表、展示与导出函数
# =========================================================

def make_result_tables(result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """根据计算结果生成基本参数、衍生参数、关键参数表。"""
    basic_rows = [
        ["SH", "高度/长度相似比", result.get("SH"), "模型值 / 原型值"],
        ["SE", "弹性模量相似比", result.get("SE"), "模型值 / 原型值"],
        ["Ssigma", "应力/屈服强度相似比", result.get("Ssigma"), "根据所选场景确定"],
        ["Sbs", "硬化系数相似比", result.get("Sbs"), "模型值 / 原型值"],
        ["Sy", "截面特征尺寸相似比", result.get("Sy"), "由截面主控尺寸确定"],
    ]
    if result.get("Srho") is not None:
        basic_rows.append(["Srho", "材料密度相似比", result.get("Srho"), "模型密度 / 原型密度"])

    derived_rows = [
        ["SA", "截面面积相似比", result.get("SA"), "Am / Ap"],
        ["SI", "截面惯性矩相似比", result.get("SI"), "Im / Ip"],
    ]
    if result.get("SW") is not None:
        derived_rows.append(["SW", "截面抵抗矩相似比", result.get("SW"), "Wm / Wp"])
    derived_rows.append(["SK", "等效刚度相似比", result.get("SK"), "由 E、I、l 的相似关系确定"])
    if result.get("Sq") is not None:
        derived_rows.append(["Sq", "实际均布荷载相似比", result.get("Sq"), "qm / qp"])
    derived_rows.extend([
        ["Ap", "原型截面面积", result.get("Ap"), "mm²"],
        ["Am", "模型截面面积", result.get("Am"), "mm²"],
        ["Ip", "原型截面惯性矩", result.get("Ip"), "mm⁴"],
        ["Im", "模型截面惯性矩", result.get("Im"), "mm⁴"],
    ])
    if result.get("Wp") is not None:
        derived_rows.extend([
            ["Wp", "原型截面抵抗矩", result.get("Wp"), "mm³"],
            ["Wm", "模型截面抵抗矩", result.get("Wm"), "mm³"],
        ])

    key_rows = [
        ["Sd", "位移相似比", result.get("Sd"), "关键参数"],
        ["SF", "恢复力/集中荷载相似比", result.get("SF"), "关键参数"],
    ]
    if result.get("methodType") == 4 or (result.get("methodType") is None and result.get("Ssigma") is not None):
        key_rows.append(["Ssigma", "应力相似比", result.get("Ssigma"), "关键参数"])

    basic_df = pd.DataFrame(basic_rows, columns=["参数", "含义", "数值", "说明"])
    derived_df = pd.DataFrame(derived_rows, columns=["参数", "含义", "数值", "说明"])
    key_df = pd.DataFrame(key_rows, columns=["参数", "含义", "数值", "说明"])
    return basic_df, derived_df, key_df


def make_load_result_table(result: Dict[str, Any]) -> pd.DataFrame:
    """生成简支梁荷载与计算结果表。"""
    rows = [
        ["Fp", "原型跨中集中荷载", result.get("Fp"), "N"],
        ["Fm", "模型跨中集中荷载", result.get("Fm"), "N"],
        ["qp", "原型均布荷载", result.get("qp_N_per_m"), "N/m"],
        ["qm", "模型均布荷载", result.get("qm_N_per_m"), "N/m"],
        ["Sq", "实际均布荷载相似比", result.get("Sq"), "—"],
        ["SF_DA", "DA 集中荷载相似比", result.get("SF_DA"), "—"],
        ["Sd_DA", "DA 位移相似比", result.get("Sd_DA"), "—"],
        ["Sq_DA_target", "DA 目标均布荷载相似比", result.get("Sq_DA_target"), "—"],
    ]
    return pd.DataFrame(rows, columns=["参数", "含义", "数值", "单位"])

def render_result_cards(result: Dict[str, Any]) -> None:
    """渲染顶部结果摘要卡片。"""
    st.markdown("### 计算结果")

    overview = [
        ("构件形式", result.get("memberName", "悬臂梁"), "当前边界条件"),
        ("截面形式", result.get("sectionName"), "当前计算对象"),
        ("应用场景", result.get("methodName"), ""),
        ("几何基准相似比 SCB", result.get("SCB"), "用户输入"),
    ]
    columns = st.columns(len(overview), gap="small")
    for col, (title, value, foot) in zip(columns, overview):
        with col:
            st.markdown(metric_html(title, value, foot), unsafe_allow_html=True)

    key_cards = [
        ("位移相似比 Sd", result.get("Sd"), "关键参数"),
        ("恢复力/集中荷载相似比 SF", result.get("SF"), "关键参数"),
    ]
    if result.get("methodType") == 4:
        key_cards.append(("应力相似比 Ssigma", result.get("Ssigma"), "关键参数"))
    if result.get("memberType") == 2 and result.get("Fm") is not None:
        key_cards.append(("模型集中荷载 Fm（N）", result.get("Fm"), "简支梁计算结果"))

    columns = st.columns(len(key_cards), gap="small")
    for col, (title, value, foot) in zip(columns, key_cards):
        with col:
            st.markdown(metric_html(title, value, foot), unsafe_allow_html=True)

def flatten_input_params(proto: Dict[str, Any], model: Dict[str, Any]) -> pd.DataFrame:
    rows: List[List[Any]] = []
    for key in ["E", "fy", "bs", "l", "rho"]:
        if key in proto:
            rows.append(["原型构件", key, proto.get(key)])
    for key, value in proto.get("section", {}).items():
        rows.append(["原型构件", key, value])
    for key in ["E", "fy", "bs", "l", "rho"]:
        if key in model:
            rows.append(["缩尺模型", key, model.get(key)])
    for key, value in model.get("section", {}).items():
        rows.append(["缩尺模型", key, value])
    return pd.DataFrame(rows, columns=["对象", "参数", "数值"])

def make_export_long_table(
    result: Dict[str, Any],
    proto: Dict[str, Any],
    model: Dict[str, Any],
    user_inputs: Dict[str, Optional[float]],
    basic_df: pd.DataFrame,
    derived_df: pd.DataFrame,
    key_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(category: str, name: str, meaning: str, value: Any, note: str = "") -> None:
        rows.append({"类别": category, "参数": name, "含义": meaning, "数值": value, "说明": note})

    add("工具信息", "工具名称", "软件名称", "基于分类相似律的缩尺拟静力试验非完全相似比确定工具", "")
    add("工具信息", "计算时间", "结果生成时间", result.get("computedAt", ""), "")
    add("工具信息", "构件形式", "当前边界条件", result.get("memberName", "悬臂梁"), "")
    add("工具信息", "截面形式", "当前截面", result.get("sectionName", ""), "")
    add("工具信息", "应用场景", "当前计算任务", result.get("methodName", ""), "")
    add("工具信息", "SCB", "几何基准相似比", result.get("SCB"), "")

    metadata = result.get("componentMetadata") or {}
    if metadata:
        add("组件信息", "sectionId", "截面组件 ID", metadata.get("sectionId"), "")
        add("组件信息", "sectionVersion", "截面组件版本", metadata.get("sectionVersion"), "")
        add("组件信息", "sectionSource", "截面组件来源", format_component_source(metadata.get("sectionSource", "")), "")
        add("组件信息", "methodId", "计算方法组件 ID", metadata.get("methodId"), "")
        add("组件信息", "methodVersion", "计算方法组件版本", metadata.get("methodVersion"), "")
        add("组件信息", "methodSource", "计算方法组件来源", format_component_source(metadata.get("methodSource", "")), "")

    for key, value in (result.get("methodInputs") or {}).items():
        add("扩展方法附加参数", key, "用户扩展方法输入", value, "配置驱动输入")
    for key, value in (result.get("extraOutputs") or {}).items():
        add("扩展组件附加输出", key, "用户扩展方法输出", value, "配置驱动输出")

    for obj_name, data in [("原型构件", proto), ("缩尺模型", model)]:
        for key in ["E", "fy", "bs", "l", "rho"]:
            if key in data:
                add("输入参数", key, obj_name, data.get(key), "材料或构件参数")
        for key, value in data.get("section", {}).items():
            add("输入参数", key, obj_name, value, "截面参数")

    if result.get("memberType") == 2:
        for key, meaning, unit in [
            ("Fp", "原型跨中集中荷载", "N"),
            ("Fm", "模型跨中集中荷载", "N"),
            ("qp_N_per_m", "原型均布荷载", "N/m"),
            ("qm_N_per_m", "模型均布荷载", "N/m"),
            ("Sq", "实际均布荷载相似比", "—"),
            ("SF_DA", "DA 集中荷载相似比", "—"),
            ("Sd_DA", "DA 位移相似比", "—"),
            ("Sq_DA_target", "DA 目标均布荷载相似比", "—"),
        ]:
            add("简支梁荷载与结果", key, meaning, result.get(key), unit)

    for key, value in user_inputs.items():
        if value is not None:
            add("人工指定", key, OVERRIDE_LABELS.get(key, key), value, "手动覆盖")

    for category, df in [("基本参数相似比", basic_df), ("衍生参数相似比", derived_df), ("关键参数相似比", key_df)]:
        for _, row in df.iterrows():
            add(category, row["参数"], row["含义"], row["数值"], row["说明"])

    return pd.DataFrame(rows)

def export_to_excel(
    result: Dict[str, Any],
    proto: Dict[str, Any],
    model: Dict[str, Any],
    user_inputs: Dict[str, Optional[float]],
    basic_df: pd.DataFrame,
    derived_df: pd.DataFrame,
    key_df: pd.DataFrame,
) -> bytes:
    """生成 Excel 下载内容。"""
    output = io.BytesIO()
    input_df = flatten_input_params(proto, model)
    export_long_df = make_export_long_table(result, proto, model, user_inputs, basic_df, derived_df, key_df)

    summary_df = pd.DataFrame(
        [
            ["工具名称", "基于分类相似律的缩尺拟静力试验非完全相似比确定工具"],
            ["计算时间", result.get("computedAt", "")],
            ["构件形式", result.get("memberName", "悬臂梁")],
            ["截面形式", result.get("sectionName", "")],
            ["应用场景", result.get("methodName", "")],
            ["SCB", result.get("SCB")],
            ["Sd", result.get("Sd")],
            ["SF", result.get("SF")],
            ["Ssigma", result.get("Ssigma")],
            ["Fm", result.get("Fm")],
        ],
        columns=["项目", "内容"],
    )

    raw_df = pd.DataFrame([{"字段": k, "值": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in result.items()])
    override_df = pd.DataFrame([[k, v] for k, v in user_inputs.items()], columns=["参数", "指定值"])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        input_df.to_excel(writer, sheet_name="Inputs", index=False)
        basic_df.to_excel(writer, sheet_name="Basic", index=False)
        derived_df.to_excel(writer, sheet_name="Derived", index=False)
        key_df.to_excel(writer, sheet_name="Key", index=False)
        if result.get("memberType") == 2:
            make_load_result_table(result).to_excel(writer, sheet_name="Simply_Supported", index=False)
        override_df.to_excel(writer, sheet_name="Overrides", index=False)
        export_long_df.to_excel(writer, sheet_name="All_Results", index=False)
        raw_df.to_excel(writer, sheet_name="Raw_Result", index=False)

    return output.getvalue()

def export_to_csv_bytes(export_df: pd.DataFrame) -> bytes:
    return export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================================================
# 5. 页面渲染
# =========================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 操作流程")
        st.markdown(
            """
            1. 选择构件形式与截面形式  
            2. 输入原型构件与缩尺模型参数  
            3. 选择拟静力试验应用场景  
            4. 简支梁输入集中荷载与均布荷载  
            5. 可选：覆盖部分相似比  
            6. 点击计算并查看结果
            """
        )
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空结果", use_container_width=True):
                for key in ["result", "input_snapshot", "override_snapshot", "tables_snapshot", "load_snapshot"]:
                    st.session_state.pop(key, None)
                st.rerun()
        with col2:
            if st.button("重置参数", use_container_width=True):
                reset_input_state()
                st.rerun()

        st.markdown("---")
        st.markdown("### 使用提示")
        st.caption("悬臂梁沿用原计算流程；简支梁按跨中集中荷载与全跨均布荷载共同作用进行计算。")

def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">基于分类相似律的缩尺拟静力试验非完全相似比确定工具</div>
            <div class="hero-subtitle">面向缩尺拟静力试验的参数相似比计算</div>
            <div style="opacity:0.94; line-height:1.6; max-width:1080px;">
                本工具针对低周往复加载的缩尺拟静力恢复力与位移滞回响应，根据分类相似律确定非完全相似比，
                支持悬臂梁和简支梁两类构件形式，以及五类常用截面下的基本参数、衍生参数与关键参数相似比自动求解。
            </div>
            <div class="hero-tags">
                <span class="hero-tag">分类相似律</span>
                <span class="hero-tag">缩尺拟静力试验</span>
                <span class="hero-tag">非完全相似比</span>
                <span class="hero-tag">低周往复加载</span>              
                <span class="hero-tag">加载制度确定</span>
                <span class="hero-tag">试验结果换算</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_tab(api: SimilarityAPI) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 1. 选择构件形式、截面形式与应用场景")

    section_map, method_map = get_component_maps(api)
    section_ids = list(section_map.keys())
    method_ids = list(method_map.keys())
    default_section_id = "builtin_hollow_square"
    default_method_id = "builtin_csl_n"

    if st.session_state.get("member_select") not in MEMBER_NAMES:
        st.session_state["member_select"] = 1
    if st.session_state.get("section_select") not in section_ids:
        st.session_state["section_select"] = default_section_id
    if st.session_state.get("method_select") not in method_ids:
        st.session_state["method_select"] = default_method_id

    top_col0, top_col1, top_col2, top_col3 = st.columns([0.9, 1.05, 1.35, 1.0], gap="large")
    with top_col0:
        member_type = st.selectbox(
            "构件形式",
            options=list(MEMBER_NAMES.keys()),
            format_func=lambda item: MEMBER_NAMES[item],
            key="member_select",
            help="悬臂梁采用端部恢复力关系；简支梁采用跨中集中荷载与全跨均布荷载共同作用关系。",
        )

    with top_col1:
        section_id = st.selectbox(
            "截面形式",
            options=section_ids,
            format_func=lambda item: section_map[item].definition.name,
            key="section_select",
            help="选择当前试件的截面形式。",
        )
        section_record = section_map[section_id]
        section_definition = section_record.definition
        section_type = section_definition.legacy_id

    with top_col2:
        method_id = st.selectbox(
            "拟静力试验应用场景",
            options=method_ids,
            format_func=lambda item: component_display_name(method_map[item]),
            key="method_select",
            help="选择当前计算任务。",
        )
        method_record = method_map[method_id]
        method_definition = method_record.definition
        method_label = component_display_name(method_record)
        if method_record.source == "builtin":
            st.caption(METHOD_DESCRIPTIONS[method_label])
        else:
            st.caption(method_definition.description or "当前方法由用户扩展配置文件提供。")

    with top_col3:
        scb = st.number_input(
            "几何基准相似比 SCB",
            min_value=1.0e-9,
            value=0.5,
            step=0.05,
            format="%.8f",
            key="scb_input",
            help="用于传统量纲分析及相关对照计算的基准比例。",
        )

    if member_type == 2 and method_record.source != "builtin":
        st.warning("简支梁荷载关系当前与软件内置 DA、CSL-Y 和 CSL-N 方法配套；请选择内置计算方法。")
    elif member_type == 2 and section_record.source == "user":
        st.info("当前用户扩展截面将采用配置文件中的面积、惯性矩和特征尺寸关系参与简支梁计算。")

    if section_record.source == "builtin":
        render_section_schematic(int(section_type))
    else:
        render_user_section_schematic(section_definition)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 2. 输入原型构件与缩尺模型参数")
    proto_mat, model_mat = get_material_inputs()

    simply_supported_inputs: Optional[Dict[str, Any]] = None
    if member_type == 2:
        simply_supported_inputs = get_simply_supported_load_inputs()

    st.markdown("---")
    if section_record.source == "builtin":
        proto_section, model_section = get_section_inputs(int(section_type))
    else:
        st.markdown(
            """
            <div class="info-box">
            当前选择的是用户扩展截面。参数输入项、截面面积、截面惯性矩及特征尺寸关系由已安装的配置文件提供。
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-caption">截面参数</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">根据当前用户扩展截面配置输入几何参数。</div>', unsafe_allow_html=True)
        proto_section, model_section = render_dynamic_parameter_form(
            section_definition,
            int(st.session_state.get("form_version", 0)),
        )

    method_inputs: Dict[str, float] = {}
    if method_record.source == "user":
        method_inputs = render_user_method_inputs(method_definition)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 3. 已知相似比输入")
    user_inputs = build_user_inputs()

    calculate_col, hint_col = st.columns([0.9, 2.1], gap="large")
    with calculate_col:
        do_calculate = st.button("开始计算", type="primary", use_container_width=True)
    with hint_col:
        st.markdown(
            """
            <div class="section-note" style="margin-top:0.42rem;">
            计算前系统将自动检查参数取值与截面几何约束。简支梁结果与集中荷载和均布荷载的实际组合有关。
            </div>
            """,
            unsafe_allow_html=True,
        )

    if do_calculate:
        try:
            if member_type == 2 and method_record.source != "builtin":
                raise ValidationError("简支梁计算请选择软件内置 DA、CSL-Y 或 CSL-N 方法。")

            proto, model = build_param_dict(
                proto_mat,
                model_mat,
                proto_section,
                model_section,
                simply_supported_inputs=simply_supported_inputs,
            )
            if section_record.source == "builtin" and method_record.source == "builtin":
                result = calculate_similarity(
                    section_type=int(section_type),
                    method_type=int(method_definition.legacy_id),
                    scb=float(scb),
                    proto=proto,
                    model=model,
                    user_inputs=user_inputs,
                    show_detail=1,
                    member_type=int(member_type),
                    load_params=(simply_supported_inputs or {}).get("load_params"),
                )
                result["methodName"] = method_label
                result["componentMetadata"] = {
                    "sectionId": section_definition.id,
                    "sectionVersion": section_definition.component_version,
                    "sectionSource": section_record.source,
                    "methodId": method_definition.id,
                    "methodVersion": method_definition.component_version,
                    "methodSource": method_record.source,
                }
            else:
                result = calculate_with_component_api(
                    api=api,
                    section_id=section_id,
                    method_id=method_id,
                    scb=float(scb),
                    proto=proto,
                    model=model,
                    overrides=user_inputs,
                    method_inputs=method_inputs,
                )
                if member_type == 2:
                    result = apply_simply_supported_to_base_result(
                        base_result=result,
                        proto=proto,
                        model=model,
                        method_type=int(method_definition.legacy_id),
                        scb=float(scb),
                        user_inputs=user_inputs,
                        load_params=(simply_supported_inputs or {}).get("load_params"),
                    )
                    result["methodName"] = method_label
                else:
                    result["memberType"] = 1
                    result["memberName"] = MEMBER_NAMES[1]

            result["computedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["methodDescription"] = (
                METHOD_DESCRIPTIONS[method_label]
                if method_record.source == "builtin"
                else method_definition.description or "当前结果由用户扩展计算方法得到。"
            )
            result["methodInputs"] = dict(method_inputs)
            if simply_supported_inputs:
                result["loadInputMode"] = simply_supported_inputs["q_mode"]

            basic_df, derived_df, key_df = make_result_tables(result)
            st.session_state["result"] = result
            st.session_state["input_snapshot"] = {
                "proto": proto,
                "model": model,
                "memberType": int(member_type),
                "memberName": MEMBER_NAMES[int(member_type)],
            }
            st.session_state["override_snapshot"] = user_inputs
            st.session_state["method_input_snapshot"] = method_inputs
            st.session_state["load_snapshot"] = simply_supported_inputs
            st.session_state["tables_snapshot"] = {"basic": basic_df, "derived": derived_df, "key": key_df}

            st.success("计算完成。请切换到“计算结果”或“结果导出与说明”查看结果。")
        except ValidationError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"计算失败：{type(exc).__name__}: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

def render_results_tab() -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("请先在“参数输入”页面完成计算。")
        return

    basic_df, derived_df, key_df = make_result_tables(result)
    render_result_cards(result)

    st.markdown(
        f"""
        <div class="success-box">
        {result.get("methodDescription", METHOD_DESCRIPTIONS.get(result.get("methodName", ""), ""))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for warning in result.get("calculationWarnings", []) or []:
        st.warning(warning)

    st.markdown("### 分类结果表")
    tab_basic, tab_derived, tab_key = st.tabs(["基本参数相似比", "衍生参数相似比", "关键参数相似比"])
    with tab_basic:
        st.dataframe(to_display_df(basic_df), use_container_width=True, hide_index=True)
    with tab_derived:
        st.dataframe(to_display_df(derived_df), use_container_width=True, hide_index=True)
    with tab_key:
        st.dataframe(to_display_df(key_df), use_container_width=True, hide_index=True)

    if result.get("memberType") == 2:
        st.markdown("### 简支梁荷载与计算结果")
        st.markdown(
            '<div class="info-box">简支梁结果由跨中集中荷载与全跨均布荷载共同确定。模型集中荷载 Fm 由所选方法自动计算。</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(to_display_df(make_load_result_table(result)), use_container_width=True, hide_index=True)

    extra_outputs = result.get("extraOutputs") or {}
    if extra_outputs:
        with st.expander("查看扩展组件附加结果", expanded=False):
            extra_df = pd.DataFrame([{"参数": key, "数值": format_num(value)} for key, value in extra_outputs.items()])
            st.dataframe(extra_df, use_container_width=True, hide_index=True)

def render_export_tab() -> None:
    result = st.session_state.get("result")
    input_snapshot = st.session_state.get("input_snapshot")
    user_inputs = st.session_state.get("override_snapshot")

    if not result or not input_snapshot:
        st.info("请先在“参数输入”页面完成计算。")
        return

    proto = input_snapshot["proto"]
    model = input_snapshot["model"]
    user_inputs = user_inputs or {key: None for key in OVERRIDE_KEYS}
    basic_df, derived_df, key_df = make_result_tables(result)

    st.markdown("### 结果导出")
    st.markdown(
        """
        <div class="info-box">
        导出文件包含工具名称、截面形式、应用场景、输入参数、基本参数相似比、衍生参数相似比、关键参数相似比与计算时间。
        </div>
        """,
        unsafe_allow_html=True,
    )

    export_df = make_export_long_table(result, proto, model, user_inputs, basic_df, derived_df, key_df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"similarity_result_{timestamp}.csv"
    excel_name = f"similarity_result_{timestamp}.xlsx"

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.download_button(
            label="下载 CSV 结果文件",
            data=export_to_csv_bytes(export_df),
            file_name=csv_name,
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="下载 Excel 结果文件",
            data=export_to_excel(result, proto, model, user_inputs, basic_df, derived_df, key_df),
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("### 导出内容预览")
    st.dataframe(to_display_df(export_df), use_container_width=True, hide_index=True)


def render_help_tab() -> None:
    st.markdown("### 使用说明")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="small-card">
            <div class="section-caption">功能定位</div>
            <div class="section-note">
            用于缩尺拟静力试验中的非完全相似比确定，支持悬臂梁和简支梁两类构件形式。
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="small-card">
            <div class="section-caption">构件与截面</div>
            <div class="section-note">
            悬臂梁采用端部恢复力关系；简支梁考虑跨中集中荷载与全跨均布荷载共同作用。内置圆形、圆环、方形、空心方钢和工字钢截面。
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="small-card">
            <div class="section-caption">应用场景</div>
            <div class="section-note">
            DA 用作对照计算；CSL-Y 面向已完成拟静力试验；CSL-N 面向拟进行拟静力试验。简支梁相似比随实际荷载组合变化。
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="small-card">
            <div class="section-caption">参数校验</div>
            <div class="section-note">
            系统检查材料、几何和荷载参数。简支梁可由材料密度自动计算自重均布荷载，也可直接输入实际均布荷载。
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 建议流程")
    st.markdown(
        """
        <div class="card">
        <ol>
            <li>选择悬臂梁或简支梁，并选择截面形式。</li>
            <li>输入原型构件与缩尺模型的材料、长度和截面参数。</li>
            <li>简支梁输入原型集中荷载，并选择均布荷载的确定方式。</li>
            <li>选择 DA、CSL-Y 或 CSL-N；如有需要，可覆盖部分相似比。</li>
            <li>点击“开始计算”，查看分类结果及简支梁荷载计算结果。</li>
            <li>在导出页面下载 CSV 或 Excel 文件。</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_component_manager(api: SimilarityAPI) -> None:
    manager = UserComponentManager(api.user_config_root)
    st.markdown("### 扩展组件管理")
    st.markdown(
        """
        <div class="info-box">
        用户可通过安全的 YAML/JSON 配置文件扩展截面形式或计算方法。普通组件配置不会执行 Python 代码。
        </div>
        """,
        unsafe_allow_html=True,
    )

    notice = st.session_state.pop("component_notice", None)
    if notice:
        st.success(notice)

    section_rows = [
        {
            "ID": item.definition.id,
            "名称": item.definition.name,
            "版本": item.definition.component_version,
            "来源": format_component_source(item.source),
        }
        for item in api.list_sections()
    ]
    method_rows = [
        {
            "ID": item.definition.id,
            "名称": component_display_name(item),
            "版本": item.definition.component_version,
            "来源": format_component_source(item.source),
        }
        for item in api.list_methods()
    ]

    tab_sections, tab_methods, tab_upload, tab_templates = st.tabs(
        ["截面组件", "计算方法组件", "上传与删除", "模板与安全说明"]
    )
    with tab_sections:
        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)
    with tab_methods:
        st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)
    with tab_upload:
        st.markdown("#### 安装用户扩展组件")
        kind = st.radio(
            "组件类型",
            options=["section", "method"],
            format_func=lambda value: "截面" if value == "section" else "计算方法",
            horizontal=True,
            key="component_upload_kind",
        )
        uploaded = st.file_uploader(
            "上传 YAML/JSON 配置文件",
            type=["yaml", "yml", "json"],
            key="component_uploader",
        )
        if st.button("校验并安装组件", disabled=uploaded is None, use_container_width=True):
            try:
                definition, target = manager.install(uploaded.getvalue(), uploaded.name, kind)
                api.reload_components()
                st.session_state["component_notice"] = f"已安装用户扩展组件：{definition.name}（{target.name}）"
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"安装失败：{exc}")

        st.markdown("---")
        st.markdown("#### 删除用户扩展组件")
        user_components = [
            ("section", item.definition.id, item.definition.name)
            for item in api.list_sections()
            if item.source == "user"
        ] + [
            ("method", item.definition.id, component_display_name(item))
            for item in api.list_methods()
            if item.source == "user"
        ]
        if not user_components:
            st.caption("当前没有已安装的用户扩展组件。")
        else:
            selected = st.selectbox(
                "选择要删除的组件",
                options=user_components,
                format_func=lambda item: f"{'截面' if item[0] == 'section' else '计算方法'}｜{item[2]}（{item[1]}）",
                key="component_delete_select",
            )
            if st.button("删除所选用户组件", use_container_width=True):
                try:
                    manager.delete(selected[0], selected[1])
                    if selected[0] == "section" and st.session_state.get("section_select") == selected[1]:
                        st.session_state["section_select"] = "builtin_hollow_square"
                    if selected[0] == "method" and st.session_state.get("method_select") == selected[1]:
                        st.session_state["method_select"] = "builtin_csl_n"
                    for key in ["result", "input_snapshot", "override_snapshot", "tables_snapshot", "method_input_snapshot"]:
                        st.session_state.pop(key, None)
                    api.reload_components()
                    st.session_state["component_notice"] = f"已删除用户扩展组件：{selected[2]}"
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"删除失败：{exc}")

    with tab_templates:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.download_button(
                "下载用户自定义截面模板",
                data=manager.section_template(),
                file_name="custom_section_template.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "下载用户自定义计算方法模板",
                data=manager.method_template(),
                file_name="custom_method_template.yaml",
                mime="text/yaml",
                use_container_width=True,
            )
        st.markdown(
            """
            <div class="warning-box">
            <b>安全说明：</b>普通扩展组件仅接受 YAML、YML 或 JSON 文件；页面不接受 Python 文件，
            不执行任意代码，也不会覆盖软件内置组件。数学表达式仅允许数值、已声明变量、基本算术运算、比较运算及白名单数学函数。
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 6. 主函数
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title="缩尺拟静力试验非完全相似比确定工具",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "form_version" not in st.session_state:
        st.session_state["form_version"] = 0

    set_page_style()
    render_sidebar()
    render_header()
    api = get_api()

    tab_help, tab_input, tab_result, tab_export, tab_components = st.tabs(
        ["使用说明", "参数输入", "计算结果", "结果导出与说明", "🧱 扩展组件管理"]
    )

    with tab_help:
        render_help_tab()

    with tab_input:
        render_input_tab(api)

    with tab_result:
        render_results_tab()

    with tab_export:
        render_export_tab()

    with tab_components:
        render_component_manager(api)

if __name__ == "__main__":
    main()

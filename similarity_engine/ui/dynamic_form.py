from __future__ import annotations

from typing import Any, Dict, Tuple

import streamlit as st

from ..models import SectionDefinition


def _default(parameter, side: str) -> float:
    value = parameter.prototype_default if side == "proto" else parameter.model_default
    if value is None:
        value = parameter.default
    return float(value if value is not None else 0.0)


def _kwargs(parameter, side: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "value": _default(parameter, side),
        "step": float(parameter.step or 1.0),
        "format": parameter.format or "%.6f",
    }
    lower = parameter.lower_bound
    if parameter.positive:
        lower = max(1.0e-12, lower) if lower is not None else 1.0e-12
    elif parameter.nonnegative:
        lower = max(0.0, lower) if lower is not None else 0.0
    if lower is not None:
        result["min_value"] = float(lower)
    if parameter.upper_bound is not None:
        result["max_value"] = float(parameter.upper_bound)
    return result


def render_dynamic_parameter_form(
    section: SectionDefinition,
    form_version: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    proto: Dict[str, float] = {}
    model: Dict[str, float] = {}
    left, right = st.columns(2, gap="large")
    ordered = sorted(section.parameters, key=lambda item: item.order)
    for side, column, title, target in [
        ("proto", left, "##### 原型构件", proto),
        ("model", right, "##### 缩尺模型", model),
    ]:
        with column:
            st.markdown(title)
            last_group = None
            for parameter in ordered:
                if parameter.group and parameter.group != last_group:
                    st.markdown(f"**{parameter.group}**")
                    last_group = parameter.group
                unit = f"，{parameter.unit}" if parameter.unit else ""
                target[parameter.name] = st.number_input(
                    f"{parameter.label}（{'原型构件' if side == 'proto' else '缩尺模型'}{unit}）",
                    key=f"dynamic_{side}_{section.id}_{parameter.name}_{form_version}",
                    help=parameter.description or None,
                    **_kwargs(parameter, side),
                )
    return proto, model


def collect_dynamic_parameters(
    section: SectionDefinition,
    form_version: int,
    state: Any,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    proto: Dict[str, float] = {}
    model: Dict[str, float] = {}
    for parameter in section.parameters:
        proto[parameter.name] = float(
            state.get(f"dynamic_proto_{section.id}_{parameter.name}_{form_version}", _default(parameter, "proto"))
        )
        model[parameter.name] = float(
            state.get(f"dynamic_model_{section.id}_{parameter.name}_{form_version}", _default(parameter, "model"))
        )
    return proto, model

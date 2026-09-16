# -*- coding: utf-8 -*-
"""
基于分类相似律的拟静力试验非完全相似比自动计算核心代码。

本版在原有“悬臂梁”计算基础上新增“简支梁”构件形式，并保持原函数
calculate_similarity 的原有位置参数兼容性。

主要扩展
--------
1. 构件形式：
   member_type = 1  悬臂梁（沿用原 corecode.py 计算关系）
   member_type = 2  简支梁（跨中集中荷载 + 全跨均布荷载）

2. 简支梁计算方法：
   method_type = 1  足尺
   method_type = 2  传统量纲分析（DA）
   method_type = 3  分类相似律位移约束（CSL-U）
   method_type = 4  分类相似律位移-应力约束（CSL-US）

3. 简支梁荷载约定：
   - 集中荷载 F 的单位为 N；
   - 构件长度及截面尺寸沿用原程序，单位为 mm；
   - 均布荷载在内部统一换算为 N/mm；
   - load_params 中显式输入 q 时，默认单位为 N/m，可通过 q_unit 修改；
   - 未显式输入 q 时，由 rho*g*A 自动计算构件自重均布荷载。

4. 简支梁公式（跨中响应）：
   Mmax = F*L/4 + q*L^2/8
   sigma = Mmax/W
   u = F*L^3/(48*E*I) + 5*q*L^4/(384*E*I)

编写人：刘铖
扩充说明：依据 xiuzheng_PS_DA_CSLU_CSLUS.m 中实际执行的简支梁公式整理。
"""

import math
from typing import Any, Dict, Iterable, List, Optional


SECTION_NAMES = {
    1: "圆形",
    2: "圆环",
    3: "方形",
    4: "空心方钢",
    5: "工字钢",
}

MEMBER_NAMES = {
    1: "悬臂梁",
    2: "简支梁",
}

METHOD_NAMES = {
    1: "足尺",
    2: "传统量纲分析（DA）",
    3: "分类相似律（位移约束，CSL-U）",
    4: "分类相似律（位移-应力约束，CSL-US）",
}

_EPS = 1.0e-12


def _check_positive(name: str, value: Any) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{name} 必须大于 0")


def _check_nonzero(name: str, value: Any) -> None:
    if value is None or abs(float(value)) <= _EPS:
        raise ValueError(f"{name} 不能为 0")


def _validate_section(section_type: int, params: Dict[str, float], prefix: str) -> None:
    if section_type == 1:
        _check_positive(f"{prefix}D", params.get("D"))

    elif section_type == 2:
        _check_positive(f"{prefix}D", params.get("D"))
        _check_positive(f"{prefix}t", params.get("t"))
        if 2 * params["t"] >= params["D"]:
            raise ValueError(f"{prefix}圆环截面参数错误：2t 必须小于 D")

    elif section_type == 3:
        _check_positive(f"{prefix}B", params.get("B"))

    elif section_type == 4:
        _check_positive(f"{prefix}B", params.get("B"))
        _check_positive(f"{prefix}tw", params.get("tw"))
        if 2 * params["tw"] >= params["B"]:
            raise ValueError(f"{prefix}空心方钢参数错误：2tw 必须小于 B")

    elif section_type == 5:
        _check_positive(f"{prefix}h", params.get("h"))
        _check_positive(f"{prefix}d", params.get("d"))
        _check_positive(f"{prefix}tw", params.get("tw"))
        _check_positive(f"{prefix}tf", params.get("tf"))
        if 2 * params["tf"] >= params["h"]:
            raise ValueError(f"{prefix}工字钢参数错误：2tf 必须小于 h")
        if params["tw"] >= params["d"]:
            raise ValueError(f"{prefix}工字钢参数错误：tw 必须小于 d")

    else:
        raise ValueError("section_type 输入错误，应为 1~5")


def _calc_sy(section_type: int, proto: Dict[str, float], model: Dict[str, float]) -> float:
    """外缘距离相似比。对称截面中，比例等同于对应总高度/直径之比。"""
    if section_type in (1, 2):
        return model["D"] / proto["D"]
    if section_type in (3, 4):
        return model["B"] / proto["B"]
    if section_type == 5:
        return model["h"] / proto["h"]
    raise ValueError("section_type 输入错误，应为 1~5")


def _calc_area(section_type: int, params: Dict[str, float]) -> float:
    """截面面积；输入尺寸为 mm 时，输出为 mm^2。"""
    if section_type == 1:
        return math.pi * params["D"] ** 2 / 4

    if section_type == 2:
        d_in = params["D"] - 2 * params["t"]
        return math.pi * (params["D"] ** 2 - d_in ** 2) / 4

    if section_type == 3:
        return params["B"] ** 2

    if section_type == 4:
        b_in = params["B"] - 2 * params["tw"]
        return params["B"] ** 2 - b_in ** 2

    if section_type == 5:
        return (
            params["d"] * params["h"]
            - (params["d"] - params["tw"]) * (params["h"] - 2 * params["tf"])
        )

    raise ValueError("section_type 输入错误，应为 1~5")


def _calc_inertia(section_type: int, params: Dict[str, float]) -> float:
    """截面惯性矩；输入尺寸为 mm 时，输出为 mm^4。"""
    if section_type == 1:
        return math.pi * params["D"] ** 4 / 64

    if section_type == 2:
        d_in = params["D"] - 2 * params["t"]
        return math.pi * (params["D"] ** 4 - d_in ** 4) / 64

    if section_type == 3:
        return params["B"] ** 4 / 12

    if section_type == 4:
        b_in = params["B"] - 2 * params["tw"]
        return (params["B"] ** 4 - b_in ** 4) / 12

    if section_type == 5:
        return (
            params["d"] * params["h"] ** 3
            - (params["d"] - params["tw"])
            * (params["h"] - 2 * params["tf"]) ** 3
        ) / 12

    raise ValueError("section_type 输入错误，应为 1~5")


def _calc_section_modulus(inertia: float, outer_distance: float) -> float:
    """截面抵抗矩 W=I/y；输入为 mm^4、mm，输出为 mm^3。"""
    _check_positive("外缘距离 y", outer_distance)
    return inertia / outer_distance


def _q_to_n_per_mm(q_value: float, q_unit: str) -> float:
    unit = q_unit.strip().lower().replace(" ", "")
    if unit in {"n/mm", "n·mm^-1", "n/mm^1"}:
        return float(q_value)
    if unit in {"n/m", "n·m^-1", "n/m^1"}:
        return float(q_value) / 1000.0
    raise ValueError("q_unit 仅支持 'N/m' 或 'N/mm'")


def _self_weight_q_n_per_mm(rho: float, area_mm2: float, g: float) -> float:
    """由 q=rho*g*A 计算自重，输出 N/mm。"""
    _check_positive("材料密度 rho", rho)
    _check_positive("重力加速度 g", g)
    # mm^2 -> m^2，得到 N/m 后再除以 1000 转为 N/mm。
    return rho * g * area_mm2 * 1.0e-9


def _resolve_simply_supported_loads(
    proto: Dict[str, Any],
    model: Dict[str, Any],
    load_params: Optional[Dict[str, Any]],
    area_proto: float,
    area_model: float,
) -> Dict[str, float]:
    """读取简支梁荷载，并统一转换到 F:N、q:N/mm。"""
    load_params = load_params or {}
    g = float(load_params.get("g", 9.80665))
    _check_positive("g", g)

    fp = load_params.get("Fp", proto.get("F"))
    if fp is None:
        raise ValueError('简支梁必须在 load_params["Fp"] 或 proto["F"] 中给出原型集中荷载')
    fp = float(fp)

    q_unit = str(load_params.get("q_unit", "N/m"))

    qp_input = load_params.get("qp", proto.get("q"))
    qm_input = load_params.get("qm", model.get("q"))

    if qp_input is None:
        rho_p = proto.get("rho")
        if rho_p is None:
            raise ValueError(
                "简支梁未给出原型均布荷载 qp，也未在 proto 中给出 rho，"
                "无法计算自重均布荷载。"
            )
        qp = _self_weight_q_n_per_mm(float(rho_p), area_proto, g)
    else:
        qp = _q_to_n_per_mm(float(qp_input), q_unit)

    if qm_input is None:
        rho_m = model.get("rho")
        if rho_m is None:
            raise ValueError(
                "简支梁未给出模型均布荷载 qm，也未在 model 中给出 rho，"
                "无法计算自重均布荷载。"
            )
        qm = _self_weight_q_n_per_mm(float(rho_m), area_model, g)
    else:
        qm = _q_to_n_per_mm(float(qm_input), q_unit)

    return {
        "Fp": fp,
        "qp": qp,
        "qm": qm,
        "g": g,
    }


def _safe_ratio(numerator: float, denominator: float, name: str) -> float:
    if abs(denominator) <= _EPS:
        raise ValueError(f"{name} 的分母接近 0，当前荷载组合无法稳定计算相似比")
    return numerator / denominator


def _ratio_or_nan(numerator: float, denominator: float) -> float:
    """分母为0时返回 NaN；用于荷载过零点处未定义的瞬时力相似比。"""
    if abs(denominator) <= _EPS:
        return math.nan
    return numerator / denominator


def apply_simply_supported_to_base_result(
    base_result: Dict[str, Any],
    proto: Dict[str, Any],
    model: Dict[str, Any],
    method_type: int,
    scb: float,
    user_inputs: Optional[Dict[str, Optional[float]]] = None,
    load_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将简支梁荷载关系应用到已完成截面计算的基础结果。

    该接口主要用于用户扩展截面。base_result 需至少包含 Ap、Am、
    SH、SE、SsigmaReal（或 Ssigma）、Sy、SI、SK 等基础量。
    """
    user_inputs = user_inputs or {}
    result = dict(base_result)

    if method_type not in (2, 3, 4):
        raise ValueError("简支梁扩展截面计算仅支持 DA、CSL-Y/CSL-U 和 CSL-N/CSL-US")

    Ap = float(result["Ap"])
    Am = float(result["Am"])
    SH = float(result["SH"])
    SE = float(result["SE"])
    SI = float(result["SI"])
    Sy = float(result["Sy"])
    Ssigma_real = float(
        result.get("SsigmaReal")
        if result.get("SsigmaReal") is not None
        else result.get("Ssigma", float(model["fy"]) / float(proto["fy"]))
    )

    SW = SI / Sy
    if user_inputs.get("SW") is not None:
        SW = float(user_inputs["SW"])

    rho_p = proto.get("rho")
    rho_m = model.get("rho")
    Srho = None
    if rho_p is not None and rho_m is not None:
        _check_positive("proto_rho", rho_p)
        _check_positive("model_rho", rho_m)
        Srho = float(rho_m) / float(rho_p)
    if user_inputs.get("Srho") is not None:
        Srho = float(user_inputs["Srho"])

    loads = _resolve_simply_supported_loads(
        proto=proto,
        model=model,
        load_params=load_params,
        area_proto=Ap,
        area_model=Am,
    )
    Fp = loads["Fp"]
    qp = loads["qp"]
    qm = loads["qm"]
    Sq_actual = _ratio_or_nan(qm, qp)
    if user_inputs.get("Sq") is not None:
        Sq_actual = float(user_inputs["Sq"])
        qm = qp * Sq_actual

    Lp = float(proto["l"])
    Lm = float(model["l"])
    SF_DA = Ssigma_real * scb ** 2
    Sd_DA = scb
    Sq_DA_target = SF_DA / scb

    if method_type == 2:
        Sd = Sd_DA
        SF = SF_DA
        Ssigma = Ssigma_real
        Fm = Fp * SF
        load_dependent = False
    elif method_type == 3:
        Sd = Sd_DA
        Fm = (
            Sd * SE * SI / SH ** 3 * (Fp + 5.0 / 8.0 * qp * Lp)
            - 5.0 / 8.0 * qm * Lm
        )
        SF = _ratio_or_nan(Fm, Fp)
        Ssigma = (
            SH / SW
            * _safe_ratio(
                Fm + 0.5 * qm * Lm,
                Fp + 0.5 * qp * Lp,
                "简支梁应力相似比",
            )
        )
        load_dependent = True
    else:
        Ssigma = Ssigma_real
        Fm = (
            SW / SH * Ssigma * (Fp + 0.5 * qp * Lp)
            - 0.5 * qm * Lm
        )
        SF = _ratio_or_nan(Fm, Fp)
        Sd = (
            SH ** 3 / (SE * SI)
            * _safe_ratio(
                Fm + 5.0 / 8.0 * qm * Lm,
                Fp + 5.0 / 8.0 * qp * Lp,
                "简支梁位移相似比",
            )
        )
        load_dependent = True

    if user_inputs.get("Sd") is not None:
        Sd = float(user_inputs["Sd"])
    if user_inputs.get("SF") is not None:
        SF = float(user_inputs["SF"])
        Fm = Fp * SF

    result.update(
        {
            "memberType": 2,
            "memberName": MEMBER_NAMES[2],
            "methodType": method_type,
            "SCB": float(scb),
            "Ssigma": Ssigma,
            "SsigmaReal": Ssigma_real,
            "SW": SW,
            "Srho": Srho,
            "Sd": Sd,
            "SF": SF,
            "Fp": Fp,
            "Fm": Fm,
            "qp_N_per_mm": qp,
            "qm_N_per_mm": qm,
            "qp_N_per_m": qp * 1000.0,
            "qm_N_per_m": qm * 1000.0,
            "Sq": Sq_actual,
            "SF_DA": SF_DA,
            "Sd_DA": Sd_DA,
            "Sq_DA_target": Sq_DA_target,
            "loadCombinationDependent": load_dependent,
            "forceRatioDefined": not math.isnan(SF),
            "uniformLoadRatioDefined": not math.isnan(Sq_actual),
        }
    )
    return result


def calculate_similarity(
    section_type: int,
    method_type: int,
    scb: float,
    proto: Dict[str, Any],
    model: Dict[str, Any],
    user_inputs: Optional[Dict[str, Optional[float]]] = None,
    show_detail: int = 1,
    member_type: int = 1,
    load_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    计算构件相似比。

    参数说明
    ----------
    section_type : int
        1圆形，2圆环，3方形，4空心方钢，5工字钢。
    method_type : int
        1足尺，2 DA，3 CSL-U，4 CSL-US。
    scb : float
        传统几何缩尺比，例如 1/2 缩尺输入 0.5。
    proto, model : dict
        通用字段：E、fy、bs、l、section。
        简支梁采用自重均布荷载时，还需 rho。

        示例：
        {
            "E": 2.1e5,             # MPa
            "fy": 270,              # MPa
            "bs": 0.02,
            "l": 3600,              # mm
            "rho": 7850,            # kg/m^3，简支梁需要
            "section": {"B": 450, "tw": 14}
        }
    user_inputs : dict, optional
        可覆盖 SH、SE、Ssigma、Sbs、Sy、SA、SI、SW、SK、Srho、Sq、Sd、SF。
    member_type : int
        1 悬臂梁（默认，兼容旧版调用）；2 简支梁。
    load_params : dict, optional
        简支梁 method_type=3/4 必需的荷载信息。
        {
            "Fp": 200000,           # N；也可放在 proto["F"]
            "qp": 1879.6,           # 可选；默认由 proto.rho 计算
            "qm": 469.9,            # 可选；默认由 model.rho 计算
            "q_unit": "N/m",       # 显式 q 的单位，N/m 或 N/mm
            "g": 9.80665
        }
    """
    user_inputs = user_inputs or {}

    if section_type not in SECTION_NAMES:
        raise ValueError("section_type 输入错误，应为 1~5")
    if method_type not in METHOD_NAMES:
        raise ValueError("method_type 输入错误，应为 1~4")
    if member_type not in MEMBER_NAMES:
        raise ValueError("member_type 输入错误，应为 1（悬臂梁）或 2（简支梁）")

    _check_positive("SCB", scb)
    _check_positive("proto_E", proto.get("E"))
    _check_positive("proto_fy", proto.get("fy"))
    _check_positive("proto_bs", proto.get("bs"))
    _check_positive("proto_l", proto.get("l"))
    _check_positive("model_E", model.get("E"))
    _check_positive("model_fy", model.get("fy"))
    _check_positive("model_bs", model.get("bs"))
    _check_positive("model_l", model.get("l"))

    proto_section = proto.get("section", {})
    model_section = model.get("section", {})
    _validate_section(section_type, proto_section, "原型")
    _validate_section(section_type, model_section, "模型")

    section_name = SECTION_NAMES[section_type]
    member_name = MEMBER_NAMES[member_type]
    method_name = METHOD_NAMES[method_type]

    # ---------- 截面衍生量 ----------
    if section_type in (1, 2):
        yp = proto_section["D"] / 2.0
        ym = model_section["D"] / 2.0
    elif section_type in (3, 4):
        yp = proto_section["B"] / 2.0
        ym = model_section["B"] / 2.0
    else:
        yp = proto_section["h"] / 2.0
        ym = model_section["h"] / 2.0

    Ap = _calc_area(section_type, proto_section)
    Am = _calc_area(section_type, model_section)
    Ip = _calc_inertia(section_type, proto_section)
    Im = _calc_inertia(section_type, model_section)
    Wp = _calc_section_modulus(Ip, yp)
    Wm = _calc_section_modulus(Im, ym)

    # ---------- 基本与衍生相似比 ----------
    SH = model["l"] / proto["l"]
    SE = model["E"] / proto["E"]
    Ssigma_real = model["fy"] / proto["fy"]
    Sbs = model["bs"] / proto["bs"]
    Sy = ym / yp
    SA = Am / Ap
    SI = Im / Ip
    SW = Wm / Wp

    rho_p = proto.get("rho")
    rho_m = model.get("rho")
    Srho = None
    if rho_p is not None and rho_m is not None:
        _check_positive("proto_rho", rho_p)
        _check_positive("model_rho", rho_m)
        Srho = float(rho_m) / float(rho_p)

    # 用户覆盖基础/衍生相似比。
    if user_inputs.get("SH") is not None:
        SH = float(user_inputs["SH"])
    if user_inputs.get("SE") is not None:
        SE = float(user_inputs["SE"])
    if user_inputs.get("Ssigma") is not None:
        Ssigma_real = float(user_inputs["Ssigma"])
    if user_inputs.get("Sbs") is not None:
        Sbs = float(user_inputs["Sbs"])
    if user_inputs.get("Sy") is not None:
        Sy = float(user_inputs["Sy"])
    if user_inputs.get("SA") is not None:
        SA = float(user_inputs["SA"])
    if user_inputs.get("SI") is not None:
        SI = float(user_inputs["SI"])
    if user_inputs.get("SW") is not None:
        SW = float(user_inputs["SW"])
    else:
        # 若 Sy 或 SI 被用户覆盖，SW 应与覆盖后的量保持一致。
        SW = SI / Sy
    if user_inputs.get("Srho") is not None:
        Srho = float(user_inputs["Srho"])

    # 刚度相似比必须在 SH、SE、SI 覆盖后再计算，避免旧版的“覆盖后未联动”问题。
    SK = SE * SI / SH ** 3
    if user_inputs.get("SK") is not None:
        SK = float(user_inputs["SK"])

    # ---------- 不同构件形式的关键参数计算 ----------
    result_extra: Dict[str, Any] = {}

    if member_type == 1:
        # 悬臂梁：严格保留旧版 corecode.py 的计算关系。
        if method_type == 1:
            Sd = 1.0
            SF = 1.0
            Ssigma = 1.0
        elif method_type == 2:
            Sd = scb
            SF = scb ** 2
            Ssigma = 1.0
        elif method_type == 3:
            Sd = scb
            SF = SK * Sd
            Ssigma = 1.0
        else:
            Sd = SH ** 2 * Ssigma_real / SE / Sy
            SF = SI * Ssigma_real / Sy / SH
            Ssigma = Ssigma_real

    else:
        # 简支梁：跨中集中荷载 F + 全跨均布荷载 q。
        # DA 不依赖具体 F，但 CSL-U / CSL-US 是荷载组合相关的，必须给出 Fp。
        SF_DA = Ssigma_real * scb ** 2
        Sd_DA = scb
        Sq_DA_target = SF_DA / scb

        if method_type in (3, 4):
            loads = _resolve_simply_supported_loads(
                proto=proto,
                model=model,
                load_params=load_params,
                area_proto=Ap,
                area_model=Am,
            )
            Fp = loads["Fp"]
            qp = loads["qp"]
            qm = loads["qm"]

            # 用户允许直接覆盖均布荷载相似比。覆盖后以 qp 为基准更新 qm。
            Sq_actual = _ratio_or_nan(qm, qp)
            if user_inputs.get("Sq") is not None:
                Sq_actual = float(user_inputs["Sq"])
                qm = qp * Sq_actual

            Lp = float(proto["l"])
            Lm = float(model["l"])

            if method_type == 3:
                # CSL-U：指定位移相似比 Sd=Sd_DA，反算模型集中荷载。
                Sd = Sd_DA
                Fm = (
                    Sd * SE * SI / SH ** 3 * (Fp + 5.0 / 8.0 * qp * Lp)
                    - 5.0 / 8.0 * qm * Lm
                )
                SF = _ratio_or_nan(Fm, Fp)
                Ssigma = (
                    SH
                    / SW
                    * _safe_ratio(
                        Fm + 0.5 * qm * Lm,
                        Fp + 0.5 * qp * Lp,
                        "简支梁应力相似比",
                    )
                )

            else:
                # CSL-US：指定应力相似比，反算模型集中荷载，再计算位移相似比。
                Ssigma = Ssigma_real
                Fm = (
                    SW
                    / SH
                    * Ssigma
                    * (Fp + 0.5 * qp * Lp)
                    - 0.5 * qm * Lm
                )
                SF = _ratio_or_nan(Fm, Fp)
                Sd = (
                    SH ** 3
                    / (SE * SI)
                    * _safe_ratio(
                        Fm + 5.0 / 8.0 * qm * Lm,
                        Fp + 5.0 / 8.0 * qp * Lp,
                        "简支梁位移相似比",
                    )
                )

            result_extra.update(
                {
                    "Fp": Fp,
                    "Fm": Fm,
                    "qp_N_per_mm": qp,
                    "qm_N_per_mm": qm,
                    "qp_N_per_m": qp * 1000.0,
                    "qm_N_per_m": qm * 1000.0,
                    "Sq": Sq_actual,
                    "SF_DA": SF_DA,
                    "Sd_DA": Sd_DA,
                    "Sq_DA_target": Sq_DA_target,
                    "loadCombinationDependent": True,
                    "forceRatioDefined": not math.isnan(SF),
                    "uniformLoadRatioDefined": not math.isnan(Sq_actual),
                }
            )

        elif method_type == 2:
            # 与 MATLAB 验证程序一致：集中荷载按 DA 力相似比缩放，
            # 均布荷载采用原型与模型的实际取值。
            loads = _resolve_simply_supported_loads(
                proto=proto,
                model=model,
                load_params=load_params,
                area_proto=Ap,
                area_model=Am,
            )
            Fp = loads["Fp"]
            qp = loads["qp"]
            qm = loads["qm"]
            Sq_actual = _ratio_or_nan(qm, qp)
            if user_inputs.get("Sq") is not None:
                Sq_actual = float(user_inputs["Sq"])
                qm = qp * Sq_actual

            Sd = Sd_DA
            SF = SF_DA
            Ssigma = Ssigma_real
            Fm = Fp * SF
            result_extra.update(
                {
                    "Fp": Fp,
                    "Fm": Fm,
                    "qp_N_per_mm": qp,
                    "qm_N_per_mm": qm,
                    "qp_N_per_m": qp * 1000.0,
                    "qm_N_per_m": qm * 1000.0,
                    "Sq": Sq_actual,
                    "SF_DA": SF_DA,
                    "Sd_DA": Sd_DA,
                    "Sq_DA_target": Sq_DA_target,
                    "loadCombinationDependent": False,
                    "forceRatioDefined": not math.isnan(SF),
                    "uniformLoadRatioDefined": not math.isnan(Sq_actual),
                }
            )

        else:
            Sd = 1.0
            SF = 1.0
            Ssigma = 1.0
            result_extra.update(
                {
                    "SF_DA": SF_DA,
                    "Sd_DA": Sd_DA,
                    "Sq_DA_target": Sq_DA_target,
                    "loadCombinationDependent": False,
                }
            )

    # 保留旧版关键参数手动覆盖接口。
    if user_inputs.get("Sd") is not None:
        Sd = float(user_inputs["Sd"])
    if user_inputs.get("SF") is not None:
        SF = float(user_inputs["SF"])

    result: Dict[str, Any] = {
        "memberType": member_type,
        "memberName": member_name,
        "sectionType": section_type,
        "sectionName": section_name,
        "methodType": method_type,
        "methodName": method_name,
        "SCB": scb,
        "SH": SH,
        "SE": SE,
        "Ssigma": Ssigma,
        "SsigmaReal": Ssigma_real,
        "Sbs": Sbs,
        "Sy": Sy,
        "SA": SA,
        "SI": SI,
        "SW": SW,
        "SK": SK,
        "Srho": Srho,
        "Sd": Sd,
        "SF": SF,
        "Ap": Ap,
        "Am": Am,
        "Ip": Ip,
        "Im": Im,
        "yp": yp,
        "ym": ym,
        "Wp": Wp,
        "Wm": Wm,
        "showDetail": show_detail,
    }
    result.update(result_extra)
    return result


def calculate_similarity_series(
    force_series: Iterable[float],
    section_type: int,
    method_type: int,
    scb: float,
    proto: Dict[str, Any],
    model: Dict[str, Any],
    user_inputs: Optional[Dict[str, Optional[float]]] = None,
    show_detail: int = 1,
    member_type: int = 2,
    load_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    对简支梁集中荷载时程逐点计算 SF、Sd 等结果。

    该函数对应 MATLAB 程序中“每一步根据当前 Fp 反算 Fm”的流程。
    force_series 中每个值均为原型集中荷载，单位 N，可带正负号。
    """
    if member_type != 2:
        raise ValueError("calculate_similarity_series 当前仅用于简支梁 member_type=2")
    if method_type not in (3, 4):
        raise ValueError("时程逐点计算仅适用于 CSL-U 或 CSL-US")

    base_load_params = dict(load_params or {})
    output: List[Dict[str, Any]] = []

    for step, fp in enumerate(force_series, start=1):
        current_loads = dict(base_load_params)
        current_loads["Fp"] = float(fp)
        one = calculate_similarity(
            section_type=section_type,
            method_type=method_type,
            scb=scb,
            proto=proto,
            model=model,
            user_inputs=user_inputs,
            show_detail=show_detail,
            member_type=member_type,
            load_params=current_loads,
        )
        one["step"] = step
        output.append(one)

    return output


if __name__ == "__main__":
    # 1) 旧版悬臂梁调用方式仍然可用。
    demo_proto_cantilever = {
        "E": 2.1e5,
        "fy": 270,
        "bs": 0.02,
        "l": 3600,
        "section": {"B": 450, "tw": 14},
    }
    demo_model_cantilever = {
        "E": 210000,
        "fy": 270,
        "bs": 0.02,
        "l": 1800,
        "section": {"B": 220, "tw": 8},
    }
    print(
        "悬臂梁示例：",
        calculate_similarity(
            section_type=4,
            method_type=4,
            scb=0.5,
            proto=demo_proto_cantilever,
            model=demo_model_cantilever,
        ),
    )

    # 2) 简支梁 CSL-US 示例；q 由 rho*g*A 自动计算。
    demo_proto_simple = {
        "E": 2.1e5,
        "fy": 235,
        "bs": 0.02,
        "l": 3600,
        "rho": 7850,
        "section": {"B": 450, "tw": 14},
    }
    demo_model_simple = {
        "E": 2.1e5,
        "fy": 235,
        "bs": 0.02,
        "l": 1800,
        "rho": 7850,
        "section": {"B": 225, "tw": 7},
    }
    print(
        "简支梁示例：",
        calculate_similarity(
            section_type=4,
            method_type=4,
            scb=0.5,
            proto=demo_proto_simple,
            model=demo_model_simple,
            member_type=2,
            load_params={"Fp": 200000.0},
        ),
    )

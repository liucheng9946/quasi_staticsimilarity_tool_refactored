from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, Iterable, Mapping, Set


_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "pow": pow,
}

_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class SafeExpressionError(ValueError):
    pass


def normalize_expression(expression: str) -> str:
    return str(expression).strip().replace("^", "**")


def extract_names(expression: str) -> Set[str]:
    try:
        tree = ast.parse(normalize_expression(expression), mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError(f"表达式语法错误：{expression}") from exc
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names - set(_ALLOWED_FUNCTIONS) - set(_ALLOWED_CONSTANTS)


class _Evaluator(ast.NodeVisitor):
    def __init__(self, variables: Mapping[str, Any]):
        self.variables = dict(variables)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise SafeExpressionError("表达式只允许数值和布尔常量")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.variables:
            return self.variables[node.id]
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise SafeExpressionError(f"表达式引用了未定义变量：{node.id}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError(f"不允许的运算符：{type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError(f"不允许的一元运算符：{type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise SafeExpressionError("表达式包含非白名单函数")
        if node.keywords:
            raise SafeExpressionError("函数调用不支持关键字参数")
        return _ALLOWED_FUNCTIONS[node.func.id](*[self.visit(arg) for arg in node.args])

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        for operator_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(operator_node))
            if op is None:
                raise SafeExpressionError(f"不允许的比较运算符：{type(operator_node).__name__}")
            right = self.visit(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        values = [bool(self.visit(value)) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise SafeExpressionError("不允许的布尔运算符")

    def generic_visit(self, node: ast.AST) -> Any:
        raise SafeExpressionError(f"表达式包含不允许的语法：{type(node).__name__}")


def evaluate_expression(expression: str, variables: Mapping[str, Any]) -> Any:
    expression = normalize_expression(expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError(f"表达式语法错误：{expression}") from exc
    return _Evaluator(variables).visit(tree)


def validate_expression_names(expression: str, allowed_names: Iterable[str]) -> None:
    unknown = extract_names(expression) - set(allowed_names)
    if unknown:
        raise SafeExpressionError("表达式包含未声明变量：" + "、".join(sorted(unknown)))


def split_assignment(equation: str) -> tuple[str, str]:
    text = str(equation).strip()
    if text.count("=") != 1 or any(token in text for token in ("==", ">=", "<=", "!=")):
        raise SafeExpressionError(f"方法方程必须采用“变量 = 表达式”形式：{equation}")
    left, right = [part.strip() for part in text.split("=", 1)]
    if not left.isidentifier():
        raise SafeExpressionError(f"方程左侧必须是合法变量名：{left}")
    if not right:
        raise SafeExpressionError(f"方程右侧不能为空：{equation}")
    return left, right

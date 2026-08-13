"""Symbolic / numeric math tool backed by sympy.

``math_symbolic`` runs sympy **in-process** (inside DeepTutor's own Python
venv, where sympy is a declared dependency) rather than in the sandbox. This
keeps the math tool usable even when the sandbox's interpreter differs from
the venv, and it is safe because the host builds the request from a fixed
operation set: the model picks an operation and arguments, never code.

Expression parsing uses ``sympy.parse_expr`` with a limited local dict of
symbols, so it does not evaluate arbitrary Python.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import sympy as sp

from deeptutor.core.tool_protocol import BaseTool, ToolDefinition, ToolParameter, ToolResult

_OPERATIONS = ("simplify", "solve", "diff", "integrate", "limit", "verify", "numeric_search")

_DEFAULT_VARIABLES = "x"
_TIMEOUT_S = 20.0


def _symbols(variables: str) -> tuple[tuple[sp.Symbol, ...], dict[str, sp.Symbol]]:
    names = [s.strip() for s in (variables or _DEFAULT_VARIABLES).split(",") if s.strip()]
    if not names:
        names = ["x"]
    syms = sp.symbols(" ".join(names))
    if not isinstance(syms, tuple):
        syms = (syms,)
    local = {name: sym for name, sym in zip(names, syms)}
    return syms, local


def _parse(expr: str, local: dict[str, sp.Symbol]) -> sp.Expr:
    if not expr.strip():
        raise ValueError("expression is empty")
    return sp.parse_expr(expr.strip(), local_dict=local, transformations="all")


def _numeric_search(expr: sp.Expr, sym: sp.Symbol, arg: str) -> str:
    parts = [p.strip() for p in (arg or "0,10,1").split(",")]
    try:
        start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        start, end, step = 0.0, 10.0, 1.0
    if step == 0:
        raise ValueError("numeric_search step must be non-zero")

    f = sp.lambdify(sym, expr, "math")
    zeros: list[float] = []
    sign_changes: list[tuple[float, float]] = []
    lines: list[str] = []
    x = start
    prev_y: float | None = None
    prev_x: float | None = None
    iterations = 0
    while x <= end and iterations < 100000:
        try:
            y = float(f(x))
        except Exception:
            y = float("nan")
        lines.append(f"{x:.6g}: {y}")
        if not math.isnan(y) and abs(y) < 1e-12:
            zeros.append(x)
        if (
            prev_y is not None
            and prev_x is not None
            and not math.isnan(prev_y)
            and not math.isnan(y)
            and prev_y * y < 0
        ):
            sign_changes.append((prev_x, x))
        prev_y, prev_x = y, x
        x += step
        iterations += 1

    if zeros:
        lines.append("ZEROS at: " + str(zeros))
    if sign_changes:
        lines.append("SIGN_CHANGE between: " + str(sign_changes))
    return "\n".join(lines)


def _evaluate(
    operation: str,
    expr: str,
    *,
    rhs: str,
    variables: str,
    arg: str,
) -> str:
    syms, local = _symbols(variables)
    parsed = _parse(expr, local)

    if operation == "simplify":
        return str(sp.simplify(parsed))
    if operation == "solve":
        sols = sp.solve(parsed, syms)
        return sp.pretty(sols) if sols else "No solution found"
    if operation == "diff":
        order = int(arg) if str(arg).strip().isdigit() else 1
        return str(sp.diff(parsed, syms[0], order))
    if operation == "integrate":
        var_name = arg.strip() or next(iter(local))
        return str(sp.integrate(parsed, local[var_name]))
    if operation == "limit":
        point = (arg or "0").strip()
        point_val: Any = sp.oo if point in ("oo", "inf", "infty") else sp.sympify(point)
        return str(sp.limit(parsed, syms[0], point_val))
    if operation == "verify":
        if not rhs.strip():
            raise ValueError("verify requires the 'rhs' argument")
        rhs_parsed = _parse(rhs, local)
        diff = sp.simplify(parsed - rhs_parsed)
        verdict = "EQUAL" if diff == 0 else "NOT_EQUAL"
        return f"{verdict}\ndifference simplified to: {diff}"
    if operation == "numeric_search":
        return _numeric_search(parsed, syms[0], arg)
    raise ValueError(f"unsupported operation: {operation}")


class MathSymbolicTool(BaseTool):
    """Typed sympy front-end running in-process."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="math_symbolic",
            description=(
                "Run symbolic or numeric math through sympy. Pick one fixed "
                "operation (simplify, solve, diff, integrate, limit, verify, "
                "numeric_search) and pass a math expression. Use for verifying "
                "identities, solving equations, checking proof algebra, or "
                "numerically searching for counterexamples."
            ),
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Which sympy operation to run.",
                    enum=list(_OPERATIONS),
                ),
                ToolParameter(
                    name="expr",
                    type="string",
                    description="The math expression (sympy-parseable). For verify this is the left-hand side.",
                ),
                ToolParameter(
                    name="rhs",
                    type="string",
                    description="Right-hand side expression; only used by verify.",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="variables",
                    type="string",
                    description="Comma-separated variable symbols, e.g. 'x,y'. Default 'x'.",
                    required=False,
                    default=_DEFAULT_VARIABLES,
                ),
                ToolParameter(
                    name="arg",
                    type="string",
                    description=(
                        "Operation-specific extra argument: diff=order (default 1), "
                        "integrate=variable, limit=point (default 0; 'oo' for infinity), "
                        "numeric_search='start,end,step' (default '0,10,1')."
                    ),
                    required=False,
                    default="",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        operation = str(kwargs.get("operation") or "").strip()
        if operation not in _OPERATIONS:
            return ToolResult(
                content=f"Error: operation must be one of {', '.join(_OPERATIONS)}.",
                success=False,
            )
        expr = str(kwargs.get("expr") or "").strip()
        if not expr:
            return ToolResult(content="Error: expr is required.", success=False)

        try:
            result_text = await asyncio.wait_for(
                asyncio.to_thread(
                    _evaluate,
                    operation,
                    expr,
                    rhs=str(kwargs.get("rhs") or "").strip(),
                    variables=str(kwargs.get("variables") or _DEFAULT_VARIABLES).strip(),
                    arg=str(kwargs.get("arg") or "").strip(),
                ),
                timeout=_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"math_symbolic timed out after {_TIMEOUT_S:.0f}s.",
                success=False,
            )
        except Exception as exc:  # noqa: BLE001 — surface a readable math error
            return ToolResult(content=f"math_symbolic error: {exc}", success=False)

        return ToolResult(
            content=result_text,
            success=True,
            metadata={"operation": operation, "expr": expr},
        )


__all__ = ["MathSymbolicTool"]

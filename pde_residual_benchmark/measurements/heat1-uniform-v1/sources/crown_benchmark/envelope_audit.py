"""Analytic extremum audit for the three scalar tanh envelopes.

An affine residual has extrema only at endpoints and roots of g'(x)-m.
The monotonicity partitions of g' are known analytically, so bracketed roots
cover every branch, including wide and saturated intervals. This is a
floating-point validity check, NOT outward-rounded interval certification.
It never changes the line it checks.
"""
from dataclasses import asdict, dataclass
import math
from typing import Literal

from scipy.optimize import brentq

from .types import BoundFailure

KINDS = ("tanh", "tanh-prime", "tanh-double-prime")
AUDIT_ATOL = 1e-9
AUDIT_RTOL = 1e-12
ROOT_XTOL = 2e-14
ROOT_RTOL = 8.881784197001252e-16


def _validate(kind, lower, upper):
    if kind not in KINDS:
        raise ValueError(f"Unknown envelope: {kind!r}")
    if not all(math.isfinite(v) for v in (lower, upper)) or lower > upper:
        raise ValueError("Envelope interval must be finite and ordered")


def activation_value(kind: str, x: float) -> float:
    """Stable scalar reference; sech² avoids 1-tanh² saturation cancellation."""
    t = math.tanh(x)
    q = math.exp(-2 * abs(x))
    s = 4 * q / (1 + q) ** 2
    if kind == "tanh":
        return t
    if kind == "tanh-prime":
        return s
    if kind == "tanh-double-prime":
        return -2 * t * s
    raise ValueError(f"Unknown envelope: {kind!r}")


def _derivative(kind, x):
    t = math.tanh(x)
    s = activation_value("tanh-prime", x)
    if kind == "tanh":
        return s
    if kind == "tanh-prime":
        return -2 * t * s
    return (6 * t * t - 2) * s


def _turning_points(kind):
    # Zeros of g'': tanh'', tanh''', tanh'''' respectively.
    if kind == "tanh":
        return (0.0,)
    if kind == "tanh-prime":
        a = math.atanh(1 / math.sqrt(3))
        return (-a, a)
    a = math.atanh(math.sqrt(2 / 3))
    return (-a, 0.0, a)


def residual_stationary_points(kind: str, lower: float, upper: float,
                               slope: float) -> tuple[float, ...]:
    """All interior roots of g'(x)=slope, without a sampling grid.

    Splitting at analytic turning points also handles a repeated root where
    the signs do not change. Rounding-near roots at these points need not be
    classified as roots: audit_line additionally evaluates the turning points.
    """
    lower, upper, slope = float(lower), float(upper), float(slope)
    _validate(kind, lower, upper)
    if not math.isfinite(slope):
        raise ValueError("Nonfinite line slope")
    splits = [lower, *(p for p in _turning_points(kind) if lower < p < upper), upper]
    roots = set()
    residual = lambda x: _derivative(kind, x) - slope
    for a, b in zip(splits, splits[1:]):
        fa, fb = residual(a), residual(b)
        if fa == 0 and lower < a < upper:
            roots.add(a)
        if fb == 0 and lower < b < upper:
            roots.add(b)
        if fa < 0 < fb or fb < 0 < fa:
            roots.add(float(brentq(residual, a, b, xtol=ROOT_XTOL,
                                   rtol=ROOT_RTOL, maxiter=200)))
    return tuple(sorted(roots))


@dataclass(frozen=True)
class EnvelopeAudit:
    kind: str
    lower: float
    upper: float
    slope: float
    intercept: float
    side: str
    candidates: tuple[float, ...]
    signed_violations: tuple[float, ...]
    max_violation: float
    witness: float
    tolerance: float
    passed: bool

    def to_dict(self):
        return asdict(self)


class EnvelopeViolation(BoundFailure):
    """Original authors' envelope failed; report retains a minimal reproducer."""
    def __init__(self, report: EnvelopeAudit):
        self.report = report
        super().__init__(
            f"Invalid {report.kind} {report.side} envelope on "
            f"[{report.lower:.17g}, {report.upper:.17g}]: "
            f"violation={report.max_violation:.17g} at x={report.witness:.17g}, "
            f"tolerance={report.tolerance:.3g}; "
            f"line=({report.slope:.17g})*x+({report.intercept:.17g}). "
            "No repaired bound was substituted."
        )


def audit_line(kind: str, lower: float, upper: float, slope: float,
               intercept: float, side: Literal["lower", "upper"], *,
               atol: float = AUDIT_ATOL, rtol: float = AUDIT_RTOL) -> EnvelopeAudit:
    lower, upper, slope, intercept = map(float, (lower, upper, slope, intercept))
    _validate(kind, lower, upper)
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    if not all(math.isfinite(v) for v in (slope, intercept, atol, rtol)) or min(atol, rtol) < 0:
        raise BoundFailure("Nonfinite line or invalid audit tolerance")
    points = tuple(sorted({lower, upper,
        *residual_stationary_points(kind, lower, upper, slope),
        *(p for p in _turning_points(kind) if lower < p < upper)}))
    violations = tuple((slope * x + intercept - activation_value(kind, x))
                       * (1 if side == "lower" else -1) for x in points)
    if not all(math.isfinite(v) for v in violations):
        raise BoundFailure("Nonfinite line residual")
    i = max(range(len(points)), key=violations.__getitem__)
    scale = max(1., abs(intercept), abs(slope * lower), abs(slope * upper))
    tolerance = atol + rtol * scale
    return EnvelopeAudit(kind, lower, upper, slope, intercept, side, points,
                         violations, max(0., violations[i]), points[i], tolerance,
                         violations[i] <= tolerance)


def require_valid(report: EnvelopeAudit) -> EnvelopeAudit:
    if not report.passed:
        raise EnvelopeViolation(report)
    return report


def audit_envelope(kind, lower, upper, lower_line, upper_line):
    return tuple(audit_line(kind, lower, upper, *line, side)
                 for side, line in (("lower", lower_line), ("upper", upper_line)))

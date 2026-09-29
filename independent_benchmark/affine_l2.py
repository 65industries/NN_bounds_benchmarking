"""Exact L2 integration of a CROWN affine sandwich over a 2-D cell.

WHY THIS EXISTS
---------------
Released partial_crown does not return a scalar interval as its native object. Its
backward pass produces an AFFINE PAIR, sound at every point of the cell:

    g_L(x) = A_L . x + c_L  <=  phi(x)  <=  A_U . x + c_U = g_U(x)      for all x in C

(`crown.py`: `optimized_backward_crown` -> `CROWN_coefficients`; the derivative classes
carry the same pair through `u_dxixi_crown_coefficients_*`). The scalar interval [L,U]
used elsewhere in this benchmark is a CONCRETIZATION of that pair over the box, i.e. it
discards the linear term. Discarding it caps the L2 enclosure at first order, because a
piecewise-CONSTANT enclosure cannot beat the integrand's own O(h) oscillation on a cell.

Keeping the linear term gives second order at no extra CROWN cost. Crucially this needs
NO higher-derivative information: the pair is a pointwise sandwich, not a Taylor model,
so there is no Taylor remainder to bound. NetBounds' Q1 rule is a Taylor model and does
need sup|D^2 phi| (so sup|D^4 u| for a second-derivative target); CROWN's relaxation
error is already inside A_L, c_L, A_U, c_U. Only order-2 information is used here.

WHAT IS COMPUTED
----------------
Squaring the sandwich pointwise (the same reduction used for scalars, now applied to
functions) gives, for every x in C,

    phi(x)^2 <= max(g_L(x)^2, g_U(x)^2)
    phi(x)^2 >= 0 if g_L(x) <= 0 <= g_U(x) else min(g_L(x)^2, g_U(x)^2)

Both sides are PIECEWISE QUADRATIC, with pieces separated by the straight lines where
g_L, g_U or g_L+g_U vanishes. This module integrates them EXACTLY: it clips the cell
against those lines (Sutherland-Hodgman on a convex polygon stays convex), then
integrates a quadratic over each convex piece with the triangle edge-midpoint rule,
which is exact for total degree <= 2.

Ordinary float64 throughout: exact in the sense of "no quadrature error", not
outward-rounded interval arithmetic. That matches the rest of this benchmark.
"""
import math

REDUCTION = 'exact integral of max/min of the squared CROWN affine sandwich'

POLICY = dict(
    name='crown-affine-l2-v1',
    consumes='released partial_crown affine coefficients (A_L, c_L, A_U, c_U)',
    reduction=REDUCTION,
    reduction_owner='benchmark harness, not partial_crown',
    derivative_order_required=2,
    taylor_remainder_used=False,
    rationale=('the CROWN pair is a pointwise sandwich, not a Taylor model, so squaring '
               'and integrating it needs no higher-derivative envelope'),
    exactness='piecewise-quadratic integrand integrated in closed form; no quadrature error',
    upstream_modifications=[],
)

_EPS = 0.0


def polygon_area(poly):
    """Shoelace area of a simple polygon (non-negative for CCW input)."""
    if len(poly) < 3:
        return 0.
    total = 0.
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.


def clip_halfplane(poly, normal, offset):
    """Sub-polygon where normal . x + offset >= 0 (convex in, convex out)."""
    if not poly:
        return []
    a, b = normal
    out = []
    n = len(poly)
    values = [a * x + b * y + offset for x, y in poly]
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        v0, v1 = values[i], values[(i + 1) % n]
        if v0 >= -_EPS:
            out.append((x0, y0))
        if (v0 > 0. and v1 < 0.) or (v0 < 0. and v1 > 0.):
            t = v0 / (v0 - v1)
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def integrate_affine_square(poly, A, c):
    """int_poly (A . x + c)^2 dx, exact.

    The integrand has total degree 2, so the triangle edge-midpoint rule
    (area/3 * sum of the three edge midpoints) integrates it without error.
    """
    if len(poly) < 3:
        return 0.
    a0, a1 = A
    q = lambda p: (a0 * p[0] + a1 * p[1] + c) ** 2
    total = 0.
    p0 = poly[0]
    for i in range(1, len(poly) - 1):
        p1, p2 = poly[i], poly[i + 1]
        area = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2.
        if area == 0.:
            continue
        m01 = ((p0[0] + p1[0]) / 2., (p0[1] + p1[1]) / 2.)
        m12 = ((p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.)
        m20 = ((p2[0] + p0[0]) / 2., (p2[1] + p0[1]) / 2.)
        total += area * (q(m01) + q(m12) + q(m20)) / 3.
    return total


def _box(lower, upper):
    return [(lower[0], lower[1]), (upper[0], lower[1]),
            (upper[0], upper[1]), (lower[0], upper[1])]


def cell_square_integral(A_lower, c_lower, A_upper, c_upper, lower, upper):
    """Exact [lo, hi] enclosure of int_C phi^2 for any phi inside the sandwich.

    Returns a tuple (lo, hi) of plain floats. Raises if the sandwich is reversed
    anywhere on the cell, which would mean the enclosure is not valid.
    """
    if len(lower) != 2 or len(upper) != 2:
        raise ValueError('The affine L2 reduction is implemented for 2-D cells only')
    if len(A_lower) != 2 or len(A_upper) != 2:
        raise ValueError('Affine coefficient vectors must match the cell dimension')
    for a, b in zip(lower, upper):
        if not (b > a):
            raise ValueError('Cell must have positive extent on every axis')
    values = [(A_upper[0] - A_lower[0]) * x + (A_upper[1] - A_lower[1]) * y
              + (c_upper - c_lower) for x, y in _box(lower, upper)]
    if min(values) < -1e-12:
        raise ValueError('Reversed CROWN sandwich: g_L exceeds g_U inside the cell')

    box = _box(lower, upper)
    # Upper: max(gL^2, gU^2). gU^2 >= gL^2 exactly where (gL+gU)(gU-gL) >= 0, and
    # gU >= gL on the cell, so the split line is s = gL + gU.
    As = (A_lower[0] + A_upper[0], A_lower[1] + A_upper[1])
    cs = c_lower + c_upper
    hi = (integrate_affine_square(clip_halfplane(box, As, cs), A_upper, c_upper)
          + integrate_affine_square(clip_halfplane(box, (-As[0], -As[1]), -cs), A_lower, c_lower))
    # Lower: gL^2 where gL >= 0, gU^2 where gU <= 0, zero on the straddling strip.
    lo = (integrate_affine_square(clip_halfplane(box, A_lower, c_lower), A_lower, c_lower)
          + integrate_affine_square(clip_halfplane(box, (-A_upper[0], -A_upper[1]), -c_upper),
                                    A_upper, c_upper))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < -1e-12 or hi < lo - 1e-12:
        raise ValueError(f'Invalid affine square integral [{lo}, {hi}]')
    return max(0., lo), max(0., hi)


def straddles(A_lower, c_lower, A_upper, c_upper, lower, upper):
    """True when the sandwich contains zero somewhere in the cell."""
    box = _box(lower, upper)
    lo_vals = [A_lower[0] * x + A_lower[1] * y + c_lower for x, y in box]
    hi_vals = [A_upper[0] * x + A_upper[1] * y + c_upper for x, y in box]
    return min(lo_vals) <= 0. <= max(hi_vals)

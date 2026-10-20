"""Existing degeneracy-safe squared-sandwich integration only.

Harness-owned, not a partial_crown authors API. Extracted verbatim; no
unused auto_LiRPA/device adapter is distributed here.
"""
import math
import sys
CONCRETIZATION_ROUNDOFF_FACTOR = 64 * sys.float_info.epsilon

def _finite_scalar(x, name):
    try:
        value = float(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _pair(x, name):
    try:
        if len(x) != 2:
            raise ValueError(f'{name} must have length two')
        return tuple(_finite_scalar(v, name) for v in x)
    except TypeError as exc:
        raise ValueError(f'{name} must have length two') from exc


def _value(A, c, point):
    return math.fsum((c, A[0]*point[0], A[1]*point[1]))


def _clip(poly, A, c):
    """Convex polygon intersected with A.x+c>=0, with NO epsilon expansion."""
    if not poly:
        return []
    if A == (0., 0.):
        return poly if c >= 0 else []
    values = [_value(A, c, p) for p in poly]
    out = []
    for i, p in enumerate(poly):
        q, v, w = poly[(i+1) % len(poly)], values[i], values[(i+1) % len(poly)]
        if v >= 0:
            out.append(p)
        if (v > 0 > w) or (v < 0 < w):
            ratio = v/(v-w)
            out.append((p[0]+ratio*(q[0]-p[0]), p[1]+ratio*(q[1]-p[1])))
    return out


def _square(poly, A, c):
    """Closed triangle moments, no sampled/Gauss numerical reduction.

    Integral on a triangle is area/12*(sum(v_i**2)+(sum(v_i))**2),
    where v_i are its three affine vertex values. This is the barycentric
    degree-two moment identity, written as nonnegative summands.
    """
    if len(poly) < 3:
        return 0.
    p = poly[0]
    pieces = []
    for q, r in zip(poly[1:-1], poly[2:]):
        area = abs((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]))/2
        if area == 0:
            continue
        values = [_value(A, c, vertex) for vertex in (p, q, r)]
        pieces.append(area/12*math.fsum([*(v*v for v in values), math.fsum(values)**2]))
    return math.fsum(pieces)


def cell_integrals(payload, lower, upper):
    """Return eight scalar integral/evidence fields, without modifying payload.

    The six raw keys are required; extra archived-cell keys are ignored. The
    returned square quantities INCLUDE VOLUME. Lower = int(max(l,0)^2 +
    min(u,0)^2), upper = int(max(l^2,u^2)). For l<=u the upper split is l+u=0;
    an identically zero split integrates ONE square, not two copies of the box.
    Identical affine pairs use one exact box moment for all three square fields.

    Structural guards reject nonfinite data, nonpositive geometry, reversed
    native endpoints, negative affine corner gaps, failed re-concretization,
    and invalid integrals. There is no clipping/repair, sample gate, or native
    tolerance change. Re-concretization alone allows 64*eps times the explicit
    affine evaluation scale. A tiny negative computed gap is rejected as well.
    """
    try:
        return _cell_integrals(payload, lower, upper)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError('Invalid nonfinite or unrepresentable affine square integral') from exc


def _cell_integrals(payload, lower, upper):
    lo, hi = _pair(lower, 'lower'), _pair(upper, 'upper')
    width = tuple(b-a for a, b in zip(lo, hi))
    volume = width[0]*width[1]
    if any(not math.isfinite(w) or w <= 0 for w in width) or not math.isfinite(volume) or volume <= 0:
        raise ValueError('Cell must have positive finite extent and volume')
    try:
        AL, AU = _pair(payload['A_lower'], 'A_lower'), _pair(payload['A_upper'], 'A_upper')
        cL, cU, L, U = (_finite_scalar(payload[key], key)
                       for key in ('c_lower', 'c_upper', 'lower_bound', 'upper_bound'))
    except (KeyError, TypeError) as exc:
        raise ValueError('Expected the six native affine payload fields') from exc
    if L > U:
        raise ValueError('Reversed native endpoints')
    corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    lower_values = [_value(AL, cL, p) for p in corners]
    upper_values = [_value(AU, cU, p) for p in corners]
    delta, dc = tuple(b-a for a, b in zip(AL, AU)), cU-cL
    gaps = [_value(delta, dc, p) for p in corners]
    if not all(math.isfinite(v) for v in (*lower_values, *upper_values, *gaps)):
        raise ValueError('Nonfinite affine corner evaluation')
    gap_min, gap_max = min(gaps), max(gaps)
    if gap_min < 0:
        raise ValueError('Reversed affine sandwich on the box')
    reconstruction_error = max(abs(min(lower_values)-L), abs(max(upper_values)-U))
    scale = max(1., abs(L), abs(U),
                *(math.fsum((abs(c), abs(A[0]*p[0]), abs(A[1]*p[1])))
                  for A, c in ((AL, cL), (AU, cU)) for p in corners))
    if not math.isfinite(scale) or reconstruction_error > CONCRETIZATION_ROUNDOFF_FACTOR*scale:
        raise ValueError(f'Concretization mismatch: absolute error {reconstruction_error}, scale {scale}')

    # Unit-box geometry avoids area cancellation on tiny/anisotropic cells far
    # from the origin. Coefficients change coordinates, not their native values.
    aL, aU = tuple(a*w for a, w in zip(AL, width)), tuple(a*w for a, w in zip(AU, width))
    bL, bU = _value(AL, cL, lo), _value(AU, cU, lo)
    aM = tuple(a/2+b/2 for a, b in zip(aL, aU))
    bM = bL/2+bU/2
    affine_square = volume*math.fsum((_value(aM, bM, (.5, .5))**2,
                                     aM[0]**2/12, aM[1]**2/12))
    unit = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    if AL == AU and cL == cU:
        lower_square = upper_square = affine_square
    else:
        lower_square = volume*math.fsum((_square(_clip(unit, aL, bL), aL, bL),
            _square(_clip(unit, tuple(-a for a in aU), -bU), aU, bU)))
        s, c = tuple(a+b for a, b in zip(aL, aU)), bL+bU
        if s == (0., 0.) and c == 0:
            upper_square = volume*_square(unit, aU, bU)
        else:
            upper_square = volume*math.fsum((_square(_clip(unit, s, c), aU, bU),
                _square(_clip(unit, tuple(-a for a in s), -c), aL, bL)))
    result = dict(lower_square=lower_square, upper_square=upper_square,
                  priority=upper_square-lower_square, affine_square=affine_square,
                  rho=gap_max/2, concretization_error=reconstruction_error,
                  affine_gap_min=gap_min, affine_gap_max=gap_max)
    if (not all(math.isfinite(x) for x in result.values()) or lower_square < 0
            or upper_square < lower_square or affine_square < 0):
        raise ValueError(f'Invalid affine square integral [{lower_square}, {upper_square}]')
    return result

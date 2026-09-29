"""Independent rational reduction oracles and bounded fresh-process native probes.

Run this file as a script with --native-probe METHOD TARGET COORDINATE for a
JSON timing/diagnostic record. Only the contract's selected checkpoint is read.
"""
from copy import deepcopy
from fractions import Fraction as Q
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from independent_benchmark import minimal_affine_l2_native as N


RAW_KEYS = {'lower_bound', 'upper_bound', 'A_lower', 'c_lower', 'A_upper', 'c_upper'}
INTEGRAL_KEYS = {'lower_square', 'upper_square', 'priority', 'affine_square', 'rho',
                 'concretization_error', 'affine_gap_min', 'affine_gap_max'}
BOX = ([.25, .5], [.5, .875])


def payload(AL, cL, AU, cU, lo=(0., 0.), hi=(1., 1.)):
    corners = list(itertools.product(*zip(lo, hi)))
    return dict(A_lower=list(map(float, AL)), c_lower=float(cL),
                A_upper=list(map(float, AU)), c_upper=float(cU),
                lower_bound=min(math.fsum([float(cL), *(float(a)*x for a, x in zip(AL, p))]) for p in corners),
                upper_bound=max(math.fsum([float(cU), *(float(a)*x for a, x in zip(AU, p))]) for p in corners))


def rational_polygon(lo, hi, a, c):
    """Intersect by candidate enumeration and rational convex hull, NOT clipping."""
    box = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    value = lambda p: sum(x*y for x, y in zip(a, p)) + c
    points = {p for p in box if value(p) >= 0}
    for p, q in zip(box, box[1:] + box[:1]):
        v, w = value(p), value(q)
        if v*w < 0:
            points.add(tuple(x + v/(v-w)*(y-x) for x, y in zip(p, q)))
    points = sorted(points)
    cross = lambda p, q, r: (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    halves = []
    for sequence in (points, points[::-1]):
        half = []
        for point in sequence:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def rational_moment(poly, px, py):
    """Green's theorem: integral x^p y^q = contour x^(p+1)y^q dy/(p+1)."""
    total = Q(0)
    for (x, y), (u, v) in zip(poly, poly[1:] + poly[:1]):
        for i in range(px+2):
            for j in range(py+1):
                total += (Q(math.comb(px+1, i)*math.comb(py, j), (px+1)*(i+j+1))
                          * x**(px+1-i)*(u-x)**i*y**(py-j)*(v-y)**j*(v-y))
    return total


def rational_square(poly, a, c):
    terms = ((2, 0, a[0]**2), (0, 2, a[1]**2), (0, 0, c**2),
             (1, 1, 2*a[0]*a[1]), (1, 0, 2*a[0]*c), (0, 1, 2*a[1]*c))
    return sum(coefficient*rational_moment(poly, i, j) for i, j, coefficient in terms)


def rational_oracle(AL, cL, AU, cU, lo, hi):
    AL, AU, lo, hi = [tuple(map(Q, x)) for x in (AL, AU, lo, hi)]
    cL, cU = Q(cL), Q(cU)
    full = rational_polygon(lo, hi, (Q(0), Q(0)), Q(0))
    lower = (rational_square(rational_polygon(lo, hi, AL, cL), AL, cL)
             + rational_square(rational_polygon(lo, hi, tuple(-a for a in AU), -cU), AU, cU))
    split, offset = tuple(x+y for x, y in zip(AL, AU)), cL+cU
    if split == (0, 0) and offset == 0:
        upper = rational_square(full, AU, cU)
    else:
        upper = (rational_square(rational_polygon(lo, hi, split, offset), AU, cU)
                 + rational_square(rational_polygon(lo, hi, tuple(-a for a in split), -offset), AL, cL))
    midpoint = rational_square(full, tuple((x+y)/2 for x, y in zip(AL, AU)), (cL+cU)/2)
    return lower, upper, midpoint


@pytest.mark.parametrize('L,U', [(1., 3.), (-6., -2.), (-2., 6.), (0., 4.),
                                  (-4., 0.), (0., 0.), (-3., 3.), (2., 2.)])
def test_constants_include_volume_and_symmetric_zero_split_once(L, U):
    lo, hi = BOX
    raw = payload([0, 0], L, [0, 0], U, lo, hi)
    saved = deepcopy(raw)
    result = N.cell_integrals(raw, lo, hi)
    volume = math.prod(b-a for a, b in zip(lo, hi))
    assert set(result) == INTEGRAL_KEYS
    assert result['lower_square'] == pytest.approx(volume*(0 if L <= 0 <= U else min(L*L, U*U)))
    assert result['upper_square'] == pytest.approx(volume*max(L*L, U*U))
    assert result['affine_square'] == pytest.approx(volume*((L+U)/2)**2)
    assert result['priority'] == result['upper_square']-result['lower_square']
    assert result['rho'] == (U-L)/2
    assert result['affine_gap_min'] == result['affine_gap_max'] == U-L
    assert result['concretization_error'] == 0
    assert raw == saved


@pytest.mark.parametrize('AL,cL,AU,cU', [
    ([2, -3], Q(3, 4), [2, -3], Q(3, 4)),  # identical, crosses zero
    ([1, 0], -1, [-1, 0], 1),               # symmetric nonconstant, touches edge
    ([1, 1], 0, [1, 1], 1),                # touching vertex
    ([1, 0], 0, [1, 0], 0),                # identical, touches edge
    ([1, 0], Q(-1, 3), [1, 0], Q(1, 3)),   # parallel zero lines
    ([0, 1], -1, [0, 1], -1),              # identical negative
    ([1, 1], -2, [-1, -1], 2),             # symmetric, touches corner
])
def test_degenerate_and_parallel_cases_against_rational_oracle(AL, cL, AU, cU):
    lo, hi = (0, 0), (1, 1)
    result = N.cell_integrals(payload(AL, cL, AU, cU, lo, hi), lo, hi)
    expected = rational_oracle(AL, cL, AU, cU, lo, hi)
    for key, want in zip(('lower_square', 'upper_square', 'affine_square'), expected):
        assert result[key] == pytest.approx(float(want), rel=2e-14, abs=1e-15)
    if AL == AU and cL == cU:
        assert result['lower_square'] == result['upper_square'] == result['affine_square']
        assert result['priority'] == result['rho'] == 0


def test_anisotropic_rational_sandwiches_with_independent_boundary_moments():
    rng = random.Random(307)
    for _ in range(75):
        lo = [Q(rng.randrange(-6, 6), 8), Q(rng.randrange(-6, 6), 8)]
        hi = [lo[0]+Q(rng.randrange(1, 6), 32), lo[1]+Q(rng.randrange(1, 6), 4)]
        AL, AU = [[Q(rng.randrange(-24, 25), 8) for _ in range(2)] for _ in range(2)]
        cL = Q(rng.randrange(-8, 9), 8)
        cU = cL + max(sum((a-b)*x for a, b, x in zip(AL, AU, p))
                      for p in itertools.product(*zip(lo, hi))) + Q(rng.randrange(0, 9), 16)
        result = N.cell_integrals(payload(AL, cL, AU, cU, lo, hi), lo, hi)
        exact = rational_oracle(AL, cL, AU, cU, lo, hi)
        for key, want in zip(('lower_square', 'upper_square', 'affine_square'), exact):
            assert result[key] == pytest.approx(float(want), rel=3e-13, abs=1e-16)
        gaps = [sum((b-a)*x for a, b, x in zip(AL, AU, p))+cU-cL
                for p in itertools.product(*zip(lo, hi))]
        assert result['affine_gap_min'] == pytest.approx(float(min(gaps)), abs=1e-15)
        assert result['affine_gap_max'] == pytest.approx(float(max(gaps)), abs=1e-15)
        assert result['rho'] == result['affine_gap_max']/2


@pytest.mark.parametrize('field', sorted(RAW_KEYS))
@pytest.mark.parametrize('invalid', [math.nan, math.inf, -math.inf])
def test_nonfinite_payloads_rejected_without_mutation(field, invalid):
    raw = payload([0, 0], -1, [0, 0], 1)
    raw[field] = [invalid, 0.] if field.startswith('A_') else invalid
    before = repr(raw)
    with pytest.raises(ValueError, match='finite'):
        N.cell_integrals(raw, [0, 0], [1, 1])
    assert repr(raw) == before


@pytest.mark.parametrize('raw', [payload([0, 0], 2, [0, 0], 1),
                                payload([2, 0], 0, [0, 0], 1),
                                payload([0, 0], 1e-16, [0, 0], 0)])
def test_reversed_affine_pairs_are_not_repaired(raw):
    with pytest.raises(ValueError, match='[Rr]eversed'):
        N.cell_integrals(raw, [0, 0], [1, 1])


@pytest.mark.parametrize('lo,hi', [([0], [1]), ([0, 0], [1, 0]), ([1, 0], [0, 1]),
                                   ([math.nan, 0], [1, 1]), ([0, 0], [1, math.inf])])
def test_invalid_geometry(lo, hi):
    with pytest.raises(ValueError):
        N.cell_integrals(payload([0, 0], -1, [0, 0], 1), lo, hi)


def test_overflow_is_invalid_integral_not_a_repair():
    with pytest.raises(ValueError):
        N.cell_integrals(payload([0, 0], 1e200, [0, 0], 2e200), [0, 0], [1, 1])


def test_concretization_guard_has_fixed_roundoff_scale_not_sample_gate():
    raw = payload([2, -3], .75, [2, -3], 1.75)
    raw['lower_bound'] = math.nextafter(raw['lower_bound'], -math.inf)
    result = N.cell_integrals(raw, [0, 0], [1, 1])
    assert result['concretization_error'] > 0
    for scale in (1., 1e8):
        scaled = {k: [v*scale for v in x] if isinstance(x, list) else x*scale for k, x in raw.items()}
        scaled['lower_bound'] -= 1e-6*scale
        with pytest.raises(ValueError, match='[Cc]oncretization'):
            N.cell_integrals(scaled, [0, 0], [1, 1])


def test_large_offset_small_anisotropic_box_uses_local_geometry():
    lo, hi = [1e8, -2e8], [1e8+2**-15, -2e8+2**-14]
    AL, cL, AU, cU = [2, 1], -.25, [2, 1], .5
    result = N.cell_integrals(payload(AL, cL, AU, cU, lo, hi), lo, hi)
    expected = rational_oracle(AL, cL, AU, cU, lo, hi)
    for key, want in zip(('lower_square', 'upper_square', 'affine_square'), expected):
        assert result[key] == pytest.approx(float(want), rel=2e-14, abs=1e-30)


def test_integrals_are_additive_including_volume_on_unequal_subcells():
    lo, hi = [-.25, -.5], [.75, 1.5]
    AL, AU, cL, cU = [2., -1.], [2.5, -.75], -.125, 1.125
    whole = N.cell_integrals(payload(AL, cL, AU, cU, lo, hi), lo, hi)
    subcells = [(lo, [.125, hi[1]]), ([.125, lo[1]], hi)]
    pieces = [N.cell_integrals(payload(AL, cL, AU, cU, a, b), a, b) for a, b in subcells]
    for key in ('lower_square', 'upper_square', 'priority', 'affine_square'):
        assert math.fsum(p[key] for p in pieces) == pytest.approx(whole[key], rel=2e-14)


def test_invalid_computed_integrals_raise_without_clamping(monkeypatch):
    monkeypatch.setattr(N, '_square', lambda *a: -1.)
    raw = payload([0, 0], 1, [0, 0], 2)
    with pytest.raises(ValueError, match='Invalid affine square integral'):
        N.cell_integrals(raw, [0, 0], [1, 1])

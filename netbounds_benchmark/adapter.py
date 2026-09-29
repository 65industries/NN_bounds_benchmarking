"""NetBounds affine-Q1 cell-model adapter for signed supremum enclosures.

The upstream paper defines Q1 as an L2 quadrature. This adapter uses its exact
pointwise affine model and Hessian remainder (section2.tex, equations
cell-polynomial-models/cell-remainder-bounds/cell-model-errors) and takes cell
ranges, NOT squared-integral quadrature sums. The derivative-bound algorithms
are imported from the pinned, unmodified NetBounds-dev checkout.
"""
from copy import deepcopy
from dataclasses import fields
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from crown_benchmark.models import ExactSecondDerivative, validate_scalar_model
from crown_benchmark.types import BoundBatch, BoundFailure, validate_boxes

NETBOUNDS_COMMIT = '7835f105a6731bb6c133fd3d8ddcc243c1e9326b'
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'vendor/netbounds'


def load_backend(root=None):
    from reproducers.source_integrity import verify_vendor
    root, _ = verify_vendor('netbounds', root, NETBOUNDS_COMMIT)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = {name:importlib.import_module(name) for name in (
        'derivative_bounds.activation_bounds', 'derivative_bounds.second_order',
        'derivative_bounds.fourth_order', 'derivative_centers.common')}
    for name, module in modules.items():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise BoundFailure(f'Unpinned module imported: {name}: {module.__file__}')
    return root, modules


class RawMLPView(torch.nn.Module):
    """Only the attribute ABI upstream needs; no spatial/PDE interpretation."""
    def __init__(self, model, activation_bounds):
        super().__init__()
        layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
        self.hidden_layers = torch.nn.ModuleList(layers[:-1])
        self.output_layer = layers[-1]
        self.input_dim = layers[0].in_features
        self.activation = torch.tanh
        self.AB = activation_bounds

    @property
    def input_layer(self):
        return self.hidden_layers[0] if self.hidden_layers else self.output_layer

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


def q1_radius(entries, pairs, radii):
    """1/2 sum_{q,r} M_qr h_q h_r, with symmetric pairs stored once."""
    coefficients = torch.stack([
        radii[q] * radii[r] * (0.5 if q == r else 1.0) for q, r in pairs])
    return (entries * coefficients).sum(-1)


class NetBoundsQ1Adapter:
    def __init__(self, model, target='value', coordinate=None, *, dtype=torch.float64,
                 device='cpu', n_taylor=None, backend_root=None):
        start = time.perf_counter()
        architecture = validate_scalar_model(model)
        device = torch.device(device)
        if dtype != torch.float64 or device.type not in ('cpu', 'cuda'):
            raise ValueError('This controlled adapter supports CPU or CUDA float64 only')
        if target not in ('value', 'second-diagonal'):
            raise ValueError('Unsupported scalar target')
        if target == 'value' and coordinate is not None:
            raise ValueError('Value target has no coordinate')
        if target == 'second-diagonal' and (isinstance(coordinate, bool) or
                not isinstance(coordinate, int) or not 0 <= coordinate < architecture[0]):
            raise ValueError('Invalid zero-based derivative coordinate')
        if n_taylor is not None and (isinstance(n_taylor, bool) or
                not isinstance(n_taylor, int) or not 1 <= n_taylor <= 5):
            raise ValueError('Activation Taylor depth must be None or an integer in [1,5]')
        self.root, self.modules = load_backend(backend_root)
        self.model = deepcopy(model).to(dtype=dtype, device=device).eval().requires_grad_(False)
        self.dtype, self.device = dtype, next(self.model.parameters()).device
        self.target, self.coordinate = target, coordinate
        self.input_dim, self.n_taylor = architecture[0], n_taylor
        self.net = RawMLPView(self.model, self.modules['derivative_bounds.activation_bounds'].TanhBounds(dtype=dtype))
        self.pairs = tuple((q, r) for q in range(self.input_dim) for r in range(q, self.input_dim))
        self.quartets = (tuple(sorted({tuple(sorted((coordinate, coordinate, q, r)))
                                     for q, r in self.pairs})) if target == 'second-diagonal' else ())
        self.evaluator = self.model if target == 'value' else ExactSecondDerivative(self.model, coordinate)
        self.metadata = {
            'method': 'netbounds-q1', 'backend': 'NetBounds-dev', 'backend_commit': NETBOUNDS_COMMIT,
            'backend_root': str(self.root), 'patches': [],
            'adapter': 'raw-MLP attribute view plus pointwise Q1 supremum aggregation; no quadrature integral',
            'cell_model': 'g(c)+grad(g)(c).z',
            'remainder': '0.5 sum_qr h_q h_r (abs(D_qr g(c))+E_qr g(C))',
            'signed_range': 'g(c) +/- (sum_q abs(D_q g(c))*h_q + remainder)',
            'highest_network_derivative_order': 2 if target == 'value' else 4,
            'target_hessian_pairs': self.pairs, 'network_quartets': self.quartets,
            'activation_taylor_depth_requested': n_taylor,
            'activation_taylor_depth_effective': {str(m):6-m if n_taylor is None else min(n_taylor,6-m)
                                                 for m in range(1, (2 if target=='value' else 4)+1)},
            'activation_remainder': 'upstream exponential tanh-derivative envelope; unchanged',
            'derivative_bound_meaning': 'variation from center, not an absolute derivative supremum',
            'precision': {'parameters': str(dtype), 'activation_bounds':str(self.net.AB.dtype),
                          'inputs_and_accumulation':str(dtype), 'outward_rounded':False},
            'domain_coordinates': 'full input dimension; no spatial/time convention',
            'q1_source': 'tex/IMA/section2.tex:166-215; derivative_bounds/second_order.py; derivative_bounds/fourth_order.py',
        }
        self.setup_seconds = time.perf_counter() - start

    def evaluate(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            x = x[None]
        with torch.no_grad():
            return self.evaluator(x)

    def derivative_bounds(self, centers, radii):
        if not self.net.hidden_layers:
            raise ValueError('Affine networks are handled exactly without a derivative recurrence')
        if self.target == 'value':
            result = self.modules['derivative_bounds.second_order'].compute_second_order_bounds(
                self.net, centers, radii, self.n_taylor, multi_indices=self.pairs)
        else:
            result = self.modules['derivative_bounds.fourth_order'].compute_up_to_fourth_order_bounds(
                self.net, centers, radii, self.n_taylor, multi_indices=self.quartets)
        for field in fields(result):
            value = getattr(result, field.name)
            if isinstance(value, torch.Tensor):
                if value.dtype != self.dtype or value.device != self.device or not bool(torch.isfinite(value).all()):
                    raise BoundFailure(f'NetBounds precision/finiteness failure: {field.name}')
                if field.name.endswith('bounds') and bool((value < 0).any()):
                    raise BoundFailure(f'Negative derivative-variation bound: {field.name}')
        return result

    def _one(self, lower, upper):
        center = lower * 0.5 + upper * 0.5
        radii = (upper - lower) * 0.5
        c = center[None]
        with torch.no_grad():
            if not self.net.hidden_layers:
                base = self.model(c).reshape(-1) if self.target == 'value' else torch.zeros(1, dtype=self.dtype, device=self.device)
                gradient = (self.net.output_layer.weight if self.target == 'value'
                            else torch.zeros((1, self.input_dim), dtype=self.dtype, device=self.device))
                hessian_center = torch.zeros((1, len(self.pairs)), dtype=self.dtype, device=self.device)
                hessian_variation = torch.zeros_like(hessian_center)
                derivative_indices = self.pairs if self.target == 'value' else self.quartets
            else:
                jet = self.derivative_bounds(c, radii)
                if self.target == 'value':
                    centers = self.modules['derivative_centers.common'].compute_raw_centers(self.net, c, order=1)
                    base = centers.value.reshape(-1)
                    gradient = centers.gradient
                    hessian_center, hessian_variation = jet.base_hessian, jet.second_derivative_bounds
                    derivative_indices = jet.pairs
                else:
                    pair = (self.coordinate, self.coordinate)
                    base = jet.base_hessian[:, jet.pairs.index(pair)]
                    gradient = torch.stack([jet.base_third_derivatives[:, jet.triples.index(tuple(sorted((*pair, q))))]
                                            for q in range(self.input_dim)], dim=-1)
                    indices = [jet.quartets.index(tuple(sorted((*pair, q, r)))) for q, r in self.pairs]
                    hessian_center = jet.base_fourth_derivatives[:, indices]
                    hessian_variation = jet.fourth_derivative_bounds[:, indices]
                    derivative_indices = [jet.quartets[i] for i in indices]
            hessian_sup = hessian_center.abs() + hessian_variation
            remainder = q1_radius(hessian_sup, self.pairs, radii)
            affine_radius = (gradient.abs() * radii).sum(-1)
            radius = affine_radius + remainder
            independent = self.evaluate(c).reshape(-1)
            center_discrepancy = (base - independent).abs().max().item()
            if not torch.allclose(base, independent, atol=1e-12, rtol=1e-10):
                raise BoundFailure(f'NetBounds center disagrees with independent evaluator: {center_discrepancy}')
            for tensor in (base, gradient, hessian_center, hessian_variation, hessian_sup, radius):
                if tensor.dtype != self.dtype or not bool(torch.isfinite(tensor).all()):
                    raise BoundFailure('Nonfinite or wrong-precision Q1 calculation')
            details = {'center':center.tolist(), 'radii':radii.tolist(), 'center_value':base.item(),
                'center_gradient':gradient[0].tolist(), 'hessian_pairs':self.pairs,
                'network_hessian_indices':derivative_indices,
                'hessian_center':hessian_center[0].tolist(), 'hessian_variation':hessian_variation[0].tolist(),
                'hessian_absolute_envelope':hessian_sup[0].tolist(),
                'affine_radius':affine_radius.item(), 'remainder_radius':remainder.item(),
                'center_check_abs_error':center_discrepancy, 'execution_dtype':str(self.dtype)}
            return (base-radius).item(), (base+radius).item(), details

    def bound(self, lower, upper):
        lower, upper = validate_boxes(lower, upper, self.input_dim, self.dtype, self.device)
        rows = [self._one(lo, hi) for lo, hi in zip(lower, upper)]
        result = BoundBatch(torch.tensor([r[0] for r in rows], dtype=self.dtype, device=self.device),
                            torch.tensor([r[1] for r in rows], dtype=self.dtype, device=self.device),
                            {'q1_cells':[r[2] for r in rows], 'upstream_unmodified':True})
        return result.validate(len(lower))

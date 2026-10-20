"""Faithful, fail-closed adapter of fgirbal/partial_crown at a pinned commit.

No bound-propagation kernel is copied or rewritten here. Three external source
files are loaded in private module namespaces with explicit import, dimension,
factory-dtype and noninteractive-error adaptations. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import time
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from independent_benchmark.cpu_clock import excluded

from .envelope_audit import (
    AUDIT_ATOL, AUDIT_RTOL, ROOT_RTOL, ROOT_XTOL, EnvelopeViolation,
    audit_envelope, audit_line, require_valid,
)
from .types import BoundBatch, BoundFailure, validate_boxes

UPSTREAM_COMMIT = "161377048ee92b4c09faf5ce41628168c7e01556"
UPSTREAM_URL = "https://github.com/fgirbal/partial_crown"
SOURCE_FILES = (
    "pinn_verifier/activations/activation_relaxations.py",
    "pinn_verifier/activations/tanh.py",
    "pinn_verifier/crown.py",
)


def _upstream_path(path=None):
    root = Path(__file__).resolve().parents[1] / "vendor/partial_crown"
    if path is not None and Path(path).resolve() != root.resolve():
        raise BoundFailure("Only the bundled partial_crown source pin is supported")
    return root


def _unavailable(*args, **kwargs):
    raise BoundFailure("Gurobi/ONNX normalization APIs are not loaded by the pure tanh-MLP adapter")


class _UnavailableGurobi:
    def __getattr__(self, name):
        return _unavailable()


class _AddUnavailable(nn.Module):
    __init__ = _unavailable


class _MulUnavailable(nn.Module):
    __init__ = _unavailable


def _debug_trap():
    raise BoundFailure("Upstream partial_crown reached pdb.set_trace(): invalid bound or unsupported path")


def load_authors(upstream_path=None, dtype=torch.float64):
    """Load original sources; returned ``metadata['patches']`` is an exact log.

    Public for the defect reproducer, which audits raw authors' envelopes.
    This never creates a fake solver, imports ONNX, changes sys.path, changes
    torch's default dtype, or writes into the external checkout.
    """
    if dtype is not torch.float64:
        raise ValueError("partial-crown validation adapter currently supports float64 only")
    path = _upstream_path(upstream_path)
    from reproducers.source_integrity import verify_vendor
    path, source_pin = verify_vendor("partial_crown", path, UPSTREAM_COMMIT)
    commit = source_pin["commit"]
    patches, source_hashes, runtime_hashes = [], {}, {}
    observed_factories = set()

    def factory(name):
        def make(*args, **kwargs):
            # All reached constructors create numerical coefficients, not indices.
            kwargs["dtype"] = dtype
            result = getattr(torch, name)(*args, **kwargs)
            observed_factories.add(str(result.dtype))
            return result
        return make

    common = {f"_adapter_{name}": factory(name) for name in ("tensor", "zeros", "ones", "eye")}
    common.update(grb=_UnavailableGurobi(), Add=_AddUnavailable, Mul=_MulUnavailable,
                  _debug_trap=_debug_trap)

    def load(relative, short, injections):
        original = (path / relative).read_bytes()
        if hashlib.sha256(original).hexdigest() != source_pin["files"][relative]:
            raise BoundFailure(f"Upstream bytes differ from the pinned object: {relative}")
        source_hashes[relative] = hashlib.sha256(original).hexdigest()
        source = original.decode()

        def replace(old, new, reason, expected=1):
            nonlocal source
            count = source.count(old)
            if count != expected:
                raise BoundFailure(f"Adapter patch mismatch for {relative}: {old!r} ({count} != {expected})")
            patches.append(dict(file=relative, old=old, new=new, count=count, reason=reason))
            source = source.replace(old, new)

        if short in ("activations", "tanh"):
            replace("import gurobipy as grb", "# Optional Gurobi unavailable; annotations postponed.",
                    "optional import; LP entry points fail explicitly")
        if short == "tanh":
            replace("from .activation_relaxations import ActivationRelaxation, ActivationRelaxationType, line_type",
                    "# Activation classes injected from the private authors module.", "private import namespace")
            replace("np.Inf", "np.inf", "NumPy 2 spelling compatibility", expected=4)
        if short == "crown":
            replace("from tools.custom_torch_modules import Add, Mul",
                    "# Unsupported normalization sentinels injected; raw MLPs only.",
                    "optional ONNX/custom-module import; no normalization is permitted")
            replace("from pinn_verifier.activations.activation_relaxations import ActivationRelaxationType, ActivationRelaxation",
                    "# Activation classes injected from the private authors module.", "private import namespace")
            replace("self.input_dimension = 2",
                    "self.input_dimension = model[0].in_features", "input dimension from the actual first Linear")
            replace("assert domain_bounds.shape[1] == self.input_dimension",
                    "assert domain_bounds.shape[1] == 2", "endpoint axis is always 2, distinct from input dimension")

        # Token-level callable replacements only (not annotations or comments).
        tree = ast.parse(source)
        replacements = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            f = node.func
            if isinstance(f.value, ast.Name) and f.value.id == "torch" and f.attr in ("Tensor", "tensor", "zeros", "ones", "eye"):
                new = "_adapter_" + ("tensor" if f.attr == "Tensor" else f.attr)
                replacements.append((f.lineno, f.col_offset, f.end_col_offset, new, "explicit float64 coefficient factory"))
            if isinstance(f.value, ast.Name) and f.value.id == "pdb" and f.attr == "set_trace":
                replacements.append((f.lineno, f.col_offset, f.end_col_offset, "_debug_trap", "fail closed rather than interactive debugger"))
        lines = source.splitlines(keepends=True)
        for line, start, end, new, reason in sorted(replacements, reverse=True):
            old = lines[line - 1][start:end]
            patches.append(dict(file=relative, line=line, old=old, new=new, count=1, reason=reason))
            lines[line - 1] = lines[line - 1][:start] + new + lines[line - 1][end:]
        source = "from __future__ import annotations\n" + "".join(lines)
        patches.append(dict(file=relative, old="", new="from __future__ import annotations\n", count=1,
                            reason="defer optional Gurobi type annotations"))
        runtime_hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
        module = ModuleType(f"crown_benchmark._authors_{short}")
        module.__file__ = str(path / relative)
        module.__dict__.update(common)
        module.__dict__.update(injections)
        exec(compile(source, str(path / relative), "exec"), module.__dict__)
        return module

    activations = load(SOURCE_FILES[0], "activations", {})
    injections = {name: getattr(activations, name) for name in
                  ("ActivationRelaxation", "ActivationRelaxationType", "line_type")}
    tanh = load(SOURCE_FILES[1], "tanh", injections)
    crown = load(SOURCE_FILES[2], "crown", injections)
    metadata = dict(upstream_url=UPSTREAM_URL, upstream_commit=commit, upstream_path=str(path),
                    upstream_tracked_clean=None, source_verification="vendored SHA-256 allowlist",
                    source_sha256=source_hashes,
                    runtime_source_sha256=runtime_hashes, patches=patches,
                    license_note="upstream setup.py declares MIT; no root LICENSE file at pinned revision")
    return SimpleNamespace(crown=crown, tanh=tanh, activations=activations, metadata=metadata,
                           observed_factory_dtypes=observed_factories)


class _AuditedRelaxation:
    """Inspect but never modify the exact coefficients returned upstream.

    The audit of each returned line (curve containment of the line and of its
    upstream-normalised representation) is *deferred*: ``get_bounds`` only records
    what upstream returned, and ``flush_audit`` verifies everything afterwards.
    This keeps our verification cost out of the timed bound call while remaining
    fail-closed: a bound is never accepted before its lines have been audited.
    """
    def __init__(self, raw, kind, statistics):
        self.raw, self.kind, self.statistics = raw, kind, statistics
        self.pending = []

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _check(self, report):
        stats = self.statistics
        stats["lines_checked"] += 1
        stats["max_observed_violation"] = max(stats["max_observed_violation"], report.max_violation)
        require_valid(report)

    def get_bounds(self, lower, upper):
        # Upstream per-neuron root finding stays charged. Recording the returned
        # lines is small residual instrumentation overhead, also charged.
        lines = self.raw.get_bounds(lower, upper)
        self.pending.append(("lines", float(lower), float(upper),
                             tuple((float(m), float(b)) for m, b in lines)))
        return lines

    def get_lb_ub_in_interval(self, lower, upper):
        bounds = self.raw.get_lb_ub_in_interval(lower, upper)
        if float(bounds[0]) > float(bounds[1]):
            raise BoundFailure("Upstream activation range reversed")
        self.pending.append(("range", float(lower), float(upper), (float(bounds[0]), float(bounds[1]))))
        return bounds

    def flush_audit(self):
        """Audit every recorded line; raises BoundFailure on the first invalid one."""
        pending, self.pending = self.pending, []
        for what, lower, upper, payload in pending:
            if what == "lines":
                for report in audit_envelope(self.kind, lower, upper, *payload):
                    self._check(report)
                # Upstream changes m*x+b to m*(x+b/m). Also audit the rounded intercept
                # actually represented by that normalization, without changing it.
                for side, (m, b) in zip(("lower", "upper"), payload):
                    if m == 0:
                        raise BoundFailure("Upstream zero-slope line is incompatible with its alpha*(x+beta) representation")
                    self._check(audit_line(self.kind, lower, upper, m, m * (b / m), side))
            else:
                for side, value in zip(("lower", "upper"), payload):
                    self._check(audit_line(self.kind, lower, upper, 0., value, side))
        return len(pending)


def _floating_tensors(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _floating_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floating_tensors(item)


class PartialCrownAdapter:
    """Authors' ordinary value CROWN and specialized hybrid diagonal d² CROWN.

    ``upstream_path`` is the only optional configuration; unsafe audit bypasses
    or unlabelled relaxation changes are deliberately not supported.
    """
    def __init__(self, model: nn.Sequential, target="value", coordinate=None,
                 dtype=torch.float64, device="cpu", **opts):
        started = time.perf_counter()
        upstream_path = opts.pop("upstream_path", None)
        if opts:
            raise ValueError(f"Unknown partial-CROWN options: {sorted(opts)}")
        if target not in ("value", "second-diagonal"):
            raise ValueError("target must be 'value' or 'second-diagonal'")
        if dtype is not torch.float64 or torch.device(device).type != "cpu":
            raise ValueError("Pinned partial-CROWN adapter is validated for CPU float64 only")
        if type(model) is not nn.Sequential or not len(model):
            raise ValueError("Expected nonempty nn.Sequential")
        if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
            raise ValueError("Forward hooks can change the model; hooked models are unsupported")
        layers = list(model)
        affine = all(type(layer) is nn.Linear for layer in layers)
        if not affine and (len(layers) % 2 != 1 or any(type(layer) is not (nn.Linear if i % 2 == 0 else nn.Tanh)
                                                     for i, layer in enumerate(layers))):
            raise ValueError("Only affine networks or alternating Linear/Tanh ending in scalar Linear are supported")
        linear = [layer for layer in layers if type(layer) is nn.Linear]
        if not linear or linear[-1].out_features != 1:
            raise ValueError("Expected scalar-output Linear layer")
        if any(a.out_features != b.in_features for a, b in zip(linear, linear[1:])):
            raise ValueError("Incompatible Linear layer dimensions")
        for layer in linear:
            if (type(layer.in_features) is not int or type(layer.out_features) is not int or
                    min(layer.in_features, layer.out_features) <= 0 or
                    layer.weight.layout != torch.strided or
                    tuple(layer.weight.shape) != (layer.out_features, layer.in_features) or
                    (layer.bias is not None and tuple(layer.bias.shape) != (layer.out_features,))):
                raise ValueError("Invalid Linear dimensions or tensor layout")
            if layer.weight.device.type == "meta":
                raise ValueError("Model parameters must have real storage")
        self.input_dim = linear[0].in_features
        if target == "value" and coordinate is not None:
            raise ValueError("Value target must not specify a coordinate")
        if target == "second-diagonal" and (type(coordinate) is not int or not 0 <= coordinate < self.input_dim):
            raise ValueError("Second-diagonal target requires an in-range zero-based coordinate")
        self.dtype, self.device = dtype, torch.device(device)
        self.target, self.coordinate, self._affine = target, coordinate, affine
        self.model = copy.deepcopy(model).to(device=self.device, dtype=dtype).eval()
        self.model.requires_grad_(False)
        for layer in self.model:
            if type(layer) is nn.Linear and layer.bias is None:
                # An absent bias is exactly zero, including in the derivative.
                layer.bias = nn.Parameter(torch.zeros(layer.out_features, dtype=dtype), requires_grad=False)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Nonfinite model parameter")
        self._authors = load_authors(upstream_path, dtype)
        self._audit_statistics = {}
        # By default every bound() call audits its own relaxation lines before returning
        # (fail-closed, standalone-safe). A driver that calls audit() itself right after
        # the timed call sets deferred_audit=True so the audit is not charged to the method.
        self.deferred_audit = False
        self._reset_statistics()
        single = self._authors.activations.ActivationRelaxationType.SINGLE_LINE
        t = self._authors.tanh
        self._relaxations = [_AuditedRelaxation(cls(single), kind, self._audit_statistics)
                             for cls, kind in ((t.TanhRelaxation, "tanh"),
                                (t.TanhDerivativeRelaxation, "tanh-prime"),
                                (t.TanhSecondDerivativeRelaxation, "tanh-double-prime"))]
        self.metadata = dict(self._authors.metadata,
            method="partial-crown", target=target, coordinate=coordinate,
            input_dim=self.input_dim, execution_dtype=str(dtype), device=str(self.device),
            algorithm=("authors ordinary CROWN value component" if target == "value" else
                       "authors specialized first/second derivative hybrid CROWN"),
            value_backprop_mode="FULL_BACKPROP", derivative_backprop_mode="COMPONENT_BACKPROP",
            intermediate_bounds="authors ordinary CROWN, not IBP",
            hybrid_mode="component backward propagation with forward substitution of value and derivative affine bounds",
            relaxations={"classes": [type(r.raw).__name__ for r in self._relaxations],
                "type": "SINGLE_LINE", "lower_tangent_bias": .9, "upper_tangent_bias": .2,
                "mccormick_weights": .5, "authors_line_margin": 1e-7,
                "authors_singleton_margin": 1e-5, "authors_zero_slope_tilt": 1e-6,
                "authors_double_prime_global_range_margin": 1e-4,
                "unused_authors_prime_range_optimizer_margin": 1e-3,
                "authors_root_xtol": 1e-8, "authors_root_rtol": 1e-8,
                "mathematical_patches": [], "repair_variant": None},
            envelope_audit={"policy": "fail closed before using every returned line/range",
                "atol": AUDIT_ATOL, "rtol": AUDIT_RTOL,
                "root_xtol": ROOT_XTOL, "root_rtol": ROOT_RTOL,
                "extrema": "endpoints and all residual-stationary branches, plus analytic turning points"},
            algebraic_base_case="affine" if affine else None,
            wrapper_patches=["deep-copy model and explicitly cast parameters to CPU float64",
                "materialize absent Linear biases as exact zeros",
                "fresh upstream solution/derivative state per bound call (avoid stale coefficient caches)",
                "audit envelopes without replacing them; nonfinite/reversed outputs fail closed"],
            timing_policy={
                "setup_seconds": "model copy/cast, pin checks, private source loading, relaxation construction",
                "bound_seconds": "caller times propagation plus per-line/root audits, precision/finite checks, endpoint/midpoint evaluations",
                "not_pure_kernel_timing": True},
            numerical_rigor="ordinary float64 and numerical root solving; not outward-rounded or end-to-end machine-certified")
        if affine:
            w = torch.eye(self.input_dim, dtype=dtype)
            b = torch.zeros(self.input_dim, dtype=dtype)
            for layer in self.model:
                b = layer.weight @ b + layer.bias
                w = layer.weight @ w
            self._affine_weight, self._affine_bias = w[0], b[0]
        self.setup_seconds = time.perf_counter() - started

    def _reset_statistics(self):
        self._audit_statistics.clear()
        self._audit_statistics.update(lines_checked=0, max_observed_violation=0.)

    def evaluate(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not torch.isfinite(x).all():
            raise ValueError("Expected finite [batch,input_dim] evaluation points")
        if self.target == "value":
            with torch.no_grad():
                result = self.model(x).flatten()
        elif self._affine:
            result = torch.zeros(len(x), dtype=self.dtype, device=self.device)
        else:
            # Evaluation only, NOT the bounding implementation.
            with torch.enable_grad():
                point = x.detach().clone().requires_grad_(True)
                first = torch.autograd.grad(self.model(point).sum(), point, create_graph=True)[0]
                result = torch.autograd.grad(first[:, self.coordinate].sum(), point)[0][:, self.coordinate].detach()
        if not torch.isfinite(result).all():
            raise BoundFailure("Nonfinite target evaluation")
        return result

    def bound(self, lower, upper):
        lower, upper = validate_boxes(lower, upper, self.input_dim, self.dtype, self.device)
        self._reset_statistics()
        stages = []
        with torch.no_grad():
            if self._affine:
                if self.target == "second-diagonal":
                    lb = ub = torch.zeros(len(lower), dtype=self.dtype)
                else:
                    w, b = self._affine_weight, self._affine_bias
                    lb = lower @ w.clamp(min=0) + upper @ w.clamp(max=0) + b
                    ub = upper @ w.clamp(min=0) + lower @ w.clamp(max=0) + b
            else:
                c = self._authors.crown
                solution = c.CROWNPINNSolution(list(self.model), self._relaxations[0], device=self.device)
                solution.domain_bounds = torch.stack((lower, upper), dim=1)
                try:
                    solution.compute_bounds(debug=False, backprop_mode=c.BackpropMode.FULL_BACKPROP)
                    stages.append(solution)
                    output = solution
                    if self.target == "second-diagonal":
                        first = c.CROWNPINNPartialDerivative(solution, self.coordinate, self._relaxations[1])
                        first.compute_bounds(debug=False, backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
                        stages.append(first)
                        output = c.CROWNPINNSecondPartialDerivative(first, self.coordinate, self._relaxations[2])
                        output.compute_bounds(debug=False, backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
                        stages.append(output)
                except BoundFailure:
                    raise
                except Exception as exc:
                    raise BoundFailure(f"Pinned partial_crown failed ({type(exc).__name__}): {exc}") from exc
                lb, ub = output.lower_bounds[-1].flatten(), output.upper_bounds[-1].flatten()
            with excluded():
                # Verify stored intermediate tensors AND generated coefficient factories.
                # Accounting only: these fail-closed checks are never bypassed.
                tensors = list(self.model.parameters()) + [lb, ub]
                for stage in stages:
                    tensors.extend(_floating_tensors(vars(stage)))
                observed = sorted({str(t.dtype) for t in tensors} | self._authors.observed_factory_dtypes)
                if observed != [str(self.dtype)]:
                    raise BoundFailure(f"Upstream precision mismatch: {observed}")
                if any(not torch.isfinite(t).all() for t in tensors):
                    raise BoundFailure("Nonfinite upstream intermediate tensor")
                for stage in stages:
                    for lo, hi in zip(stage.lower_bounds, stage.upper_bounds):
                        if torch.any(lo > hi):
                            raise BoundFailure("Reversed upstream intermediate range")
        with excluded():
            metadata = copy.deepcopy(self.metadata)
            metadata["envelope_audit"].update(self._audit_statistics)
            metadata["precision_audit"] = {"observed_floating_dtypes": observed,
                                           "stored_tensors_checked": len(tensors)}
            result = BoundBatch(lb.detach().clone(), ub.detach().clone(), metadata).validate(len(lower))
            # A deterministic smoke guard is not a proof. Failure excludes the result;
            # no sampled value is used to adjust or repair the returned bounds.
            points = torch.stack((lower, (lower + upper) / 2, upper), dim=1)
            actual = self.evaluate(points.reshape(-1, self.input_dim)).reshape(len(lower), 3)
            allowance = 1e-9 + 1e-10 * actual.abs()
            if ((actual < result.lower[:, None] - allowance) |
                    (actual > result.upper[:, None] + allowance)).any():
                raise BoundFailure("Endpoint/midpoint evaluation exceeds partial-CROWN enclosure; no clipping applied")
            metadata["sample_guard"] = {"points_per_box": 3, "passed": True, "is_soundness_proof": False}
            pending = sum(len(r.pending) for r in self._relaxations)
            metadata["relaxation_audit"] = {"deferred": self.deferred_audit, "pending_lines": pending,
                                            "note": ("lines audited by adapter.audit() right after the timed call; the driver refuses the bound if it raises"
                                                     if self.deferred_audit else "lines audited inline before returning")}
            self._last_metadata = metadata   # audit() writes its statistics into the metadata of the call it verifies
            if not self.deferred_audit:
                self.audit()   # inline, fail-closed: raises before the bound is returned
        return result

    def audit(self):
        """Verify every relaxation line returned during the last bound() call (fail-closed).

        Called by the subdivision driver immediately after the timed bound call, outside
        the method's CPU accounting. Raises BoundFailure on any invalid line; a bound whose
        audit did not run or did not pass is never accepted.
        """
        checked = sum(r.flush_audit() for r in self._relaxations)
        self._audit_statistics["deferred_audit_batches"] = self._audit_statistics.get("deferred_audit_batches", 0) + 1
        last = getattr(self, "_last_metadata", None)
        if last is not None:   # record the audit outcome on the bound it verified (same dict object the driver holds)
            last["envelope_audit"].update(self._audit_statistics)
            last["relaxation_audit"]["audited"] = True
        return checked

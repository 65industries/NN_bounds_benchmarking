"""Strict scalar tanh checkpoint loading and explicit pure-second derivatives.

This module has no dependency on a bound engine.  ``ExactSecondDerivative`` is
an ordinary forward graph, not a differentiation of a bound or a nested
Jacobian operator.  Its independent autograd validation is an evaluation test,
not a proof of floating-point enclosure soundness.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_ACTIVATION = "tanh after every hidden affine layer; linear output"
_STORAGE_DTYPES = {
    str(dtype): dtype
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
}
_EXECUTION_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A reconstructed, frozen model and JSON-safe provenance metadata."""

    model: nn.Sequential
    metadata: dict[str, Any]


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Manifest has duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Manifest contains nonfinite JSON constant {value!r}")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Manifest JSON number overflows finite precision: {value!r}")
    return result


def _architecture(value: Any) -> list[int]:
    if (not isinstance(value, list) or len(value) < 2
            or any(type(width) is not int or width <= 0 for width in value)):
        raise ValueError("architecture must be a list of at least two positive integer widths")
    if value[-1] != 1:
        raise ValueError("architecture must have scalar output (last width 1)")
    return list(value)


def _manifest_entry(path: Path, manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(
            stream, object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Manifest must contain a files list")
    if any(not isinstance(entry, dict) or not isinstance(entry.get("file"), str)
           for entry in manifest["files"]):
        raise ValueError("Every manifest record must have a string file identifier")
    # Supplied manifests use exact basenames.  Do not guess from architectures,
    # silently normalize identifiers, or pick the first of duplicate records.
    matches = [entry for entry in manifest["files"] if entry["file"] == path.name]
    if len(matches) != 1:
        raise ValueError(f"Manifest must contain exactly one record for {path.name!r}; found {len(matches)}")
    entry = matches[0]
    required = {"sha256", "architecture", "activation", "dtype", "bytes"}
    missing = sorted(required - entry.keys())
    if missing:
        raise ValueError(f"Manifest record is missing required fields: {', '.join(missing)}")
    _architecture(entry["architecture"])
    if entry["activation"] != _ACTIVATION:
        raise ValueError(f"Unsupported activation contract: {entry['activation']!r}")
    if not isinstance(entry["dtype"], str) or entry["dtype"] not in _STORAGE_DTYPES:
        raise ValueError(f"Unsupported storage dtype: {entry['dtype']!r}")
    if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
        raise ValueError("Manifest bytes must be a positive integer")
    if not isinstance(entry["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
        raise ValueError("Manifest sha256 must be exactly 64 lowercase hexadecimal characters")
    return entry


def _copied_linear(weight: torch.Tensor, bias: torch.Tensor | None,
                   *, dtype: torch.dtype, device: torch.device | str) -> nn.Linear:
    # Meta initialization does not consume the global random-number generator.
    # All real storage comes from owned copies of already validated tensors.
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                      device="meta", dtype=dtype)
    layer.weight = nn.Parameter(weight.detach().to(device=device, dtype=dtype).clone(), requires_grad=False)
    if bias is not None:
        layer.bias = nn.Parameter(bias.detach().to(device=device, dtype=dtype).clone(), requires_grad=False)
    return layer


def load_checkpoint(
    path: str | Path,
    manifest_path: str | Path | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """Reconstruct the exact manifest-pinned raw scalar tanh MLP.

    A manifest is mandatory (default: ``path.parent / 'manifest.json'``).
    Only the supplied ``hidden_layers.N.{weight,bias}`` and
    ``output_layer.{weight,bias}`` state-dict layout is accepted.  Architecture,
    activation, byte count, hash, exact keys, shapes, finite tensor contents and
    uniform storage dtype are checked before a frozen model is returned.
    Deserialization always uses ``weights_only=True``; there is no unsafe
    compatibility fallback.  Execution precision is explicitly float32/float64.
    """
    if dtype not in _EXECUTION_DTYPES:
        raise ValueError("Execution dtype must be torch.float32 or torch.float64")
    requested_device = torch.device(device)
    if requested_device.type == "meta":
        raise ValueError("A real execution device is required; meta is unsupported")
    path = Path(path).expanduser().resolve()
    manifest_path = (path.parent / "manifest.json" if manifest_path is None
                     else Path(manifest_path).expanduser().resolve())
    entry = _manifest_entry(path, manifest_path)
    architecture = _architecture(entry["architecture"])
    # Hash the same immutable byte snapshot that torch.load receives: replacing
    # the on-disk file after verification cannot change the loaded parameters.
    payload = path.read_bytes()
    if len(payload) != entry["bytes"]:
        raise ValueError(f"Checkpoint bytes mismatch: expected {entry['bytes']}, got {len(payload)}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Checkpoint sha256 mismatch: expected {entry['sha256']}, got {digest}")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint must be a plain state-dict mapping")
    layer_names = [f"hidden_layers.{index}" for index in range(len(architecture) - 2)] + ["output_layer"]
    expected_keys = {f"{name}.{part}" for name in layer_names for part in ("weight", "bias")}
    if set(state.keys()) != expected_keys:
        missing = sorted(expected_keys - state.keys())
        unexpected = sorted(repr(key) for key in state.keys() - expected_keys)
        raise ValueError(f"Checkpoint state-dict keys mismatch: missing={missing}, unexpected={unexpected}")
    storage_dtype = _STORAGE_DTYPES[entry["dtype"]]
    for index, name in enumerate(layer_names):
        for part, shape in (("weight", (architecture[index + 1], architecture[index])),
                            ("bias", (architecture[index + 1],))):
            key = f"{name}.{part}"
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                raise ValueError(f"Checkpoint {key} must be a dense tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Checkpoint {key} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
            if tensor.dtype != storage_dtype:
                raise ValueError(f"Checkpoint {key} storage dtype mismatch: expected {storage_dtype}, got {tensor.dtype}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Checkpoint {key} contains nonfinite parameters")
    modules = []
    for index, name in enumerate(layer_names):
        modules.append(_copied_linear(state[f"{name}.weight"], state[f"{name}.bias"],
                                      dtype=dtype, device=requested_device))
        if index < len(layer_names) - 1:
            modules.append(nn.Tanh())
    model = nn.Sequential(*modules).eval()
    # Casting can overflow even when native checkpoint contents are finite.
    validate_scalar_model(model)
    metadata = {
        **entry,
        "architecture": architecture,
        "input_dim": architecture[0],
        "output_dim": 1,
        "path": str(path),
        "manifest_path": str(manifest_path),
        "sha256": digest,
        "bytes": len(payload),
        "storage_dtype": entry["dtype"],
        "execution_dtype": str(dtype),
        "device": str(next(model.parameters()).device),
        "frozen": True,
        "state_dict_layout": "hidden_layers.N.{weight,bias}; output_layer.{weight,bias}",
    }
    return LoadedCheckpoint(model=model, metadata=metadata)


def _layers(model: nn.Module) -> list[nn.Module]:
    if type(model) is nn.Linear:
        return [model]
    if type(model) is not nn.Sequential:
        raise TypeError("Expected an ordinary nn.Sequential tanh MLP or scalar nn.Linear")
    # Sequential iteration preserves repeated module references; children()
    # deduplicates them and can silently change a shared-layer computation.
    return list(model)


def validate_scalar_model(model: nn.Module) -> list[int]:
    """Validate a scalar affine/tanh topology and return its widths.

    Accept a bare scalar ``nn.Linear`` or a nonempty alternating
    ``Linear, Tanh, ..., Linear`` sequential model.  A zero-weight affine model
    is a supported constant function; vector outputs and custom operations are
    not silently treated as scalar tanh networks.  Bias-free affine layers are
    supported for in-memory models, although supplied checkpoints require bias
    keys.  Nothing in the source model is mutated.
    """
    layers = _layers(model)
    if not layers or len(layers) % 2 != 1:
        raise ValueError("Expected a nonempty affine/tanh model ending in Linear")
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("Forward hooks can change the model; hooked models are unsupported")
    architecture = []
    dtype, device = None, None
    for index, layer in enumerate(layers):
        expected = nn.Linear if index % 2 == 0 else nn.Tanh
        if type(layer) is not expected:
            raise TypeError(f"Unsupported layer {index}: expected {expected.__name__}, got {type(layer).__name__}")
        if expected is nn.Tanh:
            continue
        if (type(layer.in_features) is not int or type(layer.out_features) is not int
                or layer.in_features <= 0 or layer.out_features <= 0):
            raise ValueError("Affine layer widths must be positive integers")
        if architecture and architecture[-1] != layer.in_features:
            raise ValueError("Inconsistent consecutive affine dimensions")
        if not architecture:
            architecture.append(layer.in_features)
        architecture.append(layer.out_features)
        for name, tensor, shape in (
            ("weight", layer.weight, (layer.out_features, layer.in_features)),
            ("bias", layer.bias, (layer.out_features,)),
        ):
            if tensor is None:
                if name == "bias":
                    continue
                raise ValueError("Affine weight is missing")
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tuple(tensor.shape) != shape):
                raise ValueError(f"Invalid affine {name} dimensions or tensor layout")
            if tensor.dtype not in _EXECUTION_DTYPES:
                raise ValueError("Model execution dtype must be torch.float32 or torch.float64")
            if tensor.device.type == "meta":
                raise ValueError("Model parameters must have real storage")
            if dtype is None:
                dtype, device = tensor.dtype, tensor.device
            elif tensor.dtype != dtype or tensor.device != device:
                raise ValueError("All parameters must have the same dtype and device")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Model contains nonfinite parameters")
    return _architecture(architecture)


def _coordinate(value: Any, input_dim: int) -> int:
    if type(value) is not int or not 0 <= value < input_dim:
        raise ValueError(f"coordinate must be an integer in [0, {input_dim}), got {value!r}")
    return value


class ExactSecondDerivative(nn.Module):
    """Normal forward graph for ``d² f / dx_coordinate²``, shape ``[batch, 1]``.

    All affine parameters are owned frozen copies.  For each hidden affine/tanh
    layer propagate ``h=tanh(Wh+b)``, ``p=Wv``, ``q=Ww``, ``r=1-h*h`` and
    ``v=r*p``, ``w=r*q-2*h*r*p*p``.  The first ``p`` is the constant selected
    weight column and its ``q`` is zero.  Output is ``W_out w`` with *no* output
    bias.  Only linear, tanh and elementwise arithmetic occur in ``forward``;
    no nonlinear tensor is detached, and no autograd/Jacobian operation runs.
    """

    def __init__(self, model: nn.Module, coordinate: int):
        super().__init__()
        architecture = validate_scalar_model(model)
        self.input_dim = architecture[0]
        self.coordinate = _coordinate(coordinate, self.input_dim)
        self.architecture = architecture
        affine = _layers(model)[::2]
        self.hidden_layers = nn.ModuleList([
            _copied_linear(layer.weight, layer.bias,
                           dtype=layer.weight.dtype, device=layer.weight.device)
            for layer in affine[:-1]
        ])
        self.output_weight = nn.Parameter(affine[-1].weight.detach().clone(), requires_grad=False)
        if self.hidden_layers:
            self.register_buffer("first_direction", self.hidden_layers[0].weight[:, self.coordinate].detach().clone().unsqueeze(0))
        else:
            self.register_buffer("zero_weight", self.output_weight.new_zeros((1, self.input_dim)))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hidden_layers:
            # Keep a normal batch-shaped linear graph, including for a constant
            # source model, rather than returning a batch-independent scalar.
            return F.linear(x, self.zero_weight, None)
        h = torch.tanh(self.hidden_layers[0](x))
        r = 1.0 - h * h
        p = self.first_direction
        v = r * p
        w = -2.0 * h * r * (p * p)
        for layer in self.hidden_layers[1:]:
            a = layer(h)
            p = F.linear(v, layer.weight, None)
            q = F.linear(w, layer.weight, None)
            h = torch.tanh(a)
            r = 1.0 - h * h
            v = r * p
            w = r * q - 2.0 * h * r * (p * p)
        return F.linear(w, self.output_weight, None)


def _float64_reference_copy(model: nn.Module) -> nn.Sequential:
    modules = []
    for layer in _layers(model):
        if type(layer) is nn.Linear:
            modules.append(_copied_linear(layer.weight, layer.bias, dtype=torch.float64, device="cpu"))
        else:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules).eval()


def _nested_autograd_second(model: nn.Sequential, points: torch.Tensor,
                            coordinate: int) -> torch.Tensor:
    """Independent oracle: differentiate only the ordinary source forward."""
    x = points.detach().clone().requires_grad_(True)
    output = model(x)
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Nonfinite source forward during derivative validation")
    first = torch.autograd.grad(output.sum(), x, create_graph=True)[0][:, coordinate]
    if not first.requires_grad:  # Affine (including constant) source networks.
        return torch.zeros_like(x[:, coordinate:coordinate + 1])
    second = torch.autograd.grad(first.sum(), x, allow_unused=True)[0]
    if second is None:
        return torch.zeros_like(x[:, coordinate:coordinate + 1])
    return second[:, coordinate:coordinate + 1].detach()


def validate_derivatives(
    model: nn.Module,
    coordinates: Iterable[int] | int | str = "all",
    seed: int = 0,
    n_points: int = 16,
    *,
    atol: float = 1e-11,
    rtol: float = 1e-9,
) -> dict[str, Any]:
    """JSON-safe float64 comparison to independent nested input autograd.

    Evaluate every requested coordinate at **all** unit-cube corners, the
    center, and ``n_points`` seeded uniform points (no global RNG mutation).
    ``n_points`` may be zero.  ``coordinates`` is ``'all'``, an integer, or a
    nonempty duplicate-free iterable of zero-based integers.  Source parameters,
    dtype/device, gradient flags, training mode and gradients are not changed.

    Finite disagreements return ``passed=False`` and errors; callers must gate
    acceptance on that flag.  Unsupported or nonfinite cases raise.  Relative
    errors use ``max(abs(reference), atol, float64.eps)`` as denominator.  This
    diagnostic tests evaluation, not a relaxation or a global supremum bound.
    """
    architecture = validate_scalar_model(model)
    input_dim = architecture[0]
    if isinstance(coordinates, str):
        if coordinates != "all":
            raise ValueError("coordinates must be 'all', an integer, or an integer iterable")
        selected = list(range(input_dim))
    elif isinstance(coordinates, int):
        selected = [coordinates]
    else:
        selected = list(coordinates)
    selected = [_coordinate(coordinate, input_dim) for coordinate in selected]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("coordinates must be nonempty and contain no duplicates")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(n_points) is not int or n_points < 0:
        raise ValueError("n_points must be a nonnegative integer")
    if (isinstance(atol, bool) or isinstance(rtol, bool)
            or not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0):
        raise ValueError("atol and rtol must be finite nonnegative numbers")
    if input_dim > 12:
        raise ValueError("All-corner validation is limited to input_dim <= 12 (4096 corners)")
    # A caller may be in no_grad/inference_mode; the independent oracle still
    # needs normal differentiable tensors, without changing the caller's mode.
    with torch.inference_mode(False), torch.enable_grad():
        reference = _float64_reference_copy(model)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        corners = torch.tensor(list(itertools.product((0., 1.), repeat=input_dim)), dtype=torch.float64)
        center = torch.full((1, input_dim), .5, dtype=torch.float64)
        random_points = torch.rand((n_points, input_dim), dtype=torch.float64, generator=generator)
        points = torch.cat((corners, center, random_points), dim=0)
        per_coordinate = []
        relative_floor = max(float(atol), torch.finfo(torch.float64).eps)
        for coordinate in selected:
            oracle = _nested_autograd_second(reference, points, coordinate)
            derivative = ExactSecondDerivative(reference, coordinate)
            with torch.no_grad():
                evaluated = derivative(points)
            if not bool(torch.isfinite(oracle).all() and torch.isfinite(evaluated).all()):
                raise ValueError(f"Nonfinite derivative evaluation for coordinate {coordinate}")
            absolute_error = (evaluated - oracle).abs()
            relative_error = absolute_error / oracle.abs().clamp_min(relative_floor)
            if not bool(torch.isfinite(absolute_error).all() and torch.isfinite(relative_error).all()):
                raise ValueError(f"Nonfinite derivative error metrics for coordinate {coordinate}")
            passed = bool(torch.all(absolute_error <= atol + rtol * oracle.abs()))
            per_coordinate.append({
                "coordinate": coordinate,
                "passed": passed,
                "max_abs_error": float(absolute_error.max()),
                "max_relative_error": float(relative_error.max()),
                "worst_point_index": int(absolute_error.flatten().argmax()),
                "reference": oracle.flatten().tolist(),
                "evaluated": evaluated.flatten().tolist(),
            })
    return {
        "passed": all(item["passed"] for item in per_coordinate),
        "reference_method": "independent nested torch.autograd.grad on source forward",
        "source_dtype": str(_layers(model)[0].weight.dtype),
        "source_device": str(_layers(model)[0].weight.device),
        "execution_dtype": "torch.float64",
        "device": "cpu",
        "architecture": architecture,
        "input_dim": input_dim,
        "coordinates": selected,
        "seed": seed,
        "n_points": len(points),
        "n_random_points": n_points,
        "corner_count": len(corners),
        "points": points.tolist(),
        "point_kinds": ["corner"] * len(corners) + ["center"] + ["seeded_uniform"] * n_points,
        "domain_lower": [0.] * input_dim,
        "domain_upper": [1.] * input_dim,
        "atol": float(atol),
        "rtol": float(rtol),
        "relative_error_floor": relative_floor,
        "max_abs_error": max(item["max_abs_error"] for item in per_coordinate),
        "max_relative_error": max(item["max_relative_error"] for item in per_coordinate),
        "per_coordinate": per_coordinate,
        "qualification": "Pointwise numerical evaluation check, not a bound or soundness proof.",
    }

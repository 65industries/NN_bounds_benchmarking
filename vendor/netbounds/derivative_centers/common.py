"""Differentiable center-derivative recurrences for PINN training.

Unlike derivative_bounds.*, these helpers do not enter torch.no_grad() and do
not compute interval/radius bounds. They only propagate exact center
input-derivatives of the raw network output, so a scalar loss built from the
returned tensors can still be differentiated with respect to network weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from derivative_bounds.activation_bounds import TanhBounds
from derivative_bounds.first_order import FirstOrderBoundAccumulator
from derivative_bounds.second_order import SecondOrderBoundAccumulator
from derivative_bounds.third_order import ThirdOrderBoundAccumulator, required_pairs_for_triples
from derivative_bounds.fourth_order import FourthOrderBoundAccumulator, required_triples_for_quads


@dataclass
class CenterDerivatives:
    input_dim: int
    value: torch.Tensor
    gradient: torch.Tensor | None = None
    pairs: tuple[tuple[int, int], ...] = ()
    hessian: torch.Tensor | None = None
    triples: tuple[tuple[int, int, int], ...] = ()
    third: torch.Tensor | None = None
    quartets: tuple[tuple[int, int, int, int], ...] = ()
    fourth: torch.Tensor | None = None


def normalize_points(model, y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        y = y.unsqueeze(0)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    return y.to(dtype=dtype, device=device)


def activation_derivatives(model, z: torch.Tensor, max_order: int) -> dict[int, torch.Tensor]:
    # NetBounds models expose AB; training FCN models generally do not.
    AB = getattr(model, "AB", None) or TanhBounds()
    return {order: AB.activation_m_prime_stable(z, order) for order in range(1, max_order + 1)}


def raw_network_value(model, y: torch.Tensor) -> torch.Tensor:
    a = y
    for layer in model.hidden_layers:
        a = torch.tanh(layer(a))
    return model.output_layer(a)


def compute_raw_centers(
    model,
    y: torch.Tensor,
    *,
    order: int,
    pairs: tuple[tuple[int, int], ...] = (),
    triples: tuple[tuple[int, int, int], ...] = (),
    quartets: tuple[tuple[int, int, int, int], ...] = (),
) -> CenterDerivatives:
    y = normalize_points(model, y)
    input_dim = int(getattr(model, "input_dim"))
    batch_size = y.shape[0]
    dtype = y.dtype
    device = y.device
    if order < 0 or order > 4:
        raise ValueError("order must be in [0,4].")
    if order >= 2 and not pairs:
        raise ValueError("pairs are required for order >= 2")
    if order >= 3 and not triples:
        raise ValueError("triples are required for order >= 3")
    if order >= 4 and not quartets:
        raise ValueError("quartets are required for order >= 4")

    pair_to_column = {pair: i for i, pair in enumerate(pairs)}
    triple_to_column = {triple: i for i, triple in enumerate(triples)}
    first_acc = FirstOrderBoundAccumulator(batch_size=batch_size, input_dim=input_dim, dtype=dtype, device=device)
    second_acc = SecondOrderBoundAccumulator(batch_size=batch_size, input_dim=input_dim, pairs=pairs, dtype=dtype, device=device) if order >= 2 else None
    third_acc = ThirdOrderBoundAccumulator(batch_size=batch_size, input_dim=input_dim, triples=triples, pair_to_column=pair_to_column, dtype=dtype, device=device) if order >= 3 else None
    fourth_acc = FourthOrderBoundAccumulator(batch_size=batch_size, input_dim=input_dim, quartets=quartets, pair_to_column=pair_to_column, triple_to_column=triple_to_column, dtype=dtype, device=device) if order >= 4 else None

    a = y
    for layer in model.hidden_layers:
        z = layer(a)
        m = activation_derivatives(model, z, max(order, 1))
        Z1_base, Z1_error, _ = first_acc.next_preactivation(layer.weight)
        Z2_base = Z2_error = Z3_base = Z3_error = Z4_base = Z4_error = None
        if order >= 2:
            assert second_acc is not None
            Z2_base, Z2_error = second_acc.next_preactivation(layer.weight)
        if order >= 3:
            assert third_acc is not None
            Z3_base, Z3_error = third_acc.next_preactivation(layer.weight)
        if order >= 4:
            assert fourth_acc is not None
            Z4_base, Z4_error = fourth_acc.next_preactivation(layer.weight)

        q1 = torch.zeros_like(m[1])
        first_acc.append_layer(m[1], q1, Z1_base, Z1_error, store_cache=False)
        if order >= 2:
            assert second_acc is not None and Z2_base is not None and Z2_error is not None
            second_acc.append_layer(
                m1_k=m[1], m2_k=m[2], Q1_k=q1, Q2_k=torch.zeros_like(m[2]),
                Z1_base=Z1_base, Z1_error=Z1_error,
                Z2_base=Z2_base, Z2_error=Z2_error,
                store_cache=False,
            )
        if order >= 3:
            assert third_acc is not None and Z2_base is not None and Z2_error is not None and Z3_base is not None and Z3_error is not None
            third_acc.append_layer(
                m1_k=m[1], m2_k=m[2], m3_k=m[3],
                Q1_k=q1, Q2_k=torch.zeros_like(m[2]), Q3_k=torch.zeros_like(m[3]),
                Z1_base=Z1_base, Z1_error=Z1_error,
                Z2_base=Z2_base, Z2_error=Z2_error,
                Z3_base=Z3_base, Z3_error=Z3_error,
                store_cache=False,
            )
        if order >= 4:
            assert fourth_acc is not None and Z2_base is not None and Z2_error is not None and Z3_base is not None and Z3_error is not None and Z4_base is not None and Z4_error is not None
            fourth_acc.append_layer(
                m1_k=m[1], m2_k=m[2], m3_k=m[3], m4_k=m[4],
                Q1_k=q1, Q2_k=torch.zeros_like(m[2]), Q3_k=torch.zeros_like(m[3]), Q4_k=torch.zeros_like(m[4]),
                Z1_base=Z1_base, Z1_error=Z1_error,
                Z2_base=Z2_base, Z2_error=Z2_error,
                Z3_base=Z3_base, Z3_error=Z3_error,
                Z4_base=Z4_base, Z4_error=Z4_error,
                store_cache=False,
            )
        a = torch.tanh(z)

    value = model.output_layer(a)
    gradient = hessian = third = fourth = None
    if order >= 1:
        gradient, _ = first_acc.final_bounds(model.output_layer.weight)
    if order >= 2:
        assert second_acc is not None
        hessian, _ = second_acc.final_bounds(model.output_layer.weight)
    if order >= 3:
        assert third_acc is not None
        third, _ = third_acc.final_bounds(model.output_layer.weight)
    if order >= 4:
        assert fourth_acc is not None
        fourth, _ = fourth_acc.final_bounds(model.output_layer.weight)

    return CenterDerivatives(input_dim=input_dim, value=value, gradient=gradient, pairs=pairs, hessian=hessian, triples=triples, third=third, quartets=quartets, fourth=fourth)


def required_sets_for_quartets(quartets: tuple[tuple[int, int, int, int], ...]):
    triples = required_triples_for_quads(quartets)
    pairs = required_pairs_for_triples(triples)
    return pairs, triples, quartets

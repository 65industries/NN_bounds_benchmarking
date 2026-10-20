"""Released partial_crown affine coefficients, read out without modifying upstream.

The benchmark's PartialCrownAdapter returns a concretized BoundBatch. The released
solver objects also store the AFFINE pair that concretization came from; this module
runs the same released chain and reads those attributes. Nothing upstream is patched,
no bound is recomputed by us, and the concretized interval is read back as well so the
two can be cross-checked against each other on every cell.

Attribute provenance (upstream `pinn_verifier/crown.py`):
  * value target          -> CROWNPINNSolution.layer_CROWN_coefficients[-1]
                             = (f_U_A_0, f_U_constant, f_L_A_0, f_L_constant)
  * second-diagonal target-> CROWNPINNSecondPartialDerivative
                             .u_dxixi_crown_coefficients_{lbs,ubs}[-1]
                             .u_dxixi_crown_constants_{lbs,ubs}[-1]
"""
import torch

from crown_benchmark.types import BoundFailure


class AffineCrownCell:
    """One released CROWN call per cell, exposing the affine pair it already computes."""

    def __init__(self, model, target, coordinate):
        from crown_benchmark.partial import PartialCrownAdapter, load_authors
        self._adapter = PartialCrownAdapter(model, target, coordinate)
        self._adapter.deferred_audit = True
        self._authors = load_authors()
        self._model = model
        self._target = target
        self._coordinate = coordinate
        self.input_dim = 2
        self.calls = 0
        self.audits = []
        self.consistency = dict(checked=0, max_gap=0.)

    def affine_bound(self, lower, upper):
        """Return (A_L, c_L, A_U, c_U, L, U) for this cell from ONE released call."""
        self.calls += 1
        c = self._authors.crown
        lo = torch.as_tensor(lower, dtype=torch.float64)
        hi = torch.as_tensor(upper, dtype=torch.float64)
        solution = c.CROWNPINNSolution(list(self._model), self._adapter._relaxations[0],
                                       device='cpu')
        solution.domain_bounds = torch.stack((lo, hi), dim=0)
        try:
            solution.compute_bounds(debug=False, backprop_mode=c.BackpropMode.FULL_BACKPROP)
            if self._target == 'value':
                UA, Uc, LA, Lc = solution.layer_CROWN_coefficients[-1]
                A_U = UA.flatten().to(torch.float64)
                A_L = LA.flatten().to(torch.float64)
                c_U = float(Uc.flatten()[0])
                c_L = float(Lc.flatten()[0])
                L = float(solution.lower_bounds[-1].flatten()[0])
                U = float(solution.upper_bounds[-1].flatten()[0])
            else:
                first = c.CROWNPINNPartialDerivative(solution, self._coordinate,
                                                     self._adapter._relaxations[1])
                first.compute_bounds(debug=False, backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
                second = c.CROWNPINNSecondPartialDerivative(first, self._coordinate,
                                                            self._adapter._relaxations[2])
                second.compute_bounds(debug=False,
                                      backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
                A_L = second.u_dxixi_crown_coefficients_lbs[-1].flatten().to(torch.float64)
                c_L = float(second.u_dxixi_crown_constants_lbs[-1].flatten()[0])
                A_U = second.u_dxixi_crown_coefficients_ubs[-1].flatten().to(torch.float64)
                c_U = float(second.u_dxixi_crown_constants_ubs[-1].flatten()[0])
                L = float(second.lower_bounds[-1].flatten()[0])
                U = float(second.upper_bounds[-1].flatten()[0])
        except BoundFailure:
            raise
        except Exception as exc:
            raise BoundFailure(
                f'Pinned partial_crown affine read-out failed ({type(exc).__name__}): {exc}') from exc

        A_L, A_U = [float(v) for v in A_L], [float(v) for v in A_U]
        for t in (*A_L, *A_U, c_L, c_U, L, U):
            if not torch.isfinite(torch.tensor(t, dtype=torch.float64)):
                raise BoundFailure('Nonfinite CROWN affine coefficient')
        self._check_consistency(A_L, c_L, A_U, c_U, lower, upper, L, U)
        return A_L, c_L, A_U, c_U, L, U

    def _check_consistency(self, A_L, c_L, A_U, c_U, lower, upper, L, U):
        """The affine pair, concretized by hand, must reproduce the released interval.

        This is a fail-closed check that we are reading the SAME object the released
        code concretizes, not an unrelated intermediate tensor.
        """
        corners = [(x, y) for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]
        lo_c = min(A_L[0] * x + A_L[1] * y + c_L for x, y in corners)
        hi_c = max(A_U[0] * x + A_U[1] * y + c_U for x, y in corners)
        scale = max(1., abs(L), abs(U))
        gap = max(abs(lo_c - L), abs(hi_c - U)) / scale
        if gap > 1e-9:
            raise BoundFailure(
                f'CROWN affine pair does not concretize to the released interval (gap {gap:.3e})')
        self.consistency['checked'] += 1
        self.consistency['max_gap'] = max(self.consistency['max_gap'], gap)

    def drain_audit(self):
        """Record the released envelope audit for the last call; never enforce it."""
        import traceback as _tb
        entry = dict(failed=False, error=None, lines_checked=0, max_observed_violation=0.)
        try:
            entry['lines_checked'] = sum(r.flush_audit() for r in self._adapter._relaxations)
            entry['max_observed_violation'] = float(
                self._adapter._audit_statistics.get('max_observed_violation', 0.))
        except Exception as exc:
            entry['failed'] = True
            entry['error'] = repr(exc)
            entry['traceback'] = _tb.format_exc()
        self.audits.append(entry)
        return entry

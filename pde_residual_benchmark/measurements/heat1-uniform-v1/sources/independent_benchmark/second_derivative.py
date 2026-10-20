"""Measured Q1 supremum path only; no adaptive search or PDE target."""
import math
import time
import numpy as np
from .cpu_clock import excluded

METHOD_POLICY = dict(
    name='raw-second-derivative-cpu-v1',
    target='pure second diagonal derivative F_ii of the raw network, not a PDE residual or a masked solution',
    coordinate_convention='zero-based input axis of the raw network: 0 = x, 1 = t',
    native_output_guard=False,
    native_output_clipping=False,
    sample_containment_stopping=False,
    early_stopping_on_diagnostics=False,
    relaxation_audit='recorded after each native call; never rejects or changes the returned interval',
    partial_crown_entry='CROWNPINNSolution -> CROWNPINNPartialDerivative -> CROWNPINNSecondPartialDerivative, '
                        'FULL_BACKPROP then COMPONENT_BACKPROP, exactly as the released adapter composes them',
    netbounds_entry='compute_up_to_fourth_order_bounds via RawQ1Integrator; affine cell model with '
                    'fourth-derivative envelope remainder; complete uniform grid',
    netbounds_center_consistency='upstream base jet compared with an independent evaluator; excluded from CPU, '
                                 'unchanged from the value-table rows',
)


def grid_reduction(base, gradient, hessian_sup, epsilon):
    """Signed NetBounds Q1 cell enclosure, reduced over the complete grid."""
    epsilon = np.asarray(epsilon, dtype=np.float64)
    radius = (np.abs(gradient) * epsilon).sum(axis=1) + .5 * np.einsum('i,nij,j->n', epsilon, hessian_sup, epsilon)
    lower = float((base - radius).min())
    upper = float((base + radius).max())
    return lower, upper, max(-lower, upper)


def run_netbounds(model, coordinate, n, clock):
    """One independent NetBounds Q1 row on a complete uniform grid."""
    from netbounds_benchmark.uniform_q1 import RawQ1Integrator
    from netbounds_benchmark.fixed_q1 import evaluate_fixed_grid
    mark = clock.mark()
    kernel = RawQ1Integrator(model, 'second-diagonal', coordinate, moments=False)
    counts = dict(cell_batch_calls=0, cells_evaluated=0)
    original = kernel.cell_batch

    def cell_batch(centers, radii):
        values = original(centers, radii)
        with excluded():
            counts['cell_batch_calls'] += 1
            counts['cells_evaluated'] += len(centers)
        return values

    kernel.cell_batch = cell_batch
    setup = clock.elapsed(mark)
    mark = clock.mark()
    wall = time.perf_counter()
    grid = evaluate_fixed_grid(kernel, n=n, batch_size=256, jets_only=True)
    arrays = grid.pop('arrays')
    lower, upper, bound = grid_reduction(arrays['base'], arrays['gradient'], arrays['hessian_sup'], grid['epsilon'])
    compute = clock.elapsed(mark)
    numerical = dict(bound=bound, signed_lower=lower, signed_upper=upper, arrays=arrays,
                     n=n, cells=grid['cells'], epsilon=grid['epsilon'], coverage=grid['coverage'],
                     stop_reason='complete_uniform_grid', counters=counts, grid_computation=grid,
                     method_policy=dict(METHOD_POLICY),
                     norm_reduction="max of this row's signed Q1 cell enclosures; no moment integration",
                     backend=dict(kernel.metadata, integration='none: L-infinity Taylor-cell enclosures; jets only'))
    timing = dict(adapter_setup_cpu_seconds=setup, reference_cpu_seconds=0.,
                  computation_cpu_seconds=compute, computation_wall_seconds=time.perf_counter() - wall,
                  harness_diagnostics_cpu_seconds=clock.excluded_seconds)
    assert grid['complete'] and grid['discarded_cells'] == 0 and grid['cpu_limit'] is None
    assert counts['cells_evaluated'] == grid['cells'] == n ** kernel.input_dim
    assert counts['cell_batch_calls'] == math.ceil(grid['cells'] / 256)
    return numerical, timing, kernel.metadata

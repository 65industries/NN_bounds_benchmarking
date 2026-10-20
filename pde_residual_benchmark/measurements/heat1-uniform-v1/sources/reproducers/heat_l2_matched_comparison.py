"""Measured matched-grid L2 workers; portable coordinator is heat_reproduce.py."""
import importlib.util
import math
import os
from pathlib import Path
import time
import json
ROOT = Path(__file__).resolve().parents[1]
NB = "netbounds-q1-l2"
PC = "partial-crown-affine-l2"
REF = "gauss-reference"
MOMENTS = ("midpoint_square", "affine_square", "cross_error", "remainder_square", "upper_square")

def helpers():
    path = Path(__file__).with_name('heat_second_order_e_comparison.py')
    spec = importlib.util.spec_from_file_location('l2_shared', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reduce_q1(arrays):
    sums = {key: math.fsum(arrays[key].tolist()) for key in MOMENTS}
    if not all(math.isfinite(v) and v >= 0 for v in sums.values()):
        raise ValueError('Invalid native moment sum; no repair')
    if not math.isclose(sums['upper_square'], math.fsum(sums[k] for k in
                       ('affine_square', 'cross_error', 'remainder_square')), rel_tol=1e-14):
        raise ValueError('Native Q1 component sum mismatch')
    # Match the heat report: one nonnegative truncation AFTER global summation.
    lower_square = max(0., sums['affine_square'] - sums['cross_error'])
    upper_square = sums['upper_square']
    if not 0 <= lower_square <= upper_square:
        raise ValueError('Reversed L2 endpoints; no min(lower,upper) repair')
    return lower_square, upper_square, sums


def target_values(model, coordinate, points):
    import torch
    from crown_benchmark.models import _nested_autograd_second
    if coordinate is None:
        with torch.no_grad():
            return model(points).reshape(-1).detach()
    return _nested_autograd_second(model, points, coordinate).reshape(-1).detach()


def point_diagnostics(model, coordinate, arrays, method):
    import numpy as np
    import torch
    offsets = torch.cartesian_prod(*[torch.tensor([-1., 0., 1.], dtype=torch.float64)] * 2).numpy()
    strict = tolerant = count = 0
    worst = -math.inf
    for first in range(0, len(arrays['centers']), 128):
        stop = first + 128
        c = arrays['centers'][first:stop]
        z = offsets[None] * ((arrays['upper'][first:stop] - arrays['lower'][first:stop]) / 2)[:, None]
        points = c[:, None] + z
        values = target_values(model, coordinate, torch.from_numpy(points.reshape(-1, 2))).numpy().reshape(-1, 9)
        if method == NB:
            a = arrays['base'][first:stop, None] + np.einsum('ni,npi->np', arrays['gradient'][first:stop], z)
            d = .5 * np.einsum('npi,nij,npj->np', abs(z), arrays['hessian_sup'][first:stop], abs(z))
            lower, upper = a - d, a + d
        else:
            lower = np.einsum('ni,npi->np', arrays['A_lower'][first:stop], points) + arrays['c_lower'][first:stop, None]
            upper = np.einsum('ni,npi->np', arrays['A_upper'][first:stop], points) + arrays['c_upper'][first:stop, None]
        excess = np.maximum(lower - values, values - upper)
        tolerance = 1e-10 + 1e-9 * np.maximum(np.maximum(abs(lower), abs(upper)), abs(values))
        strict += int((excess > 0).sum()); tolerant += int((excess > tolerance).sum())
        count += values.size; worst = max(worst, float(excess.max()))
    return dict(points_per_cell=9, point_evaluations=count, cells_checked=len(arrays['centers']),
                strict_violating_points=strict, tolerance_violating_points=tolerant,
                maximum_signed_excess=worst, atol=1e-10, rtol=1e-9,
                affects_bound=False, soundness_proof=False)


def run_row(args):
    import numpy as np
    import torch
    h = helpers()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    row = args.output.resolve(); row.mkdir(parents=True, exist_ok=False)
    seen = h.install_read_guard(args.campaign, row)
    # Import/setup is outside numerical timing.
    from crown_benchmark.models import load_checkpoint
    from independent_benchmark.cpu_clock import numerical_clock, excluded
    from independent_benchmark.crown_affine import AffineCrownCell
    from independent_benchmark.minimal_affine_l2_native import cell_integrals
    from independent_benchmark.l2_norm import reference_l2
    from netbounds_benchmark.uniform_q1 import RawQ1Integrator
    from netbounds_benchmark.fixed_q1 import evaluate_fixed_grid
    coordinate = None if args.target == 'F' else int(args.target[-1])
    target = 'value' if coordinate is None else 'second-diagonal'
    t = h.cpu(); loaded = load_checkpoint(ROOT / 'model_weights' / h.CHECKPOINT_NAME)
    loading = h.cpu() - t
    t = h.cpu()
    if args.method == NB:
        kernel = RawQ1Integrator(loaded.model, target, coordinate, moments=True, device='cpu')
    elif args.method == PC:
        kernel = AffineCrownCell(loaded.model, target, coordinate)
    setup = h.cpu() - t
    common = dict(case_id=row.name, method=args.method, target=args.target, coordinate=coordinate,
                  n=args.n, checkpoint=loaded.metadata, pid=os.getpid(), precision='CPU float64, no outward rounding',
                  checkpoint_loading_cpu_seconds=loading, backend_setup_cpu_seconds=setup,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
    with numerical_clock() as clock:
        if args.method == REF:
            reference = reference_l2(loaded.model, coordinate, clock, degree=args.n)
            h.save(row / 'result.json', dict(common, reference=reference, is_bound=False))
            h.save(row / 'diagnostics.json', dict(campaign_read_paths=sorted(seen), affects_bound=False))
            return
        start, wall = clock.mark(), time.perf_counter()
        lower, upper, centers, eps = h.grid(args.n)
        geometry = dict(lower=lower.numpy(), upper=upper.numpy(), centers=centers.numpy())
        if args.method == NB:
            native = evaluate_fixed_grid(kernel, n=args.n, batch_size=256)
            arrays = dict(native['arrays'], **geometry)
            lower_square, upper_square, sums = reduce_q1(arrays)
            native_calls = len(range(0, args.n ** 2, 256))
            metadata = kernel.metadata
        else:
            records = []
            for lo, hi in zip(geometry['lower'].tolist(), geometry['upper'].tolist()):
                AL, cL, AU, cU, L, U = kernel.affine_bound(lo, hi)
                raw = dict(A_lower=AL, c_lower=cL, A_upper=AU, c_upper=cU, lower_bound=L, upper_bound=U)
                integral = cell_integrals(raw, lo, hi)
                records.append(dict(raw, **integral))
                with excluded():
                    kernel.drain_audit()
            arrays = {key: np.asarray([r[key] for r in records], dtype=np.float64) for key in records[0]}
            arrays.update(geometry)
            lower_square = math.fsum(arrays['lower_square'].tolist())
            upper_square = math.fsum(arrays['upper_square'].tolist())
            sums = None; native_calls = kernel.calls
            metadata = dict(native_readout='AffineCrownCell.affine_bound', integration='minimal_affine_l2_native.cell_integrals')
        if not (math.isfinite(upper_square) and 0 <= lower_square <= upper_square):
            raise ValueError('Invalid global L2 interval; no repair')
        low, high = math.sqrt(lower_square), math.sqrt(upper_square)
        cpu_seconds = clock.elapsed(start); wall_seconds = time.perf_counter() - wall
        inline_cpu = clock.excluded_seconds
    result = dict(common, lower=low, upper=high, lower_square=lower_square, upper_square=upper_square,
                  moment_sums=sums, cells=args.n ** 2, native_calls=native_calls, backend=metadata,
                  bounding_cpu_seconds=cpu_seconds, bounding_wall_seconds=wall_seconds,
                  inline_diagnostics_cpu_seconds=inline_cpu, epsilon=eps.tolist(),
                  stop_reason='complete_uniform_grid', cpu_ceiling=None, finished_numerics_utc=h.now(),
                  coverage=dict(complete=True, nonoverlapping=True, volume_sum_exact='1',
                                cell_volume_exact=f'1/{args.n ** 2}', adaptive_splits=0),
                  timing='Geometry, native bounds, existing moment/polygon integration, collection, sums and roots. '
                         'Excludes imports, checkpoint/backend setup, diagnostic spans and serialization. '
                         'Native internal timers, shape checks and residual bookkeeping remain charged; wall includes inline diagnostics.')
    np.savez(row / 'cells.npz', **arrays)
    h.save(row / 'result.json', result)
    frozen = {name: h.digest(row / name) for name in ('cells.npz', 'result.json')}
    t = h.cpu()
    diagnostic = point_diagnostics(loaded.model, coordinate, arrays, args.method)
    with numerical_clock() as clock:
        diagnostic['own_quadrature'] = reference_l2(loaded.model, coordinate, clock, degree=64)
    diagnostic['own_quadrature_inside'] = low <= diagnostic['own_quadrature']['value'] <= high
    if args.method == PC:
        diagnostic['native_envelope_audits'] = dict(calls=len(kernel.audits),
            failed_calls=sum(r['failed'] for r in kernel.audits),
            lines_checked=sum(r['lines_checked'] for r in kernel.audits),
            maximum_observed_violation=max(r['max_observed_violation'] for r in kernel.audits),
            failures=[r for r in kernel.audits if r['failed']])
        diagnostic['affine_concretization'] = kernel.consistency
    diagnostic['diagnostics_cpu_seconds'] = h.cpu() - t
    diagnostic['numerical_artifact_sha256'] = frozen
    diagnostic['numerical_artifacts_unchanged'] = all(h.digest(row / k) == v for k, v in frozen.items())
    assert diagnostic['numerical_artifacts_unchanged']
    diagnostic['campaign_read_paths'] = sorted(seen)
    h.save(row / 'diagnostics.json', diagnostic)
    print(json.dumps({k: result[k] for k in ('case_id', 'lower', 'upper', 'bounding_cpu_seconds')}), flush=True)

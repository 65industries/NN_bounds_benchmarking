"""Measured E2/partial-CROWN workers; portable coordinator is heat_reproduce.py.

Definitions below are retained from the frozen measured worker.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "heat_1d_separable_L2_W128_netbounds.pt"
NB_ROOT = ROOT / "vendor/netbounds"
PC_ROOT = ROOT / "vendor/partial_crown"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.rename(path)


def cpu():
    return time.process_time_ns() * 1e-9


def now():
    return datetime.now(timezone.utc).isoformat()


def grid(n):
    import torch
    if n < 1 or n & (n - 1):
        raise ValueError('Use a power-of-two grid for exact binary geometry')
    axis = torch.arange(n, dtype=torch.float64)
    ij = torch.cartesian_prod(axis, axis)
    lower, upper = ij / n, (ij + 1) / n
    centers = (ij + .5) / n
    return lower, upper, centers, torch.full((2,), .5 / n, dtype=torch.float64)


def partial_setup(model):
    from crown_benchmark.partial import load_authors
    authors = load_authors(dtype=__import__('torch').float64)
    single = authors.activations.ActivationRelaxationType.SINGLE_LINE
    t = authors.tanh
    relaxations = [c(single) for c in (t.TanhRelaxation, t.TanhDerivativeRelaxation,
                                      t.TanhSecondDerivativeRelaxation)]
    return authors, relaxations


def partial_call(model, coordinate, authors, relaxations, lower, upper):
    import torch
    c = authors.crown
    with torch.no_grad():
        solution = c.CROWNPINNSolution(list(model), relaxations[0], device=torch.device('cpu'))
        solution.domain_bounds = torch.stack((lower[None], upper[None]), dim=1)
        solution.compute_bounds(debug=False, backprop_mode=c.BackpropMode.FULL_BACKPROP)
        first = c.CROWNPINNPartialDerivative(solution, coordinate, relaxations[1])
        first.compute_bounds(debug=False, backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
        second = c.CROWNPINNSecondPartialDerivative(first, coordinate, relaxations[2])
        second.compute_bounds(debug=False, backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
        return second.lower_bounds[-1].item(), second.upper_bounds[-1].item()


def netbounds_setup(model):
    import torch
    from netbounds_benchmark.adapter import RawMLPView, load_backend
    root, modules = load_backend()
    net = RawMLPView(model, modules['derivative_bounds.activation_bounds'].TanhBounds(dtype=torch.float64))
    return net, modules['derivative_bounds.second_order'].compute_second_order_bounds


def netbounds_call(net, kernel, centers, radii, coordinate):
    result = kernel(net, centers, radii, n_taylor=None,
                    multi_indices=[(coordinate, coordinate)], return_cache=False)
    assert result.pairs == ((coordinate, coordinate),)
    base = result.base_hessian[:, 0]
    variation = result.second_derivative_bounds[:, 0]
    return base, variation


def install_read_guard(campaign, row):
    """Fail if a row reads another row or any historical numerical campaign."""
    campaign = campaign.resolve()
    results = Path(os.environ['HEAT_E_ORIGINAL_ROOT']) / 'results'
    seen = set()
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode, flags = args[1], args[2]
        reading = (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if not reading:
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(results):
            allowed = path.is_relative_to(campaign / 'sources') or path.is_relative_to(row)
            if not allowed:
                raise PermissionError(f'Independent row cannot read numerical input: {path}')
            seen.add(str(path))
    sys.addaudithook(hook)
    return seen


def diagnose(model, coordinate, lower, upper, centers, lbs, ubs, base=None, variation=None):
    import numpy as np
    import torch
    from crown_benchmark.models import _nested_autograd_second
    offsets = torch.cartesian_prod(*[torch.tensor([-1., 0., 1.], dtype=torch.float64)] * 2)
    values = []
    for start in range(0, len(centers), 128):
        c, h = centers[start:start+128], (upper[start:start+128]-lower[start:start+128]) / 2
        points = c[:, None] + offsets[None] * h[:, None]
        values.append(_nested_autograd_second(model, points.reshape(-1, 2), coordinate).reshape(-1, 9))
    v = torch.cat(values).numpy()
    excess = np.maximum(lbs[:, None] - v, v - ubs[:, None])
    tol = 1e-10 + 1e-9 * np.maximum(np.maximum(abs(lbs), abs(ubs))[:, None], abs(v))
    flat = int(np.argmax(abs(v)))
    cell, point = divmod(flat, 9)
    witness = centers[cell] + offsets[point] * (upper[cell]-lower[cell]) / 2
    out = dict(cells_checked=len(centers), points_per_cell=9, point_evaluations=int(v.size),
               strict_violating_points=int((excess > 0).sum()),
               tolerance_violating_points=int((excess > tol).sum()),
               maximum_signed_excess=float(excess.max()),
               largest_observed=float(abs(v).max()), witness=witness.tolist(),
               atol=1e-10, rtol=1e-9, affects_bound=False, soundness_proof=False)
    if base is not None:
        out['center_max_abs_discrepancy'] = float(abs(base - v[:, 4]).max())
        residual = abs(v-base[:, None])-variation[:, None]
        out['variation_max_signed_excess'] = float(residual.max())
        out['variation_strict_violating_points'] = int((residual > 0).sum())
        out['variation_tolerance_violating_points'] = int((residual > tol).sum())
    return out


class RecordRelaxation:
    """Only used in an UNTIMED independent replay, never in measured bounds."""
    def __init__(self, raw, kind):
        self.raw, self.kind, self.records = raw, kind, []
    def __getattr__(self, name):
        return getattr(self.raw, name)
    def get_bounds(self, lo, hi):
        lines = self.raw.get_bounds(lo, hi)
        self.records.append(('lines', float(lo), float(hi), [(float(a),float(b)) for a,b in lines]))
        return lines
    def get_lb_ub_in_interval(self, lo, hi):
        bounds = self.raw.get_lb_ub_in_interval(lo, hi)
        self.records.append(('range', float(lo), float(hi), tuple(float(v) for v in bounds)))
        return bounds


def replay_partial(model, coordinate, authors, indices, lower, upper, lbs, ubs):
    from crown_benchmark.envelope_audit import audit_line
    _, raw = partial_setup(model)
    relax = [RecordRelaxation(r, k) for r, k in zip(raw, ('tanh','tanh-prime','tanh-double-prime'))]
    endpoints = []
    for index in indices:
        l, u = partial_call(model, coordinate, authors, relax, lower[index], upper[index])
        endpoints.append(dict(cell=index, max_abs_endpoint_gap=max(abs(l-lbs[index]), abs(u-ubs[index]))))
    reports = 0
    violations = []
    strict = 0
    worst = 0.
    for r in relax:
        for what, lo, hi, payload in r.records:
            lines = ([(s,a,b) for s,(a,b) in zip(('lower','upper'),payload)] if what == 'lines'
                     else [(s,0.,v) for s,v in zip(('lower','upper'),payload)])
            for side, a, b in lines:
                intercepts = [b] + ([a*(b/a)] if what == 'lines' and a != 0 else [])
                for intercept in intercepts:
                    report = audit_line(r.kind,lo,hi,a,intercept,side)
                    reports += 1
                    worst = max(worst, report.max_violation)
                    strict += report.max_violation > 0
                    if not report.passed:
                        violations.append(report.to_dict())
    return dict(cells=indices, endpoint_replay=endpoints, lines_checked=reports,
                strict_positive_line_excesses=strict, failed_lines=len(violations),
                maximum_line_excess=worst, failures=violations, scope='selected cells only',
                affects_bound=False)


def run_row(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    row = args.output.resolve()
    row.mkdir(parents=True, exist_ok=False)
    seen = install_read_guard(args.campaign, row)
    from crown_benchmark.models import load_checkpoint
    t = cpu()
    loaded = load_checkpoint(ROOT / 'model_weights' / CHECKPOINT_NAME)
    loading = cpu()-t
    t = cpu()
    if args.method == 'netbounds-e2':
        net, kernel = netbounds_setup(loaded.model)
        metadata = dict(method=args.method, native_entry='compute_second_order_bounds',
                        backend_root=str(NB_ROOT), activation_taylor_depths={'1':5,'2':4},
                        network_derivative_order=2, mathematical_patches=[], outward_rounded=False)
    else:
        authors, relaxations = partial_setup(loaded.model)
        metadata = dict(authors.metadata, method=args.method, mathematical_patches=[],
                        backprop_modes=['FULL_BACKPROP','COMPONENT_BACKPROP','COMPONENT_BACKPROP'],
                        relaxation_type='SINGLE_LINE', repair_variant=None, outward_rounded=False)
    setup = cpu()-t
    t, w = cpu(), time.perf_counter()
    lower, upper, centers, radii = grid(args.n)
    if args.method == 'netbounds-e2':
        base, variation = netbounds_call(net, kernel, centers, radii, args.coordinate)
        lbs, ubs = base-variation, base+variation
        cell_bound = base.abs()+variation
        bound = cell_bound.max().item()
    else:
        endpoints = [partial_call(loaded.model,args.coordinate,authors,relaxations,l,u)
                     for l,u in zip(lower,upper)]
        lbs = torch.tensor([p[0] for p in endpoints],dtype=torch.float64)
        ubs = torch.tensor([p[1] for p in endpoints],dtype=torch.float64)
        cell_bound = torch.maximum(-lbs,ubs)
        bound = cell_bound.max().item()
        base = variation = None
    bounding_cpu, bounding_wall = cpu()-t, time.perf_counter()-w
    arrays = {k:v.detach().numpy().copy() for k,v in dict(lower=lower,upper=upper,centers=centers,
                    lower_bound=lbs,upper_bound=ubs,absolute_bound=cell_bound).items()}
    if base is not None:
        arrays.update(base=base.numpy().copy(), variation=variation.numpy().copy())
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError('Nonfinite native output; no replacement returned')
    max_cell = int(cell_bound.argmax().item())
    result = dict(case_id=row.name, method=args.method, coordinate=args.coordinate,n=args.n,
        cells=args.n**2, cell_evaluations=args.n**2, native_calls=1 if base is not None else args.n**2,
        bound=bound,signed_lower=lbs.min().item(),signed_upper=ubs.max().item(),
        max_cell=max_cell,max_cell_center=centers[max_cell].tolist(),epsilon=radii.tolist(),
        bounding_cpu_seconds=bounding_cpu,bounding_wall_seconds=bounding_wall,
        checkpoint_loading_cpu_seconds=loading,backend_setup_cpu_seconds=setup,
        checkpoint=loaded.metadata,backend=metadata,pid=os.getpid(),finished_numerics_utc=now(),
        grid_convention='n cell centers per axis, n^2 closed cells, radius 1/(2*n)',
        stop_reason='complete_uniform_grid',cpu_ceiling=None,
        timing='process user+system; grid construction, native bounds and max reduction only; '
               'no imports, model loading, backend/relaxation setup, I/O, independent checks or replay',
        precision='float64; native parameters copied without value changes; no outward rounding')
    if base is not None:
        result.update(max_cell_base=float(base[max_cell]),max_cell_variation=float(variation[max_cell]))
    np.savez(row/'cells.npz',**arrays)
    save(row/'result.json',result)
    frozen = {p.name:digest(p) for p in (row/'result.json',row/'cells.npz')}
    t = cpu()
    diagnostic = diagnose(loaded.model,args.coordinate,lower,upper,centers,
                          arrays['lower_bound'],arrays['upper_bound'],arrays.get('base'),arrays.get('variation'))
    indices = sorted(set([0,max_cell,args.n**2-1]))
    if base is None:
        diagnostic['native_replay'] = replay_partial(loaded.model,args.coordinate,authors,indices,lower,upper,
                                                     arrays['lower_bound'],arrays['upper_bound'])
    else:
        replay_base,replay_var = netbounds_call(net,kernel,centers[indices],radii,args.coordinate)
        diagnostic['native_replay'] = dict(cells=indices,
            base_max_abs_gap=float(abs(replay_base.numpy()-arrays['base'][indices]).max()),
            variation_max_abs_gap=float(abs(replay_var.numpy()-arrays['variation'][indices]).max()))
    diagnostic['diagnostics_cpu_seconds'] = cpu()-t
    diagnostic['numerical_artifacts_unchanged'] = all(digest(row/name)==h for name,h in frozen.items())
    assert diagnostic['numerical_artifacts_unchanged']
    diagnostic['numerical_artifact_sha256'] = frozen
    diagnostic['campaign_read_paths'] = sorted(seen)
    save(row/'diagnostics.json',diagnostic)
    print(json.dumps({k:result[k] for k in ('case_id','bound','bounding_cpu_seconds')}) ,flush=True)


def run_reference(args):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    row=args.output.resolve()
    row.mkdir(parents=True,exist_ok=False)
    seen=install_read_guard(args.campaign,row)
    from crown_benchmark.models import load_checkpoint, _nested_autograd_second
    loaded=load_checkpoint(ROOT/'model_weights'/CHECKPOINT_NAME)
    t=cpu()
    axis=torch.linspace(0,1,257,dtype=torch.float64)
    pts=torch.cartesian_prod(axis,axis)
    vals=torch.cat([_nested_autograd_second(loaded.model,p,args.coordinate)
                    for p in pts.split(2048)]).flatten()
    k=int(vals.abs().argmax())
    result=dict(coordinate=args.coordinate,point_evaluations=len(pts),grid_nodes_per_axis=257,
                observed_maximum=vals.abs().max().item(),signed_value=vals[k].item(),witness=pts[k].tolist(),
                reference_cpu_seconds=cpu()-t,reference_type='independent nested autograd on raw forward',
                is_upper_bound=False,checkpoint_sha256=loaded.metadata['sha256'],pid=os.getpid())
    np.savez(row/'reference.npz',points=pts.numpy(),values=vals.numpy())
    save(row/'result.json',result)
    print(json.dumps(result),flush=True)

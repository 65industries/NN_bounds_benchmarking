"""Measured Q1 supremum worker; portable coordinator is heat_reproduce.py."""
import importlib.util
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
METHOD = "netbounds-q1-expansion"

def helpers():
    path=Path(__file__).with_name('heat_second_order_e_comparison.py')
    spec=importlib.util.spec_from_file_location('e2_shared',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run_row(args):
    import numpy as np
    import torch
    h=helpers()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    row=args.output.resolve();row.mkdir(parents=True,exist_ok=False)
    seen=h.install_read_guard(args.campaign,row)
    from crown_benchmark.models import load_checkpoint
    from independent_benchmark.cpu_clock import numerical_clock
    from independent_benchmark.second_derivative import run_netbounds
    from netbounds_benchmark.uniform_q1 import RawQ1Integrator
    t=h.cpu();loaded=load_checkpoint(ROOT/'model_weights'/h.CHECKPOINT_NAME);loading=h.cpu()-t
    with numerical_clock() as clock:
        numerical,timing,metadata=run_netbounds(loaded.model,args.coordinate,args.n,clock)
    jets=numerical.pop('arrays')
    lower,upper,centers,eps=h.grid(args.n)
    e=eps.numpy()
    affine=(abs(jets['gradient'])*e).sum(-1)
    remainder=.5*np.einsum('i,nij,j->n',e,jets['hessian_sup'],e)
    radius=affine+remainder
    arrays=dict(**jets,lower=lower.numpy(),upper=upper.numpy(),centers=centers.numpy(),
                affine_radius=affine,remainder_radius=remainder,variation=radius,
                lower_bound=jets['base']-radius,upper_bound=jets['base']+radius,
                absolute_bound=abs(jets['base'])+radius)
    assert all(np.isfinite(x).all() for x in arrays.values())
    assert numerical['bound']==float(arrays['absolute_bound'].max())
    k=int(arrays['absolute_bound'].argmax())
    result=dict(case_id=row.name,method=METHOD,coordinate=args.coordinate,n=args.n,
        cells=numerical['cells'],cell_evaluations=numerical['counters']['cells_evaluated'],
        native_calls=numerical['counters']['cell_batch_calls'],batch_size=256,
        bound=numerical['bound'],signed_lower=numerical['signed_lower'],signed_upper=numerical['signed_upper'],
        epsilon=numerical['epsilon'],coverage=numerical['coverage'],
        bounding_cpu_seconds=timing['computation_cpu_seconds'],
        bounding_wall_seconds=timing['computation_wall_seconds'],
        inline_diagnostics_cpu_seconds=timing['harness_diagnostics_cpu_seconds'],
        checkpoint_loading_cpu_seconds=loading,backend_setup_cpu_seconds=timing['adapter_setup_cpu_seconds'],
        checkpoint=loaded.metadata,backend=numerical['backend'],pid=os.getpid(),finished_numerics_utc=h.now(),
        max_cell=k,max_cell_center=centers[k].tolist(),max_cell_base=float(jets['base'][k]),
        max_cell_affine_radius=float(affine[k]),max_cell_remainder_radius=float(remainder[k]),
        max_cell_variation=float(radius[k]),stop_reason=numerical['stop_reason'],cpu_ceiling=None,
        timing='Existing run_netbounds compute timer: full fixed-grid jets, native fourth-order propagation, '
               'array collection and signed supremum reduction. Setup/loading/I/O/post-run diagnostics excluded; '
               'inline midpoint consistency and call-count diagnostics excluded by the existing CPU ledger. '
               'Wall time includes those inline diagnostics.',
        precision='CPU float64, one thread, no outward rounding',
        grid_convention='n cell centers per axis, n^2 closed cells, radius 1/(2*n)')
    np.savez(row/'cells.npz',**arrays);h.save(row/'result.json',result)
    frozen={name:h.digest(row/name) for name in ('cells.npz','result.json')}
    t=h.cpu()
    diagnostic=h.diagnose(loaded.model,args.coordinate,lower,upper,centers,arrays['lower_bound'],
                          arrays['upper_bound'],arrays['base'],arrays['variation'])
    indices=sorted(set([0,k,args.n**2-1]))
    replay=RawQ1Integrator(loaded.model,'second-diagonal',args.coordinate,moments=False)
    block=replay.cell_batch(centers[indices],eps)
    diagnostic['native_replay']=dict(cells=indices,fields={key:dict(
        max_abs_gap=float(np.max(abs(value.numpy()-jets[key][indices]))),
        max_scaled_gap=float(np.max(abs(value.numpy()-jets[key][indices])/(1+abs(jets[key][indices])))))
        for key,value in block.items()})
    diagnostic['diagnostics_cpu_seconds']=h.cpu()-t
    diagnostic['numerical_artifacts_unchanged']=all(h.digest(row/name)==digest for name,digest in frozen.items())
    assert diagnostic['numerical_artifacts_unchanged']
    diagnostic['numerical_artifact_sha256']=frozen;diagnostic['campaign_read_paths']=sorted(seen)
    h.save(row/'diagnostics.json',diagnostic)
    print(json.dumps({key:result[key] for key in ('case_id','bound','bounding_cpu_seconds')}),flush=True)

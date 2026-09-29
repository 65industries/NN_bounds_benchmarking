"""Fixed-grid functions extracted verbatim from the measured harness."""
from fractions import Fraction
import math
import time
import numpy as np
import torch
from crown_benchmark.types import BoundFailure
from .uniform_q1 import SUM_FIELDS, ARRAY_FIELDS

def grid_centers(indices, n, d, *, device=None):
    """Row-major cell centres of the uniform n^d grid: index -> (i_0,...,i_{d-1}), i_0 slowest."""
    indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
    coords = []
    for axis in range(d):
        coords.append((indices // n**(d-1-axis)) % n)
    return (torch.stack(coords, dim=-1).to(torch.float64)+.5)/n


def evaluate_fixed_grid(kernel, *, n, batch_size=256, jets_only=False, device=None,
                        cpu_clock=time.process_time, wall_clock=time.perf_counter):
    """Evaluate each cell once on exactly the specified n^d uniform grid (d = kernel input dimension)."""
    if isinstance(n,bool) or not isinstance(n,int) or n<1 or n & (n-1):
        raise ValueError('n must be a positive power of two')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('batch_size must be a positive integer')
    d=getattr(kernel,'input_dim',2); total=n**d
    device=torch.device(getattr(kernel,'device','cpu') if device is None else device)
    if device.type not in ('cpu','cuda'): raise ValueError('Require a CPU or CUDA grid device')
    if device.type=='cuda': torch.cuda.synchronize(device)
    cpu,wall=cpu_clock(),wall_clock()
    eps=torch.full((d,),1/(2*n),dtype=torch.float64,device=device)
    fields = ('base','gradient','hessian_sup') if jets_only else ARRAY_FIELDS
    chunks={key:[] for key in fields}
    processed=0
    for first in range(0,total,batch_size):
        indices=torch.arange(first,min(first+batch_size,total),dtype=torch.int64,device=device)
        centers=grid_centers(indices,n,d)
        block=kernel.cell_batch(centers,eps)
        for key in fields:
            values=block[key].detach().cpu().numpy().copy()
            shape=(len(indices),d) if key=='gradient' else (len(indices),d,d) if key=='hessian_sup' else (len(indices),)
            if values.shape!=shape or values.dtype!=np.float64 or not np.isfinite(values).all():
                raise BoundFailure(f'Invalid fixed-grid Q1 output: {key}')
            if key in SUM_FIELDS or key=='hessian_sup':
                if (values<0).any(): raise BoundFailure(f'Negative fixed-grid Q1 output: {key}')
            chunks[key].append(values)
        processed+=len(indices)
    if processed!=total: raise BoundFailure('Incomplete fixed grid')
    arrays={key:np.concatenate(parts,axis=0) for key,parts in chunks.items()}
    sums={key:math.fsum(arrays[key].tolist()) for key in (() if jets_only else SUM_FIELDS)}
    if not all(math.isfinite(v) and v>=0 for v in sums.values()):
        raise BoundFailure('Invalid fixed-grid reduction')
    if not jets_only and not math.isclose(sums['upper_square'],math.fsum(sums[k] for k in ('affine_square','cross_error','remainder_square')),rel_tol=1e-14,abs_tol=0):
        raise BoundFailure('Fixed-grid component sum mismatch')
    result=dict(n=n,cells=total,input_dim=d,epsilon=eps.tolist(),complete=True,arrays=arrays,sums=sums,
        l2_g_upper=None if jets_only else math.sqrt(sums['upper_square']),jets_only=jets_only,batch_size=batch_size,cpu_limit=None,
        processed_cells_total=processed,discarded_cells=0,stop_reason='fixed_grid_complete',
        coverage=dict(complete=True,nonoverlapping=True,uniform=True,adaptive_splits=0,
            index_start=0,index_stop=total,index_order=f'row major over {d} axes: axis k = (index // n**({d}-1-k)) % n',
            cell_volume_exact=str(Fraction(1,total)),volume_sum_exact='1',
            proof='all Cartesian grid indices exactly once; no coarser or finer grid evaluated'),
        selection='single prescribed complete uniform grid')
    if device.type=='cuda': torch.cuda.synchronize(device)
    result.update(cpu_seconds=cpu_clock()-cpu,wall_seconds=wall_clock()-wall)
    return result

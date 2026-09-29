"""Uniform raw-network Q1 L2 quadrature with upstream NetBounds derivative bounds.

Integration is the literal coordinatewise moment formula in pinned NetBounds-dev
tex/IMA/section2.tex:174-275, not a PDE residual or a constant-supremum integral.
Q = integral P^2; E = 2 integral A D + integral D^2;
P=a+b.z, A=|a|+sum |b_i||z_i|, D=.5 sum H_ij |z_i||z_j|.
All arithmetic is ordinary float64, not outward-rounded interval arithmetic.
"""
from fractions import Fraction
import math
import time

import numpy as np
import torch
from crown_benchmark.types import BoundFailure
from .adapter import NetBoundsQ1Adapter


def q1_moment_cell(base, gradient, hessian_sup, radii):
    """Per-cell squared-integral contributions; final sqrt is taken globally."""
    if radii.ndim != 1 or not 1 <= len(radii) <= 3:
        raise ValueError('Require one common radius vector, dimension 1..3')
    d, count = len(radii), len(base)
    if base.shape != (count,) or gradient.shape != (count,d) or hessian_sup.shape != (count,d,d):
        raise ValueError('Bad cell-jet shapes; implicit broadcasting is prohibited')
    for t in (base, gradient, hessian_sup, radii):
        if (t.dtype != torch.float64 or t.device.type not in ('cpu', 'cuda')
                or t.device != base.device or not bool(torch.isfinite(t).all())):
            raise ValueError('Require finite float64 tensors on the same CPU or CUDA device')
    if not bool((radii > 0).all()) or not bool((hessian_sup >= 0).all()):
        raise ValueError('Require positive radii and nonnegative Hessian envelopes')
    if not torch.equal(hessian_sup, hessian_sup.transpose(-1,-2)):
        raise ValueError('Hessian envelopes must be symmetric')
    volume = (2*radii).prod()
    if not bool(torch.isfinite(volume)) or volume <= 0:
        raise ValueError('Invalid cell volume')
    moments = {}
    def moment(indices):
        counts = tuple(indices.count(i) for i in range(d))
        if counts not in moments:
            moments[counts] = volume * math.prod(radii[i]**k / (k+1) for i,k in enumerate(counts))
        return moments[counts]
    pairs = [(i,j) for i in range(d) for j in range(i,d)]
    # The off-diagonal entries occur twice in .5 sum_{i,j} H_ij |z_i z_j|.
    terms = [(pair, hessian_sup[:,pair[0],pair[1]] * (.5 if pair[0]==pair[1] else 1.)) for pair in pairs]
    cross = torch.zeros_like(base)
    remainder = torch.zeros_like(base)
    for pair, coefficient in terms:
        affine_moment = base.abs()*moment(pair)
        for i in range(d):
            affine_moment = affine_moment + gradient[:,i].abs()*moment((*pair,i))
        cross = cross + 2*coefficient*affine_moment
        for pair2, coefficient2 in terms:
            remainder = remainder + coefficient*coefficient2*moment((*pair,*pair2))
    midpoint = volume*base.square()
    affine = midpoint + volume*(gradient.square()*radii.square()/3).sum(-1)
    upper = affine + cross + remainder
    out = dict(midpoint_square=midpoint, affine_square=affine, cross_error=cross,
               remainder_square=remainder, upper_square=upper)
    if any(not bool(torch.isfinite(x).all()) or bool((x<0).any()) for x in out.values()):
        raise BoundFailure('Nonfinite/negative Q1 moment contribution')
    return out


class RawQ1Integrator:
    """Batched raw scalar target; reuses unchanged upstream derivative recurrences."""
    def __init__(self, model, target='value', coordinate=None, *, moments=True, device='cpu'):
        self.moments = moments
        self.adapter = NetBoundsQ1Adapter(model, target, coordinate, device=device)
        self.device = self.adapter.device
        if not 1 <= self.adapter.input_dim <= 3:
            raise ValueError('This benchmark is scoped to checkpoints with 1..3 inputs')
        self.input_dim = self.adapter.input_dim
        self.metadata = dict(self.adapter.metadata,
            method='netbounds-q1-uniform',
            adapter='raw-MLP view; manuscript Q1 moment integration wrapper',
            q1_source='tex/IMA/section2.tex:174-275; eq:cell-quadrature-values, eq:q1-cell-moment-correction',
            integration='Q1 plus coordinatewise moment correction E1; sqrt(sum(Q1+E1))',
            remainder='D(z)=0.5 sum_qr H_qr |z_q||z_r|, H_qr=abs(D_qr g(c))+variation_qr',
            grid_policy='complete uniform grids only; no adaptive leaves; epsilon=1/(2n) per axis')
        self.metadata.pop('signed_range',None)
        self.setup_seconds = self.adapter.setup_seconds

    @torch.no_grad()
    def cell_batch(self, centers, radii):
        d = self.input_dim
        if centers.ndim != 2 or centers.shape[1] != d or len(centers)==0:
            raise ValueError(f'Require nonempty N x {d} midpoint batch')
        if centers.dtype != torch.float64 or centers.device != self.device or not bool(torch.isfinite(centers).all()):
            raise ValueError('Require finite float64 midpoints on the kernel device')
        if (radii.dtype != torch.float64 or radii.device != self.device or radii.shape != (d,)
                or not bool((radii>0).all()) or not bool(torch.isfinite(radii).all())):
            raise ValueError('Invalid common half-width vector')
        if bool((centers-radii < 0).any()) or bool((centers+radii > 1).any()):
            raise ValueError('Cells must lie in the raw unit-cube input domain')
        a = self.adapter
        if not a.net.hidden_layers:
            base = a.evaluate(centers).reshape(-1)
            gradient = (a.net.output_layer.weight.expand(len(centers),-1) if a.target=='value' else torch.zeros_like(centers))
            hessian = torch.zeros((len(centers),d,d),dtype=torch.float64,device=self.device)
        else:
            jet = a.derivative_bounds(centers, radii)
            if a.target == 'value':
                raw = a.modules['derivative_centers.common'].compute_raw_centers(a.net, centers, order=1)
                base, gradient = raw.value.reshape(-1), raw.gradient
                columns = [jet.pairs.index(p) for p in a.pairs]
                envelopes = jet.base_hessian[:,columns].abs()+jet.second_derivative_bounds[:,columns]
            else:
                pair = (a.coordinate,a.coordinate)
                base = jet.base_hessian[:,jet.pairs.index(pair)]
                gradient = torch.stack([jet.base_third_derivatives[:,jet.triples.index(tuple(sorted((*pair,i))))] for i in range(d)],dim=-1)
                columns = [jet.quartets.index(tuple(sorted((*pair,i,j)))) for i,j in a.pairs]
                envelopes = jet.base_fourth_derivatives[:,columns].abs()+jet.fourth_derivative_bounds[:,columns]
            hessian = torch.empty((len(centers),d,d),dtype=torch.float64,device=self.device)
            for k,(i,j) in enumerate(a.pairs):
                hessian[:,i,j] = hessian[:,j,i] = envelopes[:,k]
        from independent_benchmark.cpu_clock import excluded
        with excluded():
            independent = a.evaluate(centers).reshape(-1)
            if not torch.allclose(base,independent,atol=1e-12,rtol=1e-10):
                raise BoundFailure('Upstream midpoint jet disagrees with independent target evaluator')
        jets = dict(base=base, gradient=gradient, hessian_sup=hessian)
        if not self.moments:
            return jets
        return dict(q1_moment_cell(base,gradient,hessian,radii), **jets)


SUM_FIELDS = ('midpoint_square','affine_square','cross_error','remainder_square','upper_square')
ARRAY_FIELDS = (*SUM_FIELDS,'base','gradient','hessian_sup')


def uniform_grid_search(kernel, *, cpu_limit=60., batch_size=256, max_n=4096,
                        cpu_clock=time.process_time, wall_clock=time.perf_counter):
    """Evaluate n=1,2,4,... full grids, keeping the finest completed one.

All coarser grids and unfinished next-grid batches are charged. A partial grid
never contributes to a global norm. One synchronous batch may overrun the cap.
Setup, final artifact I/O and independent audit are separately accounted.
"""
    if not math.isfinite(cpu_limit) or cpu_limit<=0:
        raise ValueError('CPU limit must be positive and finite')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError('Invalid batch size')
    if isinstance(max_n,bool) or not isinstance(max_n,int) or max_n<1 or max_n & (max_n-1):
        raise ValueError('max_n must be a positive power of two')
    start, wall_start = cpu_clock(), wall_clock()
    levels, selected = [], None
    processed_total = 0
    n = 1
    while n<=max_n:
        if cpu_clock()-start>=cpu_limit:
            break
        level_start, level_wall = cpu_clock(), wall_clock()
        chunks = {key:[] for key in ARRAY_FIELDS}
        eps = torch.full((2,),1/(2*n),dtype=torch.float64)
        processed = 0
        for first in range(0,n*n,batch_size):
            if cpu_clock()-start>=cpu_limit:
                break
            indices = torch.arange(first,min(first+batch_size,n*n),dtype=torch.int64)
            centers = (torch.stack((indices//n,indices%n),dim=-1).to(torch.float64)+.5)/n
            block = kernel.cell_batch(centers,eps)
            for key in ARRAY_FIELDS:
                value = block[key].detach().cpu().numpy().copy()
                if len(value)!=len(indices) or not np.isfinite(value).all():
                    raise BoundFailure('Invalid Q1 batch output')
                chunks[key].append(value)
            processed += len(indices)
            processed_total += len(indices)
        complete = processed==n*n
        level = dict(n=n,cells=n*n,processed_cells=processed,complete=complete)
        if complete:
            arrays = {key:np.concatenate(values,axis=0) for key,values in chunks.items()}
            sums = {key:math.fsum(arrays[key].tolist()) for key in SUM_FIELDS}
            if not all(math.isfinite(x) and x>=0 for x in sums.values()):
                raise BoundFailure('Invalid global Q1 sum')
            if not math.isclose(sums['upper_square'],math.fsum(sums[k] for k in ('affine_square','cross_error','remainder_square')),rel_tol=1e-14,abs_tol=0):
                raise BoundFailure('Q1 components disagree after reduction')
            level.update(sums=sums,l2_g_upper=math.sqrt(sums['upper_square']))
            selected = dict(n=n,cells=n*n,epsilon=eps.tolist(),complete=True,
                sums=sums,l2_g_upper=level['l2_g_upper'],arrays=arrays,
                coverage=dict(complete=True,nonoverlapping=True,uniform=True,adaptive_splits=0,
                    index_start=0,index_stop=n*n,index_order='row major: (index//n,index%n)',
                    cell_volume_exact=str(Fraction(1,n*n)),volume_sum_exact='1',
                    proof='all row-major Cartesian grid indices exactly once; uniform dyadic endpoints tile [0,1]^2'))
        level.update(cpu_seconds=cpu_clock()-level_start,wall_seconds=wall_clock()-level_wall)
        levels.append(level)
        if not complete:
            break
        n *= 2
    if selected is None:
        raise BoundFailure('CPU ceiling reached without a complete uniform grid')
    selected.update(levels=levels,cpu_seconds=cpu_clock()-start,wall_seconds=wall_clock()-wall_start,
        cpu_limit=cpu_limit,batch_size=batch_size,processed_cells_total=processed_total,
        discarded_cells=sum(l['processed_cells'] for l in levels if not l['complete']),
        completed_grid_cells_total=sum(l['cells'] for l in levels if l['complete']),
        stop_reason='cpu_limit' if n<=max_n else 'max_n',
        selection='finest completed uniform dyadic grid, not a mixture of levels')
    return selected

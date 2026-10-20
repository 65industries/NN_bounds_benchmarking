"""Native derivative readout plus harness-owned signed PDE/affine integration.

No native recurrence, relaxation, tolerance or returned coefficient is repaired.
The same value solution is shared by Fx/Fxx and Ft *inside one cell/row* only.
"""
from pathlib import Path
import itertools
import math
from typing import Any
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'pde_residual_benchmark/inputs/heat1-l2-w128.pt'
CHECKPOINT_SHA = '514531c1d38d138b053de85842e4c4a92de0bc82a629d448437ad1c8b5028ba6'
GRIDS = (8, 16, 32, 64)
ALPHA = .1


def load_heat1():
    from crown_benchmark.models import load_checkpoint
    loaded = load_checkpoint(ROOT / CHECKPOINT)
    if loaded.metadata['sha256'] != CHECKPOINT_SHA:
        raise ValueError('This case requires the exact PDEbenchmarking Heat1 checkpoint')
    return loaded


def grid_boxes(n):
    if type(n) is not int or n < 1:
        raise ValueError('Positive integer grid required')
    lower = np.asarray(list(itertools.product(range(n), repeat=2)), dtype=np.float64)/n
    return lower, lower + 1./n


def concretize(pair, lower, upper):
    corners = itertools.product(*zip(lower, upper))
    points = list(corners)
    def value(A, c, p): return math.fsum((c, A[0]*p[0], A[1]*p[1]))
    return (min(value(pair['A_lower'], pair['c_lower'], p) for p in points),
            max(value(pair['A_upper'], pair['c_upper'], p) for p in points))


def combine_residual(ft, xx, lower, upper):
    # alpha > 0: subtraction swaps the xx lower and upper sandwiches.
    result: dict[str, Any] = dict(A_lower=[a-ALPHA*b for a,b in zip(ft['A_lower'],xx['A_upper'])],
                  c_lower=ft['c_lower']-ALPHA*xx['c_upper'],
                  A_upper=[a-ALPHA*b for a,b in zip(ft['A_upper'],xx['A_lower'])],
                  c_upper=ft['c_upper']-ALPHA*xx['c_lower'])
    result['lower_bound'],result['upper_bound'] = concretize(result,lower,upper)
    return result


def scalar_square(lower, upper, volume):
    if not all(math.isfinite(v) for v in (lower,upper,volume)) or lower>upper or volume<=0:
        raise ValueError('Invalid scalar enclosure')
    low = 0. if lower<=0<=upper else min(lower*lower,upper*upper)
    return volume*low, volume*max(lower*lower,upper*upper)


class HeatResidualCrown:
    def __init__(self, model, audit=False):
        from crown_benchmark.partial import load_authors, PartialCrownAdapter
        self.model, self.calls = model, 0
        self.audit_adapter = PartialCrownAdapter(model,'second-diagonal',0) if audit else None
        if self.audit_adapter is not None:
            self.authors = self.audit_adapter._authors
            self.relaxations = self.audit_adapter._relaxations
        else:
            self.authors = load_authors()
            single = self.authors.activations.ActivationRelaxationType.SINGLE_LINE
            t = self.authors.tanh
            self.relaxations = [cls(single) for cls in
                (t.TanhRelaxation,t.TanhDerivativeRelaxation,t.TanhSecondDerivativeRelaxation)]
        self.metadata = dict(self.authors.metadata, value_backprop='FULL_BACKPROP',
            derivative_backprop='COMPONENT_BACKPROP', relaxation='SINGLE_LINE',
            mathematical_patches=[], repair_variant=None,
            operator='F_t - 0.1 F_xx', coordinates=['x','t'], shared_value_solution_per_cell=True)

    @staticmethod
    def _read(obj, prefix):
        return dict(A_lower=getattr(obj,prefix+'_crown_coefficients_lbs')[-1].flatten().tolist(),
                    c_lower=float(getattr(obj,prefix+'_crown_constants_lbs')[-1].flatten()[0]),
                    A_upper=getattr(obj,prefix+'_crown_coefficients_ubs')[-1].flatten().tolist(),
                    c_upper=float(getattr(obj,prefix+'_crown_constants_ubs')[-1].flatten()[0]),
                    lower_bound=float(obj.lower_bounds[-1].flatten()[0]),
                    upper_bound=float(obj.upper_bounds[-1].flatten()[0]))

    def bound(self, lower, upper, journal):
        self.calls += 1
        journal.update(lower=list(lower),upper=list(upper),completed_stages=[])
        c=self.authors.crown
        solution=c.CROWNPINNSolution(list(self.model),self.relaxations[0],device='cpu')
        solution.domain_bounds=torch.tensor([lower,upper],dtype=torch.float64)
        solution.compute_bounds(debug=False,backprop_mode=c.BackpropMode.FULL_BACKPROP)
        journal['completed_stages'].append('F')
        x=c.CROWNPINNPartialDerivative(solution,0,self.relaxations[1])
        x.compute_bounds(debug=False,backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
        journal['completed_stages'].append('Fx')
        xx=c.CROWNPINNSecondPartialDerivative(x,0,self.relaxations[2])
        xx.compute_bounds(debug=False,backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
        journal['Fxx']=self._read(xx,'u_dxixi')
        journal['completed_stages'].append('Fxx')
        t=c.CROWNPINNPartialDerivative(solution,1,self.relaxations[1])
        t.compute_bounds(debug=False,backprop_mode=c.BackpropMode.COMPONENT_BACKPROP)
        journal['Ft']=self._read(t,'u_dxi')
        journal['completed_stages'].append('Ft')
        journal['residual']=combine_residual(journal['Ft'],journal['Fxx'],lower,upper)
        return journal['residual']


def residual_components(model, points):
    with torch.enable_grad():
        x=points.detach().clone().requires_grad_(True)
        f=model(x)
        first=torch.autograd.grad(f.sum(),x,create_graph=True)[0]
        second=torch.autograd.grad(first[:,0].sum(),x)[0][:,0]
        return first[:,1].detach(),second.detach()


def residual_values(model, points):
    t,xx=residual_components(model,points)
    return t-ALPHA*xx


def gauss_reference(model, degree):
    nodes,weights=np.polynomial.legendre.leggauss(degree)
    nodes=(nodes+1.)/2.;weights=weights/2.
    points=np.asarray(list(itertools.product(nodes,repeat=2)))
    w=np.outer(weights,weights).reshape(-1)
    values=[]
    for start in range(0,len(points),256):
        r=residual_values(model,torch.from_numpy(points[start:start+256])).numpy()
        values.extend((r*r*w[start:start+256]).tolist())
    return math.fsum(values)

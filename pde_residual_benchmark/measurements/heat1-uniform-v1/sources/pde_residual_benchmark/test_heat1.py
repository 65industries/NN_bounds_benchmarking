"""Heat1 extension: analytic fixtures, native smoke and failure-contract tests."""
import copy
import math
import numpy as np
import pytest
import torch
from pde_residual_benchmark.core import (HeatResidualCrown, combine_residual,
    concretize, grid_boxes, load_heat1, residual_values, scalar_square)
from independent_benchmark.minimal_affine_l2_native import cell_integrals


def pair(A, c, B=None, d=None):
    return dict(A_lower=A, c_lower=c, A_upper=A if B is None else B,
                c_upper=c if d is None else d, lower_bound=0., upper_bound=0.)


def test_negative_pde_coefficient_reverses_second_derivative_pair():
    ft=pair([2.,3.],4.,[5.,6.],7.)
    xx=pair([11.,13.],17.,[19.,23.],29.)
    r=combine_residual(ft,xx,[0.,0.],[1.,1.])
    np.testing.assert_allclose(r['A_lower'],[.1,.7],atol=2e-15)
    np.testing.assert_allclose(r['A_upper'],[3.9,4.7],atol=2e-15)
    assert r['c_lower']==4.-.1*29.
    assert r['c_upper']==7.-.1*17.
    assert r['lower_bound']==concretize(r,[0.,0.],[1.,1.])[0]


@pytest.mark.parametrize('n',[8,16,32,64])
def test_exact_complete_uniform_geometry(n):
    lo,hi=grid_boxes(n)
    assert lo.shape==hi.shape==(n*n,2)
    assert len(set(map(tuple,lo)))==n*n
    np.testing.assert_array_equal(hi-lo,np.full((n*n,2),1/n))
    assert math.fsum(np.prod(hi-lo,axis=1))==1.
    np.testing.assert_array_equal(lo[:,0],np.repeat(np.arange(n)/n,n))
    np.testing.assert_array_equal(lo[:,1],np.tile(np.arange(n)/n,n))


def test_affine_residual_fixture_has_exact_square_integral():
    r=combine_residual(pair([0.,1.],1.),pair([0.,0.],2.),[0.,0.],[1.,1.])
    a=cell_integrals(r,[0.,0.],[1.,1.])
    assert a['lower_square']==a['upper_square']
    assert a['lower_square']==pytest.approx(.8**2+.8+1/3,abs=1e-15)


@pytest.mark.parametrize('L,U,lo,hi',[(-2.,3.,0.,9.),(2.,3.,4.,9.),(-3.,-2.,4.,9.),(0.,0.,0.,0.)])
def test_scalar_lift_is_diagnostic_only(L,U,lo,hi):
    assert scalar_square(L,U,1.)==(lo,hi)


def test_original_forward_autograd_uses_time_last_and_plain_heat():
    class Polynomial(torch.nn.Module):
        def forward(self,x):return (x[:,0]**3+2*x[:,1]**2)[:,None]
    z=torch.tensor([[.2,.3],[.6,.8]],dtype=torch.float64)
    r=residual_values(Polynomial(),z)
    np.testing.assert_allclose(r.numpy(),4*z[:,1].numpy()-.6*z[:,0].numpy(),rtol=0,atol=5e-16)


def test_exact_pde_checkpoint_is_not_old_nn_checkpoint():
    loaded=load_heat1()
    assert loaded.metadata['sha256']=='514531c1d38d138b053de85842e4c4a92de0bc82a629d448437ad1c8b5028ba6'
    assert all(p.dtype==torch.float64 and not p.requires_grad for p in loaded.model.parameters())


def test_native_smoke_reads_both_affine_objects_and_preserves_raw():
    torch.set_num_threads(1)
    model=load_heat1().model
    kernel=HeatResidualCrown(model)
    raw={}
    r=kernel.bound([.25,.5],[.265625,.515625],raw)
    assert set(raw)>={'Ft','Fxx','residual','completed_stages','lower','upper'}
    assert raw['completed_stages']==['F','Fx','Fxx','Ft']
    before=copy.deepcopy(raw)
    v=cell_integrals(r,raw['lower'],raw['upper'])
    assert raw==before and v['lower_square']<=v['upper_square']
    z=torch.tensor([[.25,.5],[.265625,.515625],[.2578125,.5078125]],dtype=torch.float64)
    y=residual_values(model,z).numpy()
    l=np.asarray(r['A_lower'])@z.numpy().T+r['c_lower']
    u=np.asarray(r['A_upper'])@z.numpy().T+r['c_upper']
    assert np.all(l<=y+1e-10) and np.all(y<=u+1e-10)
    assert kernel.calls==1


def test_new_matrix_is_separate_and_complete():
    from pde_residual_benchmark.campaign import case_matrix
    matrix=case_matrix()
    assert [c['n'] for c in matrix if 'n' in c]==[8,16,32,64]
    assert sum(c.get('cells',0) for c in matrix)==5440
    assert len(matrix)==5 and len({c['id'] for c in matrix})==5
    assert len(case_matrix(True))==2
    from reproducers.heat_reproduce import cases
    assert len(cases())==60


def test_cross_row_and_retained_reads_are_denied_in_fresh_process(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    campaign=tmp_path/'campaign';own=campaign/'rows/one';other=campaign/'rows/two'
    original=tmp_path/'original'
    for d in (own,other,original/'results',original/'pde_residual_benchmark/inputs/retained_heat1'):
        d.mkdir(parents=True)
        (d/'payload.json').write_text('{}')
    script='''
import json,sys
from pathlib import Path
from pde_residual_benchmark.campaign import install_read_guard
c,o,r=map(Path,sys.argv[1:4]);install_read_guard(c,o,r)
assert (o/'payload.json').read_text()=='{}'
answers=[]
for p in [c/'rows/two/payload.json',r/'results/payload.json',r/'pde_residual_benchmark/inputs/retained_heat1/payload.json']:
 try:p.read_text()
 except PermissionError:answers.append(True)
 else:answers.append(False)
print(json.dumps(answers))
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(campaign),str(own),str(original)],
        capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==[True,True,True]

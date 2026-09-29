"""Contract tests for the direct manuscript E2 benchmark (not Q1)."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from crown_benchmark.models import load_checkpoint, _nested_autograd_second
from reproducers.heat_second_order_e_comparison import (
    ROOT, CHECKPOINT_NAME, grid, netbounds_setup, netbounds_call,
    partial_setup, partial_call,
)


@pytest.mark.parametrize('n', [8, 16, 32])
def test_exact_closed_square_cover(n):
    lower, upper, centers, radii = grid(n)
    assert lower.shape == upper.shape == centers.shape == (n*n, 2)
    assert torch.equal(centers-radii, lower)
    assert torch.equal(centers+radii, upper)
    assert torch.unique(centers,dim=0).shape[0] == n*n
    assert torch.equal(lower.min(0).values,torch.zeros(2,dtype=torch.float64))
    assert torch.equal(upper.max(0).values,torch.ones(2,dtype=torch.float64))
    assert Fraction(1,n)**2 * n*n == 1
    for i in (0,1):
        assert torch.equal(torch.unique(lower[:,i]),torch.arange(n,dtype=torch.float64)/n)
        assert torch.equal(torch.unique(upper[:,i]),torch.arange(1,n+1,dtype=torch.float64)/n)


@pytest.mark.parametrize('coordinate',[0,1])
def test_paper_product_formula_center_and_no_fourth_order(monkeypatch, coordinate):
    torch.set_num_threads(1)
    loaded=load_checkpoint(ROOT/'model_weights'/CHECKPOINT_NAME)
    net,kernel=netbounds_setup(loaded.model)
    import derivative_bounds.fourth_order as fourth
    def forbidden(*a,**k):
        raise AssertionError('E2 must not invoke the Q1 fourth-order kernel')
    monkeypatch.setattr(fourth,'compute_up_to_fourth_order_bounds',forbidden)
    _,_,centers,radii=grid(8)
    centers=centers[[0,15,31,63]]
    base,var=netbounds_call(net,kernel,centers,radii,coordinate)
    oracle=_nested_autograd_second(loaded.model,centers,coordinate).flatten()
    torch.testing.assert_close(base,oracle,atol=1e-12,rtol=1e-12)
    result=kernel(net,centers,radii,n_taylor=None,multi_indices=[(coordinate,coordinate)],return_cache=True)
    # Independently specialize the paper's product-of-envelopes formula,
    # rather than reusing the production expansion of nonnegative terms.
    for k in range(len(net.hidden_layers)):
        z=result.z[k]
        a=net.AB.activation_m_prime_stable(z,1)
        b=net.AB.activation_m_prime_stable(z,2)
        p=result.preactivation_gradients[k,:,:,coordinate]
        P=result.preactivation_error_bounds[k,:,:,coordinate]
        r=result.preactivation_hessians[k,:,:,0]
        R=result.preactivation_hessian_error_bounds[k,:,:,0]
        Q1,Q2=result.Q[:,k]
        paper=(a.abs()+Q1)*(r.abs()+R)-a.abs()*r.abs()
        paper+=(b.abs()+Q2)*(p.abs()+P)**2-b.abs()*p.abs()**2
        torch.testing.assert_close(paper,result.activation_hessian_error_bounds[k,:,:,0],atol=1e-12,rtol=1e-12)
    paper_output=(result.activation_hessian_error_bounds[-1,:,:,0]*net.output_layer.weight.abs()).sum(-1)
    torch.testing.assert_close(paper_output,var,atol=1e-12,rtol=1e-12)
    zero=kernel(net,centers,torch.zeros(2,dtype=torch.float64),multi_indices=[(coordinate,coordinate)])
    assert torch.equal(zero.second_derivative_bounds,torch.zeros_like(zero.second_derivative_bounds))
    np.testing.assert_allclose((base.abs()+var).numpy(),np.maximum((base+var).numpy(),-(base-var).numpy()),rtol=0,atol=0)

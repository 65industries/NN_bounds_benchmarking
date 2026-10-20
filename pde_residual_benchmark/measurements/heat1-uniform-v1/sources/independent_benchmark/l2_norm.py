"""Independent norm reference, extracted from the measured harness."""
import math
import numpy as np
import torch

def reference_l2(model, coord, clock, *, degree=200):
    """Tensor Gauss-Legendre reference with a half-order self-check."""
    if degree % 2:
        raise ValueError('degree must be even for the half-order self-check')
    from crown_benchmark.models import ExactSecondDerivative

    fn = (lambda p: model(p).flatten() if coord is None
          else ExactSecondDerivative(model, coord)(p).flatten())
    start = clock.mark()

    def quad(deg):
        q, w = np.polynomial.legendre.leggauss(deg)
        q, w = (q + 1) / 2, w / 2
        X, Y = np.meshgrid(q, q, indexing='ij')
        pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float64)
        with torch.no_grad():
            v = fn(pts).detach().numpy()
        return math.sqrt(math.fsum((np.outer(w, w).ravel() * v * v).tolist()))

    value, half = quad(degree), quad(degree // 2)
    return dict(value=value, half_value=half, degree=degree, half_degree=degree // 2,
                self_check_abs_difference=abs(value - half),
                computation_cpu_seconds=clock.elapsed(start),
                method='tensor Gauss-Legendre of g^2 on [0,1]^2, not a certified bound')
